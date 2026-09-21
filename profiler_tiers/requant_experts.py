#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
requant_experts.py — requantizer GGUF **expert-grain** pour le 35B Q8_K_P.

Méthode d2_layer_profiler_v3 (streaming : jamais >1 tenseur chargé ×3 buffers),
étendue au GRAIN EXPERT :
  - SNR EXACT par expert (quantize→dequantize gguf-py vs source Q8_0/F16),
    agrégé par couche (mean/min/argmin sur les 256 experts)
  - tailles candidats exactes (n//blk)*sz par tenseur
  - --write : GGUF complet avec KVs copiés OCTET POUR OCTET (section KV
    reproduite verbatim), seuls les tenseurs ciblés sont requantizés.

Règles : --tensor-type "REGEX=TYPE" (répétable) · --exclude REGEX
Types dispo : Q4_K Q5_K Q6_K Q8_0 TQ3_1S TQ4_1S (enums vérifiés du fork)

Exemples :
  # SNR candidat M1 (gate/up → Q4_K), couches 0-7 :
  py requant_experts.py --src "D:/Hermes...Q8_K_P.gguf" \
      --tensor-type "ffn_(gate|up)_exps=Q4_K" --snr --layers 0-7
  # écriture du candidat (KVs identiques) :
  py requant_experts.py ... --write "E:/oneplus/genesis_expQ4K.gguf"
"""
import argparse, json, math, os, re, struct, sys, time
import numpy as np

GGUF_DIR = r"E:\oneplus\geniex_harness\npu-rtx\reference\llama-cpp-turboquant\gguf-py"
if GGUF_DIR not in sys.path:
    sys.path.insert(0, GGUF_DIR)
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
from gguf.quants import quantize, dequantize

ALIAS = {n: n for n in ("F32", "F16", "Q8_0", "Q4_0", "Q5_0", "Q4_K",
                        "Q5_K", "Q6_K", "TQ3_1S", "TQ4_1S")}


def enum_of(name):
    u = name.upper()
    if u not in ALIAS:
        raise SystemExit(f"[X] type inconnu: {name} (dispo: {', '.join(ALIAS)})")
    return GGMLQuantizationType[u]


def bpw(ttype):
    blk, sz = GGML_QUANT_SIZES[ttype]
    return sz * 8.0 / blk


def packed_size(n_elems, ttype):
    blk, sz = GGML_QUANT_SIZES[ttype]
    if n_elems % blk:
        raise ValueError(f"n_elems {n_elems} pas multiple de {blk} ({ttype})")
    return (n_elems // blk) * sz


def read_raw(f, data_start, off, size):
    f.seek(data_start + off)
    d = f.read(size)
    if len(d) != size:
        raise IOError("lecture courte")
    return d


def to_f32(raw, ttype, n):
    if ttype == GGMLQuantizationType.F32:
        return np.frombuffer(raw, dtype=np.float32, count=n).astype(np.float32)
    if ttype == GGMLQuantizationType.F16:
        return np.frombuffer(raw, dtype=np.float16, count=n).astype(np.float32)
    return dequantize(np.frombuffer(raw, dtype=np.uint8), ttype).astype(np.float32)


# ----------------------------------------------------------------- parsing GGUF
def parse_header(src):
    """Parse magic/version/counts + tous les KVs + toutes les tensor infos.
    Retourne dict avec section KV [kv_start, ti_start) pour copie verbatim."""
    f = open(src, "rb")
    fsize = os.path.getsize(src)
    magic = f.read(4)
    assert magic == b"GGUF", "pas un GGUF"
    version, n_tensors, n_kvs = struct.unpack("<IQQ", f.read(20))
    kv_start = f.tell()

    def rd_str():
        (ln,) = struct.unpack("<Q", f.read(8))
        return f.read(ln)

    def skip_value(vt):
        if vt == 8:                      # STRING
            rd_str()
        elif vt == 9:                    # ARRAY
            (st,) = struct.unpack("<I", f.read(4))
            (cnt,) = struct.unpack("<Q", f.read(8))
            for _ in range(cnt):
                skip_value(st)
        else:
            size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                    10: 8, 11: 8, 12: 8}[vt]
            f.read(size)

    kvs = []
    for _ in range(n_kvs):
        key = rd_str().decode("utf-8")
        (vt,) = struct.unpack("<I", f.read(4))
        kvs.append((key, vt))
        skip_value(vt)
    ti_start = f.tell()

    tensors = []   # (name, dims list, dtype, offset)
    for _ in range(n_tensors):
        name = rd_str().decode("utf-8")
        (nd,) = struct.unpack("<I", f.read(4))
        dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
        (dt,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        tensors.append((name, dims, dt, off))
    data_start = f.tell()
    pad = (-data_start) % 32
    data_start += pad
    f.close()
    return dict(fsize=fsize, version=version, n_tensors=n_tensors, n_kvs=n_kvs,
                kv_start=kv_start, ti_start=ti_start, data_start=data_start,
                tensors=tensors, kv_pairs=kvs)


def file_offsets(info):
    """Tailles sur disque par tenseur : gap offsets, dernier -> fin de données."""
    offs = [t[3] for t in info["tensors"]]
    ds = info["data_start"]
    sizes = []
    for i in range(len(offs)):
        end = offs[i + 1] if i + 1 < len(offs) else info["fsize"] - ds
        sizes.append(end - offs[i])
    return sizes


# ------------------------------------------------------------- plan de requant
def build_plan(info, sizes, rules, excludes):
    plan = []  # (name, dims, src_dtype, new_dtype|None, src_size, new_size)
    for (name, dims, dt, off), sz in zip(info["tensors"], sizes):
        n = int(np.prod(dims))
        src_t = GGMLQuantizationType(dt)
        new_t = None
        if not any(re.search(x, name) for x in excludes):
            for rx, ttype in rules:
                if re.search(rx, name):
                    if n % GGML_QUANT_SIZES[ttype][0] == 0 and src_t not in (
                            GGMLQuantizationType.F32,):
                        new_t = ttype
                    break
        new_sz = packed_size(n, new_t) if new_t else sz
        plan.append(dict(name=name, dims=dims, src=src_t, new=new_t,
                         src_sz=sz, new_sz=new_sz, n=n))
    return plan


# ------------------------------------------------------- SNR grain expert
def snr_expert(raw_exp, src_t, new_t, n_exp_elems):
    """SNR exact d'UN expert : quantize→dequantize vs source (Q8_0/F16)."""
    ref = to_f32(raw_exp, src_t, n_exp_elems)
    q = quantize(ref, new_t)
    est = dequantize(q, new_t).astype(np.float32)
    err = ref - est
    nr = float(np.linalg.norm(ref))
    ne_ = float(np.linalg.norm(err))
    return 99.0 if ne_ == 0 else 20.0 * math.log10(max(nr, 1e-30) / ne_)


def expert_family(name):
    if "ffn_down_exps" in name:
        return "down"
    if "ffn_gate_exps" in name:
        return "gate"
    if "ffn_up_exps" in name:
        return "up"
    return None


# --------------------------------------------------------------------- mode SNR
def run_snr(args, plan, info):
    """SNR EXACT par expert : streaming (1 expert ~1 Mo en RAM à la fois),
    agrégé par (couche, famille gate/up/down)."""
    data_start = info["data_start"]
    layers = parse_layers_arg(args.layers)
    src_f = open(args.src, "rb")
    by_layer = {}
    t0 = time.time()
    done = 0
    for p in plan:
        if p["new"] is None:
            continue
        fam = expert_family(p["name"])
        if fam is None:
            continue
        m = re.search(r"blk\.(\d+)\.", p["name"])
        if not m:
            continue
        blk = int(m.group(1))
        if layers and blk not in layers:
            continue
        if not re.search(args.match, p["name"]):
            continue
        # dims (a, b, ne) : expert en dernière dim ; slice 3D -> expert contigu
        a, b, ne = p["dims"]
        n_exp = a * b                       # éléments par expert
        L = by_layer.setdefault(blk, {})
        snrs = []
        src_t = p["src"]
        pack_bytes_per_exp = packed_size(n_exp, src_t)
        for e in range(ne):
            off_e = offset_of(info, p["name"]) + e * pack_bytes_per_exp
            raw_e = read_raw(src_f, data_start, off_e, pack_bytes_per_exp)
            snrs.append(snr_expert(raw_e, src_t, p["new"], n_exp))
        snr = np.array(snrs)
        glob = float(snr.mean())
        L.setdefault(fam, []).append((p["name"], snr, glob))
        done += 1
        print(f"  [{done}] {p['name']}: mean {snr.mean():.2f} dB · "
              f"min {snr.min():.2f} (e{int(snr.argmin())}) · "
              f"p5 {np.percentile(snr, 5):.2f} "
              f"({time.time()-t0:.0f}s cumulés)", flush=True)
        del snr
    src_f.close()

    print("\n=== SNR EXPERT-GRAIN PAR COUCHE (dB, plus haut = meilleur) ===")
    print(f"{'blk':>4} {'fam':<5} {'mean':>7} {'min':>7} {'argmin':>7} {'p5':>7} {'n':>4}")
    worst = []
    for blk in sorted(by_layer):
        for fam in ("gate", "up", "down"):
            if fam not in by_layer[blk]:
                continue
            entries = by_layer[blk][fam]
            allsnr = np.concatenate([s for _, s, _ in entries])
            worst_e = int(allsnr.argmin())
            gmean = float(allsnr.mean())
            print(f"{blk:>4} {fam:<5} {gmean:7.2f} {allsnr.min():7.2f} "
                  f"e{worst_e:>4} {np.percentile(allsnr, 5):7.2f} "
                  f"{len(allsnr):4d}")
            worst.append((float(allsnr.min()), blk, fam, worst_e))
    worst.sort()
    print("\n=== TOP 10 EXPERTS LES PLUS FRAGILES (à garder en haute précision) ===")
    for v, blk, fam, e in worst[:10]:
        print(f"  blk.{blk:<2} {fam:<5} expert {e:<4} SNR {v:.2f} dB")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"snr_experts_{args.tag}.json")
    json.dump({
        "src": args.src, "tag": args.tag,
        "rules": [f"{rx}={t.name}" for rx, t in parse_rules(args.tensor_type)],
        "per_layer": {str(b): {fam: [{"tensor": n,
                                      "mean": float(s.mean()),
                                      "min": float(s.min()),
                                      "argmin": int(s.argmin()),
                                      "p5": float(np.percentile(s, 5)),
                                      "global": float(g)}
                                     for n, s, g in ent]
                               for fam, ent in L.items()}
                      for b, L in by_layer.items()},
    }, open(out, "w"), indent=1)
    print(f"\n[OK] {out}")


def offset_of(info, name):
    for t in info["tensors"]:
        if t[0] == name:
            return t[3]
    raise KeyError(name)


def parse_rules(specs):
    rules = []
    for r in specs:
        if "=" not in r:
            raise SystemExit(f"[X] --tensor-type attend REGEX=TYPE : {r}")
        rx, tn = r.split("=", 1)
        rules.append((rx, enum_of(tn)))
    return rules


def parse_layers_arg(s):
    if not s:
        return set()
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


# ------------------------------------------------------------------ mode WRITE
def run_write(args, plan, info):
    out = os.path.abspath(args.write)
    est = sum(p["new_sz"] for p in plan)
    n_req = sum(1 for p in plan if p["new"])
    print(f"[write] {out}")
    print(f"  source {info['fsize']/2**30:.2f} GiB -> sortie {est/2**30:.2f} GiB "
          f"({n_req} tenseurs requantizés)")
    if est / 2**30 > args.free_gib:
        raise SystemExit(f"[X] sortie {est/2**30:.1f} > {args.free_gib} GiB libres — annulé")

    src = open(args.src, "rb")
    dst = open(out, "wb")
    # 1) header
    dst.write(b"GGUF")
    dst.write(struct.pack("<IQQ", 3, info["n_tensors"], info["n_kvs"]))
    # 2) KVs VERBATIM
    src.seek(info["kv_start"])
    remaining = info["ti_start"] - info["kv_start"]
    while remaining:
        chunk = src.read(min(1 << 20, remaining))
        dst.write(chunk)
        remaining -= len(chunk)
    # 3) tensor infos reconstruites + offsets séquentiels
    off = 0
    infos = []
    for p in plan:
        infos.append((p["name"], p["dims"],
                      (p["new"] or p["src"]).value if hasattr(p["new"] or p["src"], "value") else int(p["new"] or p["src"]),
                      off))
        off += p["new_sz"]
    for name, dims, dt, o in infos:
        nb = name.encode("utf-8")
        dst.write(struct.pack("<Q", len(nb)))
        dst.write(nb)
        dst.write(struct.pack("<I", len(dims)))
        dst.write(struct.pack(f"<{len(dims)}Q", *[int(d) for d in dims]))
        dst.write(struct.pack("<I", int(dt)))
        dst.write(struct.pack("<Q", o))
    pos = dst.tell()
    pad = (-pos) % 32
    if pad:
        dst.write(b"\x00" * pad)
    # 4) données
    data_start = info["data_start"]
    t0 = time.time()
    for i, p in enumerate(plan):
        src.seek(data_start + offset_of(info, p["name"]))
        if p["new"] is None:
            remaining = p["src_sz"]
            while remaining:
                chunk = src.read(min(1 << 22, remaining))
                dst.write(chunk)
                remaining -= len(chunk)
        else:
            raw = src.read(p["src_sz"])
            ref = to_f32(raw, p["src"], p["n"])
            q = quantize(ref, p["new"])
            dst.write(np.ascontiguousarray(q).tobytes())
            del raw, ref, q
        if (i + 1) % 50 == 0 or i == len(plan) - 1:
            el = time.time() - t0
            print(f"  [{i+1}/{len(plan)}] {p['name']} "
                  f"{'-> ' + p['new'].name if p['new'] else 'copy'} "
                  f"({el:.0f}s, ETA {el/(i+1)*(len(plan)-i-1):.0f}s)", flush=True)
    src.close()
    dst.close()
    print(f"[OK] {out} = {os.path.getsize(out)/2**30:.2f} GiB (estimé {est/2**30:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tensor-type", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--snr", action="store_true")
    ap.add_argument("--match", default=r"ffn_(gate|up|down)_exps")
    ap.add_argument("--layers", default="")
    ap.add_argument("--write", default=None)
    ap.add_argument("--free-gib", type=float, default=38.0)
    ap.add_argument("--tag", default="cand")
    args = ap.parse_args()

    rules = parse_rules(args.tensor_type)
    info = parse_header(args.src)
    sizes = file_offsets(info)
    plan = build_plan(info, sizes, rules, args.exclude)
    old = sum(p["src_sz"] for p in plan)
    new = sum(p["new_sz"] for p in plan)
    pool_old = sum(p["src_sz"] for p in plan if "exps" in p["name"])
    pool_new = sum(p["new_sz"] for p in plan if "exps" in p["name"])
    print(f"source {old/2**30:.2f} GiB (fichier {info['fsize']/2**30:.2f}) -> "
          f"candidat {new/2**30:.2f} GiB")
    print(f"pool experts {pool_old/2**30:.2f} -> {pool_new/2**30:.2f} GiB")
    print("règles:", [(rx, t.name) for rx, t in rules], "exclu:", args.exclude)

    if args.write:
        run_write(args, plan, info)
    elif args.snr:
        run_snr(args, plan, info)
    else:
        print("(ajoute --snr ou --write)")


if __name__ == "__main__":
    main()
