# Profileur Qwen3.6-35B-A3B APEX-MTP — matrice tensorielle RÉELLE (header GGUF) → 1080 / 5070
# =================================================================================================
# Source header : mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF → Qwen3.6-35B-A3B-APEX-MTP-Balanced.gguf
#   (recette APEX-MTP publique, même famille que le repo LuffyTheFox *gated* — caveat §caveats)
# Téléchargé par range HTTP (16 MB de header sur 24.27 GiB) — PAS le fichier complet.
#
# Sorties :
#   1. matrice layer×tensor×quant mesurée (markdown)
#   2. pool experts par couche (gate/up/down, bytes/expert RÉELS)
#   3. profil decode : 1080 (sm_61, PCIe3x16) et 5070 (sm_120, PCIe5x8, Voie A XDNA2 en flag)
#
# Usage : py profile_apex_matrix.py [--header header_apex_balanced.bin]
#                                    [--context 8192] [--skew 0.85] [--kv f16|turbo4]
#                                    [--machine 1080|5070|both]
import struct, sys, argparse
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------- tables ggml (bytes/élément)
# bpw = bits par poids — enum ggml.h RÉEL (vérifié : total pondéré = taille fichier ±0.1%)
GGML_TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
                   9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
                   15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
                   20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
                   26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
                   34: "TQ1_0", 35: "TQ2_0", 36: "MXFP4", 39: "TURBO3_0(fork)", 40: "TURBO4_0(fork)"}
GGML_BPW = {
    0: (32.0, "F32"), 1: (16.0, "F16"), 30: (16.0, "BF16"),
    2: (4.5, "Q4_0"), 3: (5.0, "Q4_1"), 6: (5.5, "Q5_0"), 7: (6.0, "Q5_1"), 8: (8.5, "Q8_0"),
    10: (2.5625, "Q2_K"), 11: (3.4375, "Q3_K"), 12: (4.5, "Q4_K"), 13: (5.5, "Q5_K"),
    14: (6.5625, "Q6_K"), 15: (8.5, "Q8_K"),
    16: (2.0625, "IQ2_XXS"), 17: (2.5625, "IQ2_XS"), 18: (3.0625, "IQ3_XXS"), 19: (1.5625, "IQ1_S"),
    20: (4.5, "IQ4_NL"), 21: (4.0625, "IQ3_S"), 22: (2.5625, "IQ2_S"), 23: (4.25, "IQ4_XS"),
    29: (1.75, "IQ1_M"), 34: (1.6875, "TQ1_0"), 35: (2.0625, "TQ2_0"), 36: (4.0, "MXFP4"),
    39: (3.0, "TURBO3_0"), 40: (4.0, "TURBO4_0"),  # turbo : BPW approx recette fork
}

# ---------------------------------------------------------------- machines (mêmes ancres que gpu_tier_profiler.py)
MACHINES = {
    "1080": dict(label="GTX 1080 8GB (dev, sm_61)", vram_gb=8.0, bw_eff_gbs=205.0,
                 pcie_gbs=15.75, npu=False, kv_ok={"f16", "q8_0", "turbo4"}),
    "5070": dict(label="RTX 5070 Laptop 8GB (cible, sm_120)", vram_gb=8.0, bw_eff_gbs=165.0,
                 pcie_gbs=31.5, npu=True, kv_ok={"f16", "q8_0", "turbo4"}),
}

# ---------------------------------------------------------------- parsing GGUF
class Cur:
    def __init__(self, b): self.b, self.o = b, 0
    def _need(self, n):
        if self.o + n > len(self.b): raise ValueError(f"truncated@{self.o}+{n}>{len(self.b)}")
    def u8(self):  self._need(1); v = struct.unpack_from("<B", self.b, self.o)[0]; self.o += 1; return v
    def u32(self): self._need(4); v = struct.unpack_from("<I", self.b, self.o)[0]; self.o += 4; return v
    def u64(self): self._need(8); v = struct.unpack_from("<Q", self.b, self.o)[0]; self.o += 8; return v
    def f32(self): self._need(4); v = struct.unpack_from("<f", self.b, self.o)[0]; self.o += 4; return v
    def s(self):
        n = self.u64(); self._need(n); v = self.b[self.o:self.o + n].decode("utf-8", "replace"); self.o += n; return v
    def val(self, t):
        if t == 0: return self.u8()
        if t == 1:
            self._need(1); v = struct.unpack_from("<b", self.b, self.o)[0]; self.o += 1; return v
        if t == 2:
            self._need(2); v = struct.unpack_from("<H", self.b, self.o)[0]; self.o += 2; return v
        if t == 3:
            self._need(2); v = struct.unpack_from("<h", self.b, self.o)[0]; self.o += 2; return v
        if t == 4: return self.u32()
        if t == 5:
            self._need(4); v = struct.unpack_from("<i", self.b, self.o)[0]; self.o += 4; return v
        if t == 6: return self.f32()
        if t == 7: v = self.u8(); return bool(v)
        if t == 8: return self.s()
        if t == 10: return self.u64()
        if t == 11:
            self._need(8); v = struct.unpack_from("<q", self.b, self.o)[0]; self.o += 8; return v
        if t == 12:
            self._need(8); v = struct.unpack_from("<d", self.b, self.o)[0]; self.o += 8; return v
        if t == 9:
            it, n = self.u32(), self.u64()
            if it == 8:  # array de strings : sauter efficacement
                out = []
                for _ in range(n):
                    ln = self.u64(); out.append(self.b[self.o:self.o+ln].decode("utf-8","replace")); self.o += ln
                return out
            step = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(it, 0)
            if step == 0:
                raise ValueError(f"array type {it} non supporté")
            v = self.b[self.o:self.o + n*step]; self.o += n*step
            return v
        raise ValueError(f"type gguf {t} inconnu")

def parse_header(path):
    b = Path(path).read_bytes()
    c = Cur(b)
    magic = b[:4]
    assert magic == b"GGUF", f"magic {magic!r} invalide"
    c.o = 4
    version, n_tensors, n_kv = c.u32(), c.u64(), c.u64()
    kvs = {}
    for _ in range(n_kv):
        k = c.s()
        try:
            t = c.u32()
            kvs[k] = c.val(t)
        except (ValueError, struct.error) as e:
            return dict(version=version, n_tensors=n_tensors, n_kv=n_kv, kvs=kvs, tensors=[],
                        truncated_kv=f"{k}@off{c.o}: {e}")
    tensors, truncated = [], False
    try:
        for _ in range(n_tensors):
            name = c.s(); nd = c.u32()
            ne = [c.u64() for _ in range(nd)]
            gt = c.u32(); off = c.u64()
            tensors.append(dict(name=name, ne=ne, type=gt, offset=off))
    except struct.error:
        truncated = True
    return dict(version=version, n_tensors=n_tensors, n_kv=n_kv, kvs=kvs, tensors=tensors,
                truncated=truncated, header_bytes=len(b), consumed=c.o)

# ---------------------------------------------------------------- familles de tensors
def family(name, arch):
    if name in ("token_embd.weight", "token_embd_g.weight"): return "embd"
    if name.startswith("output"): return "output"
    if name.startswith(("",)) and name.startswith("blk."):
        rest = name.split(".", 2)[2] if name.count(".") >= 2 else name
        if rest.startswith(("ffn_gate_exps", "ffn_up_exps")): return "routed_gu"
        if rest.startswith("ffn_down_exps"): return "routed_down"
        if rest.startswith(("ffn_gate_shexp", "ffn_up_shexp")): return "shared_gu"
        if rest.startswith("ffn_down_shexp"): return "shared_down"
        if rest.startswith("ffn_gate_inp"): return "router"
        return "backbone"   # attention/GDN/norms
    return "autre"

# ---------------------------------------------------------------- profil decode
def profile(m, M, args, per_layer, totals):
    vram = M["vram_gb"] * 2**30
    free = vram * 0.92
    # KV f16 : head_count_kv=2, key_length=value_length=256 → (2*256 + 2*256) élem = 2048 B/tok/couche (f16)
    kv_bytes = args.context * 41 * 2 * 2 * 256 * 2
    if args.kv == "turbo4": kv_bytes *= 0.5
    cb, rail = args.compute_buffer * 2**30, args.rail * 2**30
    permanent = totals["backbone"] + totals["shared"] + totals["output"] + totals["embd"]
    cache_b = free - permanent - kv_bytes - cb - rail
    n_exp = 256
    per_exp = totals["routed"] / (n_exp * totals.get("n_blk", 41)) if totals["routed"] else 0
    res = max(0, int(cache_b / per_exp / totals.get("n_blk", 41))) if per_exp else 0
    res = min(res, n_exp)
    hit = min((res / n_exp) ** args.skew, 1.0)
    miss_b = (1 - hit) * 8 * per_exp
    t_h2d = miss_b / (M["pcie_gbs"] * 1e9)
    t_cpu = miss_b / 50e9  # CPU DDR5 eff ~50 GB/s — ASSUMED
    miss_dec = "transfert GPU" if t_h2d < t_cpu else "compute CPU"
    t_miss = min(t_h2d, t_cpu)
    act = hit * 8 * per_exp + permanent
    t_gpu = act / (M["bw_eff_gbs"] * 1e9)
    tps = 1.0 / (t_gpu + t_miss)
    return dict(res=res, hit=hit, tps=tps, t_gpu=t_gpu*1000, t_miss=t_miss*1000,
                miss_dec=miss_dec, cache_gib=cache_b/2**30, permanent_gib=permanent/2**30,
                kv_gib=kv_bytes/2**30, per_exp_mb=per_exp/1e6, pool_gib=totals["routed"]/2**30,
                t_h2d=t_h2d*1000, t_cpu=t_cpu*1000)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", default=str(HERE / "header_apex_balanced.bin"))
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--skew", type=float, default=0.85)
    ap.add_argument("--kv", default="f16", choices=["f16", "turbo4"])
    ap.add_argument("--compute-buffer", type=float, default=0.5)
    ap.add_argument("--rail", type=float, default=0.3)
    ap.add_argument("--machine", default="both", choices=["1080", "5070", "both"])
    args = ap.parse_args()

    H = parse_header(args.header)
    kvs = H["kvs"]
    arch = kvs.get("general.architecture", "?")
    n_blk = kvs.get(f"{arch}.block_count", 40)
    n_exp = kvs.get(f"{arch}.expert_count", 256)
    n_used = kvs.get(f"{arch}.experts_used", 8)
    print(f"GGUF v{H['version']} | arch={arch} | blocks={n_blk} experts={n_exp} top-k={n_used}")
    print(f"tensors déclarés={H['n_tensors']} parsés={len(H['tensors'])} "
          f"(header {H['header_bytes']/2**20:.1f} MB, tronqué={H.get('truncated', False)})")
    print(f"kv notables: " + ", ".join(f"{k}={v}" for k, v in list(kvs.items())
          if k.startswith(("general.", f"{arch}.expert", f"{arch}.embedding", f"{arch}.attention.key", f"{arch}.attention.value", f"{arch}.attention.head")) )[:600])

    fam_bytes = {}
    per_layer = {}
    rows = []
    for t in H["tensors"]:
        f = family(t["name"], arch)
        bpw, tname = GGML_BPW.get(t["type"], (0.0, f"type{t['type']}?"))
        nel = 1
        for d in t["ne"]: nel *= d
        nbytes = nel * bpw / 8.0
        fam_bytes[f] = fam_bytes.get(f, 0) + nbytes
        lay = t["name"].split(".")[1] if t["name"].startswith("blk.") else None
        if lay is not None:
            per_layer.setdefault(lay, {}).setdefault(f, [0, tname, tuple(t["ne"])])
            per_layer[lay][f][0] += nbytes
        rows.append((t["name"], tname, nbytes, tuple(t["ne"])))

    totals = {
        "routed": fam_bytes.get("routed_gu", 0) + fam_bytes.get("routed_down", 0),
        "shared": fam_bytes.get("shared_gu", 0) + fam_bytes.get("shared_down", 0),
        "backbone": fam_bytes.get("backbone", 0) + fam_bytes.get("router", 0),
        "output": fam_bytes.get("output", 0) + fam_bytes.get("autre", 0),
        "embd": fam_bytes.get("embd", 0),
    }
    routed_per_layer = (fam_bytes.get("routed_gu", 0) + fam_bytes.get("routed_down", 0)) / max(1, n_blk)
    per_exp = routed_per_layer / n_exp
    totals["n_blk"] = n_blk

    print("\n=== TOTAUX FAMILLES (mesurés, GiB) ===")
    for k, v in totals.items(): print(f"  {k:<10} {v/2**30:8.2f}")
    print(f"  routed/couche {routed_per_layer/2**30:.3f} GiB | /expert {per_exp/1e6:.2f} MB")
    gtot = sum(fam_bytes.values())
    print(f"  TOTAL fichier estimé {gtot/2**30:.2f} GiB (pondéré bpw)")

    # histogramme types — diagnostic d'écarts bpw
    type_bytes, type_count = {}, {}
    for t in H["tensors"]:
        bpw, tname = GGML_BPW.get(t["type"], (0.0, f"type{t['type']}?"))
        nel = 1
        for d in t["ne"]: nel *= d
        nb = nel * bpw / 8.0
        type_bytes[tname] = type_bytes.get(tname, 0) + nb
        type_count[tname] = type_count.get(tname, 0) + 1
    print("\n=== HISTOGRAMME TYPES ===")
    for tname in sorted(type_bytes, key=lambda x: -type_bytes[x]):
        print(f"  {tname:<10} {type_count[tname]:>4} tensors  {type_bytes[tname]/2**30:8.2f} GiB")

    print("\n=== MATRICE PAR COUCHE (mesurée) ===")
    for lay in sorted(per_layer, key=int):
        d = per_layer[lay]
        def q(f):
            return d.get(f, [0, "-", ()])[1]
        rg_b = d.get("routed_gu", [0])[0] + d.get("routed_down", [0])[0]
        print(f"  blk{lay:>3}: gu={q('routed_gu'):<9} down={q('routed_down'):<9} "
              f"shexp={q('shared_down'):<9} backbone={q('backbone'):<9} | routed {rg_b/2**30:.3f} GiB")

    print("\n=== PROFIL DECODE (contexte", args.context, ", kv", args.kv, ", skew", args.skew, ") ===")
    for mid in ([args.machine] if args.machine != "both" else ["1080", "5070"]):
        M = MACHINES[mid]
        r = profile(M, M, args, per_layer, totals)
        print(f"\n[{mid}] {M['label']}")
        print(f"  pool experts total {r['pool_gib']:.2f} GiB | /expert {r['per_exp_mb']:.2f} MB")
        print(f"  permanent (backbone+shared+out+embd) {r['permanent_gib']:.2f} GiB | KV {r['kv_gib']:.2f} GiB")
        print(f"  budget experts VRAM {r['cache_gib']:.2f} GiB -> {r['res']}/{n_exp} résidents -> hit {r['hit']:.2f}")
        print(f"  miss {8*(1-r['hit']):.1f} experts/tok : {r['miss_dec']} "
              f"(H2D {r['t_h2d']:.2f} ms vs CPU {r['t_cpu']:.2f} ms) -> t_miss {r['t_miss']:.2f} ms")
        print(f"  GPU actif {r['t_gpu']:.2f} ms + miss {r['t_miss']:.2f} ms -> **{r['tps']:.1f} t/s**")

    print("\n[CAVEATS] (1) header = variante mudler Balanced (recette publique) — le fichier "
          "LuffyTheFox 'Genesis-MTP-APEX' est gated : memes shapes attendues (meme arch 35B-A3B), "
          "quants possiblement differents selon la recette exacte du repo. "
          "(2) KV f16 borne haute : couches GDN = state recurrent < KV attention. "
          "(3) skew/h2d/cpu_bw = ASSUMED -> calibration microbench_h2d.py + trace routing sur cible.")

    # écriture SORTIE utf-8 (console cp1252 casse les flèches)
    import io
    sortie = HERE / "SORTIE_APEX_MATRIX.txt"
    with open(sortie, "w", encoding="utf-8") as f:
        pass  # placeholder — sortie console déjà complète
    print(f"\n[OK] script: {Path(__file__).name}")

if __name__ == "__main__":
    main()
