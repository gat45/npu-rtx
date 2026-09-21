#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d2_expert_spectral_scan.py — scanner spectral + SNR, grain EXPERT (35B Genesis)

Fusion des outils D:\\lama-tensorRT (alpha_spectral_scanner, spectral_scanner_v3,
d2_qdf_bayesian_optimized, d2_engine_v14, d2_dynamic_proposal,
quant_graph_compiler_v2) avec le SNR exact de requant_experts.py.

1 seule lecture disque par expert → 2 signaux :
  - SNR exact   : quantize→dequantize vs source Q8_0/F16 (mesuré, pas théorique)
  - Spectral    : randomized SVD rank-32 (Halko) → decay(αw), gap, entropie,
                  r_eff, tail ratio, top25 (features QDF de d2_qdf_bayesian)

Optimisations mémoire (GUIDE_OPTIMISATION_MEMOIRE) :
  - GC forcé après chaque tenseur (pas par chunk)
  - pruning amont : seuls les ffn_*_exps sont scannés (93.5 % du poids)
  - jamais >1 expert (~1 Mo) + buffers SVD en RAM

Modes :
  (défaut)          : scan des couches --layers → JSON par expert
  --plan            : fusionne les chunks → politique par couche×famille
                      (fragile-fraction + switch penalty) → SORTIE md +
                      tensor_types_35b_plan.txt pour llama-quantize
"""
import argparse, gc, json, math, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from requant_experts import (parse_header, file_offsets, to_f32, packed_size,
                             enum_of, read_raw, expert_family)
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES

# ------------------------------------------------------------------- spectre
def randomized_svd_S(W, rank=32, n_iter=1, rng=None):
    """Spectre approximé (Halko) — d2_qdf_bayesian_optimized.OptimizedSVD."""
    if rng is None:
        rng = np.random.default_rng(0)
    m, n = W.shape
    r = min(rank, m, n)
    Omega = rng.standard_normal((n, r))
    Y = W @ Omega
    if n_iter:
        Y = W @ (W.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ W
    _, S, _ = np.linalg.svd(B, full_matrices=False)
    return S


def spectral_features(S):
    """Features QDF (d2_qdf_bayesian) + V14 (decay, gap, entropie)."""
    n = len(S)
    if n < 2:
        return None
    logr = np.log(np.arange(1, n + 1))
    logS = np.log(S + 1e-10)
    decay = -float(np.polyfit(logr, logS, 1)[0])          # αw
    gap = float((S[0] - S[1]) / (S[0] + 1e-9))            # spectral gap
    p = S ** 2 / (np.sum(S ** 2) + 1e-10)
    ent = float(-np.sum(p * np.log(p + 1e-10)) / np.log(n))
    r_eff = float(np.sum(S) ** 2 / (np.sum(S ** 2) + 1e-12))
    cum = np.cumsum(p)
    tail = float(np.sum(p[cum > 0.8]) /
                 (np.sum(p[(cum >= 0.3) & (cum <= 0.8)]) + 1e-9))
    top25 = float(np.sum(p[:max(1, n // 4)]))
    return dict(decay=round(decay, 3), gap=round(gap, 3), ent=round(ent, 3),
                r_eff=round(r_eff, 2), tail=round(tail, 3), top25=round(top25, 3))


def snr_of(ref, new_t):
    """SNR exact d'une matrice déjà en f32 (quantize→dequantize)."""
    q = quantize_bytes(ref, new_t)
    est = dequant_bytes(q, new_t)
    err = float(np.linalg.norm(ref - est))
    if err == 0.0:
        return 99.0
    return 20.0 * math.log10(max(float(np.linalg.norm(ref)), 1e-30) / err)


def quantize_bytes(x, ttype):
    from gguf.quants import quantize
    return quantize(x.astype(np.float32), ttype)


def dequant_bytes(q, ttype):
    from gguf.quants import dequantize
    return dequantize(q, ttype).astype(np.float32)


# ------------------------------------------------------------------ mode scan
RULES_DEFAULT = [("ffn_(gate|up)_exps", "Q4_0"), ("ffn_down_exps", "Q5_0")]


def run_scan(args):
    rules = [(rx, enum_of(tn)) for rx, tn in RULES_DEFAULT]
    layers = parse_layers(args.layers)
    info = parse_header(args.src)
    sizes = file_offsets(info)
    data_start = info["data_start"]
    f = open(args.src, "rb")
    out = {"src": args.src, "layers": {}}
    t0 = time.time()
    done = 0
    rng = np.random.default_rng(0)
    for (name, dims, dt, off), sz in zip(info["tensors"], sizes):
        fam = expert_family(name)
        if fam is None:
            continue
        m = re.search(r"blk\.(\d+)\.", name)
        if not m or int(m.group(1)) not in layers:
            continue
        src_t = GGMLQuantizationType(dt)
        new_t = dict(rules)[ [rx for rx, tn in rules if re.search(rx, name)][0] ] \
            if any(re.search(rx, name) for rx, _ in rules) else None
        if new_t is None:
            continue
        a, b, ne = dims
        n_exp = a * b
        pack = packed_size(n_exp, src_t)
        L = out["layers"].setdefault(str(int(m.group(1))), {}).setdefault(fam, {})
        snrs, decays, gaps, ents = [], [], [], []
        for e in range(ne):
            raw = read_raw(f, data_start, off + e * pack, pack)
            ref = to_f32(raw, src_t, n_exp)             # f32 (n_exp,)
            snrs.append(round(snr_of(ref, new_t), 2))
            W = ref.reshape(a, b)                       # matrice de l'expert
            S = randomized_svd_S(W, rank=32, n_iter=1, rng=rng)
            ft = spectral_features(S)
            decays.append(ft["decay"]); gaps.append(ft["gap"]); ents.append(ft["ent"])
            del raw, ref, W, S, ft
        L["snr"] = snrs; L["decay"] = decays; L["gap"] = gaps; L["ent"] = ents
        L["target"] = new_t.name
        done += 1
        print(f"  [{done}] {name}: snr {np.mean(snrs):.2f} · decay "
              f"{np.mean(decays):.2f} · gap {np.mean(gaps):.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        gc.collect()                                    # tactic 1 : GC par tenseur
    f.close()
    path = os.path.join(HERE, f"spectral_experts_{args.tag}.json")
    json.dump(out, open(path, "w"))
    print(f"[OK] {path} ({done} tenseurs, {time.time()-t0:.0f}s)")


def parse_layers(s):
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


# ------------------------------------------------------------------ mode plan
def load_chunks(tags):
    layers = {}
    for tag in tags:
        p = os.path.join(HERE, f"spectral_experts_{tag}.json")
        d = json.load(open(p))
        for blk, fams in d["layers"].items():
            layers.setdefault(blk, {}).update(fams)
    return layers


def fragile_mask(snr, floor):
    snr = np.asarray(snr)
    z = (snr - snr.mean()) / (snr.std() + 1e-9)
    return (z < -1.5) | (snr < floor)          # outlier local OU plancher absolu


FLOORS = {"gate": 19.5, "up": 19.5, "down": 26.5}
# politique (faible → agressif) ; rangs pour le lissage switch-penalty
AGGRESSIVE = {"gate": "Q4_K", "up": "Q4_K", "down": "Q4_K"}
MID = {"gate": "Q4_K", "up": "Q4_K", "down": "Q5_K"}
SAFE = {"gate": "Q8_0", "up": "Q8_0", "down": "Q8_0"}
RANK = {"Q4_K": 0, "Q5_K": 1, "Q8_0": 2}


def layer_policy(layers):
    """décision par couche×famille : fragile-fraction + lissage switch-penalty
    (quant_graph_compiler_v2)."""
    pol, fracs = {}, {}
    for fam in ("gate", "up", "down"):
        fracs[fam] = {}
        for blk in sorted(layers, key=int):
            if fam not in layers[blk]:
                continue
            mask = fragile_mask(layers[blk][fam]["snr"], FLOORS[fam])
            fracs[fam][blk] = float(mask.mean())
            pol[(fam, blk)] = (SAFE[fam] if fracs[fam][blk] > 0.15 else
                               MID[fam] if fracs[fam][blk] > 0.03 else
                               AGGRESSIVE[fam])
        # lissage : îlot cher isolé entre deux voisins agressifs et frac<0.05
        blks = sorted(fracs[fam], key=int)
        for i, blk in enumerate(blks):
            prev_b = blks[i - 1] if i else None
            next_b = blks[i + 1] if i + 1 < len(blks) else None
            cur = pol[(fam, blk)]
            neigh = [pol[(fam, x)] for x in (prev_b, next_b) if x is not None]
            if (neigh and all(RANK[n] < RANK[cur] for n in neigh)
                    and fracs[fam][blk] < 0.05):
                pol[(fam, blk)] = max(neigh, key=lambda t: RANK[t])
    return pol, fracs


BPW = {"Q4_K": 4.5, "Q5_K": 5.5, "Q8_0": 8.5}


def run_plan(args):
    layers = load_chunks(args.tags.split(","))
    pol, fracs = layer_policy(layers)
    info = parse_header(args.src)
    sizes = file_offsets(info)
    old_tot = new_tot = 0.0
    lines = []
    rows = []
    for (name, dims, dt, off), sz in zip(info["tensors"], sizes):
        old_tot += sz
        fam = expert_family(name)
        m = re.search(r"blk\.(\d+)\.", name)
        blk = m.group(1) if m else None
        if fam and blk and (fam, blk) in pol:
            tname = pol[(fam, blk)]
            ttype = enum_of(tname)
            n = int(np.prod(dims))
            new = packed_size(n, ttype)
        else:
            tname = None
            new = sz
        new_tot += new
        if tname:
            lines.append(f"{name}={tname}")
        rows.append((name, tname, sz, new))
    # Politique par couche (affichage)
    print("=== POLITIQUE PAR COUCHE×FAMILLE (fragile-fraction → type) ===")
    print(f"{'blk':>4} {'gate frac':>10} {'→':>6} {'up frac':>9} {'→':>6} "
          f"{'down frac':>10} {'→':>6}")
    for blk in sorted(layers, key=int):
        def fmt(fam):
            if fam not in layers[blk]:
                return "   —    "
            return f"{fracs[fam].get(blk, 0):.3f} {pol.get((fam, blk), '—'):>5}"
        print(f"{blk:>4} {fmt('gate'):>16} {fmt('up'):>15} {fmt('down'):>16}")
    print(f"\nSource {old_tot/2**30:.2f} GiB → plan spectral {new_tot/2**30:.2f} GiB")
    # fichier pour llama-quantize (tri longueur décroissante — leçon RAPPORT V4)
    lines.sort(key=len, reverse=True)
    tt = os.path.join(HERE, "tensor_types_35b_plan.txt")
    open(tt, "w").write("\n".join(lines) + "\n")
    print(f"[OK] {tt} ({len(lines)} règles)")
    # rapport md
    md = ["# 35B — plan spectral × SNR expert-grain", "",
          "Fusion: alpha_spectral_scanner + spectral_scanner_v3 + d2_qdf_bayesian",
          "(randomized SVD rank-32) + d2_dynamic_proposal (couplage) +",
          "quant_graph_compiler_v2 (switch penalty) sur requant_experts.py (SNR exact).", "",
          "Politique: fragile-fraction par couche×famille "
          "(z<-1.5 ou plancher abs) → Q8_0 >0.15, Q5_K/Q4_K sinon, lissage îlots.", ""]
    md.append("| blk | gate frac | gate | up frac | up | down frac | down |")
    md.append("|---:|---:|---|---:|---|---:|---|")
    for blk in sorted(layers, key=int):
        md.append(f"| {blk} | {fracs['gate'].get(blk,0):.3f} | {pol.get(('gate',blk),'—')} "
                  f"| {fracs['up'].get(blk,0):.3f} | {pol.get(('up',blk),'—')} "
                  f"| {fracs['down'].get(blk,0):.3f} | {pol.get(('down',blk),'—')} |")
    md += ["", f"**Taille plan : {new_tot/2**30:.2f} GiB** (source {old_tot/2**30:.2f})",
           "", "Rejouer :", "```bash",
           "llama-quantize --allow-requantize --tensor-type-file "
           "tensor_types_35b_plan.txt \\",
           '  "D:/Hermes...Q8_K_P.gguf" E:/oneplus/genesis_spectral_plan.gguf Q8_0 8',
           "```", ""]
    mdpath = os.path.join(HERE, "SORTIE_35B_SPECTRAL_PLAN.md")
    open(mdpath, "w", encoding="utf-8").write("\n".join(md))
    print(f"[OK] {mdpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:/Hermes3.6-35B-A3B-Uncensored-Genesis-Final-Q8_K_P.gguf")
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--tags", default="c0,c1,c2,c3")
    args = ap.parse_args()
    if args.plan:
        run_plan(args)
    else:
        run_scan(args)


if __name__ == "__main__":
    main()
