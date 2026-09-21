#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""critical_path_5070.py — décomposition T_token + reconstruction du chemin critique
pour 5070 8 Go + XDNA2 (Ryzen AI 9 HX 365). Reprend les constantes de moe_axis_profiler.

T_token ≠ Σ segments : le chemin critique est le MAX des chemins du graphe de
dépendances (leçon #25859 : le chemin llama.cpp actuel est sérieux
router -> wait -> copy -> compute, le GPU idle pendant les H2D).

3 stratégies comparées :
  S0 série        : router -> H2D miss (WARM/COLD) -> GPU compute      (chemin upstream)
  S1 prefetch     : router(N) déclenche H2D(N+1) async -> GPU compute   (masque si t_compute >= t_h2d)
  S2 Voie A split : GPU = backbone + experts résidents ; NPU XDNA2 = overflow
                    sur-package (zéro PCIe) -> overlap double-flux

Ledger final : 5 leviers (CACHE, PREFETCH, PINNED, QUANT, NPU_SPLIT) en
Δchemin-critique vs S0, avec coûts explicites. Tout paramètre non mesuré = ASSUMED.
"""
import argparse
import json
import math
import sys

from moe_axis_profiler import (MACHINES, MODEL, GI,
                               budget, warm_tier, kv_gib, hit_skewed)

# Ancres / hypothèses (ASSUMED sauf mention)
T_ROUTER_LAYER_MS = 0.02     # dispatch router + argmax top-k par couche (ASSUMED)
T_SYNC_LAYER_MS = 0.005      # sync CPU->GPU par couche (ASSUMED)
H2D_PAGEABLE_GBS = 6.5       # ancre MaxDam
SYNC_COST_NPU_MS = 0.15      # coût dispatch+sync NPU par token (ASSUMED, à instrumenter XRT)
# DÉCOUPLAGE OP15 (2026-09-21) : plus AUCUNE constante OP15 en dur — 21.93 GB/s (FLM OP15)
# et overflow x16 (bench Phase 1 OP15/FLM) étaient des chiffres OP15. Ils doivent être
# fournis pour la cible HX 365 via flags (--npu-ddr-gbs / --npu-overflow-cost) ou
# microbench/sonde XRT. Sans eux : le segment NPU est marqué UNKNOWN, pas simulé.


def segments(m, args, hit, res):
    """Segments du graphe de dépendances par token (ms)."""
    per_exp = MODEL["expert_mb"][args.expert_fmt] * 1e6
    t_router = MODEL["n_layers"] * T_ROUTER_LAYER_MS
    t_sync = MODEL["n_layers"] * T_SYNC_LAYER_MS
    _, f_ram, h_warm, h_cold, t_warm, t_cold, _ = warm_tier(m, args, hit)
    t_h2d = t_warm + t_cold
    bytes_miss = (h_warm + h_cold) * MODEL["top_k"] * per_exp
    # GPU : backbone + experts résidents lus en BW
    act_gpu = MODEL["top_k"] * per_exp * hit + MODEL["dense_backbone_bytes"]
    t_gpu = act_gpu / (m["bw_eff_gbs"] * 1e9) * 1000
    # NPU : overflow sur-package (lit la RAM unifiée à npu_ddr_gbs — fourni par l'utilisateur/cible)
    t_npu = bytes_miss / (args.npu_ddr_gbs * 1e9) * 1000 + SYNC_COST_NPU_MS
    # Contre-factuel pageable : swap UNIQUEMENT les jambes H2D (WARM + jambe H2D du COLD),
    # la jambe SSD reste à ssd_gbs — sinon PINNED paraît négatif en régime COLD-dominant.
    bytes_warm = seg_warm = h_warm * MODEL["top_k"] * per_exp
    bytes_cold = h_cold * MODEL["top_k"] * per_exp
    t_h2d_pg = bytes_warm / (H2D_PAGEABLE_GBS * 1e9) * 1000 \
        + bytes_cold * (1.0 / (args.ssd_gbs * 1e9) + 1.0 / (H2D_PAGEABLE_GBS * 1e9)) * 1000
    return {"t_router": t_router, "t_sync": t_sync, "t_h2d": t_h2d,
            "t_h2d_pageable": t_h2d_pg, "t_gpu": t_gpu, "t_npu": t_npu,
            "h_warm": h_warm, "h_cold": h_cold, "bytes_miss": bytes_miss}


def strategies(seg, prefetch, npu_ok=True):
    """Chemins critiques des 3 stratégies (ms)."""
    s0 = seg["t_router"] + seg["t_h2d"] + seg["t_gpu"] + seg["t_sync"]
    if prefetch:
        # H2D(N+1) masqué dans le compute(N) si t_gpu >= t_h2d ; résiduel sinon
        s1 = seg["t_router"] + max(seg["t_gpu"], seg["t_h2d"]) + seg["t_sync"]
    else:
        s1 = s0
    # Voie A : GPU ne lit que résidents, NPU prend l'overflow en parallèle
    s2 = seg["t_router"] + max(seg["t_gpu"], seg["t_npu"]) + seg["t_sync"] if npu_ok else math.inf
    return {"S0_serial": s0, "S1_prefetch": s1, "S2_npu_split": s2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="5070", choices=list(MACHINES))
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--kv", default="turbo4", choices=["f16", "q8_0", "turbo4", "turbo3"])
    ap.add_argument("--expert-fmt", default="nvfp4", choices=["q4", "nvfp4"])
    ap.add_argument("--skew", type=float, default=0.85)
    ap.add_argument("--compute-buffer", dest="compute_buffer", type=float, default=0.5)
    ap.add_argument("--rail", type=float, default=0.3)
    ap.add_argument("--ram-gb", dest="ram_gb", type=float, default=32.0)
    ap.add_argument("--ssd-gbs", dest="ssd_gbs", type=float, default=5.0)
    ap.add_argument("--npu-ddr-gbs", dest="npu_ddr_gbs", type=float, default=None,
                    help="BW NPU HX 365 (GB/s) — OBLIGATOIRE pour S2 (aucune valeur OP15 par défaut)")
    ap.add_argument("--npu-overflow-cost", dest="npu_overflow_cost", type=float, default=None,
                    help="coût overflow NPU en x hit GPU — OBLIGATOIRE pour le ledger NPU_SPLIT")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    m = MACHINES[args.machine]
    # Garde-fou découplage : sans BW NPU fournie, S2/ledger NPU sont UNKNOWN (pas simulés)
    npu_ok = args.npu_ddr_gbs is not None and args.npu_ddr_gbs > 0
    if not npu_ok:
        print("[DECOUPLAGE] --npu-ddr-gbs absent -> S2 NPU-split = UNKNOWN (constante OP15 refusee, "
              "calibrer via sonde XRT / microbench cible)")
    if args.npu_overflow_cost is None:
        print("[DECOUPLAGE] --npu-overflow-cost absent -> ligne ledger NPU_SPLIT = UNKNOWN")
    cache_b, per_exp, res_max, kv = budget(m, args)
    res = min(int(res_max), MODEL["experts"])
    hit = hit_skewed(res, args.skew)
    seg = segments(m, args, hit, res)
    _, f_ram, _, _, _, _, _ = warm_tier(m, args, hit)
    strat = strategies(seg, prefetch=True, npu_ok=npu_ok)

    print(f"=== CRITICAL PATH 5070/XDNA2 — {m['label']} — {MODEL['label']} ===")
    print(f"ctx={args.ctx} KV={args.kv} experts={args.expert_fmt} skew={args.skew}"
          f" | résidence {res}/256, hit {hit:.2f} (WARM {seg['h_warm']*100:.0f}% / COLD {seg['h_cold']*100:.0f}%)")
    print("\n[DÉCOMPOSITION T_token — segments indépendants, PAS une somme]")
    print(f"  T_router (CPU, 40 couches)   {seg['t_router']:.2f} ms")
    print(f"  T_sync   (40 couches)        {seg['t_sync']:.2f} ms")
    print(f"  T_H2D    (miss WARM+COLD)    {seg['t_h2d']:.2f} ms  [pageable serait {seg['t_h2d_pageable']:.2f}]")
    print(f"  T_GPU    (backbone+résidents){seg['t_gpu']:.2f} ms")
    print(f"  T_NPU    (overflow XDNA2)    {seg['t_npu']:.2f} ms  [sur-package, overlap possible]")
    print("\n[CHEMIN CRITIQUE par stratégie]")
    for k, v in strat.items():
        tag = "  <- optimal" if v == min(strat.values()) else ""
        print(f"  {k:<14} {v:.2f} ms -> {1000/v:.1f} t/s{tag}")
    best = min(strat.values())
    print(f"\n[AXE 01] wall {sum(v for k, v in seg.items() if k.startswith('t_') and k != 't_h2d_pageable'):.2f} ms"
          f" vs critical path {best:.2f} ms -> overlap récupère"
          f" {100*(1-best/sum(v for k, v in seg.items() if k.startswith('t_') and k != 't_h2d_pageable')):.0f}%")

    print("\n[LEDGER — Δchemin critique vs S0 (coûts inclus)]")
    s0 = strat["S0_serial"]
    # CACHE : gain = CP(zéro résident, tout miss) − CP(courant). Honnête et vérifiable.
    miss_all = (1.0) * MODEL["top_k"] * per_exp
    t_h2d_all = miss_all / (m["h2d_pinned"] * 1e9) * 1000
    t_gpu_all = (MODEL["top_k"] * per_exp + MODEL["dense_backbone_bytes"]) / (m["bw_eff_gbs"] * 1e9) * 1000
    cp_no_cache = seg["t_router"] + t_h2d_all + t_gpu_all + seg["t_sync"]
    rows = [
        (f"CACHE ({res} résidents)", cp_no_cache - s0,
         f"coût: VRAM {cache_b/GI:.2f} GiB (opportunity) ; CP sans cache {cp_no_cache:.2f} ms"
         + (" [ATTENTION: contre-factuel suppose RAM>=pool — irréaliste ici, f_ram<1]" if f_ram < 1 else "")),
        ("PREFETCH (S1 vs S0)", s0 - strat["S1_prefetch"], "coût: PCIe gaspillé si prédiction fausse (usefulness à mesurer)"),
        ("PINNED (vs pageable)", seg["t_h2d_pageable"] - seg["t_h2d"], "coût: RAM pinned soustraite à l'OS (registration)"),
        ("QUANT (NVFP4 vs Q4)", None, "coût: requant + dequant compute (voir moe_axis_profiler ledger)"),
        ("NPU_SPLIT (S2 vs best)", (min(strat["S0_serial"], strat["S1_prefetch"])) - strat["S2_npu_split"],
         f"coût: sync NPU {SYNC_COST_NPU_MS} ms/tok + instabilité runtime (à instrumenter XRT)"),
    ]
    for name, gain, cost in rows:
        g = f"{gain:+.2f} ms" if gain is not None else "voir ledger moe_axis"
        print(f"  {name:<24} ΔCP {g:>16} | {cost}")

    if args.npu_overflow_cost is not None:
        print(f"\n[LEDGER — NPU_SPLIT coût overflow x{args.npu_overflow_cost} (fourni cible, à valider sonde XRT)]")
    else:
        print("\n[LEDGER — NPU_SPLIT coût overflow UNKNOWN (constante OP15 refusée)")
    print("\n[XDNA2 — axes à instrumenter sur cible (aucun mesuré ici)]")
    print("  xrt.dispatch, xrt.bo.alloc, dma.bytes, cpu/npu gap, overlap GPU/NPU réel")
    print("\n[PROVENANCE] T_ROUTER/T_SYNC/SYNC_NPU/NPU_DDR = ASSUMED ; h2d = ancre MaxDam ;"
          " NPU overflow x16 = bench.json Phase 1 (OP15/FLM, transposition HX 365 à valider)")


if __name__ == "__main__":
    main()
