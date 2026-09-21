#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""moe_axis_profiler.py — profiler multi-axes MoE pour 5070 8 Go + 35B-A3B.
Reprend les principes validés de la littérature/communauté (voir AXES_5070.md) :
  1. budget expert = free_vram − permanent − KV − compute_buffer − rail (leçon #24528)
  2. hit skewed (2 mondes : localité forte vs routing uniforme)
  3. H2D pinned vs pageable (leçon MaxDam : 6.5 → 20 GB/s)
  4. pipeline prefetch N+1 masqué sur t_compute (leçon #25859/wackMall)
  5. miss → décision transfert vs compute CPU (RFC #24528)
  6. ledger NET_HARDWARE_GAIN + marginal dG/dCache + Pareto
  7. tier WARM = pinned RAM en cache L2 explicite ; éviction VRAM→RAM GRATUITE
     (poids read-only : la copie RAM reste intacte, zéro DMA d'éviction) ;
     le tier SSD n'est entré QUE si le pool experts dépasse la RAM (leçon
     SSD-LLaMA/mmap : SSD = capacité, pas chemin de decode)
Usage : py moe_axis_profiler.py [--ctx 8192] [--kv turbo4] [--expert-fmt nvfp4]
          [--skew 0.6] [--compute-buffer 0.5] [--rail 0.3] [--ram-gb 32] [--ssd-gbs 5]
"""
import argparse
import json
import math
import sys

# --- profils machine (cohérents gpu_tier_profiler.py) ---
MACHINES = {
    "5070": {"label": "RTX 5070 Laptop 8 Go (cible)", "vram_gb": 8.0,
             "bw_eff_gbs": 165.0, "pcie_gbs": 31.5, "h2d_pageable": 6.5, "h2d_pinned": 20.0,
             "pcie_gen": 5.0, "lanes": 8},
    "1080": {"label": "GTX 1080 8 Go (dev)", "vram_gb": 8.0,
             "bw_eff_gbs": 205.0, "pcie_gbs": 15.75, "h2d_pageable": 6.0, "h2d_pinned": 12.0,
             "pcie_gen": 3.0, "lanes": 16},
}
MODEL = {"label": "Qwen3.5/3.6-35B-A3B", "n_layers": 40, "experts": 256, "top_k": 8,
         "expert_mb": {"q4": 2.45, "nvfp4": 1.25},
         "permanent_gib": 1.5,          # emb + LM head (vocab 248k) + 30 GDN + 10 attn + 40 routers + 40 shared experts
         "dense_backbone_bytes": 1.5e9, # 85.2 % du trafic (vLLM #51197)
         "compute_buffer_gib": 0.5, "rail_gib": 0.3}
# DÉCOUPLAGE OP15 (2026-09-21) : 16× et 61.5 t/s venaient de la simu Phase 1 calée
# sur OP15/FLM (Snapdragon/HTP) — MATÉRIEL DIFFÉRENT de XDNA2 (HX 365). Plus de
# valeur par défaut : fournie via --npu-overflow-cost (sonde XRT cible), sinon UNKNOWN.
NPU_OVERFLOW_COST = None  # ex-16.0 (OP15) — refusé en dur
GI = 1024 ** 3


def budget(m, args):
    """Leçon #24528 : le budget experts soustrait TOUT, pas juste la KV."""
    kv = kv_gib(args)
    free = m["vram_gb"] * 0.92 * GI
    cache_bytes = free - MODEL["permanent_gib"] * GI - kv * GI \
        - args.compute_buffer * GI - args.rail * GI
    per_exp = MODEL["expert_mb"][args.expert_fmt] * 1e6
    return max(cache_bytes, 0.0), per_exp, cache_bytes / per_exp / MODEL["n_layers"], kv


def kv_gib(args):
    per_tok = 2 * 2 * 4 * 128 * 2  # K+V × heads × head_dim × f16
    total = per_tok * MODEL["n_layers"] * args.ctx
    f = {"f16": 1.0, "q8_0": 0.531, "turbo4": 0.2578, "turbo3": 0.203}[args.kv]
    return total * f / GI


def hit_skewed(res, skew):
    """Hit-rate avec skew : skew=1 uniforme (res/256), skew<1 → localité forte.
    Calibration pratique : hit ≈ res^skew / 256^skew borne [uniforme, parfait]."""
    return min((res / MODEL["experts"]) ** skew, 1.0)


def tier_costs(m, args, hit):
    """Coûts par token du chemin WARM/COLD : H2D pinned, miss→CPU, prefetch."""
    miss_bytes = (1 - hit) * MODEL["top_k"] * MODEL["expert_mb"][args.expert_fmt] * 1e6
    t_h2d = miss_bytes / (m["h2d_pinned"] * 1e9) * 1000           # ms, leçon MaxDam
    t_h2d_pg = miss_bytes / (m["h2d_pageable"] * 1e9) * 1000
    t_cpu_miss = (1 - hit) * MODEL["top_k"] / MODEL["top_k"] * 0.9  # ~0.9 ms par expert CPU (128B/elt ref) — ASSUMED
    # prefetch N+1 : masqué si t_compute(n) > t_h2d(n+1). t_compute ≈ decode total.
    t_compute = decode_tps(m, args)[1]
    t_dma_visible = max(t_h2d - t_compute, 0.0) if args.prefetch else t_h2d
    return miss_bytes, t_h2d, t_h2d_pg, t_cpu_miss, t_dma_visible


def warm_tier(m, args, hit_vram):
    """WARM = pinned RAM comme cache L2 explicite.
    - pool experts total (toutes couches) vs RAM dispo -> fraction couverte
    - miss VRAM -> WARM (fetch pinned H2D) si l'expert est en RAM, sinon COLD (SSD+H2D)
    - éviction VRAM→RAM : metadata seule (pas de write-back, poids read-only)
    Retour: (pool_gib, f_ram, h_warm, h_cold, t_warm_ms, t_cold_ms, evict_per_tok)"""
    per_exp_b = MODEL["expert_mb"][args.expert_fmt] * 1e6
    pool_gib = MODEL["n_layers"] * MODEL["experts"] * per_exp_b / GI
    ram_avail = args.ram_gb - args.rail - MODEL["permanent_gib"] - kv_gib(args)
    f_ram = 1.0 if pool_gib <= ram_avail else max(ram_avail / pool_gib, 0.0)
    miss = 1.0 - hit_vram
    h_warm = miss * f_ram
    h_cold = miss * (1.0 - f_ram)
    bytes_warm = h_warm * MODEL["top_k"] * per_exp_b
    bytes_cold = h_cold * MODEL["top_k"] * per_exp_b
    t_warm = bytes_warm / (m["h2d_pinned"] * 1e9) * 1000
    t_cold = bytes_cold * (1.0 / (args.ssd_gbs * 1e9) + 1.0 / (m["h2d_pinned"] * 1e9)) * 1000
    evict = h_warm + h_cold  # chaque fetch VRAM comble un slot -> 1 éviction metadata
    return pool_gib, f_ram, h_warm, h_cold, t_warm, t_cold, evict


def decode_tps(m, args):
    """t/token (ms) et t/s : BW-bound backbone + temps d'attente transferts."""
    _, per_exp, res_max, _ = budget(m, args)
    res = min(int(res_max), MODEL["experts"])
    hit = hit_skewed(res, args.skew)
    act = MODEL["top_k"] * per_exp * hit + MODEL["dense_backbone_bytes"]
    tps = m["bw_eff_gbs"] * 1e9 / act
    return hit, tps


def ledger(args, m):
    """Ledger NET_HARDWARE_GAIN par levier (coût − gain, honnête sur ASSUMED)."""
    lines = []
    base_hit, base_tps = decode_tps(m, args)
    lines.append(("baseline (cache plein budget)", base_hit, base_tps))
    # Levier 1 : NVFP4 vs Q4
    if args.expert_fmt == "q4":
        args_nv = argparse.Namespace(**{**vars(args), "expert_fmt": "nvfp4"})
        _, _, res_nv, _ = budget(m, args_nv)
        hit_nv = hit_skewed(min(int(res_nv), 256), args.skew)
        act_nv = MODEL["top_k"] * MODEL["expert_mb"]["nvfp4"] * 1e6 * hit_nv + MODEL["dense_backbone_bytes"]
        tps_nv = m["bw_eff_gbs"] * 1e9 / act_nv
        lines.append(("levier NVFP4 (vs Q4)", hit_nv, tps_nv))
    # Levier 2 : prefetch
    args_pf = argparse.Namespace(**{**vars(args), "prefetch": True})
    if not args.prefetch:
        miss_b, t_h2d, t_pg, _, t_vis = tier_costs(m, args_pf, base_hit)
        gain = (t_h2d - t_vis) * 1e-3
        lines.append(("levier prefetch N+1", base_hit, 1.0 / max(1.0 / base_tps - gain * (1 - base_hit), 1e-9)))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="5070", choices=list(MACHINES))
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--kv", default="turbo4", choices=["f16", "q8_0", "turbo4", "turbo3"])
    ap.add_argument("--expert-fmt", default="nvfp4", choices=["q4", "nvfp4"])
    ap.add_argument("--skew", type=float, default=0.85, help="1.0=uniforme, <1=localité")
    ap.add_argument("--compute-buffer", dest="compute_buffer", type=float, default=0.5)
    ap.add_argument("--rail", type=float, default=0.3)
    ap.add_argument("--ram-gb", dest="ram_gb", type=float, default=32.0)
    ap.add_argument("--ssd-gbs", dest="ssd_gbs", type=float, default=5.0)
    ap.add_argument("--npu-overflow-cost", dest="npu_overflow_cost", type=float, default=None,
                    help="coût overflow NPU (x hit GPU) — aucune valeur OP15 par défaut (matériel différent)")
    ap.add_argument("--prefetch", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    m = MACHINES[args.machine]
    cache_b, per_exp, res_max, kv = budget(m, args)
    res = min(int(res_max), MODEL["experts"])
    hit = hit_skewed(res, args.skew)
    miss_b, t_h2d, t_h2d_pg, t_cpu_miss, _ = tier_costs(m, args, hit)
    act = MODEL["top_k"] * per_exp * hit + MODEL["dense_backbone_bytes"]
    tps = m["bw_eff_gbs"] * 1e9 / act
    pool_gib, f_ram, h_warm, h_cold, t_warm, t_cold, evict = warm_tier(m, args, hit)
    t_miss = t_warm + t_cold
    t_compute = 1000.0 / tps
    t_dma_vis = max(t_miss - t_compute, 0.0) if args.prefetch else t_miss
    t_tok = 1000.0 / tps + t_dma_vis

    print(f"=== MoE AXIS PROFILER — {m['label']} — {MODEL['label']} ===")
    print(f"ctx={args.ctx} KV={args.kv} ({kv:.2f} GiB) experts={args.expert_fmt}"
          f" ({per_exp/1e6:.2f} MB) skew={args.skew} prefetch={'ON' if args.prefetch else 'OFF'}")
    print(f"\n[BUDGET #24528] free {m['vram_gb']*0.92:.2f} Go − permanent {MODEL['permanent_gib']} − KV {kv:.2f}"
          f" − compute_buffer {args.compute_buffer} − rail {args.rail} = **{cache_b/GI:.2f} GiB**"
          f" -> résidence {res}/256 (plafond poids-seuls gpu_tier_profiler: comparable mais TOUT soustrait ici)")
    print(f"[AXE 09/10] hit skew {hit:.2f} | miss {1-hit:.2f}")
    print(f"[TIER L2 WARM] pool experts {pool_gib:.1f} GiB | RAM dispo {args.ram_gb-args.rail-MODEL['permanent_gib']-kv_gib(args):.1f} GiB"
          f" -> WARM couvre {f_ram*100:.0f}% des miss | évictions {evict:.2f}/tok (metadata seule, 0 DMA)")
    print(f"  HOT (VRAM) {hit*100:.0f}% | WARM (pinned) {h_warm*100:.0f}% -> {t_warm:.2f} ms | COLD (SSD {args.ssd_gbs} GB/s) {h_cold*100:.0f}% -> {t_cold:.2f} ms | total miss {t_miss:.2f} ms/tok")
    print(f"[AXE 03/04] miss {miss_b/1e6:.1f} MB/tok : H2D pinned {t_h2d:.2f} ms | pageable {t_h2d_pg:.2f} ms"
          f" (ratio ×{t_h2d_pg/max(t_h2d,1e-9):.1f}) | miss→CPU {t_cpu_miss:.2f} ms"
          f" -> décision WARM: {'transfert GPU' if t_warm < t_cpu_miss else 'compute CPU'}")
    if args.prefetch:
        print(f"[AXE 11/14] prefetch N+1: DMA visible {t_dma_vis:.2f} ms (masqué partiellement, t_compute {1000/tps:.2f} ms)")
    print(f"[AXE 05] decode {tps:.1f} t/s (BW-bound) | t/token {t_tok:.2f} ms"
          f" -> effectif {1000/t_tok:.1f} t/s")
    print(f"[AXE 18] marginal: +1 GiB cache = +{GI/per_exp/MODEL['n_layers']:.0f} experts"
          f" -> dHit/dGiB = {hit_skewed(min(int((cache_b+GI)/per_exp/MODEL['n_layers']),256), args.skew)-hit:+.3f}")
    # NPU overflow (Voie A) : UNKNOWN sans constante fournie (matériel XDNA2 ≠ OP15)
    if args.npu_overflow_cost is not None:
        print(f"[Voie A] overflow {1-hit:.2f} sur NPU XDNA2 (x{args.npu_overflow_cost:.0f}/hit, sur-package) — à valider par sonde XRT")
    else:
        print(f"[Voie A] overflow {1-hit:.2f} sur NPU XDNA2 — COÛT UNKNOWN (constante OP15 refusée ; --npu-overflow-cost requis)")
    print("\n[LEDGER NET_HARDWARE_GAIN]")
    for name, h, t in ledger(args, m):
        print(f"  {name:<32} hit {h:.2f} | {t:.1f} t/s")
    print("\n[PROVENANCE] permanent/compute_buffer/rail/skew/h2d = ASSUMED (AXES_5070 §6 pour MEASURED)")


if __name__ == "__main__":
    main()
