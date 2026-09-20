#!/usr/bin/env python3
"""full_matrix.py - les 10 tests P0-P9 qui debloquent reellement le projet.

Sur machine cible HX365 + RTX 5070. Chaque test produit le MEME format de trace
(vecteur complet) pour alimenter l'oracle. P0-P9 definis dans le rapport 35B.

P0 H2D 4K->64M pinned/pageable   : cout reel transfert PCIe
P1 SSD->RAM->VRAM expert         : cout Flash reel
P2 cache hit/miss expert         : valeur reelle du cache
P3 prefetch next-layer           : overlap reel
P4 prefetch prediction           : gain net apres waste
P5 RTX NVFP4 kernel              : vrai cout compute
P6 XDNA2 INT8 kernel             : vrai cout NPU
P7 RTX + XDNA2 simultanes        : contention reelle
P8 decode 1 token                : modele principal
P9 prefill 512/2K/8K             : second modele
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "static"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle"))

from bytes_per_token import load_model, bytes_moe_active

TESTS = [
    {"id": "P0", "name": "h2d_4k_64m", "variable": "pinned/pageable"},
    {"id": "P1", "name": "ssd_ram_vram_expert", "variable": "chemin Flash"},
    {"id": "P2", "name": "cache_hit_miss", "variable": "hit rate 0/50/90/99%"},
    {"id": "P3", "name": "prefetch_next_layer", "variable": "overlap"},
    {"id": "P4", "name": "prefetch_prediction", "variable": "accuracy vs waste"},
    {"id": "P5", "name": "rtx_nvfp4_kernel", "variable": "compute reel"},
    {"id": "P6", "name": "xdna2_int8_kernel", "variable": "compute NPU reel"},
    {"id": "P7", "name": "rtx_xdna_simultanee", "variable": "contention"},
    {"id": "P8", "name": "decode_1_token", "variable": "modele principal"},
    {"id": "P9", "name": "prefill_512_2k_8k", "variable": "second modele"},
]

VECTOR_KEYS = [
    "token_id", "layer", "expert_ids", "cache_hit", "cache_miss",
    "SSD_read_bytes", "SSD_read_us", "RAM_stage_bytes", "RAM_stage_us",
    "PCIe_read_bytes", "PCIe_read_us", "DMA_us", "conversion_us",
    "RTX_compute_us", "XDNA_compute_us", "sync_us",
    "GDN_us", "QSA_us", "KV_bytes", "VRAM_peak", "RAM_peak",
    "prefetch_hit", "prefetch_waste", "total_us", "tok_s",
]


def empty_vector(test_id):
    return {"test": test_id, "ts": time.strftime("%Y%m%d_%H%M%S"),
            **{k: 0 for k in VECTOR_KEYS}}


def run_p0():
    """P0 : simulation H2D (placeholder - a remplacer par microbench CUDA reel)."""
    v = empty_vector("P0")
    # expert Q4 = 1.77 MB ; simulate miss -> PCIe read
    v["PCIe_read_bytes"] = 1_769_472
    v["PCIe_read_us"] = round(1_769_472 / (20e9) * 1e6, 1)  # placeholder 20 GB/s
    v["sync_us"] = 20
    v["total_us"] = v["PCIe_read_us"] + v["sync_us"]
    return v


def run_p2(m):
    """P2 : cache hit/miss. hit 0% vs 90% -> PCIe bytes."""
    out = []
    for hit in [0.0, 0.5, 0.9, 0.99]:
        v = empty_vector(f"P2_hit{int(hit*100)}")
        active = bytes_moe_active(m["num_layers"], m["top_k"], m["params_per_expert"], "Q4")
        v["PCIe_read_bytes"] = int(active["total_bytes"] * (1 - hit))
        v["cache_hit"] = int(hit * 100)
        v["cache_miss"] = int((1 - hit) * 100)
        out.append(v)
    return out


if __name__ == "__main__":
    m = load_model()
    print(f"CIBLE = 35B-A3B ({m['num_layers']} layers, top-{m['top_k']}, "
          f"{m['params_per_expert']/1e6:.3f}M params/expert)")
    print("\n=== 10 tests P0-P9 (squelette executable, placeholders) ===")
    for t in TESTS:
        print(f"  {t['id']} {t['name']:24s} variable={t['variable']}")
    print("\n=== P0 (H2D, placeholder) ===")
    print(json.dumps(run_p0()))
    print("\n=== P2 (cache hit/miss, Q4) ===")
    for v in run_p2(m):
        print(f"  hit {v['cache_hit']:3d}% : PCIe {v['PCIe_read_bytes']/1024**2:7.1f} MiB/token")
    print("\nNOTE : ces valeurs sont des PLACEHOLDERS. Sur machine cible, remplacer par")
    print("les mesures profiler-v3 (microbench CUDA/XDNA2, cache reel, kernel reel).")
    print("Chaque test doit produire le vecteur complet (VECTOR_KEYS) pour l'oracle.")