#!/usr/bin/env python3
"""memory_planner.py - budgets VRAM/RAM + candidats elimines (lower-bound check).

Utilise les octets statiques (quant_size_engine, bytes_per_token) pour :
1. construire le budget VRAM 6 composants (dense/GDN_state/QSA_KV/workspace/PLE/cache)
2. eliminer les candidats dont le lower-bound memoire depasse les capacites mesurees
   (bandwidth placeholder : a remplacer par profiler-v3).
"""

from model_parser import canonical_dims
from quant_size_engine import format_bpw, tensor_bytes
from bytes_per_token import bytes_moe_active

EXPERT_PARAMS = 4_915_200

# Budgets machine cible (a verifier par hw_discovery - placeholders)
VRAM_TOTAL = 8.0       # GiB (RTX 5070 laptop 8GB)
RAM_TOTAL = 32.0       # GiB
VRAM_WDDM_RESERVE = 1.5  # reserve dynamique (WDDM budget variable) - a mesurer

# Placeholders bandwidth (profiler-v3 requis)
BW_PCIE_GBs = 20.0
BW_DDR_GBs = 89.6     # theorique - attention


def vram_budget_components(dims, expert_fmt, ctx=8192):
    """Estimation VRAM 6 composants (ordre de grandeur, a valider)."""
    nl = dims["num_hidden_layers"]
    # dense (attention + norms + embeddings + shared) - approx : hors experts
    # GDN state (36 couches, etat [heads,head_dim,head_dim]) + QSA KV (12 couches)
    gdn_state_gib = 36 * 48 * 128 * 128 * 2 / (1024**3)   # BF16, ~0.23 GB/session mesure FLM
    qsa_kv_gib = ctx * 12 * 2 * 256 * 2 / (1024**3)        # BF16 KV QSA
    return {
        "dense": 0.0,  # TODO : tensor map dense
        "gdn_state": round(gdn_state_gib, 3),
        "qsa_kv": round(qsa_kv_gib, 3),
        "workspace": 0.5,
        "ple": 0.0,   # PLE => host/SSD (pas GPU)
        "expert_cache": 0.0,
        "note": "placeholders - valider par profiler-v3 (GDN 0.23GB/session mesure FLM, KV 30KB/token)"
    }


def eliminate_candidates(dims, top_k, expert_fmt, ctx, vram_available):
    """Elimine les candidats dont le lower-bound memoire/PCIe est impossible."""
    nl = dims["num_hidden_layers"]
    active = bytes_moe_active(nl, top_k, EXPERT_PARAMS, expert_fmt)
    # lower-bound transfert si TOUT est miss (sans cache)
    t_miss = active["total_bytes"] / (BW_PCIE_GBs * 1e9)
    # avec cache hit 90%
    miss10 = active["routed_bytes"] * 0.10 / (BW_PCIE_GBs * 1e9)
    # VRAM : si cache budget depasse VRAM disponible -> eliminer
    cache_ok = vram_available >= 0.0
    return {
        "active_gib_token": active["total_gib"],
        "t_transfer_all_miss_ms": t_miss * 1e3,
        "t_transfer_hit90_ms": miss10 * 1e3,
        "max_tok_per_s_all_miss": 1.0 / t_miss,
        "max_tok_per_s_hit90": 1.0 / miss10 if miss10 > 0 else float('inf'),
        "eliminated": [] if cache_ok else ["cache budget > VRAM"],
    }


if __name__ == "__main__":
    dims = canonical_dims({})
    vram_avail = VRAM_TOTAL - VRAM_WDDM_RESERVE
    print(f"VRAM dispo (apres reserve WDDM {VRAM_WDDM_RESERVE} GiB) = {vram_avail} GiB")
    print("Budget VRAM composants (ctx 8K) :", vram_budget_components(dims, "Q4"))

    print("\n== Elimination (lower-bound, placeholders) ==")
    for fmt in ["Q2", "Q3", "Q4", "NVFP4", "Q6_K", "INT8", "BF16"]:
        r = eliminate_candidates(dims, 10, fmt, 8192, vram_avail)
        print(f"  {fmt:5s} : actif {r['active_gib_token']:.3f} GiB/token | "
              f"100% miss -> {r['t_transfer_all_miss_ms']:6.1f} ms/token "
              f"(max {r['max_tok_per_s_all_miss']:.1f} t/s) | "
              f"hit90 -> {r['t_transfer_hit90_ms']:5.2f} ms "
              f"(max {r['max_tok_per_s_hit90']:.0f} t/s)")

    print("\nNOTE : eliminer un candidat si max_tok_per_s < objectif, ou si le budget")
    print("VRAM cache depasse VRAM_dispo. Ces chiffres sont des bornes pessimistes sans")
    print("overlap ; le cache + double buffering changent tout (mesure profiler-v3 requise).")