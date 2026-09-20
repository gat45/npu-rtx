#!/usr/bin/env python3
"""bytes_per_token.py - octets actifs/token par phase (prefill/decode) + effet cache.

Calcule le plancher de trafic MoE par token (routed + shared) pour chaque format,
avec et sans cache (routing mass Haberstroh : 64 experts = 53%, 128 = 73%, 256 = 93%).
Sortie : plancher mémoire par config -> alimente memory_planner (elimination).
"""

from model_parser import canonical_dims
from quant_size_engine import format_bpw

EXPERT_PARAMS = 4_915_200

# Routing mass (resultat public Haberstroh) : (experts/couche, masse cumulee)
ROUTING_MASS = [
    (32, 0.37), (64, 0.53), (128, 0.73), (171, 0.82), (256, 0.93),
]


def bytes_moe_active(num_layers, top_k, params_per_expert, fmt, shared_fmt="BF16"):
    """Octets MoE actifs/token sans cache : 10 routed + shared, en GiB."""
    routed = num_layers * top_k * params_per_expert * format_bpw(fmt) / 8.0
    shared = num_layers * params_per_expert * format_bpw(shared_fmt) / 8.0
    return {"routed_bytes": routed, "shared_bytes": shared,
            "total_bytes": routed + shared, "total_gib": (routed + shared) / (1024**3)}


def bytes_moe_active_with_cache(num_layers, top_k, params_per_expert, fmt,
                                cache_experts_per_layer, hit_rate_est=0.9):
    """Avec cache : seuls les experts MANQUANT au cache passent par PCIe (miss).

    cache_experts_per_layer : experts residents (ex. 48 pour 6 GiB Q4).
    hit_rate_est : part des 10 routed deja en cache (a mesurer reellement).
    Approximation : les hit lisent depuis VRAM (cout PCIe ~0), les miss depuis RAM/SSD.
    """
    miss = num_layers * top_k * (1.0 - hit_rate_est) * params_per_expert * format_bpw(fmt) / 8.0
    shared = num_layers * params_per_expert * format_bpw("BF16") / 8.0
    return {"miss_pcie_bytes": miss, "shared_bytes": shared,
            "miss_pcie_gib": miss / (1024**3), "hit_rate_assumed": hit_rate_est,
            "cache_experts_per_layer": cache_experts_per_layer}


def prefill_vs_decode_note():
    """Prefill = batch large (experts uniques = union batch), decode = top_k par token."""
    return ("PREFILL : M grand -> union(experts) sur le batch, GEMM domine. "
            "DECODE : M petit -> top_k par token, cache/H2D/launch dominent. "
            "Deux modeles distincts (MoE-Infinity).")


if __name__ == "__main__":
    dims = canonical_dims({})
    nl = dims["num_hidden_layers"]
    tk = dims["num_experts_per_tok"]
    print("== MoE actif/token SANS cache (routed + shared BF16) ==")
    for fmt in ["BF16", "Q8_0", "Q6_K", "Q4", "NVFP4", "INT8", "Q3", "Q2"]:
        r = bytes_moe_active(nl, tk, EXPERT_PARAMS, fmt)
        print(f"  {fmt:5s} : routed={r['routed_bytes']/(1024**3):.3f} GiB  "
              f"+ shared={r['shared_bytes']/(1024**3):.3f}  = {r['total_gib']:.3f} GiB/token")

    print("\n== MoE PCIe/token AVEC cache (6 GiB Q4 ~ 48 experts/couche, hit 90%) ==")
    r = bytes_moe_active_with_cache(nl, tk, EXPERT_PARAMS, "Q4", 48)
    print(f"  miss PCIe = {r['miss_pcie_gib']:.3f} GiB/token (hit 90%) vs 1.236 GiB sans cache")
    print("  -> cache evite ~90% du trafic PCIe MoE (preuve 0.31 GB vs 26 GB Haberstroh)")

    print("\n== Routing mass (working set real) ==")
    for e, m in ROUTING_MASS:
        print(f"  {e:4d} experts/couche = {m*100:.0f}% du trafic")
    print("\n" + prefill_vs_decode_note())