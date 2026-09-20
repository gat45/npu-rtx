#!/usr/bin/env python3
"""bytes_per_token.py - octets actifs/token par phase (prefill/decode) + effet cache.

⚠️ CIBLE = Qwen3.6/3.5-35B-A3B (40 layers, 256 experts, top-8 + 1 shared).
Les parametres sont charges depuis models/*/config.json (multi-modele).
"""

import json
import os

from model_parser import canonical_dims
from quant_size_engine import format_bpw

ROUTING_MASS = [
    (32, 0.37), (64, 0.53), (128, 0.73), (171, 0.82), (256, 0.93),
]

MODEL_CFG = os.path.join(os.path.dirname(__file__), "..", "models",
                         "qwen36_35b_a3b", "config.json")


def load_model():
    with open(MODEL_CFG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    h = cfg["hidden_size"]
    i = cfg["moe_intermediate_size"]
    ppe = h * (2 * i) + i * h  # gate/up merged + down
    return {"num_layers": cfg["num_hidden_layers"],
            "top_k": cfg["num_experts_per_tok"],
            "params_per_expert": ppe,
            "num_experts": cfg["num_experts"]}


def bytes_moe_active(num_layers, top_k, params_per_expert, fmt, shared_fmt="BF16"):
    """Octets MoE actifs/token sans cache : top_k routed + shared, en GiB."""
    routed = num_layers * top_k * params_per_expert * format_bpw(fmt) / 8.0
    shared = num_layers * params_per_expert * format_bpw(shared_fmt) / 8.0
    return {"routed_bytes": routed, "shared_bytes": shared,
            "total_bytes": routed + shared, "total_gib": (routed + shared) / (1024**3)}


def bytes_total_per_token(cfg, expert_fmt, dense_fmt="BF16"):
    """FINDING CRITIQUE (vLLM #51197) : bytes/token = dense backbone + experts + states.

    Le backbone dense GDN peut representer ~80% du trafic poids/decode sur Qwen3.6.
    Optimiser le cache expert sans compter le dense = optimiser le mauvais objet.
    """
    h = cfg["hidden_size"]
    i = cfg["moe_intermediate_size"]
    nl = cfg["num_hidden_layers"]
    tk = cfg["num_experts_per_tok"]
    # dense backbone (attention + norms + router) - approximation par couche : ~12-16x hidden^2
    # + GDN/attention projections. Valeur a affiner depuis le manifest reel.
    dense_per_layer_params = 14 * h * h  # ordre de grandeur Qwen3.5/3.6 dense par layer
    dense_bytes = nl * dense_per_layer_params * format_bpw(dense_fmt) / 8.0
    moe = bytes_moe_active(nl, tk, h * (2 * i) + i * h, expert_fmt)
    return {"dense_backbone_gib": dense_bytes / (1024**3),
            "moe_routed_gib": moe["routed_bytes"] / (1024**3),
            "moe_shared_gib": moe["shared_bytes"] / (1024**3),
            "total_gib": (dense_bytes + moe["total_bytes"]) / (1024**3),
            "dense_share_pct": round(dense_bytes / (dense_bytes + moe["total_bytes"]) * 100, 1)}


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
    m = load_model()
    nl, tk, ppe = m["num_layers"], m["top_k"], m["params_per_expert"]
    print(f"CIBLE = Qwen3.6/3.5-35B-A3B : {nl} layers, {m['num_experts']} experts, top-{tk}, "
          f"{ppe/1e6:.3f}M params/expert")

    print("\n== FINDING CRITIQUE (vLLM #51197) : bytes/token TOTAL = dense + experts ==")
    with open(MODEL_CFG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    tot = bytes_total_per_token(cfg, "Q4")
    print(f"  dense_backbone (Q4 experts) : {tot['dense_backbone_gib']:.3f} GiB/token "
          f"({tot['dense_share_pct']}% du total)")
    print(f"  moe routed + shared Q4        : {tot['moe_routed_gib']:.3f} + {tot['moe_shared_gib']:.3f} GiB")
    print(f"  TOTAL                         : {tot['total_gib']:.3f} GiB/token")

    print("\n== MoE actif/token SANS cache (routed + shared BF16) ==")
    for fmt in ["BF16", "Q8_0", "Q6_K", "Q4", "NVFP4", "INT8", "Q3", "Q2"]:
        r = bytes_moe_active(nl, tk, ppe, fmt)
        print(f"  {fmt:5s} : routed={r['routed_bytes']/(1024**3):.3f} GiB  "
              f"+ shared={r['shared_bytes']/(1024**3):.3f}  = {r['total_gib']:.3f} GiB/token")

    print("\n== MoE PCIe/token AVEC cache (6 GiB Q4 ~ 48 experts/couche, hit 90%) ==")
    r = bytes_moe_active_with_cache(nl, tk, ppe, "Q4", 48)
    print(f"  miss PCIe = {r['miss_pcie_gib']:.3f} GiB/token (hit 90%) vs "
          f"{bytes_moe_active(nl, tk, ppe, 'Q4')['routed_bytes']/(1024**3):.3f} GiB sans cache")

    print("\n== Routing mass (working set real) ==")
    for e, m_ in ROUTING_MASS:
        print(f"  {e:4d} experts/couche = {m_*100:.0f}% du trafic")
    print("\n" + prefill_vs_decode_note())