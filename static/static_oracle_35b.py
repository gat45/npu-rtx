#!/usr/bin/env python3
"""static_oracle_35b.py - Static Oracle pour Qwen3.6-35B-A3B.

Reprend le pipeline Flash (19299f6) recalculé pour le 35B-A3B.
Separation : FAITS DOCUMENTES (config) / DERIVATIONS (calculs) / HYPOTHESES (notes).
ATTENTION : ne jamais figer "Q4 = X GB" - les tailles GGUF varient selon le quantizer.
A verifier contre le checkpoint/convert.log reel.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "static"))

from quant_size_engine import format_bpw, tensor_bytes
from bytes_per_token import bytes_moe_active

HERE = os.path.dirname(__file__)
CFG = os.path.join(HERE, "..", "models", "qwen36_35b_a3b", "config.json")


def load_cfg():
    with open(CFG, "r", encoding="utf-8") as f:
        return json.load(f)


def expert_params(hidden, intermediate):
    """gate/up merged = 2048x1024 (2x512), down = 512x2048 -> total/expert."""
    # merged gate+up : hidden x (2*intermediate) ; down : intermediate x hidden
    return hidden * (2 * intermediate) + intermediate * hidden


def run():
    cfg = load_cfg()
    h, i = cfg["hidden_size"], cfg["moe_intermediate_size"]
    nl, ne, tk = cfg["num_hidden_layers"], cfg["num_experts"], cfg["num_experts_per_tok"]
    ppe = expert_params(h, i)

    out = {"model": "Qwen3.6-35B-A3B",
           "nomenclature_warning": cfg.get("nomenclature_warning", ""),
           "faits_documentes": cfg,
           "derivations": {}}
    out["derivations"]["params_per_expert"] = ppe
    out["derivations"]["total_experts"] = nl * ne
    out["derivations"]["total_expert_params"] = nl * ne * ppe

    # layer_type[layer_id] : 30 GDN + 10 Attention (3:1)
    out["derivations"]["layer_type"] = {
        "pattern": "linear x3 + full, x10",
        "gdN_layers": cfg.get("gated_delta_net_layers", 30),
        "attention_layers": cfg.get("gated_attention_layers", 10),
    }

    # tailles expert par format
    sizes = {}
    for fmt in ["BF16", "INT8", "Q8_0", "Q6_K", "Q4", "NVFP4", "Q3", "Q2"]:
        sizes[fmt] = round(tensor_bytes(ppe, fmt) / (1024**2), 3)  # MiB
    out["derivations"]["expert_mib"] = sizes

    # traffic froid (sans cache) par token
    cold = {}
    for fmt in ["Q4", "Q6_K", "Q8_0", "BF16"]:
        b = bytes_moe_active(nl, tk, ppe, fmt, shared_fmt="BF16")
        cold[fmt] = {"gib_token": round(b["total_gib"], 3),
                     "mib_token": round(b["total_bytes"] / (1024**2), 1)}
    out["derivations"]["cold_traffic"] = cold

    # cache : traffic = miss_rate x cold
    out["derivations"]["cache_traffic"] = {}
    for hit in [0.50, 0.75, 0.90, 0.95, 0.99]:
        out["derivations"]["cache_traffic"][f"hit{int(hit*100)}"] = {
            "mib_token": round(cold["Q4"]["mib_token"] * (1 - hit), 2),
            "byte_hit_rate_note": "byte_hit_rate = hit_bytes/requested_bytes - a mesurer (count hit != byte hit)"}

    # provenance + confiance (correction angle mort 46/65)
    out["provenance"] = {
        "faits_documentes": "MEASURED_source", "derivations": "DERIVED",
        "gguf_sizes_public": "ASSUMED (a verifier GGUF reel)",
        "bandwidth": "ASSUMED/UNKNOWN (mesures profiler-v3 requises)",
        "confidence": {"static_bounds": 0.8, "traffic": 0.6, "latency": 0.3},
        "model_hash": "UNKNOWN (a extraire du checkpoint)",
    }

    # lower bounds (placeholder BW - a remplacer par mesures profiler-v3)
    out["derivations"]["lower_bounds_placeholder"] = {
        "pcie_20gbs_q4_cold_ms": round(cold["Q4"]["mib_token"] * 1024**2 / (20e9) * 1e3, 1),
        "pcie_20gbs_q4_hit90_ms": round(cold["Q4"]["mib_token"] * 0.10 * 1024**2 / (20e9) * 1e3, 2),
        "note": "placeholders - remplacer par SSD/DDR/PCIe mesures (hw_discovery + profiler-v3)"
    }

    # hypotheses / contraintes
    out["hypotheses"] = {
        "vram_budget_gib": 6.5,
        "l1_tile_kib": 64,
        "expert_tile_multiple_of_8": (h % 8 == 0) and (i % 8 == 0),
        "note": "dims 2048/512 multiples de 8 -> pas de penalite alignement mmul 8x8x8 XDNA2",
        "gguf_q4_size_varies": "GGUF publics Q4_K_M ~20-22 GB (varie selon quantizer) - NE PAS figer",
    }
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    print("\n--- SYNTHESE 35B ---")
    print(f"params/expert = {r['derivations']['params_per_expert']/1e6:.3f}M")
    print("expert MiB :", r["derivations"]["expert_mib"])
    print("cold traffic Q4 :", r["derivations"]["cold_traffic"]["Q4"])
    print("cache traffic (Q4, MiB/token) :", r["derivations"]["cache_traffic"])