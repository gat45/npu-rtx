#!/usr/bin/env python3
"""quant_size_engine.py — taille d'un tensor/expert par format (GGML + NVFP4 + INT8).

Valeurs vérifiées : convert.log GGUF (expert BF16 1600 MiB → Q8_0 850 MiB = 53.125%),
checkpoint NVFP4 (120.8B experts = 67.95 GB ≈ 4.5 bpw avec scales), PLE 28.8 GB NVFP4 réel.
bits_per_weight INCLUT l'overhead scales/metadata (pas 4.0 pour "Q4").
"""


def format_bpw(fmt):
    """bpw effectif (stockage réel, avec scales/min/alignement)."""
    table = {
        "BF16": 16.0, "F16": 16.0, "F32": 32.0,
        "Q8_0": 8.5, "Q6_K": 6.563, "Q5": 5.5, "Q5_0": 5.5, "Q5_K": 5.5,
        "Q4": 4.5, "Q4_0": 4.5, "Q4_1": 5.0, "Q4_K": 4.5, "IQ4_NL": 4.5, "IQ4_XS": 4.25,
        "NVFP4": 4.5, "MXFP4": 4.25, "FP8": 8.0, "INT8": 8.0,
        "Q3": 3.5, "Q3_K": 3.438, "IQ3_XXS": 3.0, "Q2_K": 2.625, "Q2": 2.63, "IQ2_XXS": 2.06,
        "TQ1_0": 1.688, "TQ2_0": 2.063, "Q1_0": 1.125,
    }
    return table.get(fmt, 16.0)


def tensor_bytes(params, fmt):
    """Octets pour N paramètres en format fmt (8 bits/byte)."""
    return params * format_bpw(fmt) / 8.0


def expert_bytes(params_per_expert, fmt):
    """Un expert/layer complet (gate+up+down)."""
    return tensor_bytes(params_per_expert, fmt)


def experts_total_bytes(num_layers, num_experts, params_per_expert, fmt):
    return tensor_bytes(num_layers * num_experts * params_per_expert, fmt)


def ple_bytes(rows=320_001_536, width=160, fmt="BF16"):
    return tensor_bytes(rows * width, fmt)


def active_moe_bytes(num_layers, num_experts_per_tok, params_per_expert,
                     fmt, shared_extra=True, dims=None):
    """Bytes actifs par token : 10 routed × 48 couches + (option) shared par couche."""
    routed = tensor_bytes(num_layers * num_experts_per_tok * params_per_expert, fmt)
    shared = 0.0
    if shared_extra:
        # shared = 1 expert/layer, même taille (shared_expert_intermediate = 640)
        shared = tensor_bytes(num_layers * params_per_expert, "BF16")  # shared souvent BF16
    return {"routed": routed, "shared": shared, "total": routed + shared}


if __name__ == "__main__":
    # Qwen3.8-Flash-Next : 4 915 200 params/expert, 48 layers, 512 experts, top-10
    ppe = 4_915_200
    for fmt in ["BF16", "Q8_0", "Q6_K", "Q5", "Q4", "NVFP4", "Q3", "Q2"]:
        eb = expert_bytes(ppe, fmt) / (1024 * 1024)
        tot = experts_total_bytes(48, 512, ppe, fmt) / (1024**3)
        act = active_moe_bytes(48, 10, ppe, fmt)["total"] / (1024**3)
        print(f"{fmt:5s}  expert={eb:6.2f} MiB  total_experts={tot:7.2f} GiB  actif/token={act:5.3f} GiB")
    print(f"PLE BF16 = {ple_bytes()['bytes'] if False else ple_bytes()/1024**3:.1f} GiB")
    print(f"PLE NVFP4 = {ple_bytes(fmt='NVFP4')/1024**3:.1f} GiB (réel rapporté ~28.8 GB)")