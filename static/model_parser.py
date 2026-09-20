#!/usr/bin/env python3
"""model_parser.py — lit config.json Qwen3.8-Flash-Next et produit les dimensions canoniques.

Créé dans npu-rtx/ (jamais dans profiler_v3). Sources : config.json officiel
(huggingface.co/Qwen/Qwen3.8-Flash-Next) + convert.log GGUF. Lecture seule des fichiers.
"""

import json


def parse_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def canonical_dims(cfg):
    """Extrait les dimensions documentées (hypothèse à vérifier contre le checkpoint réel)."""
    return {
        "hidden_size": cfg.get("hidden_size", 2560),
        "moe_intermediate_size": cfg.get("moe_intermediate_size", 640),
        "num_experts": cfg.get("num_experts", 512),
        "num_experts_per_tok": cfg.get("num_experts_per_tok", 10),
        "shared_expert_intermediate_size": cfg.get("shared_expert_intermediate_size", 640),
        "num_hidden_layers": cfg.get("num_hidden_layers", 48),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "num_key_value_heads": cfg.get("num_key_value_heads"),
        "head_dim": cfg.get("head_dim", 256),
        "vocab_size": cfg.get("vocab_size", 248320),
        "context_length": cfg.get("context_length", 262144),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
    }


def expert_shape(dims):
    """gate/up/down shapes : [2560, 640, 512] ×2 + [640, 2560, 512]."""
    h = dims["hidden_size"]
    i = dims["moe_intermediate_size"]
    e = dims["num_experts"]
    return {
        "gate": (h, i, e),
        "up": (h, i, e),
        "down": (i, h, e),
    }


def params_per_expert(dims):
    h = dims["hidden_size"]
    i = dims["moe_intermediate_size"]
    return 2 * (h * i) + (i * h)  # gate + up + down


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
    dims = canonical_dims(parse_config(cfg_path)) if cfg_path else canonical_dims({})
    print("dims:", dims)
    print("shapes:", expert_shape(dims))
    print("params/expert/layer:", params_per_expert(dims))