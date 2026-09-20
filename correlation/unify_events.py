#!/usr/bin/env python3
"""unify_events.py - adaptateur vers le format JSONL natif profiler-v3.

Respecte le schema de profiler-v3 (expert_profile.normalize_access_event +
profiler.py --live-trace) : un evenement = un acces expert/tensor avec timestamps.
Le format cible est compatible JSONL ; chaque evenement peut etre etendu avec les
champs D2 (bytes_pcie, t_dequant, critical_path) sans casser le schema existant.
"""

import json
import sys


def normalize(event, static_map=None):
    """Normalise un evenement en entree (dict brut) vers le schema profiler-v3."""
    out = {
        "timestamp_ns": int(event.get("timestamp_ns", 0)),
        "phase": event.get("phase", "decode"),       # decode / prefill
        "layer": int(event.get("layer", 0)),
        "expert_id": int(event.get("expert_id", -1)),
        "tensor": event.get("tensor", "gate_exps"),  # gate/up/down_exps, shared, router
        "weight_precision": event.get("weight_precision", "Q4_K"),
        "activation_precision": event.get("activation_precision", "BF16"),
        "storage": event.get("storage", "RAM"),       # SSD/RAM/PINNED/VRAM
        "compute_device": event.get("compute_device", "RTX"),
        "cache_hit": bool(event.get("cache_hit", False)),
        "prefetch": bool(event.get("prefetch", False)),
        # extension D2 (non-blocking pour profiler-v3)
        "bytes_storage": int(event.get("bytes_storage", 0)),
        "bytes_pcie": int(event.get("bytes_pcie", 0)),
        "t_quant_ns": int(event.get("t_quant_ns", 0)),
        "t_dequant_ns": int(event.get("t_dequant_ns", 0)),
        "t_gemm_ns": int(event.get("t_gemm_ns", 0)),
        "t_sync_ns": int(event.get("t_sync_ns", 0)),
        "critical_path_ns": int(event.get("critical_path_ns", 0)),
    }
    return out


def to_jsonl(events, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(normalize(e)) + "\n")
    return path


def from_jsonl(path):
    evs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                evs.append(json.loads(line))
    return evs


if __name__ == "__main__":
    # demo : 3 evenements synthetiques -> trace.jsonl
    demo = [
        {"timestamp_ns": 1_000, "layer": 0, "expert_id": 17, "tensor": "gate_exps",
         "weight_precision": "Q4_K", "storage": "VRAM", "cache_hit": True,
         "bytes_pcie": 0, "t_gemm_ns": 210_000},
        {"timestamp_ns": 1_250, "layer": 0, "expert_id": 92, "tensor": "up_exps",
         "weight_precision": "Q4_K", "storage": "RAM", "cache_hit": False,
         "bytes_pcie": 2_764_800, "t_pcie_ns": 138_000, "t_gemm_ns": 210_000},
        {"timestamp_ns": 1_500, "layer": 0, "expert_id": 31, "tensor": "down_exps",
         "weight_precision": "NVFP4", "storage": "VRAM", "cache_hit": True,
         "t_dequant_ns": 30_000, "t_gemm_ns": 190_000},
    ]
    out = to_jsonl(demo, "trace_demo.jsonl")
    print("Ecrit :", out)
    for e in from_jsonl(out):
        print(" ", e["expert_id"], e["tensor"], e["weight_precision"], "hit" if e["cache_hit"] else "miss")
    print("\nSchema compatible profiler-v3 (expert_profile.normalize_access_event).")