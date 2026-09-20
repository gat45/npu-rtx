#!/usr/bin/env python3
"""kernel_registry.py - KernelCapabilityDB (format x backend x arch x kernel).

Repond au trou n°12 : le planner doit connaitre les kernels REELlement disponibles.
Niveau 1 (bruit de fond, SM120) et XDNA2. Valeurs ASSUMED/UNVERIFIED a verifier
sur la machine cible (RAPPORT_FLASH_35B_CORRIGE §12).
"""

import json
import os

REGISTRY = {
    "sm120_rtx5070": {
        "nvfp4_gemm": {"backend": "cutlass", "supported": True, "shapes": [], "measured": False,
                        "note": "CUTLASS 79d_blackwell_geforce_nvfp4_grouped_gemm (SM120)"},
        "nvfp4_gemm_flashinfer": {"backend": "flashinfer", "supported": True, "shapes": [],
                                  "measured": False, "note": "b12x cute_dsl SM120/SM121"},
        "fp8_rowwise": {"backend": None, "supported": False,
                        "note": "TensorRT-LLM: FP8 rowwise NON supporte sur sm120"},
        "fp8_pertensor": {"backend": "flashinfer", "supported": True, "shapes": [], "measured": False},
        "fp8_blockscale": {"backend": None, "supported": False,
                           "note": "TensorRT-LLM: FP8 block-scale NON supporte sm120"},
        "q4k_gemm": {"backend": "llamacpp_mmq", "supported": True, "shapes": [], "measured": False},
        "q6k_gemm": {"backend": "llamacpp_mmq", "supported": True, "shapes": [], "measured": False},
        "w4a16": {"backend": None, "supported": False, "note": "TRT-LLM sm120: NON supporte"},
    },
    "xdna2_hx365": {
        "int8_gemm": {"backend": "iron/mmal", "supported": True, "shapes": ["8x8x8"], "measured": False,
                      "note": "INT8 chemin natif P0 (6.65-8.69 TOPS mesure DESCENT)"},
        "int4_gemm": {"backend": None, "supported": False, "note": "INT4 = stockage seul, compute INT8/BF16"},
        "bfp16_gemm": {"backend": "iron", "supported": True, "shapes": ["8x8x8"], "measured": False,
                       "note": "BFP16 natif P0bis (4.64 TFLOPS)"},
        "bf16_gemm": {"backend": "iron", "supported": False,
                      "note": "BF16 = EMULE via BFP16 (1/4 debit) - eviter"},
        "gemv_int8": {"backend": "iron", "supported": True, "shapes": [], "measured": False},
    },
}


def save(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "runs", "kernel_registry.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(REGISTRY, f, indent=2)
    return path


def query(arch, kernel):
    return REGISTRY.get(arch, {}).get(kernel, {"supported": "UNKNOWN"})


if __name__ == "__main__":
    p = save()
    print("KernelCapabilityDB ->", p)
    for arch, kernels in REGISTRY.items():
        for name, k in kernels.items():
            mark = "OK" if k["supported"] else "X "
            print(f"  [{mark}] {arch} / {name:22s} backend={str(k.get('backend')):12s} "
                  f"measured={k.get('measured', 'N/A')}")
    print("\nNB : UNVERIFIED (mesure sur machine cible requise). Le planner ne doit jamais")
    print("traiter un kernel 'supported' non mesure comme performant.")