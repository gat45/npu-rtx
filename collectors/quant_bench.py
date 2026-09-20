#!/usr/bin/env python3
"""quant_bench.py - mesure reelle de la conversion NVFP4->INT8 (premier test D2).

Objectif : le premier test D2 de OUBLIS_DECISIONS §2. On ne peut pas executer de
kernel NPU ici (machine dev GTX 1080, pas de XDNA2), mais on mesure le COUT CPU de
conversion NVFP4->INT8 (et BF16->INT8) qui alimente conversion_matrix, avec le
schema de provenance MEASURED/DERIVED/ASSUMED.

Formats reels :
- NVFP4 : E2M1 4-bit + scale FP8 (per-block 16) -> 2 bytes/value (nibbles)
- INT8  : 1 byte/value (packed avec scale)
- BF16  : 2 bytes/value

Mesure : time de conversion par taille -> GB/s effectif (remplace CONV_INT8_GBs).
"""

import os
import random
import sys
import time


def convert_nvfp4_to_int8(values_f32):
    """NVFP4(E2M1) -> INT8 : quantification scale par bloc de 16."""
    n = len(values_f32)
    out = bytearray(n)
    for base in range(0, n, 16):
        block = values_f32[base:base + 16]
        if not block:
            break
        scale = max(max(block), -min(block)) / 127.0 if block else 1.0
        scale = scale or 1.0
        for j, v in enumerate(block):
            out[base + j] = max(0, min(255, int(round(v / scale)) + 128))
    return out


def convert_bf16_to_int8(values_f32):
    """BF16 -> INT8 : quantification per-tensor (reference simple)."""
    scale = (max(values_f32) - min(values_f32)) / 255.0 or 1.0
    return bytearray(max(0, min(255, int(round(v / scale)) + 128)) for v in values_f32)


def bench_conversion(name, converter, n):
    values = [random.uniform(-5.0, 5.0) for _ in range(n)]
    converter(values)  # warmup
    t0 = time.perf_counter()
    reps = max(1, 4_000_000 // n)
    for _ in range(reps):
        converter(values)
    dt = time.perf_counter() - t0
    bytes_in = n * 2  # NVFP4/BF16 stocke 2 bytes/value
    gbps = (bytes_in * reps) / dt / 1e9
    return round(gbps, 2)


if __name__ == "__main__":
    print("Conversion CPU (Python, reference naive) - GB/s effectif d'entree")
    print("NOTE : Python pur = BEAUCOUP plus lent que C/numpy. Ceci est un MINIMUM.")
    for n in [4 * 1024, 64 * 1024, 1024 * 1024]:
        g_nv = bench_conversion("nvfp4->int8", convert_nvfp4_to_int8, n)
        g_bf = bench_conversion("bf16->int8", convert_bf16_to_int8, n)
        print(f"  {n//1024:6d} KiB : NVFP4->INT8 {g_nv:5.2f} GB/s | BF16->INT8 {g_bf:5.2f} GB/s")
    print("\nA remplacer par une conversion C/numpy (ktransformers-style) + GPU/NPU,")
    print("puis injecter dans conversion_matrix (CONV_INT8_GBs).")