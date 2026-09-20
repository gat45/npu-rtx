#!/usr/bin/env python3
"""conversion_matrix.py - cout des 4 chemins de conversion NVFP4->INT8 (premier test D2).

OUBLIS_DECISIONS §2 : le stockage est NVFP4/Q4, mais XDNA2 execute INT8 (chemin natif P0).
Il faut comparer :
  A: NVFP4 -> RTX  (aucune conversion, GEMM NVFP4)
  B: NVFP4 -> INT8 -> XDNA2   (conversion + GEMM INT8)
  C: BF16  -> INT8 -> XDNA2   (conversion depuis maitre)
  D: BF16  -> BFP16 -> XDNA2  (conversion + GEMM BFP16)

Cout modelise : conversion_time + transfert + GEMM, sans chevauchement (borne pessimiste).
Les debits sont des PLACEHOLDERS a remplacer par les mesures profiler-v3 (P(size,qdepth)).
"""

from expert_mapper import map_expert
from quant_size_engine import format_bpw

# Placeholders - a remplacer par profiler-v3 (jamais theoriques en conclusion)
BW_PCIE_GBs = 20.0      # placeholder
CONV_INT8_GBs = 20.0    # placeholder (KTransformers CPU, a mesurer CPU/GPU/NPU)

NPU_INT8_TOPS = 8.0     # mesure DESCENT (kernel-only 2x2 pipeline), a reproduire HX365
NPU_BFP16_TFLOPS = 4.64
RTX_NVFP4_TFLOPS = 988.0  # 5070 marketing - placeholder
EXPERT_PARAMS = 4_915_200


def bytes_fmt(params, fmt):
    return params * format_bpw(fmt) / 8.0


def compute_time(params, tops):
    flops = 2 * params
    return flops / (tops * 1e12)


def path_cost(params, storage_fmt, compute_fmt, device_tops, conv_bw=None, pcie_bw=BW_PCIE_GBs):
    """Cout d'un expert : conversion (si != None) + transfert + GEMM (non chevauche)."""
    store_b = bytes_fmt(params, storage_fmt)
    compute_b = bytes_fmt(params, compute_fmt)
    conv = 0.0
    if conv_bw:
        conv = store_b / (conv_bw * 1e9)
    transfer = store_b / (pcie_bw * 1e9)
    gemm = compute_time(params, device_tops)
    return {
        "storage": storage_fmt, "compute": compute_fmt,
        "bytes_storage": store_b, "bytes_compute": compute_b,
        "t_conversion_us": conv * 1e6, "t_transfer_us": transfer * 1e6,
        "t_gemm_us": gemm * 1e6, "t_total_no_overlap_us": (conv + transfer + gemm) * 1e6,
    }


def build_matrix():
    m = {}
    p = EXPERT_PARAMS
    m["A_nvfp4_rtx"] = path_cost(p, "NVFP4", "NVFP4", RTX_NVFP4_TFLOPS)
    m["B_nvfp4_int8_xdna"] = path_cost(p, "NVFP4", "INT8", NPU_INT8_TOPS, conv_bw=CONV_INT8_GBs)
    m["C_bf16_int8_xdna"] = path_cost(p, "BF16", "INT8", NPU_INT8_TOPS, conv_bw=CONV_INT8_GBs)
    m["D_bf16_bfp16_xdna"] = path_cost(p, "BF16", "BFP16", NPU_BFP16_TFLOPS, conv_bw=CONV_INT8_GBs)
    return m


if __name__ == "__main__":
    m = build_matrix()
    print("Cout d'UN expert/layer (placeholders - mesures profiler-v3 requises)")
    print(f"{'chemin':22s} {'conv_us':>8s} {'transfer_us':>12s} {'gemm_us':>8s} {'total_us':>10s}")
    for k, v in m.items():
        print(f"{k:22s} {v['t_conversion_us']:8.1f} {v['t_transfer_us']:12.1f} "
              f"{v['t_gemm_us']:8.1f} {v['t_total_no_overlap_us']:10.1f}")
    print("\nNOTE : estimations statiques, borne pessimiste sans overlap.")
    print("Premier test D2 reel = mesurer la conversion NVFP4->INT8 (CPU/GPU/NPU)")
    print("+ le GEMM INT8 NPU reel, puis verifier si A < B (pas de conversion)")
    print("et si le gain INT8 couvre le cout de conversion (avec cache/overlap).")