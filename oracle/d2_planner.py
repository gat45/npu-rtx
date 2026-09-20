#!/usr/bin/env python3
"""d2_planner.py - D2 Planner : Static Oracle + Dynamic Oracle + feasibility -> Pareto.

Pipeline : Static Oracle (bornes) -> feasibility filter (rejet impossible) ->
score(plan) avec lambda calibres -> frontiere de Pareto.

Les lambda sont ASSUMED par defaut ; ils doivent etre CALIBRES par les mesures
profiler-v3 (Phase C). Chaque prediction porte mean/variance/samples/confidence.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "static"))
sys.path.insert(0, os.path.dirname(__file__))

from feasibility import check, Plan
from bytes_per_token import bytes_moe_active

EXPERT_PARAMS = 4_915_200


def score(plan, lambdas=None, cache_hit=0.9, overlap=0.0, planner_overhead_us=50.0):
    """T(plan) = T_route+T_SSD+T_RAM+T_PCIe+T_DMA+T_conversion+T_compute+T_sync - T_overlap.

    Corrections angle morts 21/52/66 :
    - T_overlap soustrait le chevauchement (DMA masque par compute)
    - planner_overhead_us : ne pas replan chaque token si planning > savings
    - CACHE=OFF possible : cache_hit=0 (miss_rate=1) est une vraie strategie
    """
    if lambdas is None:
        lambdas = {"l1": 1e-3, "l2": 1e-3, "l3": 1e-3, "l4": 1e-3}
    active = bytes_moe_active(48, 10, EXPERT_PARAMS, plan.fmt)["total_bytes"]
    miss_b = active * (1.0 - cache_hit)
    pcie_bytes = miss_b
    # latence : transfert miss / BW + GEMM (placeholder) + sync + overhead planner
    t_transfer = miss_b / (plan.pcie_bw_gbs * 1e9)
    t_gemm = 1e-3  # placeholder
    t_sync = 0.2e-3  # placeholder
    t_overlap = overlap * t_transfer  # part du transfert masquee
    t_planner = planner_overhead_us * 1e-6
    latency = t_transfer + t_gemm + t_sync - t_overlap + t_planner
    return {
        "name": plan.name,
        "fmt": plan.fmt,
        "device": plan.device,
        "latency_s": latency,
        "pcie_bytes_token": pcie_bytes,
        "memory_pressure_gib": plan.vram_needed_gib,
        "cache_hit": cache_hit,
        "t_overlap_s": t_overlap,
        "t_planner_s": t_planner,
        "score": latency + lambdas["l1"] * pcie_bytes + lambdas["l4"] * plan.vram_needed_gib,
        "confidence": 0.5,  # ASSUMED -> calibration requise
        "samples": 0,
    }


def pareto_front(plans, vram_available_gib=6.5):
    """Filtre feasibility puis retourne les plans non-domines (latence, VRAM, PCIe)."""
    feasible = []
    for p in plans:
        ok, reasons = check(p, vram_available_gib)
        if ok:
            s = score(p)
            feasible.append(s)
    # dominance : plan A domine B si meilleur sur tout et strictement sur au moins un
    front = []
    for a in feasible:
        dominated = False
        for b in feasible:
            if b is a:
                continue
            b_better = (b["latency_s"] <= a["latency_s"]
                        and b["memory_pressure_gib"] <= a["memory_pressure_gib"]
                        and b["pcie_bytes_token"] <= a["pcie_bytes_token"])
            b_strict = (b["latency_s"] < a["latency_s"]
                        or b["memory_pressure_gib"] < a["memory_pressure_gib"]
                        or b["pcie_bytes_token"] < a["pcie_bytes_token"])
            if b_better and b_strict:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


if __name__ == "__main__":
    plans = [
        Plan("C0_q4_rtx", "Q4", 48, "RTX", vram_needed_gib=4.0),
        Plan("C0_q4_xdna_int8", "Q4", 48, "XDNA2-INT8", vram_needed_gib=4.0),
        Plan("C1_q2_xdna", "Q2", 48, "XDNA2", vram_needed_gib=3.0),
        Plan("C3_q6_rtx", "Q6_K", 32, "RTX", vram_needed_gib=5.0),
        Plan("C5_bf16_rtx", "BF16", 0, "RTX", vram_needed_gib=7.5),
    ]
    front = pareto_front(plans)
    print("Front de Pareto (feasible + non-domine) :")
    for f in front:
        print(f"  {f['name']:18s} {f['fmt']:5s} {f['device']:10s} "
              f"latency={f['latency_s']*1e3:6.1f}ms pcie={f['pcie_bytes_token']/1024**3:.3f}GiB/t "
              f"vram={f['memory_pressure_gib']}GiB score={f['score']:.4f}")
    print("\nNOTE : lambdas et temps = ASSUMED. Calibration profiler-v3 requise (Phase C).")