#!/usr/bin/env python3
"""feasibility.py - filtre de faisabilite : rejette les plans impossibles AVANT le Pareto.

Elimine un plan si une contrainte physique est violee :
  VRAM > disponible          -> REJECT
  L1 tile > 64 KiB           -> REJECT
  PCIe budget impossible     -> REJECT (lower-bound transfert > fenetre)
  workspace insuffisant      -> REJECT
  DMA impossible             -> REJECT (si contrainte connue)

Chaque plan rejete porte la raison (traceable). Seuls les plans FEASIBLE entrent
dans le score D2 (voir d2_planner).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "static"))

from expert_mapper import L1_CORE_BYTES
from bytes_per_token import bytes_moe_active

EXPERT_PARAMS = 4_915_200


class Plan:
    def __init__(self, name, fmt, experts_cache, device, ctx=8192,
                 vram_needed_gib=6.0, l1_footprint=32 * 1024,
                 pcie_bw_gbs=12.0, workspace_gib=0.5):
        self.name = name
        self.fmt = fmt
        self.experts_cache = experts_cache
        self.device = device
        self.ctx = ctx
        self.vram_needed_gib = vram_needed_gib
        self.l1_footprint = l1_footprint
        self.pcie_bw_gbs = pcie_bw_gbs
        self.workspace_gib = workspace_gib


def check(plan, vram_available_gib=6.5, cache_hit=0.9):
    """Retourne (feasible: bool, reasons: list)."""
    reasons = []
    if plan.vram_needed_gib > vram_available_gib:
        reasons.append(f"VRAM {plan.vram_needed_gib} GiB > dispo {vram_available_gib}")
    if plan.l1_footprint > L1_CORE_BYTES:
        reasons.append(f"L1 tile {plan.l1_footprint} B > {L1_CORE_BYTES} B")
    # lower-bound PCIe : actif/token * (1-hit) / BW
    active = bytes_moe_active(48, 10, EXPERT_PARAMS, plan.fmt)["total_bytes"]
    miss_b = active * (1.0 - cache_hit)
    t_miss = miss_b / (plan.pcie_bw_gbs * 1e9)
    # fenetre raisonnable : >= 4 t/s objectif
    if t_miss > 0.25:
        reasons.append(f"PCIe lower-bound {t_miss*1e3:.0f} ms/token > 250 ms (cache {cache_hit:.0%})")
    if plan.workspace_gib <= 0:
        reasons.append("workspace insuffisant")
    return (len(reasons) == 0, reasons)


def run_checks(plans, vram_available_gib=6.5):
    out = []
    for p in plans:
        ok, reasons = check(p, vram_available_gib)
        out.append({"plan": p.name, "fmt": p.fmt, "device": p.device,
                    "feasible": ok, "reasons": reasons})
    return out


if __name__ == "__main__":
    plans = [
        Plan("C0_q4_rtx", "Q4", 48, "RTX", vram_needed_gib=4.0),
        Plan("C0_q4_xdna_int8", "Q4", 48, "XDNA2-INT8", vram_needed_gib=4.0),
        Plan("C5_bf16_rtx", "BF16", 0, "RTX", vram_needed_gib=7.5),
        Plan("C1_q2_xdna", "Q2", 48, "XDNA2", vram_needed_gib=3.0),
        Plan("bad_l1", "Q4", 48, "XDNA2", l1_footprint=200 * 1024),
    ]
    for r in run_checks(plans):
        print(f"{'FEASIBLE ' if r['feasible'] else 'REJECT  '} {r['plan']:18s} "
              f"{r['fmt']:5s} {r['device']:10s} {r['reasons']}")