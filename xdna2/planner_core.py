# Phase 1.1 — planner_core.py : module unique du planner D2 (npu-rtx).
# Objectif (PLAN_EXECUTION_5070_NPU.md §1.1) : un point d'entrée qui importe
# place_expert + cost_contention et expose un résultat de plan structuré
# (PlanResult) — sans jamais modifier governor/ ni profiler_v3.
#
# RÈGLE D'OR (README npu-rtx) : chaque constante porte sa provenance
# ASSUMED / MEASURED / CALIBRATED. Le planner ne transforme JAMAIS
# silencieusement ASSUMED en MEASURED.

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

# Import des modules existants du dossier npu-rtx (le dossier xdna2 est un
# sous-module logique : on ajoute le parent au path pour rester importable
# depuis n'importe quel cwd).
_HERE = os.path.dirname(os.path.abspath(__file__))
_NPURT = os.path.dirname(_HERE)
if _NPURT not in sys.path:
    sys.path.insert(0, _NPURT)

from place_expert import (  # noqa: E402
    BW_RTX, BW_DDR, BW_GTT, NPU_TOPS,
    place_expert, planner_route, cost_ssd_ram_xdna, tile_util,
)
from cost_contention_patch import (  # noqa: E402
    contention_cost, xdna_expert_cost, npu_decode_tps, usage_note,
)

# ---------------------------------------------------------------------------
# Constantes calibrées (provenance explicite — RAPPORT_SYNTHESE_CORRECTIONS §1,
# CARTE_FONCTIONNELLE_XDNA2_FLM.md, mesure FLM Qwen3.5-9B 06/2026)
# ---------------------------------------------------------------------------
CONSTANTS = {
    "BW_RTX":          {"value": BW_RTX,          "prov": "MEASURED_SPEC",   "src": "NVIDIA RTX 5070 Laptop 384 GB/s GDDR7"},
    "BW_DDR":          {"value": BW_DDR,          "prov": "MEASURED_SPEC",   "src": "DDR5-5600 dual 89.6 GB/s"},
    "BW_GTT":          {"value": BW_GTT,          "prov": "MEASURED_OTHER_HW", "src": "1bit Strix Halo dma-buf 56 GB/s — Strix Point A RE-MESURER"},
    "BW_NPU_DDR_EFF":  {"value": 21.93e9,         "prov": "CALIBRATED",      "src": "FLM roofline decode Qwen3.5-9B (24.5% du peak)"},
    "NPU_TOPS_INT8":   {"value": NPU_TOPS,        "prov": "ASSUMED",         "src": "plafond 8-col Strix Halo ; Strix Point = 4 cols actives A MESURER"},
    "GPU_MOE_TPS":     {"value": 75.65,           "prov": "MEASURED_OTHER_HW", "src": "llama.cpp-Vulkan MoE 35B Strix Halo — RTX 5070 attendu plus haut"},
    "NPU_MOE_TPS":     {"value": 11.66,           "prov": "MEASURED_OTHER_HW", "src": "FLM NPU MoE 35B @1k ctx"},
    "IOCTL_PER_TOKEN": {"value": 3064,            "prov": "MEASURED",        "src": "reverse FLM : MCDM dispatch sans force_cmdlist (Windows)"},
    "VRAM_USABLE_GIB": {"value": 6.5,             "prov": "CALIBRATED",      "src": "8 GB - ~1.5 GB WDDM"},
    # Leçon §2 TIERS_HARDWARE : débit ∝ 1/poids lus par token
    "CONTENTION_THR":  {"value": 0.7,             "prov": "ASSUMED",         "src": "seuil communauté Strix Halo (cost_contention_patch)"},
}


@dataclass
class ExpertSpec:
    """Un expert à placer. freq_score ∈ [0,1] = fréquence d'activation."""
    expert_id: str
    freq_score: float
    size_bytes: int
    hidden_size: int = 2048
    layer: int = 0


@dataclass
class PlanResult:
    """Résultat structuré d'un plan de placement — le contrat du D2 planner."""
    placement: dict            # expert_id -> tier ("RTX" | "XDNA2" | "CPU" | "SSD")
    costs: dict                # expert_id -> {"RTX": s, "XDNA2": s, "CPU": s, "SSD": s}
    constraints: list = field(default_factory=list)   # violations détectées
    contention_penalty: float = 0.0
    provenance: dict = field(default_factory=dict)    # constantes utilisées + provenance
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2,
                          default=lambda o: getattr(o, "__dict__", str(o)))


def plan_experts(experts, bw_used=0.0, gpu_free_vram=6.5 * (1 << 30),
                 npu_cols_free=4, thermal_state=50, hidden_size=2048):
    """Planifie le placement de chaque expert via place_expert().

    experts : liste de ExpertSpec (ou tuples compatibles place_expert).
    Retourne un PlanResult avec coûts, contraintes et provenance.
    """
    placement, costs = {}, {}
    constraints = []

    # Colonnes XDNA2 actives : Strix Point expose 4 colonnes (cap 8) —
    # on borne par les colonnes libres passées en contexte.
    cols_active = max(1, min(4, npu_cols_free))
    for e in experts:
        tier = place_expert(
            e.expert_id, e.freq_score, e.size_bytes, e.hidden_size,
            gpu_free_vram, npu_cols_free, thermal_state,
            bw_used, CONSTANTS["BW_RTX"]["value"],
        )
        placement[e.expert_id] = tier
        c = xdna_expert_cost(e.size_bytes, e.hidden_size,
                             cols_active=cols_active)
        costs[e.expert_id] = {
            "RTX": e.size_bytes / CONSTANTS["BW_RTX"]["value"],
            "XDNA2": c,
            "CPU": e.size_bytes / CONSTANTS["BW_DDR"]["value"],
            "SSD": e.size_bytes / BW_GTT * 1.5,
        }

    # Contraintes dures (mirroir de oracle/feasibility.py)
    if gpu_free_vram <= 0:
        constraints.append("VRAM_gpu_epuisee: tout expert RTX sera rejeté")
    if npu_cols_free <= 0:
        constraints.append("XDNA2_cols_epuisees: tier NPU indisponible")
    if thermal_state > 70:
        constraints.append("thermie>70C: tier RTX gelé")

    pen = contention_cost(bw_used, 0.0, 0.0,
                          total_bw=CONSTANTS["BW_RTX"]["value"] / 1e9)

    return PlanResult(
        placement=placement,
        costs=costs,
        constraints=constraints,
        contention_penalty=pen,
        provenance={**CONSTANTS, "cost_model_v2": usage_note()},
        meta={"n_experts": len(experts), "bw_used": bw_used,
              "npu_cols_free": npu_cols_free, "thermal_state": thermal_state,
              "cols_active": cols_active},
    )


# Réexport pour adapter_bridge / tests
__all__ = ["CONSTANTS", "ExpertSpec", "PlanResult", "plan_experts",
           "place_expert", "planner_route", "cost_ssd_ram_xdna",
           "xdna_expert_cost", "npu_decode_tps", "tile_util",
           "contention_cost"]
