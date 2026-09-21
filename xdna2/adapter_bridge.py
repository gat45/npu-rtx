# Phase 1.1 — adapter_bridge.py : pont conforme au contrat governor/adapter.py
# (PLAN_EXECUTION_5070_NPU.md §1.1) : inclusion DYNAMIQUE de l'adapter npu-rtx
# dans le governor, AUCUNE modification de governor/adapter.py.
#
# Le governor découvre l'adapter via :
#   from adapter_bridge import NpuRtxAdapter
#   governor.register_adapter(NpuRtxAdapter())   # si l'API le permet
# — ou en dernier recours NpuRtxAdapter se substitue à la classe ProjectAdapter
# locale (fallback, même contrat, même signatures).

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NPURT = os.path.dirname(_HERE)
if _NPURT not in sys.path:
    sys.path.insert(0, _NPURT)

from planner_core import CONSTANTS, ExpertSpec, PlanResult, plan_experts  # noqa: E402

# --- Contrat : on tente d'hériter du vrai contrat governor (dynamique) -------
try:
    _GH = os.path.dirname(_NPURT)          # geniex_harness/
    if _GH not in sys.path:
        sys.path.insert(0, _GH)
    from governor.adapter import ProjectAdapter as _Base  # type: ignore
    _BASE_PROV = "governor.adapter.ProjectAdapter (dynamique)"
except Exception:                           # governor absent / import lourd KO
    class _Base:                            # fallback minimal, même contrat
        id = "base"
        def action_space(self): raise NotImplementedError
        def resources(self): return {"compute_min": 120, "tool_calls": 20, "test_runs": 10}
        def metrics(self): return [("quality", "max")]
        def test_pool(self): return []
        def constraints(self): return {"rmax": 0.5, "gmin": 0.0}
        def observe_state(self): return {}
        def execute(self, spec, ctx): raise NotImplementedError
        def measure(self, raw): return {}
        def rollback_last(self): return {"rolled_back": False}
    _BASE_PROV = "fallback local (governor absent)"


class NpuRtxAdapter(_Base):
    """Adapter du planner D2 npu-rtx (5070 + XDNA2 + bus mémoire).

    Conforme au contrat ProjectAdapter : ACTIONS / RESOURCES / METRICS /
    TESTS / CONSTRAINTS / EXECUTOR. Toutes les actions sont réversibles
    (simulation pure côté dev — aucun effet device).
    """

    id = "npu-rtx-d2"

    def __init__(self, npurt_root=None):
        self.root = npurt_root or _NPURT
        self._last_plan = None            # pour rollback_last()
        self._base_prov = _BASE_PROV

    # -- ACTIONS ------------------------------------------------------------
    def action_space(self):
        # Import paresseux : ActionSpec peut être indisponible si governor
        # est absent (fallback) — on retombe sur des dicts compatibles.
        try:
            from governor.adapter import ActionSpec
            def spec(name, kind, cost, risk, gain, reversible=True):
                return ActionSpec(name, kind, cost, risk, gain, reversible)
        except Exception:
            def spec(name, kind, cost, risk, gain, reversible=True):
                return {"name": name, "kind": kind, "cost_min": cost,
                        "risk": risk, "info_gain": gain,
                        "reversible": reversible, "executor_ref": None}
        return [
            spec("d2_plan_experts", "evaluate", 0.5, 0.02, 0.5),
            spec("d2_benchmark_sim", "evaluate", 1.0, 0.05, 0.7),
            spec("d2_cost_model", "evaluate", 0.5, 0.02, 0.4),
        ]

    # -- RESOURCES / METRICS / TESTS / CONSTRAINTS ---------------------------
    def resources(self):
        return {"compute_min": 60, "tool_calls": 10, "test_runs": 5}

    def metrics(self):
        return [("tok_s_est", "max"), ("contention_penalty", "min"),
                ("n_constraints", "min")]

    def test_pool(self):
        return ["T_d2_plan_smoke", "T_d2_bench_smoke"]

    def constraints(self):
        return {"rmax": 0.3, "gmin": -1.0}

    # -- OBSERVE / EXECUTE / MEASURE / ROLLBACK ------------------------------
    def observe_state(self):
        # Machine dev : GTX 1080, pas de NPU XDNA2 — état déclaré honnête.
        return {"machine": "dev_gtx1080",
                "xdna2_present": False,
                "target_available": False,
                "constants_provenance": {k: v["prov"]
                                         for k, v in CONSTANTS.items()}}

    def execute(self, spec, ctx):
        name = spec.name if hasattr(spec, "name") else spec.get("name")
        try:
            if name == "d2_plan_experts":
                experts = ctx.get("experts") or _default_experts()
                plan = plan_experts(
                    [e if isinstance(e, ExpertSpec) else ExpertSpec(**e)
                     for e in experts],
                    bw_used=ctx.get("bw_used", 0.0),
                    npu_cols_free=ctx.get("npu_cols_free", 4),
                )
                self._last_plan = plan
                return {"ok": True, "action": name,
                        "metrics": {"tok_s_est": None,
                                    "n_constraints": len(plan.constraints)},
                        "raw": plan.to_json()[:2000]}

            if name == "d2_benchmark_sim":
                from benchmark_sim import run_all
                res = run_all(seed=ctx.get("seed"))
                self._last_plan = None
                return {"ok": True, "action": name, "metrics": res["metrics"],
                        "raw": json.dumps({"rows": res["rows"],
                                           "voie_a": res["voie_a"]},
                                          ensure_ascii=False)[:2000]}

            if name == "d2_cost_model":
                experts = ctx.get("experts") or _default_experts()
                plan = plan_experts([e if isinstance(e, ExpertSpec)
                                     else ExpertSpec(**e) for e in experts])
                return {"ok": True, "action": name,
                        "metrics": {"contention_penalty":
                                    plan.contention_penalty},
                        "raw": str(plan.costs)[:2000]}

            return {"ok": False, "action": name, "error": "action inconnue"}
        except Exception as e:  # jamais lever hors du contrat
            return {"ok": False, "action": name, "error": str(e)}

    def measure(self, raw):
        m = (raw or {}).get("metrics") or {}
        return {k: v for k, v in m.items() if v is not None}

    def rollback_last(self):
        p = self._last_plan
        self._last_plan = None
        return {"rolled_back": p is not None, "variant": "d2_plan" if p else None}


def _default_experts():
    """Set de test : 8 experts type Qwen MoE (3.146M params Q4 ≈ 1.69 MiB)."""
    return [ExpertSpec(f"exp_{i}", freq_score=0.9 - i * 0.1,
                       size_bytes=1_771_674, hidden_size=2048, layer=i // 4)
            for i in range(8)]
