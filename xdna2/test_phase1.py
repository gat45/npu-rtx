# Tests unitaires Phase 1 — planner_core + adapter_bridge + benchmark_sim
# Exécution : py -m unittest test_phase1 -v   (depuis npu-rtx/xdna2/)

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planner_core import CONSTANTS, ExpertSpec, PlanResult, plan_experts
from adapter_bridge import NpuRtxAdapter
from benchmark_sim import hit_rate, resident_count, run_all
from cost_contention_patch import (BW_DDR_EFF_FLM, xdna_expert_cost,
                                   npu_decode_tps, w_eff_contended,
                                   W_EFF_CLEAN, BUS_PATHS, usage_note)


class TestPlannerCore(unittest.TestCase):
    def test_plan_basic(self):
        experts = [ExpertSpec(f"e{i}", freq_score=0.9 - i * 0.1,
                              size_bytes=1_771_674) for i in range(8)]
        plan = plan_experts(experts, npu_cols_free=4)
        self.assertIsInstance(plan, PlanResult)
        self.assertEqual(len(plan.placement), 8)
        self.assertIn("MEASURED", plan.provenance["BW_RTX"]["prov"])
        self.assertIn("RTX", plan.provenance["BW_RTX"]["src"])
        # Coûts positifs et provenance explicite (règle d'or npu-rtx)
        self.assertIn("MEASURED", plan.provenance["BW_RTX"]["prov"])
        self.assertIn("RTX", plan.provenance["BW_RTX"]["src"])
        self.assertGreater(plan.costs["e0"]["XDNA2"], 0)
        self.assertGreater(plan.costs["e0"]["CPU"], 0)

    def test_constraints_reported(self):
        experts = [ExpertSpec("e0", 0.9, 1_771_674)]
        plan = plan_experts(experts, thermal_state=80)
        self.assertTrue(any("thermie" in c for c in plan.constraints))
        plan2 = plan_experts(experts, npu_cols_free=0)
        self.assertTrue(any("cols" in c for c in plan2.constraints))

    def test_provenance_never_silent(self):
        plan = plan_experts([ExpertSpec("e0", 0.9, 1_771_674)])
        prov = plan.provenance["cost_model_v2"]
        self.assertIn("host_coherent", prov["bus_paths"])
        self.assertIn("UNKNOWN", prov["bus_paths"]["host_coherent"])


class TestCostModelV2(unittest.TestCase):
    def test_anchor_npu_tps(self):
        # Ancre FLM : 21.93 / 2.87 ≈ 7.64 t/s (mesuré 7.06-7.43, écart ~5%)
        self.assertAlmostEqual(npu_decode_tps(W_EFF_CLEAN), 7.64, places=1)

    def test_contention_inflation(self):
        self.assertGreater(w_eff_contended(), W_EFF_CLEAN)
        self.assertAlmostEqual(w_eff_contended() / W_EFF_CLEAN, 1.72, places=1)

    def test_xdna_cost_components_positive(self):
        c = xdna_expert_cost(1_771_674, 2048, cols_active=4)
        self.assertGreater(c, 0)
        c2 = xdna_expert_cost(1_771_674, 2048, cols_active=8)
        self.assertLess(c2, c)  # plus de colonnes = compute plus rapide

    def test_host_coherent_unknown_flagged(self):
        self.assertIsNone(BUS_PATHS["host_coherent"]["bw"])
        note = usage_note()
        self.assertIn("P0.6", note["unknown"][0])


class TestBenchmarkSim(unittest.TestCase):
    def test_hit_rate_monotonic_in_alpha(self):
        k = resident_count()
        h1 = hit_rate(1.0, k)
        h116 = hit_rate(1.16, k)
        self.assertGreaterEqual(h116, h1)   # plus skewé = plus de hit top-k

    def test_run_all_produces_outputs(self):
        res = run_all()
        self.assertIn("rows", res)
        self.assertIn("voie_a", res)
        self.assertGreater(len(res["rows"]), 0)
        for r in res["rows"]:
            self.assertGreater(r["tok_s_est"], 0)
        # Voie A : agrégat > GPU seul (2 flux indépendants)
        self.assertGreater(res["voie_a"]["aggregate"],
                           res["voie_a"]["tps_gpu"])

    def test_bench_files_written(self):
        run_all()
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench")
        self.assertTrue(os.path.exists(os.path.join(d, "bench.json")))
        with open(os.path.join(d, "bench.json"), encoding="utf-8") as f:
            json.load(f)  # JSON valide


class TestAdapterBridge(unittest.TestCase):
    def test_contract(self):
        ad = NpuRtxAdapter()
        self.assertEqual(ad.id, "npu-rtx-d2")
        actions = ad.action_space()
        names = [a["name"] if isinstance(a, dict) else a.name
                 for a in actions]
        self.assertIn("d2_plan_experts", names)
        self.assertTrue(ad.metrics())
        self.assertIn("rmax", ad.constraints())
        st = ad.observe_state()
        self.assertFalse(st["xdna2_present"])  # machine dev : honnête

    def test_execute_plan(self):
        ad = NpuRtxAdapter()
        spec = {"name": "d2_plan_experts"}
        r = ad.execute(spec, {})
        self.assertTrue(r["ok"])
        self.assertIsNone(r["metrics"]["tok_s_est"])  # plan seul : pas de débit
        rb = ad.rollback_last()
        self.assertTrue(rb["rolled_back"])

    def test_execute_bench(self):
        ad = NpuRtxAdapter()
        r = ad.execute({"name": "d2_benchmark_sim"}, {"seed": 42})
        self.assertTrue(r["ok"])
        self.assertGreater(r["metrics"]["tok_s_best"], 0)
        m = ad.measure(r)
        self.assertIn("tok_s_best", m)

    def test_execute_unknown_action(self):
        ad = NpuRtxAdapter()
        r = ad.execute({"name": "inconnue"}, {})
        self.assertFalse(r["ok"])
        self.assertIn("inconnue", r["error"])


if __name__ == "__main__":
    unittest.main()
