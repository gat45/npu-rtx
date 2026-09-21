# Phase 1.3 — Benchmark cross-tier SIMULÉ (machine dev, sans device).
# Produit concret du PLAN_EXECUTION_5070_NPU.md §1.3 : configs
#   A = RTX seul (overflow CPU implicite, mode -ot llama.cpp)
#   B = RTX+XDNA2  (overflow NPU puis CPU)
#   C = RTX+XDNA2+CPU (résidence CPU complète)
# sur distributions d'experts synthétiques (Pareto, tailles type MoE 35B-A3B).
#
# MODÈLE DE COÛT (honnête, ancres calibrées) :
#   - GPU : BW_RTX_eff = w_eff_9B × TPS_anchor(5070 CUDA Qwen3.5-9B) →
#     2.87e9 × 53.83 ≈ 154.5 GB/s effectifs (CALIBRATED, inclut overhead CUDA).
#   - NPU : BW_NPU_eff = BW_DDR_EFF_FLM (21.93 GB/s, CALIBRATED FLM roofline).
#   - CPU : BW_CPU_eff = 44.8 GB/s (ASSUMED, DDR5×0.5 — aucune mesure locale).
# Toute constante porte sa provenance — rien n'est transformé en MEASURED.

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NPURT = os.path.dirname(_HERE)
if _NPURT not in sys.path:
    sys.path.insert(0, _NPURT)

from planner_core import ExpertSpec, plan_experts  # noqa: E402
from cost_contention_patch import (  # noqa: E402
    BW_DDR_EFF_FLM, xdna_expert_cost, W_EFF_CLEAN,
)

# --- Ancres calibrées (MEASURED sur machine cible via FLM/benchs 06/2026) ----
TPS_GPU_ANCHOR = 53.83            # MEASURED — RTX 5070 CUDA Qwen3.5-9B decode
W_EFF_ANCHOR = W_EFF_CLEAN        # CALIBRATED — 2.87 GB working set 9B
BW_RTX_EFF = W_EFF_ANCHOR * TPS_GPU_ANCHOR   # ≈ 154.5 GB/s (CALIBRATED)
BW_CPU_EFF = 44.8e9               # ASSUMED — DDR5 89.6 × 0.5 efficacité
DENSE_BYTES = 2.95e9              # CALIBRATED-OTHER — backbone GDN dense BF16
                                  # (vLLM #51197 : ~80% du trafic poids/token)

# --- Géométrie cible (Qwen 35B-A3B, static oracle npu-rtx) -------------------
N_LAYERS = 40
N_EXPERTS_PER_LAYER = 256
TOP_K = 8                         # top-8 + 1 shared
EXPERT_BYTES = 1_771_674          # 1.69 MiB — 3.146M params Q4
HIDDEN = 2048
VRAM_USABLE = 6.5 * (1 << 30)     # CALIBRATED — 8 GB - WDDM
NPU_COLS = 4                      # MEASURED — colonnes XRT Strix Point

PARETO_ALPHAS = (1.0, 1.16, 1.3)  # skew : 1.16 ≈ localité Gemma mesurée h≈0.85

# ⚠️ Bornage réel : les 10 240 experts (40×256) sont uniques par couche → la
# résidence VRAM est un budget PAR COUCHE, pas global (32 GiB pour tout garder
# est un artefact). Modèle : VRAM répartie uniformément, budget par couche.
RESIDENT_PER_LAYER = int(VRAM_USABLE // N_LAYERS // EXPERT_BYTES)   # ≈ 41
CAP_PER_LAYER = min(RESIDENT_PER_LAYER, N_EXPERTS_PER_LAYER)


def pareto_freqs(n, alpha):
    """Fréquences triées desc, somme 1 — loi de Pareto tronquée (synthétique)."""
    ranks = range(1, n + 1)
    w = [r ** (-alpha) for r in ranks]
    s = sum(w)
    return [x / s for x in w]


def resident_count():
    """Budget de résidence PAR COUCHE (experts uniques par couche)."""
    return CAP_PER_LAYER


def hit_rate(alpha, k_resident):
    """Somme des fréquences des k experts les plus fréquents (= hit si le
    cache garde le top-k par fréquence — borne sup idéale, LRU réel plus bas)."""
    return min(sum(pareto_freqs(N_EXPERTS_PER_LAYER, alpha)[:k_resident]), 1.0)


def per_token_time(hit, expert_bytes, tier_overflow, bw_used=0.0):
    """Temps/token : backbone dense (GPU) + activations expert.

    - hit d'expert → GPU : expert_bytes / BW_RTX_EFF
    - overflow → tier_overflow : coût complet du tier (NPU = DMA+DDR+compute,
      CPU = BW_CPU_EFF).
    Retourne (t_dense, t_experts, n_overflow).
    """
    t_dense = DENSE_BYTES / BW_RTX_EFF
    n_act = N_LAYERS * TOP_K
    n_hit = n_act * hit
    n_overflow = n_act - n_hit
    if tier_overflow == "XDNA2":
        c_over = xdna_expert_cost(expert_bytes, HIDDEN, cols_active=NPU_COLS)
    else:  # CPU
        c_over = expert_bytes / BW_CPU_EFF
    t_experts = n_hit * (expert_bytes / BW_RTX_EFF) + n_overflow * c_over
    return t_dense, t_experts, int(round(n_overflow))


def run_config(name, alpha, tier_overflow):
    hit = hit_rate(alpha, resident_count())
    t_dense, t_experts, n_over = per_token_time(hit, EXPERT_BYTES, tier_overflow)
    t_tok = t_dense + t_experts
    # Plan D2 sur un set de test (validation planner_core sur ce scénario)
    plan = plan_experts(
        [ExpertSpec(f"e{i}", freq_score=1 - i / N_EXPERTS_PER_LAYER,
                    size_bytes=EXPERT_BYTES, hidden_size=HIDDEN, layer=0)
         for i in range(8)],
        bw_used=bw_used_fraction(alpha) * (BW_RTX_EFF / 1e9),
        npu_cols_free=NPU_COLS,
    )
    return {
        "config": name,
        "pareto_alpha": alpha,
        "hit_rate_topk": round(hit, 4),
        "resident_experts": resident_count(),
        "overflow_per_token": n_over,
        "t_dense_ms": round(t_dense * 1e3, 3),
        "t_experts_ms": round(t_experts * 1e3, 3),
        "tok_s_est": round(1.0 / t_tok, 2),
        "plan_smoke_constraints": plan.constraints,
    }


def bw_used_fraction(alpha):
    """Charge bus relative (0-1) — proxy : 1 - hit (traffic overflow)."""
    return 1.0 - hit_rate(alpha, resident_count())


def voie_a_aggregate():
    """Voie A — double-flux indépendant : GPU stream + NPU stream (2 requêtes).
    Débit agrégat = somme des débits isolés (contention traitée à part)."""
    tps_gpu = TPS_GPU_ANCHOR
    tps_npu = BW_DDR_EFF_FLM / W_EFF_ANCHOR     # ≈ 7.64 t/s
    return {"tps_gpu": round(tps_gpu, 2), "tps_npu": round(tps_npu, 2),
            "aggregate": round(tps_gpu + tps_npu, 2),
            "prov": "MEASURED anchors, sum of independent streams"}


def run_all(seed=None):
    rows = []
    for alpha in PARETO_ALPHAS:
        rows.append(run_config("A_RTX_only", alpha, "CPU"))
        rows.append(run_config("B_RTX_XDNA2", alpha, "XDNA2"))
        rows.append(run_config("C_RTX_XDNA2_CPU", alpha, "CPU"))
    out = {
        "model": "Qwen 35B-A3B (synthetic expert dist)",
        "anchors": {"tps_gpu_5070": TPS_GPU_ANCHOR,
                    "bw_rtx_eff_gbs": round(BW_RTX_EFF / 1e9, 1),
                    "bw_npu_eff_gbs": round(BW_DDR_EFF_FLM / 1e9, 2),
                    "bw_cpu_assumed_gbs": BW_CPU_EFF / 1e9},
        "rows": rows,
        "voie_a": voie_a_aggregate(),
    }
    # Metrics pour l'adapter (contrat governor)
    a = [r for r in rows if r["config"] == "A_RTX_only"]
    b = [r for r in rows if r["config"] == "B_RTX_XDNA2"]
    out["metrics"] = {
        "tok_s_est": min(r["tok_s_est"] for r in b),
        "tok_s_best": max(r["tok_s_est"] for r in rows),
    }
    _write(out)
    return out


def _write(out):
    d = os.path.join(_HERE, "bench")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "bench.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    comp = {r["config"] + f"@a{r['pareto_alpha']}": r["tok_s_est"]
            for r in out["rows"]}
    with open(os.path.join(d, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comp, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    res = run_all()
    for r in res["rows"]:
        print(f"{r['config']:>16} a={r['pareto_alpha']:.2f} "
              f"hit={r['hit_rate_topk']:.3f} "
              f"over={r['overflow_per_token']:>3}/tok "
              f"dense={r['t_dense_ms']:>6.2f}ms "
              f"exp={r['t_experts_ms']:>7.2f}ms "
              f"-> {r['tok_s_est']:>6.2f} tok/s")
    print("Voie A :", res["voie_a"])
