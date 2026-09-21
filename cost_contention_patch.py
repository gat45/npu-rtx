# Étape 3 (v2) — Métrique contention mémoire + cost model NPU calibré FLM
# v2 (Phase 1.2, PLAN_EXECUTION_5070_NPU.md) :
#   - scission du bus NPU : host_coherent vs GTT (dma-buf)
#   - constantes calibrées FLM injectées (BW_eff 21.93 GB/s, η 0.731,
#     W_eff 2.87 / 4.93 GB, KV 30000 tokens)
#   - pénalités DDR5 mesurées (iGPU, page file, colonnes idle)
# Compat : contention_cost() garde sa signature d'origine (planner_core l'importe).

import sys
import warnings

from place_expert import tile_util, NPU_TOPS as _NPU_TOPS_8COL  # noqa: E402

# ---------------------------------------------------------------------------
# Bus NPU : deux chemins de coût distinct (scission v2)
# ---------------------------------------------------------------------------
BUS_PATHS = {
    # DMA host->NPU via dma-buf/GTT — mesuré 1bit sur Strix Halo ; Strix Point
    # (machine cible) : A RE-MESURER (test P0.6).
    "gtt": {"bw": 56.0e9, "prov": "MEASURED_OTHER_HW (1bit Strix Halo)"},
    # Chemin host-coherent : aucune mesure locale -> UNKNOWN jusqu'à P0.6.
    "host_coherent": {"bw": None, "prov": "UNKNOWN (P0.6 requis)"},
}

# ---------------------------------------------------------------------------
# Constantes calibrées FLM (CARTE_FONCTIONNELLE_XDNA2_FLM.md, Qwen3.5-9B 06/2026)
# ---------------------------------------------------------------------------
BW_DDR_PEAK = 89.6e9        # MEASURED_SPEC — DDR5-5600 dual
BW_DDR_EFF_FLM = 21.93e9    # CALIBRATED — roofline FLM decode (24.5% du peak).
#                            ⚠️ inclut DÉJÀ les causes ci-dessous (iGPU, colonnes
#                            idle, page file, overhead XRT ×3.7) : ne pas
#                            re-décompter une seconde fois.
ETA_ROOFLINE = 0.731        # CALIBRATED — efficacité roofline (branche compute)
W_EFF_CLEAN = 2.87e9        # CALIBRATED — working set Qwen3.5-9B propre
W_EFF_CONTENDED = 4.93e9    # CALIBRATED — même modèle sous contention (+72%)
KV_CTX_TOKENS = 30000       # CALIBRATED — budget KV de référence (30k ctx)
PEN_IGPU = 3.5e9            # MEASURED — framebuffer 2560x1600@180Hz (iGPU 880M)
PEN_IDLE_COLS = 0.40        # MEASURED — column-mapping tuiles idle (-40%)
PEN_PAGEFILE = 0.10         # MEASURED — 92.4% RAM utilisée (-10%)
XRT_OVERHEAD = 3.7          # MEASURED — overhead dispatch XRT (MCDM sans
#                            force_cmdlist sous Windows) — cause incluse dans
#                            BW_DDR_EFF_FLM, exposé ici pour analyse.
CONTENTION_THR = 0.7        # ASSUMED — seuil communauté Strix Halo
NPU_COLS_MAX = 8            # MEASURED (reverse 1bit) ; Strix Point : 4 exposées
DENSE_BACKBONE_BYTES = 2.95e9   # CALIBRATED-OTHER — vLLM #51197 : backbone GDN
#                            dense ≈ 80% du trafic poids/token sur Qwen3.6-35B

CONTENDED_RATIO = W_EFF_CONTENDED / W_EFF_CLEAN   # ~1.72 (inflation contention)


def sync_cost_v2(size_bytes, path="gtt", bw_override=None):
    """Coût DMA host<->NPU sur un chemin bus donné (scission v2).

    Chemin 'host_coherent' non mesuré : fallback GTT explicite (warning),
    jamais silencieux — règle d'or npu-rtx.
    """
    spec = BUS_PATHS.get(path)
    if spec is None:
        raise ValueError(f"chemin bus inconnu : {path} (attendu gtt|host_coherent)")
    bw = bw_override or spec["bw"]
    if bw is None:
        warnings.warn("BW host_coherent UNKNOWN (P0.6 requis) -> fallback GTT",
                      stacklevel=2)
        bw = BUS_PATHS["gtt"]["bw"]
    return size_bytes / bw


def effective_ddr_bw(ram_used_frac=0.5, igpu_active=True, idle_cols=True):
    """BW DDR5 modélisée (peak - pénalités mesurées). Modèle de diagnostic :
    l'ancre de prédiction reste BW_DDR_EFF_FLM (mesurée, toutes causes incluses).
    """
    bw = BW_DDR_PEAK - (PEN_IGPU if igpu_active else 0.0)
    if idle_cols:
        bw *= (1.0 - PEN_IDLE_COLS)
    if ram_used_frac > 0.9:
        bw *= (1.0 - PEN_PAGEFILE)
    return bw


def npu_decode_tps(w_eff_bytes, bw_eff=BW_DDR_EFF_FLM):
    """Débit decode NPU memory-bound : tps = BW_eff / W_eff.

    Référence : 21.93e9 / 2.87e9 = 7.64 t/s (mesuré FLM : 7.06-7.43, écart ~5%).
    Sous contention : W_eff inflaté (mesuré 4.45 vs 4.27-4.65 t/s).
    """
    return bw_eff / max(w_eff_bytes, 1.0)


def w_eff_contended(w_eff_bytes=W_EFF_CLEAN, contended=True):
    """Working set effectif sous contention (mesuré : +~72% sur Qwen3.5-9B)."""
    return w_eff_bytes * CONTENDED_RATIO if contended else w_eff_bytes


def xdna_expert_cost(size_bytes, hidden, cols_active=4, path="gtt"):
    """Coût complet de service d'un expert sur XDNA2 (DMA + DDR + compute).

    tops scalé linéairement par colonnes actives (ASSUMED linéaire — à
    calibrer P0.6). Strix Point = 4 colonnes exposées XRT.
    """
    tops = _NPU_TOPS_8COL * (cols_active / float(NPU_COLS_MAX))
    t_dma = sync_cost_v2(size_bytes, path)
    t_ddr = size_bytes / BW_DDR_EFF_FLM          # chemin DDR effectif mesuré
    t_cmp = (2.0 * size_bytes * hidden) / (tops * 1e12) / tile_util(hidden)
    return t_dma + t_ddr + t_cmp


def contention_cost(bw_used_rtx, bw_used_npu, bw_used_cpu, total_bw=226):
    """Compat v1 : pénalité quadratique au-delà du seuil (unités GB/s)."""
    contention = (bw_used_rtx + bw_used_npu + bw_used_cpu) / total_bw
    return contention**2 * 0.25 if contention > CONTENTION_THR else 0.0


def usage_note():
    return {
        "bus_paths": {k: v["prov"] for k, v in BUS_PATHS.items()},
        "anchors": {
            "npu_decode_tps_clean": npu_decode_tps(W_EFF_CLEAN),
            "npu_decode_tps_contended": npu_decode_tps(w_eff_contended()),
        },
        "warning": "BW_DDR_EFF_FLM inclut déjà XRT_OVERHEAD — ne pas redécompter",
        "unknown": ["BW host_coherent (P0.6)", "scaling cols non-linéaire?"],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(usage_note(), indent=2, ensure_ascii=False))
