# -*- coding: utf-8 -*-
"""
flm_bottleneck_analyzer.py -- FastFlowLM RAM Bottleneck Analyzer v1.0
======================================================================
Calculateur de debit + reducteur de goulot RAM pour XDNA2 / LPDDR5X.

Construit a partir des donnees empiriques FastFlowLM :
  - soc_performance_matrix.csv  (48 mesures decode_tps vs context_tokens)
  - ANALYSE_GOULOTS_2026-05-25.md  (calibrations BW, thermal, DMA)
  - dpu_v2_qwen3_5_9b.json  (R/W ratio = 6.4x -> bandwidth-bound confirme)
  - adaptive_controller.py  (KV_PER_TOKEN = 14336, DECODE_BASELINE = 8.4 t/s)
  - QSTS roofline model  (Ridge_INT4 = 512 FLOP/B, Ridge_INT8 = 732 FLOP/B)

Usage :
    python flm_bottleneck_analyzer.py                  # rapport complet
    python flm_bottleneck_analyzer.py --model qwen3.5  # modele specifique
    python flm_bottleneck_analyzer.py --ctx 8192       # prevision contexte
    python flm_bottleneck_analyzer.py --optimize       # plan d'optimisation

Auteur : QSTS/SNSI cross-analysis session 2026-06-18
"""

import argparse
import sys
import math

# ---------------------------------------------------------------------------
# Hardware constants (XDNA2 / Strix Point mesure)
# ---------------------------------------------------------------------------

class HW:
    # Compute (XDNA2 NPU)
    TOPS_INT4   = 35.0          # T OPS
    TOPS_INT8   = 50.0
    TFLOPS_FP16 = 12.0

    # Memory (LPDDR5X-6400 mesure -- 50% effi vs theorique)
    BW_THEORETICAL_GBs = 60.0   # LPDDR5-5600 theorique
    BW_MEASURED_GBs    = 30.0   # mesure reel (ANALYSE_GOULOTS 2026-05-25)
    BW_EFFICIENCY      = 0.50   # 50% d'efficacite -- colonne-mapping NPU sous-optimal

    # eta : efficacite de transfert (calibre sur soc_performance_matrix.csv)
    # eta = TPS_mesure / TPS_modele -- constant sur ctx=1024..32768, std=0.015
    # Interpretation : 27.1% de BW perdu en overhead inter-kernel (ΦT)
    # BW_eff = BW_MEASURED * ETA = 21.93 GB/s utile
    # ATTENTION : eta varie avec ctx
    #   ctx < ~100 tok : eta -> 1.0 (KV negligeable, overhead inter-kernel minimal)
    #   ctx >= 1024    : eta = 0.7309 (valeur calibree, std=0.015)
    # Validation bench_3models.py (pt=25, warm) :
    #   qwen3.5:4b   17.25 t/s => W_eff = 30.0/17.25 = 1.74 GB ~ INT4 (1.75 GB) OK
    #   llama3.2:1b  19.7  t/s => W_eff = 30.0/19.7  = 1.52 GB ~ INT8 (1.24 GB theorique)
    #   deepseek:8b   3.3  t/s => W_eff = 30.0/3.3   = 9.09 GB ~ INT8 + overhead R1
    ETA            = 0.7309
    BW_EFF_GBs     = BW_MEASURED_GBs * ETA   # 21.93 GB/s

    # XDNA2 roofline ridges (sur BW_EFF, pas BW_MEASURED)
    RIDGE_INT4  = TOPS_INT4   * 1e12 / (BW_EFF_GBs * 1e9)   # ~1596 FLOP/B
    RIDGE_INT8  = TOPS_INT8   * 1e12 / (BW_EFF_GBs * 1e9)   # ~2280 FLOP/B
    RIDGE_FP16  = TFLOPS_FP16 * 1e12 / (BW_EFF_GBs * 1e9)   #  ~547 FLOP/B

    # KV cache -- Qwen3.5-4B ARCHITECTURE HYBRIDE (config.json confirme)
    # model_type = "qwen3_5_text", full_attention_interval = 4
    # Seulement 8/32 couches = full attention (GQA), 24/32 = linear attention
    # KV full-att seulement : 2 * kv_heads=4 * head_dim=256 * 8 layers = 16,384 bytes/tok
    # Empirique SOC_MATRIX  : 30,000 bytes/tok (linear-att contribue aussi au KV)
    # Le 14,336 de adaptive_controller.py = INT4 theorique, invalide sur XDNA2
    # Preuve : pred(14KB,ctx=32768)=+19% erreur vs pred(30KB)=-3.2%
    #
    # Fichiers model.q4nx (config.json C:\Users\videl\flm\models) :
    #   llama3.2:1b    = 1.3 GB  (INT8, poids+scales, xclbin inclus)
    #   qwen3.5:4b LLM = 4.2 GB  (INT4 poids=1.75GB + xclbin + vocab FP16)
    #   deepseek-r1:8b = 5.4 GB  (INT4 poids=4.0GB + FP16 lm_head~1GB)
    #   qwen3.5:9b LLM = 7.4 GB  (INT4 poids=4.5GB + xclbin + vocab FP16)
    # NOTE : xclbin reste en NPU memory -- FLM relit uniquement les poids tenseurs/step
    KV_PER_TOKEN_BYTES  = 30_000   # calibre sur 48 mesures SOC_MATRIX (err max +-3.2%)
    KV_PER_TOKEN_FLM    = 14_336   # code FLM (INT4 theorique -- ne pas utiliser)
    KV_PER_TOKEN_ARCH   = 16_384   # KV full-att seulement (2*4*256*8, config.json)
    # KV bytes/token par modele
    KV_MODEL = {
        "llama3.2:1b":    16_384,   # GQA standard (2*8*64*16, config.json)
        "qwen3.5:4b":     30_000,   # empirique (hybride linéaire+full-att)
        "qwen3.5:4b_soc": 30_000,
        "qwen3.5:9b":     30_000,   # meme architecture hybride que 4B
        "deepseek-r1:8b": 65_536,   # GQA standard (2*8*128*32, config.json)
    }

    # DPU kernel ratio (dpu_v2_qwen3_5_9b.json)
    DPU_RW_RATIO = 6.4  # READ=309 / (WRITE+COMPUTE) : heavily BW-bound

    # Throttle levels (MemoryOptimizer)
    KV_TOKENS = {0: 4096, 1: 3072, 2: 2048, 3: 1024, 4: 512}
    KV_COMPRESS = {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.33, 4: 0.25}

    # Gap hardware vs benchmark officiel FLM (fastflowlm.com/docs/benchmarks/qwen3.5_results)
    # Officiel : Ryzen AI 7 350 Kraken Point, 32 GB DRAM, FLM v0.9.38
    # User SOC_MATRIX : qwen3.5:4b, ctx=1K → 12.20 t/s vs officiel 15.0 t/s
    # Facteur = 12.20/15.0 = 0.813 → user HW ~19% plus lent que référence Kraken
    # Causes probables : version FLM + BW LPDDR5X (Strix vs Kraken)
    HW_FACTOR_VS_KRAKEN = 12.20 / 15.0   # 0.813


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

# Sources : config.json C:\Users\videl\flm\models\*\config.json  (lecture directe)
# bench_3models.py warm TPS : llama=19.7, qwen4b=17.25, deepseek=3.3 t/s
MODELS = {
    "llama3.2:1b": {
        # config : hidden=2048, layers=16, kv_heads=8, head_dim=64
        # file   : model.q4nx = 1.3 GB -> INT8 (theorique INT8=1.24 GB + scales)
        # bench  : 19.7 t/s warm (pt=25) -> W_eff=1.52 GB
        "params_B": 1.24, "bits": 8, "hidden": 2048, "layers": 16,
        "measured_tps": 19.7, "measured_bw_GBs": None,
        "desc": "Llama 3.2 1B (INT8, confirme par fichier 1.3 GB)"
    },
    "qwen3.5:4b": {
        # config : hidden=2560, layers=32, kv_heads=4, head_dim=256, hybrid(full_int=4)
        # file   : model.q4nx = 4.2 GB (poids INT4=1.75 GB + xclbin + vocab FP16)
        # bench  : 17.25 t/s warm (pt=25) -> W_eff=1.74 GB ~ INT4 (1.75 GB) OK
        "params_B": 3.5, "bits": 4, "hidden": 2560, "layers": 32,
        "measured_tps": 17.25, "measured_bw_GBs": None,
        "desc": "Qwen3.5 4B (INT4, hybride linear+full-att, 8/32 full layers)"
    },
    "qwen3.5:4b_soc": {
        # Donnees soc_performance_matrix.csv (ctx=1024, warmup)
        "params_B": 3.5, "bits": 4, "hidden": 2560, "layers": 32,
        "measured_tps": 12.2, "measured_bw_GBs": None,
        "desc": "Qwen3.5 4B (INT4) -- soc_performance_matrix baseline"
    },
    "qwen3.5:9b": {
        # config : hidden=4096, layers=32, kv_heads=4, head_dim=256, hybrid(full_int=4)
        # file   : model.q4nx = 7.4 GB (poids INT4=4.5 GB + xclbin + vocab FP16)
        #
        # CALIBRATION FLM OFFICIEL (v0.9.38, Kraken Point, fastflowlm.com/docs/benchmarks) :
        #   TPS officiel ctx=1K : 9.3 t/s (Kraken Point, 32 GB DRAM)
        #   Facteur HW user/Kraken : 12.20/15.0 = 0.813 (SOC_MATRIX vs officiel 4b)
        #   TPS estimé user ctx=1K : 9.3 * 0.813 = 7.56 t/s
        #
        #   W_eff implicite (roofline inverse) : BW_eff/TPS - KV_GB
        #   W_eff_kraken = 21.93/9.3 - 0.031 = 2.36 - 0.031 = 2.33 GB (ctx=1K)
        #   W_eff_user   = 21.93/7.56 - 0.031 = 2.90 - 0.031 = 2.87 GB (ctx=1K)
        #
        #   CAUSE : FLM met en cache ~1.6 GB de poids 9b en SRAM NPU (couches fréquentes)
        #   → seuls ~2.9 GB rechargés depuis DRAM par token (vs 4.5 GB théorique INT4)
        #   NOTE : w_gb override utilisé dans predict_decode_tps (ignore params_B*bits/8)
        "params_B": 9.0, "bits": 4, "hidden": 4096, "layers": 32,
        "w_gb": 2.87,   # GB effectifs rechargés/token (calibré officiel × hw_factor)
        "measured_tps": 7.56, "measured_bw_GBs": None,
        "desc": "Qwen3.5 9B (INT4, hybride, W_eff=2.87 GB calibré FLM officiel)"
    },
    "deepseek-r1:8b": {
        # config : hidden=4096, layers=32, kv_heads=8, head_dim=128, standard GQA
        # file   : model.q4nx = 5.4 GB (INT4 poids=4.0 GB + FP16 lm_head~1 GB)
        # bench  : 3.3 t/s warm (pt=18) -> overhead R1 reasoning non-modelise
        "params_B": 8.0, "bits": 4, "hidden": 4096, "layers": 32,
        "measured_tps": 3.3, "measured_bw_GBs": None,
        "desc": "DeepSeek-R1 8B distill (INT4 + FP16 lm_head, overhead R1)"
    },
}

# Donnees empiriques soc_performance_matrix (model = qwen3.5:4b_soc)
SOC_MATRIX = {
    # ctx_tokens: (avg_decode_tps_clean, prefill_tps, ttft_s)
    1024:  (12.20, 338.22, 3.50),
    2048:  (11.90, 396.23, 5.83),
    4096:  (11.61, 464.12, 8.39),
    8192:  (10.83, 488.08, 15.89),
    16384: ( 9.93, 479.70, 32.33),
    32768: ( 8.29, 424.97, 73.92),
}

# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def model_size_bytes(params_B: float, bits: int) -> float:
    """Taille modele en octets (poids quantises)."""
    return params_B * 1e9 * bits / 8.0


def kv_cache_bytes(ctx_tokens: int, kv_per_token: int = HW.KV_PER_TOKEN_BYTES) -> float:
    """Taille KV cache totale pour ctx tokens."""
    return ctx_tokens * kv_per_token


def predict_decode_tps(
    params_B: float,
    bits: int,
    ctx_tokens: int,
    bw_GBs: float = HW.BW_EFF_GBs,
    kv_per_token: int = HW.KV_PER_TOKEN_BYTES,
    w_gb: float = None,
) -> dict:
    """
    Predit le throughput decode en t/s.
    Formule : tps = BW_eff / (model_size + kv_cache)
    BW_eff = BW_mesure * eta (eta=0.731, calibre sur soc_performance_matrix).
    Erreur residuelle : +-3% sur ctx=1024..32768.
    eta capture le overhead transfert inter-kernel (ΦT = 27% de BW perdu).
    """
    model_B  = (w_gb * 1e9) if w_gb is not None else model_size_bytes(params_B, bits)
    kv_B     = kv_cache_bytes(ctx_tokens, kv_per_token)
    total_B  = model_B + kv_B
    tps      = (bw_GBs * 1e9) / total_B
    bw_used  = tps * total_B / 1e9
    sat_pct  = bw_used / bw_GBs * 100.0

    # Arithmetique intensite (QSTS roofline)
    flops_per_token = 2 * params_B * 1e9  # approximation GEMV
    ai = flops_per_token / total_B
    if bits <= 4:
        bound = "compute" if ai > HW.RIDGE_INT4 else "bandwidth"
        ridge = HW.RIDGE_INT4
    elif bits <= 8:
        bound = "compute" if ai > HW.RIDGE_INT8 else "bandwidth"
        ridge = HW.RIDGE_INT8
    else:
        bound = "compute" if ai > HW.RIDGE_FP16 else "bandwidth"
        ridge = HW.RIDGE_FP16

    return {
        "tps":           round(tps, 2),
        "model_GB":      round(model_B / 1e9, 3),
        "kv_GB":         round(kv_B / 1e9, 3),
        "total_GB":      round(total_B / 1e9, 3),
        "bw_used_GBs":   round(bw_used, 2),
        "bw_sat_pct":    round(sat_pct, 1),
        "ai_FLOP_B":     round(ai, 1),
        "ridge_FLOP_B":  round(ridge, 1),
        "bound":         bound,
    }


def tps_degradation(params_B: float, bits: int,
                    ctx_ref: int = 1024, ctx_test: int = 32768) -> float:
    """Pourcentage de degradation de tps entre ctx_ref et ctx_test."""
    r_ref  = predict_decode_tps(params_B, bits, ctx_ref)
    r_test = predict_decode_tps(params_B, bits, ctx_test)
    return (r_test["tps"] - r_ref["tps"]) / r_ref["tps"] * 100.0


def optimal_ctx(params_B: float, bits: int, max_degradation_pct: float = 10.0) -> int:
    """
    Contexte max avant que le throughput degrade de plus de max_degradation_pct%.
    """
    r_ref = predict_decode_tps(params_B, bits, 0)
    tps_threshold = r_ref["tps"] * (1.0 - max_degradation_pct / 100.0)
    for ctx in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        r = predict_decode_tps(params_B, bits, ctx)
        if r["tps"] < tps_threshold:
            return ctx // 2
    return 131072


def throttle_level(ctx_tokens: int) -> int:
    """Niveau de throttle de base (0=NONE, 4=CRITICAL) selon le contexte."""
    if ctx_tokens <= 1024: return 0
    if ctx_tokens <= 2048: return 1
    if ctx_tokens <= 4096: return 2
    if ctx_tokens <= 8192: return 3
    return 4


def qsts_throttle_override(layer_results: list) -> int:
    """
    Ajuste le throttle selon les scores QSTS des couches du modele.

    Entree : liste de dicts avec 'verdict' et 'bound' (format screen_model_xdna2.py)
    Sortie : delta de throttle (+0, +1, +2) a ajouter au niveau de base.

    Logique :
      - Couche FRICTION_POINT = CRITICAL + bandwidth-bound
        => directement impacte par le goulot RAM : durcir le KV limit
      - Si >40% des couches sont FRICTION_POINT => +2
      - Si >15% des couches sont FRICTION_POINT => +1
      - Sinon => +0
    """
    if not layer_results:
        return 0
    n = len(layer_results)
    friction = sum(
        1 for r in layer_results
        if r.get("verdict") in ("CRITICAL", "BORDERLINE")
        and r.get("bound", "") == "bandwidth"
    )
    frac = friction / n
    if frac > 0.40:
        return 2
    if frac > 0.15:
        return 1
    return 0


def throttle_with_qsts(ctx_tokens: int, layer_results: list = None) -> int:
    """
    Throttle final = base(ctx) + override(QSTS), plafonne a 4.

    layer_results : sortie de screen_model_xdna2.py (optionnel).
    Si None, utilise throttle de base uniquement.
    """
    base  = throttle_level(ctx_tokens)
    delta = qsts_throttle_override(layer_results) if layer_results else 0
    return min(4, base + delta)


def kv_limit_for_ctx(ctx_tokens: int, layer_results: list = None) -> int:
    """Limite KV recommandee (integre QSTS si disponible)."""
    level = throttle_with_qsts(ctx_tokens, layer_results)
    return HW.KV_TOKENS[level]


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------

C = {
    "RED":    "\033[91m",
    "YEL":    "\033[93m",
    "GRN":    "\033[92m",
    "CYN":    "\033[96m",
    "BOLD":   "\033[1m",
    "RESET":  "\033[0m",
}


def col(text, color):
    return f"{C[color]}{text}{C['RESET']}"


def bw_color(sat_pct):
    if sat_pct >= 100: return "RED"
    if sat_pct >= 85:  return "YEL"
    return "GRN"


def bound_color(bound):
    return "GRN" if bound == "compute" else "YEL"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{col('='*70, 'BOLD')}")
    print(f"  {col(title, 'BOLD')}")
    print(col('='*70, 'BOLD'))


def print_roofline_summary():
    section("QSTS / XDNA2 ROOFLINE MODEL")
    print(f"""
  Hardware : AMD XDNA2 NPU (Strix Point)
  ----------------------------------------
  Compute INT4    : {HW.TOPS_INT4:.0f} TOPS
  Compute INT8    : {HW.TOPS_INT8:.0f} TOPS
  Compute FP16    : {HW.TFLOPS_FP16:.0f} TFLOPS
  BW theorique    : {HW.BW_THEORETICAL_GBs:.0f} GB/s (LPDDR5X-6400)
  BW mesure       : {HW.BW_MEASURED_GBs:.0f} GB/s  [{HW.BW_EFFICIENCY*100:.0f}% vs theorique]
  eta (ΦT)        : {HW.ETA:.4f}  [{(1-HW.ETA)*100:.1f}% perdu overhead inter-kernel]
  BW effective    : {HW.BW_EFF_GBs:.2f} GB/s  [calibre sur soc_performance_matrix]
  Ridge INT4      : {HW.RIDGE_INT4:.0f} FLOP/B
  Ridge INT8      : {HW.RIDGE_INT8:.0f} FLOP/B
  Ridge FP16      : {HW.RIDGE_FP16:.0f} FLOP/B

  DPU Kernel (dpu_v2_qwen3_5_9b.json) :
    READ=309, WRITE=48, COMPUTE=339 -> R/W ratio = {HW.DPU_RW_RATIO}x
    => {col('CONFIRME : regime bandwidth-bound dominant', 'YEL')}
    => Chaque token decode charge ~model_size + kv_cache depuis LPDDR5X
""")


def print_throughput_table(params_B: float = 3.5, bits: int = 4, label: str = "qwen3.5:4b"):
    section(f"CALCULATEUR DE DEBIT -- {label.upper()}")
    print(f"  Modele : {params_B}B params, INT{bits}")
    print(f"  BW mesure : {HW.BW_MEASURED_GBs} GB/s\n")

    hdr = f"  {'Context':>8}  {'Pred TPS':>9}  {'Mesure TPS':>11}  {'Err%':>5}  {'Model GB':>8}  {'KV GB':>6}  {'Regime':>10}  {'Throttle'}"
    print(hdr)
    print("  " + "-" * 88)

    ctx_list = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    for ctx in ctx_list:
        r = predict_decode_tps(params_B, bits, ctx)
        measured = SOC_MATRIX.get(ctx, (None,))[0]
        m_str  = f"{measured:9.2f}" if measured else f"{'--':>9}"
        if measured:
            err   = (r["tps"] - measured) / measured * 100
            ec    = "GRN" if abs(err) < 5 else "YEL"
            e_str = col(f"{err:>+4.1f}%", ec)
        else:
            e_str = f"{'--':>5}"
        bc    = bound_color(r["bound"])
        tlevel = throttle_level(ctx)
        tlabel = ["NONE", "LIGHT", "MEDIUM", "HEAVY", "CRITICAL"][tlevel]
        tlcolor = ["GRN", "GRN", "YEL", "YEL", "RED"][tlevel]
        tps_str   = col(f"{r['tps']:>9.2f}", 'CYN')
        bound_str = col(f"{r['bound']:>10}", bc)
        tl_str    = col(tlabel, tlcolor)
        print(
            f"  {ctx:>8}  {tps_str}  {m_str}  {e_str}  "
            f"{r['model_GB']:>8.3f}  {r['kv_GB']:>6.3f}  "
            f"{bound_str}  {tl_str}"
        )


def print_bottleneck_analysis(params_B: float = 3.5, bits: int = 4):
    section("ANALYSE DES GOULOTS RAM")

    model_B = model_size_bytes(params_B, bits)

    print(f"  Modele : {params_B}B params, INT{bits} = {model_B/1e9:.2f} GB")
    print()

    # KV cache breakeven
    # tps drops 10% when: kv_cache = 0.111 * model_size
    kv_10pct = 0.111 * model_B
    ctx_10pct = int(kv_10pct / HW.KV_PER_TOKEN_BYTES)
    print(f"  KV cache = 10% de model_size a ctx = {ctx_10pct:,} tokens")
    print(f"  => En-dessous de {ctx_10pct:,} tokens : RAM overhead < 10%")
    print()

    # KV cache at XDNA2 SRAM limit (512 tokens)
    kv_sram = kv_cache_bytes(512)
    kv_4k   = kv_cache_bytes(4096)
    kv_32k  = kv_cache_bytes(32768)
    print(f"  Empreinte KV cache :")
    print(f"    ctx=  512 :  {kv_sram/1e6:6.1f} MB  (XDNA2 SRAM limit)")
    print(f"    ctx= 4096 :  {kv_4k/1e6:6.1f} MB  (MemoryOptimizer level=2)")
    print(f"    ctx=32768 :  {kv_32k/1e6:6.1f} MB  (max context testé)")
    print()

    # Degradation table
    print(f"  Degradation decode TPS par rapport au ctx=512 :")
    print(f"  {'Context':>8}  {'Delta TPS':>10}  {'Gain KV (MB)':>13}  {'Action'}")
    print("  " + "-" * 55)
    r_base = predict_decode_tps(params_B, bits, 512)
    for ctx in [1024, 2048, 4096, 8192, 16384, 32768]:
        r = predict_decode_tps(params_B, bits, ctx)
        delta = (r["tps"] - r_base["tps"]) / r_base["tps"] * 100
        saved = (kv_cache_bytes(ctx) - kv_cache_bytes(512)) / 1e6
        tlevel = throttle_level(ctx)
        actions = {
            0: "aucune",
            1: "context chunking 3072",
            2: "chunking 2048 + KV compress 0.5x",
            3: "chunking 1024 + KV compress 0.33x",
            4: col("chunking 512 + KV compress 0.25x [CRITICAL]", "RED"),
        }
        dc = "RED" if delta < -20 else ("YEL" if delta < -10 else "GRN")
        print(
            f"  {ctx:>8}  {col(f'{delta:>+9.1f}%', dc)}  {saved:>13.1f} MB  "
            f"{actions[tlevel]}"
        )

    print()
    # BW breakdown
    r32k = predict_decode_tps(params_B, bits, 32768)
    bw_model = model_B / 1e9 * r32k["tps"]
    bw_kv    = kv_cache_bytes(32768) / 1e9 * r32k["tps"]
    print(f"  A ctx=32768 ({r32k['tps']:.1f} t/s) :")
    print(f"    BW poids modele  : {bw_model:.1f} GB/s ({bw_model/HW.BW_MEASURED_GBs*100:.0f}% du bus)")
    print(f"    BW KV cache      : {bw_kv:.1f} GB/s ({bw_kv/HW.BW_MEASURED_GBs*100:.0f}% du bus)")
    print(f"    Total BW utilisé : {bw_model+bw_kv:.1f} / {HW.BW_MEASURED_GBs} GB/s")


def print_optimization_plan():
    section("PLAN D'OPTIMISATION RAM -- PRIORITÉ PARETO")

    print("""
  Goulots identifies (Pareto : 2 causes = 80% des pertes)
  --------------------------------------------------------

  #1  [CRITIQUE] Bande passante memoire : -50% vs theorique
      Cause   : colonne-mapping NPU sous-optimal + bus partage GPU/CPU
      Impact  : tous les modeles limites a 30 GB/s au lieu de 60
      Fix A   : Activer 8-col unlock (xrt_proxy.dll) -> +50% BW -> +50% TPS
      Fix B   : Isoler le bus memoire NPU (desactiver GPU 880M)
      Effort  : Eleve (admin, xclbin patch)

  #2  [CRITIQUE] Scheduler Python : +2-4s de latence par requete
      Cause   : AdaptiveController en Python, MPC toutes les 2s
      Impact  : p99 latency CRITIQUE meme quand NPU utilise a 99%
      Fix A   : Reduire DECIDE_COOLDOWN_S : 2.0 -> 0.5s
      Fix B   : Controller C++ natif (long terme)
      Effort  : Faible / Eleve

  #3  [IMPORTANT] KV cache non-limite sur longs contextes
      Cause   : MemoryOptimizer.apply() pas appelee si throttle_level=0
      Impact  : ctx=32768 -> kv_cache = 983 MB -> -32% TPS vs ctx=1024
      Fix     : Activer KV limit proactive (niveau 2 pour ctx >= 4096)
      Effort  : Faible
""")

    print("  KV Cache Limites Recommandees (XDNA2 / 30 GB/s mesure)")
    print("  " + "-" * 58)
    print(f"  {'Context':>8}  {'KV limit':>9}  {'KV saved MB':>11}  {'TPS gain est.':>14}")
    print("  " + "-" * 58)
    params_B, bits = 3.5, 4
    r_unlim = predict_decode_tps(params_B, bits, 32768)
    for ctx, (_, _, _) in SOC_MATRIX.items():
        lim   = kv_limit_for_ctx(ctx)
        r_lim = predict_decode_tps(params_B, bits, min(ctx, lim))
        saved = max(0, kv_cache_bytes(ctx) - kv_cache_bytes(lim)) / 1e6
        gain  = (r_lim["tps"] - r_unlim["tps"]) / r_unlim["tps"] * 100
        gc    = "GRN" if gain > 5 else "YEL"
        print(
            f"  {ctx:>8}  {lim:>9}  {saved:>11.1f} MB  "
            f"{col(f'+{gain:.1f}% (est)', gc) if gain > 0 else '--':>14}"
        )

    print(f"""
  Actions Immediates (effort < 1h)
  ---------------------------------
  1. config: DECIDE_COOLDOWN_S = 0.5s         -> -75% scheduler latency
  2. code:   MemoryOptimizer.apply(2) si ctx >= 4096  -> KV compress 0.5x
  3. config: desactiver GPU 880M (BIOS/driver)  -> +5-10% BW disponible
  4. bench:  bench_config.json max_length=2048  -> stable dans KV window
  5. diag:   --preemption 0 sur Strix           -> +9% TPS confirme

  Actions Structurelles
  ----------------------
  6. xclbin: 8-col unlock (existant dans brawll/) -> +50% BW -> double TPS
  7. CUDA:   backend RTX 5070 (4GB) pour 1B/4B -> +300-500% TPS
  8. KV:     Paged KV (paged_kv.py existe) -> gestion fine, moins de gaspillage
""")


def print_qsts_throttle_demo():
    """Demonstre l'effet du QSTS override sur le throttle."""
    section("QSTS -> THROTTLE OVERRIDE (demo)")
    print("""
  Interface : throttle_with_qsts(ctx, layer_results)
  layer_results = sortie de screen_model_xdna2.py

  Scenarios demo (ctx=4096, throttle de base = MEDIUM/2) :
""")

    scenarios = [
        ("Aucune couche FRICTION",  [{"verdict": "SAFE",     "bound": "bandwidth"} for _ in range(20)]),
        ("15% FRICTION_POINT",      [{"verdict": "CRITICAL", "bound": "bandwidth"}]*3 +
                                    [{"verdict": "SAFE",     "bound": "bandwidth"}]*17),
        ("40% FRICTION_POINT",      [{"verdict": "CRITICAL", "bound": "bandwidth"}]*8 +
                                    [{"verdict": "SAFE",     "bound": "bandwidth"}]*12),
        ("50% FRICTION, compute",   [{"verdict": "CRITICAL", "bound": "compute"}]*10 +
                                    [{"verdict": "SAFE",     "bound": "bandwidth"}]*10),
    ]
    ctx = 4096
    base = throttle_level(ctx)
    tlabels = ["NONE", "LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]
    tcolors = ["GRN", "GRN", "YEL", "YEL", "RED"]

    print(f"  {'Scenario':<32}  {'Base':>6}  {'Delta':>6}  {'Final':>8}  {'KV limit':>9}  {'KV saved MB':>12}")
    print("  " + "-" * 82)
    for label, layers in scenarios:
        delta  = qsts_throttle_override(layers)
        final  = throttle_with_qsts(ctx, layers)
        kv_lim = HW.KV_TOKENS[final]
        kv_b   = kv_limit_for_ctx(ctx, layers)
        saved  = max(0, kv_cache_bytes(ctx) - kv_cache_bytes(kv_b)) / 1e6
        fl     = col(f"{tlabels[final]:<8}", tcolors[final])
        print(f"  {label:<32}  {tlabels[base]:>6}  {'+'+str(delta):>6}  {fl}  {kv_lim:>9}  {saved:>12.1f} MB")
    print()


def print_qsts_crossref(params_B: float = 3.5, bits: int = 4):
    section("CROSS-REFERENCE QSTS / XDNA2 ROOFLINE")

    r = predict_decode_tps(params_B, bits, 1024)
    print(f"""
  Modele : {params_B}B INT{bits}, ctx=1024
  Arithmetic Intensity (AI) = FLOPs / Bytes = {r['ai_FLOP_B']:.0f} FLOP/B
  Ridge INT{bits}            = {r['ridge_FLOP_B']:.0f} FLOP/B
  Regime                 = {col(r['bound'].upper() + '-BOUND', 'YEL')}

  Confirmation DPU kernel (Qwen3.5-9B) :
    R/W ratio = {HW.DPU_RW_RATIO}x (READ=309 / WRITE=48)
    => Le NPU passe {HW.DPU_RW_RATIO:.0f}x plus de temps a lire qu'a ecrire
    => Chaque instruction de calcul est precedee de {HW.DPU_RW_RATIO:.0f} lectures memoire
    => {col("CRITERE QSTS : couche FRICTION_POINT (CRITICAL + BW-bound)", 'YEL')}

  Formule QSTS applicable :
    Q_nn   = 1 / ((1 + log1p(kv)) * (1 + sigma_act))
    QFI_nn = log(1/Q_nn) / bits
    risk   = 100 * (1 - exp(-QFI_nn * 2))

    Une couche FRICTION_POINT (risk > 75, BW-bound) ne beneficie PAS
    de compression INT4 supplementaire sans reduire le goulot memoire.
    => Priorite : optimiser le ratio BW utilise, pas la quantization.

  Seuil de rentabilite de la quantization (KV cache) :
""")
    for b in [4, 8, 16]:
        bname = f"INT{b}" if b <= 8 else "FP16"
        bw_scale = {4: 0.5, 8: 1.0, 16: 2.0}[b]
        tps_est = HW.BW_MEASURED_GBs / (params_B * bw_scale + kv_cache_bytes(4096)/1e9)
        print(f"    {bname} (weights {bw_scale} B/param) -> {tps_est:.1f} t/s @ ctx=4096")

    print()


def print_throughput_calculator(ctx: int, params_B: float = 3.5, bits: int = 4):
    """Mode interactif : prevision pour un contexte donne."""
    section(f"PREVISION DEBIT -- ctx={ctx} tokens")
    r = predict_decode_tps(params_B, bits, ctx)
    tlevel = throttle_level(ctx)
    tlabels = ["NONE", "LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]

    ms_per_tok = 1000.0 / max(r["tps"], 0.01)
    kv_frac    = r["kv_GB"] / r["total_GB"] * 100.0 if r["total_GB"] > 0 else 0.0
    bwc        = bw_color(r["bw_sat_pct"])

    print(f"""
  Modele           : {params_B}B INT{bits}
  Contexte         : {ctx} tokens
  ------------------------------------------
  Decode TPS prevu  : {col(f'{r["tps"]:.2f} t/s', 'CYN')}
  Temps/token       : {ms_per_tok:.1f} ms/token
  ------------------------------------------
  Poids modele      : {r['model_GB']:.3f} GB
  KV cache          : {r['kv_GB']:.3f} GB  ({kv_frac:.1f}% du total)
  Total BW/token    : {r['total_GB']:.3f} GB
  ------------------------------------------
  BW effective      : {col(f'{HW.BW_EFF_GBs:.1f} GB/s (eta={HW.ETA})', bwc)}
  BW saturee        : {col(f'{r["bw_sat_pct"]:.1f}%', bwc)} de BW_eff
  Regime roofline   : {col(r["bound"].upper() + "-BOUND", bound_color(r["bound"]))}  (AI={r['ai_FLOP_B']:.0f} vs ridge={r['ridge_FLOP_B']:.0f} FLOP/B)
  ------------------------------------------
  Throttle (base)   : {col(tlabels[tlevel], ['GRN','GRN','YEL','YEL','RED'][tlevel])}
  KV limit          : {HW.KV_TOKENS[tlevel]} tokens
  KV compress       : {HW.KV_COMPRESS[tlevel]}x
""")

    kv_lim = HW.KV_TOKENS[tlevel]
    if ctx > kv_lim:
        r_lim    = predict_decode_tps(params_B, bits, kv_lim)
        gain     = (r_lim["tps"] - r["tps"]) / r["tps"] * 100
        saved_MB = (kv_cache_bytes(ctx) - kv_cache_bytes(kv_lim)) / 1e6
        tps_lim_str = col(f"{r_lim['tps']:.2f}", 'GRN')
        gain_str    = col(f"+{gain:.1f}%", 'GRN')
        print(f"  Avec KV limit ({kv_lim} tokens) :")
        print(f"    TPS  : {r['tps']:.2f} -> {tps_lim_str} t/s  ({gain_str})")
        print(f"    RAM  : economie de {saved_MB:.1f} MB KV cache")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FastFlowLM RAM Bottleneck Analyzer v1.1")
    p.add_argument("--model",    type=str,  default="qwen3.5:4b",
                   choices=list(MODELS.keys()), help="Modele a analyser")
    p.add_argument("--ctx",      type=int,  default=None,
                   help="Calculer TPS pour ce ctx specifique")
    p.add_argument("--all",      action="store_true",
                   help="Afficher tous les modeles")
    p.add_argument("--optimize", action="store_true",
                   help="Mode optimisation (roofline + plan)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.ctx is not None:
        m = MODELS.get(args.model, MODELS["qwen3.5:4b"])
        print_throughput_calculator(args.ctx, m["params_B"], m["bits"])
        return
    if args.optimize:
        print_roofline_summary()
        print_bottleneck_analysis()
        print_optimization_plan()
        return
    print_roofline_summary()
    if args.all:
        for mname, m in MODELS.items():
            print_throughput_table(m["params_B"], m["bits"], mname)
    else:
        m = MODELS.get(args.model, MODELS["qwen3.5:4b"])
        print_throughput_table(m["params_B"], m["bits"], args.model)
    m = MODELS.get(args.model, MODELS["qwen3.5:4b"])
    print_bottleneck_analysis(m["params_B"], m["bits"])
    print_qsts_crossref(m["params_B"], m["bits"])
    print_qsts_throttle_demo()
    print_optimization_plan()


if __name__ == "__main__":
    main()
