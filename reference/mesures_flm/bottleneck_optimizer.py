"""
bottleneck_optimizer.py — Évaluation et ajustement adaptatif du goulot XDNA2
==============================================================================
Mesure live TPS à plusieurs contextes, calibre BW_EFF et W_eff depuis les
données réelles, identifie le régime de goulot, et recommande / exporte les
constantes optimisées pour test_gains.py / test_model_compare.py.

Usage :
  python bottleneck_optimizer.py --model qwen3.5:9b
  python bottleneck_optimizer.py --model qwen3.5:9b-cust
  python bottleneck_optimizer.py --model qwen3.5:9b --full        # scan étendu
  python bottleneck_optimizer.py --model qwen3.5:9b --export calibration.json
  python bottleneck_optimizer.py --model qwen3.5:9b --patch-test-gains
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# ── Constantes initiales (overridées après calibration) ─────────────────────────
# CORRECTION 2026-06-25 : DDR5-5600 dual-channel = 89.6 GB/s peak (pas 128 GB/s)
# 128 GB/s = LPDDR5X théorique max, jamais atteint sur Strix Point réel
# Preuve : roofline collapse à batch=1 → AI = 0.67 FLOPS/byte (Q4NX)
# Bandwidth ceiling = BW_EFF × AI = 21.927 × 0.67 = 14.7 GFLOPS (jamais compute-bound)
BW_NOMINAL_GBs  = 89.6    # DDR5-5600 dual-channel peak réel (Strix Point, corrigé)
BW_MEASURED_GBs = 30.0    # BW effective GEMV (calibré SOC_MATRIX)
ETA_PRIOR       = 0.7309  # η = BW_eff / BW_measured
BW_EFF_PRIOR    = BW_MEASURED_GBs * ETA_PRIOR   # 21.927 GB/s
W_EFF_PRIOR     = 3.83    # GB (calibré live warm Strix, 2026-06-18)
KV_BYTES_TOKEN  = 30_000  # B/token — architecture hybride Qwen3.5

# ── Nouvelles constantes RE FLM (2026-06-25) ────────────────────────────────────
# Bottleneck confirmé par 3 méthodes : RVA 0x310D0 = decode loop = 100ms/tok
# Budget per-token = compute_ms + ddr5_ms + dispatch_ms
DISPATCH_OVERHEAD_MS = 60.0  # ms — overhead middleware FLM (0x310D0, mesuré VEH+Dr0)
COMPUTE_MS_NOMINAL   = 10.0  # ms — NPU compute pur (estimé Wang/Estévez)
DDR5_MS_NOMINAL      = 30.0  # ms — DDR5 transfer (W_eff / BW_EFF × 1000)
# DISPATCH_OVERHEAD_MS = 100 - COMPUTE_MS - DDR5_MS ≈ 60ms incompressible

# ── Constantes H6 (cold-start per-request) ──────────────────────────────────────
H6_RECONFIG_MS  = 2092.0  # ms — xrtHwContextDestroy+Create per HTTP request (mesuré)
H6_FIXED        = False   # True si h6_hook.dll injecté (O6 dans bus_optimizer)

COLORS = {
    "RED": "\033[91m", "GRN": "\033[92m", "YEL": "\033[93m",
    "CYN": "\033[96m", "MAG": "\033[95m", "BOLD": "\033[1m", "RST": "\033[0m",
}
def col(t, c): return COLORS.get(c,"") + str(t) + COLORS["RST"]
def sep(n=78): print("=" * n)
def hdr(t): sep(); print(col("  " + t, "BOLD")); sep()

# ── Roofline ────────────────────────────────────────────────────────────────────
def tps_pred(ctx, bw_eff, w_eff, kv_b=KV_BYTES_TOKEN):
    d = w_eff + ctx * kv_b / 1e9
    return bw_eff / d if d > 0 else 0.0

def fit_roofline(data):
    """
    data = [(ctx, tps_warm), ...]
    Modèle : TPS = BW_eff / (W_eff + ctx * KV/1e9)
    => 1/TPS = W_eff/BW_eff + ctx*(KV/1e9)/BW_eff
    Régression linéaire : y = a + b*x  avec y=1/TPS, x=ctx
    => BW_eff = 1/b * (KV/1e9) ; W_eff = a * BW_eff
    """
    if len(data) < 2:
        return None, None, None

    xs = [ctx for ctx, _ in data]
    ys = [1.0 / tps for _, tps in data if tps > 0]
    xs = [ctx for (ctx, tps) in data if tps > 0]

    n   = len(xs)
    sx  = sum(xs);   sy  = sum(ys)
    sxx = sum(x*x for x in xs)
    sxy = sum(x*y for x, y in zip(xs, ys))

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None, None, None

    b = (n * sxy - sx * sy) / denom   # slope  = KV/(1e9 * BW_eff)
    a = (sy - b * sx) / n              # intercept = W_eff / BW_eff

    if b <= 0:
        return None, None, None

    bw_eff_fit = (KV_BYTES_TOKEN / 1e9) / b
    w_eff_fit  = a * bw_eff_fit

    # R²
    y_mean = sy / n
    ss_tot = sum((y - y_mean)**2 for y in ys)
    y_pred = [a + b * x for x in xs]
    ss_res = sum((y - yp)**2 for y, yp in zip(ys, y_pred))
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return round(bw_eff_fit, 4), round(w_eff_fit, 4), round(r2, 4)


# ── Arithmetic Intensity (roofline collapse à batch=1) ──────────────────────────
def arithmetic_intensity_q4(d_model=3584, kv_heads=4, head_dim=128):
    """
    Intensité arithmétique decode Q4NX à batch=1.
    AI = 2 × d_model / bytes_per_weight_per_token
    Pour Q4NX : 4 bits/weight → 0.5 bytes/weight
    AI ≈ 2 × 3584 / (3584 × 0.5) = 4 FLOPS/byte (attention layer)
    Mais avec GQA et SSM layers : AI_effectif ≈ 0.67 FLOPS/byte moyen
    Bandwidth ceiling = BW_EFF × AI_eff = 21.927 × 0.67 = 14.7 GFLOPS
    → JAMAIS compute-bound à batch=1, même avec 50 TOPS NPU
    """
    # Q4NX : 4 bits = 0.5 bytes par weight
    bytes_per_weight = 0.5
    # FLOPs par token par layer (GEMV) = 2 × d_model × d_intermediate
    flops_per_token = 2 * d_model
    # Bytes lus par token = d_model × bytes_per_weight (poids de 1 row du GEMV)
    bytes_per_token = d_model * bytes_per_weight
    ai = flops_per_token / bytes_per_token  # ≈ 4 FLOPS/byte pour dense
    # Avec KV + GQA + SSM layers hybrides → AI effectif réduit
    ai_effective = ai * 0.167  # facteur empirique Qwen3.5 hybride (8/32 full-att)
    bw_ceiling_gflops = BW_EFF_PRIOR * ai_effective
    return {
        "ai_dense": round(ai, 3),
        "ai_effective": round(ai_effective, 3),
        "bw_ceiling_gflops": round(bw_ceiling_gflops, 2),
        "npu_peak_tops": 50.0,
        "is_bw_bound": bw_ceiling_gflops < 50_000,  # toujours True
        "note": "batch=1 decode → toujours bandwidth-bound sur XDNA2",
    }


# ── Ingestion VEH+Dr0 CSV (parse_forward_log) ───────────────────────────────────
VEH_CSV_PATHS = [
    r"C:\flm_forward_veh.csv",
    r"C:\flm_callsite.csv",
]

def load_veh_csv(path=None):
    """
    Charge le CSV produit par veh_dr0_hook.dll ou callsite_hook.dll.
    Colonnes : call_n, token_id, delta_us, tps, tsc, tid
    Retourne les métriques per-token calibrées depuis des mesures réelles.
    """
    candidates = [path] + VEH_CSV_PATHS if path else VEH_CSV_PATHS
    for p in candidates:
        if p and Path(p).exists():
            rows = []
            with open(p, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        rows.append({
                            "n":        int(r["call_n"]),
                            "token_id": int(r["token_id"]),
                            "delta_us": int(r["delta_us"]),
                            "tps":      float(r["tps"]),
                        })
                    except (ValueError, KeyError):
                        pass
            if not rows:
                continue
            # Filtre : 10ms < delta < 5000ms (génération réelle, pas init)
            gen = [r for r in rows if 10_000 < r["delta_us"] < 5_000_000]
            if not gen:
                continue
            deltas = [r["delta_us"] for r in gen]
            tps_vals = [r["tps"] for r in gen]
            med_delta_ms = statistics.median(deltas) / 1000
            med_tps      = 1e6 / statistics.median(deltas)
            # Calibrer W_eff depuis TPS médian et ctx=0 (decode pur, pas de KV)
            # TPS = BW_EFF / W_eff → W_eff = BW_EFF / TPS
            w_eff_veh = BW_EFF_PRIOR / med_tps if med_tps > 0 else None
            return {
                "source": p,
                "n_calls": len(rows),
                "n_gen_calls": len(gen),
                "median_delta_ms": round(med_delta_ms, 2),
                "median_tps": round(med_tps, 3),
                "p5_tps":  round(1e6 / sorted(deltas)[int(0.95 * len(deltas))], 3),
                "p95_tps": round(1e6 / sorted(deltas)[int(0.05 * len(deltas))], 3),
                "w_eff_veh_gb": round(w_eff_veh, 4) if w_eff_veh else None,
                "dispatch_ms_est": round(med_delta_ms - DDR5_MS_NOMINAL - COMPUTE_MS_NOMINAL, 1),
            }
    return None


# ── Modèle H6 (cold-start per-request) ─────────────────────────────────────────
def h6_tps_apparent(n_tokens, tps_steady, reconfig_ms=H6_RECONFIG_MS, h6_fixed=False):
    """
    TPS apparent selon la longueur de la requete (amortissement H6).
    Sans fix : total_ms = reconfig_ms + n_tokens * (1000/tps_steady)
    Avec fix  : total_ms = n_tokens * (1000/tps_steady)
    """
    if tps_steady <= 0:
        return 0.0, 0.0
    ms_per_token = 1000.0 / tps_steady
    total_no_fix  = reconfig_ms + n_tokens * ms_per_token
    total_with    = n_tokens * ms_per_token
    tps_no_fix    = n_tokens / (total_no_fix  / 1000) if total_no_fix  > 0 else 0
    tps_with      = n_tokens / (total_with    / 1000) if total_with    > 0 else 0
    return round(tps_no_fix, 3), round(tps_with, 3)


def print_h6_table(tps_steady=7.45):
    """Affiche l'impact H6 selon la longueur de requête."""
    print(f"\n  H6 cold-start impact (TPS steady-state = {tps_steady:.2f} t/s) :")
    print(f"  {'tokens':>7}  {'sans h6_hook':>14}  {'avec h6_hook':>14}  {'gain':>8}")
    print("  " + "-" * 52)
    for n in [5, 10, 20, 50, 100, 200, 500]:
        t_no, t_yes = h6_tps_apparent(n, tps_steady)
        gain = (t_yes - t_no) / t_no * 100 if t_no > 0 else 0
        print(f"  {n:7d}  {t_no:14.3f}  {t_yes:14.3f}  {gain:+7.1f}%")


# ── API ─────────────────────────────────────────────────────────────────────────
STORY = (
    "In a distant future where Earth had fallen into quiet ruin, humanity lived in "
    "fragments, scattered across domed outposts and deep underground vaults. The sky "
    "was no longer blue. It shimmered with artificial auroras, remnants of weather-"
    "control systems left unattended for centuries. Among the last settlements was "
    "Bastion-9, a circular enclave powered by forgotten technologies and guarded by "
    "an ancient AI named Solen. "
)

def build_prompt(ctx_tokens):
    target = max(20, ctx_tokens - 40) * 4
    p = STORY
    while len(p) < target:
        p += " " + STORY
    return p[:target] + "\n\nSummarize briefly:"

def api_call(url, model, prompt, ctx, num_predict=60):
    try:
        payload = json.dumps({
            "model":   model,
            "prompt":  prompt,
            "stream":  False,
            "num_predict": num_predict,
            "options": {"num_ctx": ctx + 128, "temperature": 0.0},
        }).encode()
        req = urllib.request.Request(
            url + "/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read().decode())
        tok_out   = d.get("eval_count", 0)
        eval_ns   = d.get("eval_duration", 0)
        prompt_ns = d.get("prompt_eval_duration", 0)
        tok_in    = d.get("prompt_eval_count", 0)
        tps       = tok_out / (eval_ns / 1e9) if eval_ns > 0 else 0.0
        ttft      = prompt_ns / 1e9 if prompt_ns > 0 else None
        pref      = tok_in / (prompt_ns / 1e9) if prompt_ns > 0 else None
        return {"ok": True, "tps": round(tps, 3), "ttft": ttft,
                "prefill_tps": pref, "tok_out": tok_out, "tok_in": tok_in}
    except Exception as e:
        return {"ok": False, "tps": 0.0, "error": str(e)}


# ── Scan ─────────────────────────────────────────────────────────────────────────
def scan(url, model, ctx_list, runs, verbose=True):
    """Retourne {ctx: {"warm": [tps...], "cold": tps|None, "ttft": ...}}"""
    results = {}
    for ctx in ctx_list:
        prompt = build_prompt(ctx)
        tps_all, ttft_all, pref_all = [], [], []

        for i in range(runs):
            r = api_call(url, model, prompt, ctx)
            if not r["ok"]:
                if verbose:
                    print(col(f"  [ERR] ctx={ctx} run={i}: {r.get('error','?')}", "RED"))
                continue
            tps_all.append(r["tps"])
            if r["ttft"]:   ttft_all.append(r["ttft"])
            if r["prefill_tps"]: pref_all.append(r["prefill_tps"])
            if verbose:
                tag = "[COLD]" if i == 0 else f"run{i+1} "
                print(f"    ctx={ctx:6d}  {tag}  TPS={r['tps']:7.3f}  TTFT={r['ttft']:.2f}s" if r["ttft"]
                      else f"    ctx={ctx:6d}  {tag}  TPS={r['tps']:7.3f}")

        if not tps_all:
            results[ctx] = {"ok": False}
            continue

        cold = tps_all[0] if len(tps_all) >= 2 and tps_all[0] < tps_all[1] * 0.85 else None
        warm = tps_all[1:] if cold is not None and len(tps_all) > 1 else tps_all

        results[ctx] = {
            "ok": True,
            "cold": cold,
            "warm": warm,
            "tps_avg": round(sum(warm)/len(warm), 3) if warm else 0.0,
            "tps_std": round(math.sqrt(sum((x-sum(warm)/len(warm))**2 for x in warm)/max(len(warm)-1,1)), 3) if len(warm) > 1 else 0.0,
            "ttft_avg": round(sum(ttft_all)/len(ttft_all), 3) if ttft_all else None,
            "prefill_avg": round(sum(pref_all)/len(pref_all), 1) if pref_all else None,
        }
    return results


# ── Analyse goulot ───────────────────────────────────────────────────────────────
def analyze_bottleneck(scan_results, bw_eff, w_eff):
    """
    Identifie le régime dominant et retourne un dict d'analyse.
    Régimes:
      DRAM_LATENCY  : TPS > roofline BW → accès DRAM latency-bound (overhead driver/reconfig)
      DRAM_BW       : TPS ≈ roofline BW ±10%  → DRAM bande passante sature
      KV_PRESSURE   : TPS chute plus vite que modèle à grand ctx → KV goulot dominant
      COMPUTE       : TPS plateau (pas de déclin avec ctx) → NPU compute saturé
    """
    warm_data = [(ctx, r["tps_avg"]) for ctx, r in scan_results.items()
                 if r.get("ok") and r.get("tps_avg", 0) > 0]
    if len(warm_data) < 2:
        return {"regime": "INCONNU", "detail": "données insuffisantes"}

    warm_data.sort(key=lambda x: x[0])
    errs = []
    for ctx, tps in warm_data:
        pred = tps_pred(ctx, bw_eff, w_eff)
        errs.append((tps - pred) / pred * 100 if pred > 0 else 0.0)

    avg_err = sum(errs) / len(errs)
    max_err = max(errs)
    min_err = min(errs)

    # Détecter plateau (compute-bound) : std TPS < 8% sur tous les ctx
    tps_vals = [tps for _, tps in warm_data]
    tps_range = (max(tps_vals) - min(tps_vals)) / max(tps_vals) * 100 if max(tps_vals) > 0 else 0

    # Détecter KV pressure accru : TPS décline plus vite que modèle linéaire
    # => erreur devient plus négative aux grands ctx
    ctx_vals = [ctx for ctx, _ in warm_data]
    large_ctx_errs = [errs[i] for i, ctx in enumerate(ctx_vals) if ctx >= 4096]
    small_ctx_errs = [errs[i] for i, ctx in enumerate(ctx_vals) if ctx < 4096]

    kv_drift = 0.0
    if large_ctx_errs and small_ctx_errs:
        kv_drift = sum(large_ctx_errs)/len(large_ctx_errs) - sum(small_ctx_errs)/len(small_ctx_errs)

    if tps_range < 8:
        regime = "COMPUTE_BOUND"
        detail = f"TPS quasi-constant ({tps_range:.1f}% range) — NPU TOPS saturé en decode"
    elif abs(avg_err) < 5.0:
        regime = "DRAM_BW_BOUND"
        detail = f"TPS suit roofline BW ±{abs(avg_err):.1f}% — DRAM bande passante = goulot"
    elif avg_err > 10.0:
        regime = "DRAM_LATENCY_BOUND"
        detail = f"TPS > roofline ({avg_err:+.1f}%) — overhead driver/reconfig NPU dominant"
    elif kv_drift < -10.0:
        regime = "KV_PRESSURE"
        detail = f"TPS chute extra aux grands ctx (drift={kv_drift:.1f}%) — KV cache pressure"
    else:
        regime = "DRAM_MIXED"
        detail = f"err moy={avg_err:.1f}% — latence DRAM + BW mixte"

    return {
        "regime": regime,
        "detail": detail,
        "avg_err_pct": round(avg_err, 2),
        "max_err_pct": round(max_err, 2),
        "min_err_pct": round(min_err, 2),
        "kv_drift": round(kv_drift, 2),
        "tps_range_pct": round(tps_range, 2),
        "errs_per_ctx": {ctx: round(e, 2) for (ctx, _), e in zip(warm_data, errs)},
    }


# ── Recommandations ──────────────────────────────────────────────────────────────
def recommend(scan_results, bw_eff, w_eff, bottleneck, target_tps=4.0):
    recs = []
    warm_data = [(ctx, r["tps_avg"]) for ctx, r in scan_results.items()
                 if r.get("ok") and r.get("tps_avg", 0) > 0]
    warm_data.sort(key=lambda x: x[0])

    regime = bottleneck["regime"]

    # 1. ctx max pour TPS cible
    ctx_max = None
    for ctx in range(512, 200001, 512):
        pred = tps_pred(ctx, bw_eff, w_eff)
        if pred < target_tps:
            ctx_max = ctx - 512
            break
    if ctx_max:
        recs.append({
            "param": "--ctx-len",
            "value": str(min(ctx_max, 32768)),
            "reason": f"ctx max pour TPS ≥ {target_tps:.1f} t/s (roofline)",
        })

    # 2. Threshold KV throttle
    # KV = w_eff / (KV_B/1e9) → nombre de tokens avant que KV = 10% de W_eff
    kv_thresh_tokens = int(w_eff * 0.1 * 1e9 / KV_BYTES_TOKEN)
    recs.append({
        "param": "kv_throttle_ctx",
        "value": str(kv_thresh_tokens),
        "reason": f"KV = 10% de W_eff à ctx={kv_thresh_tokens} tokens — throttler en dessous pour maximiser TPS",
    })

    # 3. Mode pmode
    recs.append({
        "param": "--pmode",
        "value": "performance",
        "reason": "Réduit latence DRAM scheduler (driver overhead -15% estimé)",
    })

    # 4. Spécifique au régime
    if regime == "COMPUTE_BOUND":
        recs.append({
            "param": "observation",
            "value": "TPS compute-bound — réduire ctx n'aidera pas",
            "reason": "NPU TOPS saturé, le goulot est le nombre d'opérations GEMV",
        })
    elif regime == "DRAM_LATENCY_BOUND":
        recs.append({
            "param": "observation",
            "value": "Overhead driver NPU — warm-up obligatoire",
            "reason": "Cold start coûteux (xclbin chargement). Garder le modèle chargé.",
        })
    elif regime == "KV_PRESSURE":
        recs.append({
            "param": "--ctx-len",
            "value": str(min(4096, 32768)),
            "reason": "KV pressure détecté aux grands ctx — limiter à 4096 si possible",
        })

    # 5. Commande flm serve optimale
    ctx_opt = str(min(ctx_max or 16384, 32768))
    recs.append({
        "param": "flm_serve_cmd",
        "value": f"flm serve <model> --ctx-len {ctx_opt} --pmode performance",
        "reason": "Commande optimisée selon calibration",
    })

    return recs


# ── Export ───────────────────────────────────────────────────────────────────────
def export_calibration(model, bw_eff, w_eff, r2, bottleneck, recs, scan_results, path):
    out = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "calibration": {
            "BW_EFF_GBs": bw_eff,
            "W_EFF_GB":   w_eff,
            "ETA":        round(bw_eff / BW_MEASURED_GBs, 4),
            "R2_fit":     r2,
            "KV_BYTES_TOKEN": KV_BYTES_TOKEN,
        },
        "priors": {
            "BW_EFF_GBs": BW_EFF_PRIOR,
            "W_EFF_GB":   W_EFF_PRIOR,
            "ETA":        ETA_PRIOR,
        },
        "bottleneck": bottleneck,
        "recommendations": recs,
        "scan": {
            str(ctx): {
                "tps_avg":     r.get("tps_avg"),
                "tps_std":     r.get("tps_std"),
                "tps_pred":    round(tps_pred(ctx, bw_eff, w_eff), 3),
                "err_pct":     round((r.get("tps_avg",0) - tps_pred(ctx, bw_eff, w_eff))
                                     / tps_pred(ctx, bw_eff, w_eff) * 100, 2)
                                if r.get("ok") and tps_pred(ctx, bw_eff, w_eff) > 0 else None,
                "cold_tps":    r.get("cold"),
                "ttft_avg":    r.get("ttft_avg"),
                "prefill_tps": r.get("prefill_avg"),
            }
            for ctx, r in scan_results.items()
            if r.get("ok")
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def patch_test_gains(bw_eff, w_eff, eta, model_key, gains_path="test_gains.py"):
    """Met à jour les constantes BW_EFF_GBs, ETA, W_EFF dans test_gains.py."""
    try:
        with open(gains_path, encoding="utf-8") as f:
            src = f.read()
    except FileNotFoundError:
        print(col(f"  [WARN] {gains_path} introuvable — patch ignoré", "YEL"))
        return

    # Backup
    bak = gains_path.replace(".py", "_backup.py")
    with open(bak, "w", encoding="utf-8") as f:
        f.write(src)
    print(col(f"  Backup : {bak}", "DIM"))

    # Patcher BW_EFF_GBs
    src = re.sub(
        r"(BW_EFF_GBs\s*=\s*BW_MEASURED_GBs\s*\*\s*ETA.*?)(\n)",
        f"BW_EFF_GBs      = {bw_eff}   # recalibré live {datetime.now():%Y-%m-%d}\n",
        src, count=1
    )
    # Patcher ETA
    src = re.sub(
        r"(ETA\s*=\s*[\d.]+\s*#.*?\n)",
        f"ETA             = {eta}  # recalibré live {datetime.now():%Y-%m-%d} (modèle {model_key})\n",
        src, count=1
    )
    # Patcher w_gb du modèle
    src = re.sub(
        r'("w_gb":\s*)[\d.]+(\s*,\s*# .*?9b.*?\n)',
        lambda m: f'"w_gb": {w_eff}   # recalibré live {datetime.now():%Y-%m-%d}\n',
        src, count=1, flags=re.IGNORECASE
    )

    with open(gains_path, "w", encoding="utf-8") as f:
        f.write(src)
    print(col(f"  Patché : {gains_path}", "GRN"))


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model",  default="qwen3.5:9b")
    parser.add_argument("--url",    default="http://localhost:11434")
    parser.add_argument("--runs",   type=int, default=3,
                        help="Runs par ctx (défaut=3, inclut cold start)")
    parser.add_argument("--ctx",    type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192],
                        help="Contextes à mesurer")
    parser.add_argument("--full",   action="store_true",
                        help="Scan étendu 512→32768 (8 points)")
    parser.add_argument("--export", default="",
                        help="Export JSON calibration vers ce fichier")
    parser.add_argument("--patch-test-gains", action="store_true",
                        help="Met à jour test_gains.py avec les constantes calibrées")
    parser.add_argument("--target-tps", type=float, default=4.0,
                        help="TPS cible pour recommandation ctx max (défaut=4.0)")
    parser.add_argument("--veh-csv", default="",
                        help="Chemin CSV VEH+Dr0 (auto-détecté si absent)")
    parser.add_argument("--h6-table", action="store_true",
                        help="Affiche table d'impact H6 cold-start par longueur de requête")
    args = parser.parse_args()
    args.veh_csv = args.veh_csv or None

    ctx_list = ([512, 1024, 2048, 4096, 8192, 16384, 24576, 32768]
                if args.full else args.ctx)

    hdr(f"BOTTLENECK OPTIMIZER — {args.model}  [{datetime.now():%Y-%m-%d %H:%M}]")
    print(f"  Priors : BW_EFF={BW_EFF_PRIOR:.3f} GB/s  W_eff={W_EFF_PRIOR} GB  η={ETA_PRIOR}")
    print(f"  BW_NOM : {BW_NOMINAL_GBs} GB/s (DDR5-5600 dual-ch, corrigé depuis 128→89.6)")
    print(f"  CTX    : {ctx_list}")
    print(f"  Runs   : {args.runs} (run 0 = cold start)")

    # ── Roofline collapse check (batch=1) ────────────────────────────────────
    ai = arithmetic_intensity_q4()
    print(f"\n  [AI]  Intensité arithmétique Q4NX decode batch=1 :")
    print(f"        AI dense       = {ai['ai_dense']:.3f} FLOPS/byte")
    print(f"        AI effectif    = {ai['ai_effective']:.3f} FLOPS/byte  (Qwen3.5 hybride)")
    print(f"        BW ceiling     = {ai['bw_ceiling_gflops']:.1f} GFLOPS  vs NPU peak 50 TOPS")
    print(col(f"        → TOUJOURS bandwidth-bound à batch=1 (jamais compute-bound)", "YEL"))

    # ── VEH+Dr0 CSV si disponible ────────────────────────────────────────────
    veh = load_veh_csv(getattr(args, "veh_csv", None))
    if veh:
        print(f"\n  [VEH] CSV VEH+Dr0 : {veh['source']}")
        print(f"        {veh['n_gen_calls']} calls génération / {veh['n_calls']} total")
        print(f"        TPS médian     = {veh['median_tps']:.3f} t/s  ({veh['median_delta_ms']:.1f} ms/tok)")
        print(f"        TPS P5/P95     = {veh['p5_tps']:.3f} / {veh['p95_tps']:.3f} t/s")
        if veh["w_eff_veh_gb"]:
            print(f"        W_eff calibré  = {veh['w_eff_veh_gb']:.4f} GB  (depuis VEH, ctx≈0)")
        if veh["dispatch_ms_est"] > 0:
            print(col(f"        Dispatch estim. = {veh['dispatch_ms_est']:.1f} ms/tok  "
                      f"(= {veh['median_delta_ms']:.1f} - {DDR5_MS_NOMINAL:.0f}ms DDR5 - {COMPUTE_MS_NOMINAL:.0f}ms compute)", "YEL"))
    else:
        print(col(f"\n  [VEH] Pas de CSV VEH+Dr0 (lancer veh_dr0_hook.dll pour calibration réelle)", "DIM"))

    sep()

    # ── Phase 1 : Warm-up rapide (évite cold start sur 1er ctx) ───────────────
    print(col("\n  Phase 1 : Warm-up initial...", "CYN"))
    wu = api_call(args.url, args.model, "Hello, reply with one word.", 256, num_predict=5)
    if not wu["ok"]:
        print(col(f"  [ERREUR] FLM inaccessible : {wu.get('error','?')}", "RED"))
        print(f"  Lance d'abord : flm serve {args.model} --ctx-len 32768 --pmode performance")
        sys.exit(1)
    print(col(f"  Modèle actif (TPS warm-up = {wu['tps']:.1f} t/s)", "GRN"))

    # ── Phase 2 : Scan TPS ────────────────────────────────────────────────────
    print(col("\n  Phase 2 : Scan TPS multi-contextes...", "CYN"))
    scan_results = scan(args.url, args.model, ctx_list, args.runs, verbose=True)

    # ── Phase 3 : Calibration roofline ────────────────────────────────────────
    print(col("\n  Phase 3 : Calibration roofline...", "CYN"))

    warm_pts = [(ctx, r["tps_avg"]) for ctx, r in scan_results.items()
                if r.get("ok") and r.get("tps_avg", 0) > 0
                and (r.get("cold") is None or r["tps_avg"] > r["cold"])]

    bw_eff_fit, w_eff_fit, r2 = fit_roofline(warm_pts)

    if bw_eff_fit is None:
        print(col("  Calibration impossible (données insuffisantes) — utilise les priors", "YEL"))
        bw_eff_fit = BW_EFF_PRIOR
        w_eff_fit  = W_EFF_PRIOR
        r2         = 0.0
    else:
        eta_fit = round(bw_eff_fit / BW_MEASURED_GBs, 4)
        print(f"  BW_EFF calibré : {bw_eff_fit:.4f} GB/s  (prior={BW_EFF_PRIOR:.3f}  Δ={bw_eff_fit-BW_EFF_PRIOR:+.3f})")
        print(f"  W_eff  calibré : {w_eff_fit:.4f} GB    (prior={W_EFF_PRIOR:.2f}  Δ={w_eff_fit-W_EFF_PRIOR:+.3f})")
        print(f"  η      calibré : {eta_fit:.4f}        (prior={ETA_PRIOR:.4f})")
        print(f"  R²             : {r2:.4f}" + col("  (bon)", "GRN") if r2 > 0.95 else f"  R²: {r2:.4f}" + col("  (faible — données bruitées)", "YEL"))

    # ── Phase 4 : Tableau comparatif ──────────────────────────────────────────
    hdr("RÉSULTATS")
    print(f"  {'ctx':>7}  {'TPS warm':>9}  {'±σ':>5}  {'pred prior':>11}  {'pred calib':>11}  {'err prior':>10}  {'err calib':>10}  {'TTFT s':>7}")
    print("  " + "-" * 88)

    for ctx in sorted(scan_results):
        r = scan_results[ctx]
        if not r.get("ok"): continue
        tps  = r["tps_avg"]
        std  = r.get("tps_std", 0.0)
        pp   = tps_pred(ctx, BW_EFF_PRIOR, W_EFF_PRIOR)
        pc   = tps_pred(ctx, bw_eff_fit, w_eff_fit)
        ep   = round((tps - pp) / pp * 100, 1) if pp > 0 else 0.0
        ec   = round((tps - pc) / pc * 100, 1) if pc > 0 else 0.0
        ttft = r.get("ttft_avg")
        ec_col = "GRN" if abs(ec) < 5 else "YEL" if abs(ec) < 15 else "RED"
        ttft_s = f"{ttft:.3f}" if ttft else "  N/A"
        cold_s = col(f" [cold={r['cold']:.2f}]", "DIM") if r.get("cold") else ""
        print(f"  {ctx:7d}  {tps:9.3f}  {std:5.3f}"
              f"  {pp:11.3f}  {pc:11.3f}"
              f"  {ep:+9.1f}%  {col(f'{ec:+.1f}%', ec_col):>10}"
              f"  {ttft_s:>7}{cold_s}")

    # ── Phase 5 : Analyse goulot ──────────────────────────────────────────────
    hdr("ANALYSE GOULOT")
    bottleneck = analyze_bottleneck(scan_results, bw_eff_fit, w_eff_fit)
    regime = bottleneck["regime"]
    reg_col = {"DRAM_BW_BOUND": "CYN", "DRAM_LATENCY_BOUND": "YEL",
               "KV_PRESSURE": "RED", "COMPUTE_BOUND": "MAG",
               "DRAM_MIXED": "YEL"}.get(regime, "RST")

    print(col(f"\n  RÉGIME : {regime}", reg_col))
    print(f"  Détail : {bottleneck['detail']}")
    print(f"  Err roofline moy : {bottleneck['avg_err_pct']:+.2f}%"
          f"  (min={bottleneck['min_err_pct']:+.1f}%  max={bottleneck['max_err_pct']:+.1f}%)")
    if bottleneck["kv_drift"] != 0:
        print(f"  KV drift grands ctx : {bottleneck['kv_drift']:+.2f}%")

    print(col("\n  INTERPRÉTATION :", "BOLD"))
    interp = {
        "DRAM_BW_BOUND": (
            "DRAM bande passante = goulot primaire.\n"
            "  Le modèle est DRAM-BW bound : chaque token decode lit W_eff + KV depuis LPDDR5X.\n"
            "  ML engine % élevé = DMA actif, pas NPU compute saturé.\n"
            "  Gains possibles : réduire ctx (moins de KV), KV throttle."
        ),
        "DRAM_LATENCY_BOUND": (
            "Overhead driver NPU / reconfig xclbin dominant.\n"
            "  TPS > roofline BW : le vrai goulot est la latence driver, pas la BW.\n"
            "  Cold start coûteux. Warm-up obligatoire. '--pmode performance' aide."
        ),
        "KV_PRESSURE": (
            "KV cache pressure aux grands contextes.\n"
            "  TPS chute plus vite que prédit par W_eff seul.\n"
            "  Réduire max_prefill_len ou ctx-len."
        ),
        "COMPUTE_BOUND": (
            "NPU TOPS saturé (rare en decode GEMV sur XDNA2).\n"
            "  TPS ne décline pas avec ctx → opérations GEMV limitent.\n"
            "  Vérifier si prefill vs decode confusion."
        ),
        "DRAM_MIXED": (
            "Régime mixte latence/BW.\n"
            "  Typique à faible ctx (latence domine) → grand ctx (BW domine)."
        ),
    }
    print("  " + interp.get(regime, "").replace("\n", "\n  "))

    # ── Phase 6 : Recommandations ─────────────────────────────────────────────
    hdr("RECOMMANDATIONS")
    eta_fit = round(bw_eff_fit / BW_MEASURED_GBs, 4)
    recs = recommend(scan_results, bw_eff_fit, w_eff_fit, bottleneck, args.target_tps)

    print(f"  {'Paramètre':<22} {'Valeur':>18}  Raison")
    print("  " + "-" * 78)
    for rec in recs:
        val = str(rec["value"])[:18]
        print(f"  {rec['param']:<22} {val:>18}  {rec['reason']}")

    print(col("\n  CONSTANTES AJUSTÉES pour test_gains.py :", "BOLD"))
    print(f"    BW_EFF_GBs = {bw_eff_fit:.4f}")
    print(f"    ETA        = {eta_fit:.4f}")
    w_key = str(args.model).replace(":", "_").replace(".", "_")
    print(f"    W_EFF_{w_key[:8].upper()} = {w_eff_fit:.4f}")

    # ── Export ────────────────────────────────────────────────────────────────
    export_path = args.export or f"calibration_{args.model.replace(':','_').replace('.','_')}.json"
    cal = export_calibration(args.model, bw_eff_fit, w_eff_fit, r2,
                             bottleneck, recs, scan_results, export_path)
    print(col(f"\n  Calibration exportée : {export_path}", "GRN"))

    if args.patch_test_gains:
        print(col("\n  Patch test_gains.py...", "CYN"))
        patch_test_gains(bw_eff_fit, w_eff_fit, eta_fit, args.model)

    # ── H6 cold-start table ───────────────────────────────────────────────────
    if args.h6_table:
        hdr("H6 COLD-START IMPACT")
        tps_ss = bw_eff_fit / (w_eff_fit or W_EFF_PRIOR) if w_eff_fit else bw_eff_fit / W_EFF_PRIOR
        veh_ss = veh["median_tps"] if veh else None
        print_h6_table(veh_ss or tps_ss)
        print(col(f"\n  Source TPS steady-state : {'VEH+Dr0 mesuré' if veh_ss else 'roofline calibré'}", "CYN"))
        print(f"  H6_RECONFIG = {H6_RECONFIG_MS:.0f} ms/req")
        print(f"  Fix via h6_hook.dll : python bus_optimizer.py --h6-hook")

    sep()
    print(col("  TERMINÉ", "BOLD"))
    print()


if __name__ == "__main__":
    main()
