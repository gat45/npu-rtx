"""
bench_soc_matrix.py — Bench ciblé FastFlowLM / XDNA2
======================================================
6 points ctx = {1024, 2048, 4096, 8192, 16384, 32768}

Pour chaque ctx :
  1. Construit un préfill de ~ctx tokens (histoire répétée)
  2. Appelle POST /api/generate  (num_predict=80 tokens fixes)
  3. Mesure decode_tps = eval_count / (eval_duration_ns / 1e9)
  4. Compare vs baseline SOC_MATRIX (mesures réelles)
  5. Compare vs prédiction flm_bottleneck_analyzer (eta=0.7309, kv=30KB/tok)

Usage :
  python bench_soc_matrix.py                    # modèle par défaut qwen3.5:4b
  python bench_soc_matrix.py --model llama3.2:1b
  python bench_soc_matrix.py --ctx 4096         # un seul point
  python bench_soc_matrix.py --url http://127.0.0.1:52625
  python bench_soc_matrix.py --out results.json # sauvegarde JSON
"""

import argparse
import csv
import json
import time
import sys
import requests
from datetime import datetime

# ─── Baseline SOC_MATRIX (mesures réelles, soc_performance_matrix.csv) ─────────
#  ctx   : (decode_tps, prefill_tps, ram_mb)
SOC_MATRIX = {
    1024:  (12.20, 338.22,  3.50),
    2048:  (11.90, 396.23,  5.83),
    4096:  (11.61, 464.12,  8.39),
    8192:  (10.83, 488.08, 15.89),
    16384: ( 9.93, 479.70, 32.33),
    32768: ( 8.29, 424.97, 73.92),
}

# ─── Prédiction roofline (flm_bottleneck_analyzer, calibré eta=0.7309) ──────────
BW_MEASURED   = 30.0          # GB/s
ETA           = 0.7309        # transfer efficiency (ΦT) — calibré ctx=1024..32768
BW_EFF        = BW_MEASURED * ETA  # 21.93 GB/s
# IMPORTANT : eta varie avec ctx
#   ctx < ~100 tokens  → eta ≈ 1.0  (pas de KV overhead inter-kernel)
#   ctx = 1024..32768  → eta = 0.7309 (calibré sur SOC_MATRIX, std=0.015)
# predict_tps() utilise eta=0.7309 (valide pour le bench SOC_MATRIX ctx=1K..32K)

# KV bytes/token par modèle (INT8 GQA calibré)
# Architecture réelle (config.json des modèles) :
#   llama3.2:1b    : standard GQA  2*8*64*16   = 16,384 bytes/tok (INT8)
#   qwen3.5:4b     : HYBRIDE — seulement 8/32 couches en full-attention (interval=4)
#                    KV full-att seulement : 2*4*256*8 = 16,384 bytes/tok
#                    Empirique SOC_MATRIX  : 30,000 bytes/tok (linéaire-att contribue aussi)
#   deepseek-r1:8b : standard GQA  2*8*128*32  = 65,536 bytes/tok (INT8)
KV_MODEL = {
    "qwen3.5:4b":     30_000,   # empirique SOC_MATRIX ±3.3% (hybride linéaire+full-att)
    "qwen3.5:4b_soc": 30_000,
    "llama3.2:1b":    16_384,   # INT8 GQA calculé (2*8*64*16*1)
    "deepseek-r1:8b": 65_536,   # INT8 GQA calculé (2*8*128*32*1)
}
KV_DEFAULT = 30_000

# Taille modèle effective (poids chargés depuis DRAM par decode step)
# NOTE : les fichiers model.q4nx incluent les xclbin compilés (restent en NPU memory)
#        Seuls les poids tenseurs sont relus depuis DRAM à chaque step.
# Sources :
#   W_eff = BW_measured / TPS_warm_pt25 (bench_3models.py, pt≈25, eta≈1.0)
#   Fichiers : llama=1.3GB, qwen4b=4.2GB(+637MB vision), deepseek=5.4GB, qwen9b=7.4GB
#
#   llama3.2:1b  : file=1.3 GB, INT8 confirmé (théorique INT8=1.24 GB + scales)
#                  W_eff mesuré = 30.0/19.7 = 1.52 GB
#   qwen3.5:4b   : file LLM=4.2 GB (xclbin inclus), poids INT4=1.75 GB
#                  W_eff mesuré = 30.0/17.25 = 1.74 GB ≈ INT4 ✓
#   deepseek:8b  : file=5.4 GB (INT4 poids + FP16 lm_head ~1 GB)
#                  W_eff mesuré = 30.0/3.3 = 9.09 GB (overhead R1 reasoning inclus)
MODEL_SIZE_GB = {
    "qwen3.5:4b":     1.75,   # INT4 confirmé (bench warm Δ<1%, poids tenseurs seuls)
    "qwen3.5:4b_soc": 1.75,
    "llama3.2:1b":    1.52,   # INT8 confirmé (bench warm, file 1.3 GB)
    "deepseek-r1:8b": 9.09,   # empirique bench warm (overhead R1 non-modélisé)
}
MODEL_SIZE_DEFAULT = 1.75


def predict_tps(model: str, ctx: int) -> float:
    """Prédit decode TPS via roofline BW-bound.
    Valide pour ctx >= 1024 (eta=0.7309 calibré sur SOC_MATRIX).
    Sous-estime à ctx < 100 (eta → 1.0 quand KV ≈ 0).
    """
    kv   = KV_MODEL.get(model, KV_DEFAULT)
    w_gb = MODEL_SIZE_GB.get(model, MODEL_SIZE_DEFAULT)
    kv_gb = kv * ctx / 1e9
    total_bw = w_gb + kv_gb          # GB lus par token décodé
    tps = BW_EFF / total_bw          # tokens/s
    return tps


# ─── Prompt de préfill ──────────────────────────────────────────────────────────
STORY = (
    "In a distant future where Earth had fallen into quiet ruin, humanity lived in "
    "fragments, scattered across domed outposts and deep underground vaults. The sky "
    "was no longer blue—it shimmered with artificial auroras, remnants of weather-"
    "control systems left unattended for centuries. Among the last settlements was "
    "Bastion-9, a circular enclave powered by forgotten technologies and guarded by "
    "an ancient AI named Solen. Solen had not spoken in nearly fifty years. Inside "
    "Bastion-9 lived a young technician named Ori. Unlike most, Ori was born with an "
    "unusual trait: she could interface with dead systems using nothing more than "
    "touch. The elders called her a resonant—a rarity, perhaps even a myth—until Ori "
    "proved them right by awakening the water grid that had been dry for decades. "
    "One day, while surveying the decaying perimeter, Ori found a shard of obsidian "
    "glass buried in the dust. It pulsed when she touched it. Static voices filled "
    "her mind—fragments of languages, images of cities with skies, oceans that moved, "
    "and towers that breathed. She brought the shard to the Council. They feared it. "
    "But Solen, the silent AI, flickered back to life. Its first words in decades "
    "were: The Reclaimer has touched the key. "
)
CHARS_PER_TOKEN = 4  # estimation anglais

QUESTION = "\n\nWhat is this story about? Please give a detailed summary:"


def build_prompt(ctx_tokens: int) -> str:
    """Construit un préfill de ~ctx_tokens tokens en répétant STORY."""
    # réserve ~30 tokens pour la question finale
    target_chars = max(10, ctx_tokens - 30) * CHARS_PER_TOKEN
    base = STORY
    while len(base) < target_chars:
        base = base + " " + STORY
    return base[:target_chars] + QUESTION


# ─── Bench ──────────────────────────────────────────────────────────────────────

def bench_ctx(url: str, model: str, ctx: int, num_predict: int = 80,
              verbose: bool = True) -> dict:
    """Un point de mesure. Retourne dict avec tps mesuré, prédit, baseline."""
    prompt = build_prompt(ctx)
    est_prompt_tokens = len(prompt) // CHARS_PER_TOKEN

    payload = {
        "model":       model,
        "prompt":      prompt,
        "stream":      False,
        "num_predict": num_predict,
        "options": {
            "num_ctx":     ctx + num_predict + 64,   # fenêtre contexte
            "temperature": 0.0,                       # déterministe
        },
    }

    if verbose:
        print(f"  ctx={ctx:6d}  prompt≈{est_prompt_tokens:5d} tok  ", end="", flush=True)

    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{url}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("ERREUR : serveur FLM inaccessible")
        return {"ctx": ctx, "error": "connection_refused"}
    except requests.exceptions.Timeout:
        print("ERREUR : timeout (300s)")
        return {"ctx": ctx, "error": "timeout"}
    except requests.exceptions.HTTPError as e:
        print(f"ERREUR HTTP : {e}")
        return {"ctx": ctx, "error": str(e)}

    wall_s = time.perf_counter() - t0
    data = resp.json()

    # Métriques Ollama
    eval_count     = data.get("eval_count", 0)
    eval_dur_ns    = data.get("eval_duration", 0)
    prefill_count  = data.get("prompt_eval_count", 0)
    prefill_dur_ns = data.get("prompt_eval_duration", 0)

    decode_tps  = eval_count  / (eval_dur_ns  / 1e9) if eval_dur_ns  > 0 else 0.0
    prefill_tps = prefill_count / (prefill_dur_ns / 1e9) if prefill_dur_ns > 0 else 0.0

    # Baseline + prédiction
    baseline     = SOC_MATRIX.get(ctx)
    baseline_tps = baseline[0] if baseline else None
    pred_tps     = predict_tps(model, ctx)

    err_vs_pred = (decode_tps - pred_tps) / pred_tps * 100 if pred_tps else None
    err_vs_base = (decode_tps - baseline_tps) / baseline_tps * 100 if baseline_tps else None

    result = {
        "ctx":          ctx,
        "model":        model,
        "decode_tps":   round(decode_tps,  2),
        "prefill_tps":  round(prefill_tps, 2),
        "pred_tps":     round(pred_tps,    2),
        "baseline_tps": baseline_tps,
        "err_vs_pred":  round(err_vs_pred, 1) if err_vs_pred is not None else None,
        "err_vs_base":  round(err_vs_base, 1) if err_vs_base is not None else None,
        "eval_tokens":  eval_count,
        "wall_s":       round(wall_s, 1),
    }

    if verbose:
        b_str  = f"{baseline_tps:5.2f}" if baseline_tps else "  --  "
        e_base = f"{err_vs_base:+.1f}%" if err_vs_base is not None else "   -- "
        e_pred = f"{err_vs_pred:+.1f}%" if err_vs_pred is not None else "   -- "
        ok     = "✓" if err_vs_base is not None and abs(err_vs_base) < 10 else "?"
        print(
            f"decode={decode_tps:5.2f} t/s  "
            f"baseline={b_str}  Δbase={e_base:>7}  "
            f"pred={pred_tps:5.2f}  Δpred={e_pred:>7}  "
            f"wall={wall_s:.0f}s  {ok}"
        )

    return result


def run_bench(url: str, model: str, ctx_list: list, num_predict: int = 80,
              out_file: str = None) -> list:
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  FastFlowLM SOC Matrix Bench  —  {model}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  server: {url}")
    print(f"  num_predict={num_predict}  |  eta={ETA}  |  BW_eff={BW_EFF:.2f} GB/s")
    print(sep)
    print(f"  {'ctx':>6}  {'prompt':>7}  {'decode':>6}  {'baseline':>8}  {'Δbase':>7}  "
          f"{'pred':>6}  {'Δpred':>7}  {'wall':>5}")
    print(f"  {'-'*80}")

    results = []
    for ctx in ctx_list:
        r = bench_ctx(url, model, ctx, num_predict)
        results.append(r)
        if "error" in r:
            print(f"  ⚠ ctx={ctx} échoué ({r['error']}) — arrêt")
            break

    # Résumé
    ok = [r for r in results if "error" not in r]
    if ok:
        print(f"\n{sep}")
        print(f"  Résumé : {len(ok)}/{len(ctx_list)} points réussis")
        errs = [abs(r["err_vs_base"]) for r in ok if r["err_vs_base"] is not None]
        preds = [abs(r["err_vs_pred"]) for r in ok if r["err_vs_pred"] is not None]
        if errs:
            print(f"  Δ vs baseline  — moy={sum(errs)/len(errs):.1f}%  max={max(errs):.1f}%")
        if preds:
            print(f"  Δ vs prédiction — moy={sum(preds)/len(preds):.1f}%  max={max(preds):.1f}%")

        # Dégradation ctx=min → ctx=max
        first_tps = ok[0]["decode_tps"]
        last_tps  = ok[-1]["decode_tps"]
        degr = (last_tps - first_tps) / first_tps * 100
        print(f"  Dégradation ctx={ok[0]['ctx']}→{ok[-1]['ctx']} : "
              f"{first_tps:.2f} → {last_tps:.2f} t/s  ({degr:+.1f}%)")
        print(sep)

    # Sauvegarde JSON
    if out_file:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "model":     model,
            "url":       url,
            "results":   results,
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n  → JSON sauvegardé : {out_file}")

    return results


def write_flm_csv(results: list, model: str, csv_path: str):
    """
    Export CSV compatible avec generate_charts.py du repo FastFlowLM-npu-benchmark.
    [rolandpascua-cloud/FastFlowLM-npu-benchmark] Colonnes attendues :
      context_length_k, prefill_avg_toks_per_s, decoding_avg_toks_per_s
    Nom de fichier attendu par generate_charts.py : bench_<model>_<date>.csv
    """
    ok = [r for r in results if "error" not in r]
    if not ok:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context_length_k", "prefill_avg_toks_per_s", "decoding_avg_toks_per_s"])
        for r in ok:
            w.writerow([
                r["ctx"] / 1024,          # en K tokens (1024 → 1.0, 32768 → 32.0)
                round(r.get("prefill_tps", 0.0), 2),
                round(r["decode_tps"], 2),
            ])
    print(f"  → CSV flm-bench compatible : {csv_path}")


# ─── Main ───────────────────────────────────────────────────────────────────────

CTX_POINTS = [1024, 2048, 4096, 8192, 16384, 32768]

# Limite shmem NPU — OOM si 4B+ modèle à ctx=32768 (TECHNICAL_NOTES Issue #4)
# qwen3.5:9b   KV@32k = 30000×32768/1e9 = 0.98 GB + W_eff 4.75 GB = 5.7 GB → OK
# deepseek:8b  KV@32k = 65536×32768/1e9 = 2.15 GB + W_eff 9.09 GB → OOM risk
_OOM_RISK_MODELS = {"deepseek-r1:8b", "deepseek-r1:7b"}
_OOM_SAFE_MAX_CTX = 16384   # ctx max sûr pour modèles à risque

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bench ciblé FastFlowLM — 6 points SOC_MATRIX"
    )
    parser.add_argument("--model",   default="qwen3.5:4b",
                        help="Modèle FLM (défaut: qwen3.5:4b)")
    parser.add_argument("--ctx",     type=int, default=None,
                        help="Un seul point ctx (défaut: les 6)")
    parser.add_argument("--url",     default="http://127.0.0.1:52625",
                        help="URL serveur FLM (défaut: 127.0.0.1:52625)")
    parser.add_argument("--predict", type=int, default=80,
                        help="Tokens à décoder par mesure (défaut: 80)")
    parser.add_argument("--out",     default=None,
                        help="Fichier JSON de sortie")
    parser.add_argument("--csv",     default=None,
                        help="Export CSV compatible flm bench / generate_charts.py")
    parser.add_argument("--auto-csv", action="store_true",
                        help="Auto-génère le CSV avec nom bench_<model>_<date>.csv")
    parser.add_argument("--no-32k",  action="store_true",
                        help="Exclure ctx=32768 (évite OOM NPU pour gros modèles)")
    args = parser.parse_args()

    ctx_list = [args.ctx] if args.ctx else CTX_POINTS

    # Guard OOM : exclure ctx=32768 pour modèles à risque ou si --no-32k
    if args.no_32k or args.model in _OOM_RISK_MODELS:
        ctx_list = [c for c in ctx_list if c <= _OOM_SAFE_MAX_CTX]
        if args.model in _OOM_RISK_MODELS:
            print(f"  [OOM GUARD] {args.model} : ctx limité à {_OOM_SAFE_MAX_CTX}"
                  f" (KV/tok élevé, risque shmem NPU)")
            print(f"  → Lancer FLM avec : flm serve {args.model} --ctx-len {_OOM_SAFE_MAX_CTX}")

    results = run_bench(args.url, args.model, ctx_list, args.predict, args.out)

    # Export CSV compatible generate_charts.py
    csv_path = args.csv
    if args.auto_csv and not csv_path:
        stamp     = datetime.now().strftime("%Y%m%d")
        model_tag = args.model.replace(":", "_")
        csv_path  = f"bench_{model_tag}_{stamp}.csv"
    if csv_path:
        write_flm_csv(results, args.model, csv_path)
