# -*- coding: utf-8 -*-
"""
analyze_drift.py -- Analyseur de Derive Physique NPU / LPDDR5X
==============================================================
Lit les fichiers JSON produits par bench_perf.py (dossier bench-flm/).
Format : liste de dicts avec tps_live, w_eff_obs_gb, ctx, ttft, etc.

Usage :
    python analyze_drift.py
    python analyze_drift.py --dir bench-flm --model qwen3.5:9b-cust
"""

import os
import json
import glob
import argparse
import statistics

BW_EFF_TARGET = 21.93   # GB/s (plafond Strix calibre)

W_EFF_NOM = {
    "qwen3.5:9b":          2.95,
    "qwen3.5:9b-cust":     2.87,
    "qwen3.5:4b":          1.75,
    "llama3.2:1b":         0.55,
    "deepseek-r1-0528:8b": 2.50,
}

G = lambda s: "\033[92m" + str(s) + "\033[0m"
Y = lambda s: "\033[93m" + str(s) + "\033[0m"
R = lambda s: "\033[91m" + str(s) + "\033[0m"
C = lambda s: "\033[96m" + str(s) + "\033[0m"
B = lambda s: "\033[1m"  + str(s) + "\033[0m"


def model_from_fname(fname):
    base = os.path.basename(fname).replace(".json", "")
    parts = base.split("_")
    if len(parts) >= 3:
        model_parts = parts[1:-2]
        model = "_".join(model_parts)
        model = model.replace("qwen3.5_", "qwen3.5:")
        model = model.replace("llama3.2_", "llama3.2:")
        model = model.replace("deepseek_r1_0528_", "deepseek-r1-0528:")
        return model
    return ""


def w_eff_for_model(model_name):
    key = next(
        (k for k in sorted(W_EFF_NOM, key=len, reverse=True) if k in model_name),
        None,
    )
    return W_EFF_NOM.get(key, 2.87) if key else 2.87


def drift_label(drift_pct):
    if drift_pct > 20:
        return R("[CRITIQUE] Contention LPDDR5X active")
    elif drift_pct > 8:
        return Y("[MODERE]   Legere baisse d'efficacite bus")
    elif drift_pct > 0:
        return G("[BON]      Performance proche du plafond")
    else:
        return G("[OPTIMAL]  TPS reel >= prediction physique")


def analyze_file(fpath, model, w_nom, ceil):
    try:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("  [!] {} illisible : {}".format(os.path.basename(fpath), e))
        return []

    if not isinstance(data, list):
        data = [data]

    valid = [r for r in data if r.get("ctx", 0) > 0 and r.get("tps_live", 0) > 0]
    if not valid:
        return []

    print("  [{}] ({} runs valides)".format(os.path.basename(fpath), len(valid)))
    results = []
    for r in valid:
        tps      = r["tps_live"]
        tps_pred = r.get("tps_pred", ceil)
        ttft     = r.get("ttft", 0)
        pfill    = r.get("prefill_tps", 0)
        ram_mb   = r.get("ram_sys_mb", 0)
        w_obs    = r.get("w_eff_obs_gb", w_nom)
        dram_p   = r.get("dram_pressure_pct", 0)
        ctx      = r["ctx"]
        is_warm  = r.get("is_warmup", False)

        drift    = (ceil - tps) / ceil * 100.0
        w_infl   = (w_obs - w_nom) / w_nom * 100.0

        warm_tag = " [warmup]" if is_warm else ""
        print("    ctx={:5d}  tps={}  drift={:.1f}%  w_obs={:.3f} GB ({:+.0f}%)  {}"
              .format(ctx,
                      B("{:.2f} t/s".format(tps)),
                      drift, w_obs, w_infl,
                      drift_label(drift)) + warm_tag)
        if ttft > 0:
            print("           TTFT={:.0f} ms  prefill={:.0f} t/s  "
                  "RAM={:.1f} GB  press={:.1f}%"
                  .format(ttft * 1000, pfill, ram_mb / 1024, dram_p))

        results.append({"tps": tps, "drift": drift, "w_infl": w_infl, "is_warm": is_warm})
    print()
    return results


def print_summary(all_results, model):
    real = [r for r in all_results if not r["is_warm"]]
    if len(real) < 2:
        return

    tps_list   = [r["tps"]    for r in real]
    drift_list = [r["drift"]  for r in real]
    winf_list  = [r["w_infl"] for r in real]
    mean_drift = statistics.mean(drift_list)

    print("  " + B("RESUME") + " ({} runs sans warmup) :".format(len(real)))
    print("    TPS   : moy={:.2f}  sigma={:.3f}  min={:.2f}  max={:.2f} t/s".format(
        statistics.mean(tps_list),
        statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0,
        min(tps_list), max(tps_list)))
    print("    Derive: moy={:.1f}%  inflation W_eff={:+.1f}%".format(
        mean_drift, statistics.mean(winf_list)))

    if mean_drift > 20:
        print("    " + R("RECO URGENTE : spec_proxy.py (gain potentiel x1.5-2 TPS)"))
        print("    " + R("           OU compression KV MEDIUM/HEAVY (kv_controller.py)"))
        print("    " + Y("    Cause probable : contention LPDDR5X (GPU/navigateur actif)"))
        print("    " + Y("    -> Fermer les apps concurrentes avant l'inference"))
    elif mean_drift > 8:
        print("    " + Y("RECO : spec_proxy.py conseille (+20-30% TPS effectif)"))
    else:
        print("    " + G("OPTIMAL : systeme a {:.0f}% du plafond physique".format(
            100 - mean_drift)))
        print("    " + G("RECO    : spec_proxy.py pour gain supplementaire uniquement"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",   default="bench-flm")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    if not files:
        print(R("[-] Aucun fichier JSON dans '{}/'.".format(args.dir)))
        print("    Lance : python bench_perf.py --model qwen3.5:9b --runs 3")
        return

    by_model = {}
    for fpath in files:
        model = model_from_fname(fpath)
        if not model or ":" not in model:
            continue
        if args.model and args.model not in model:
            continue
        by_model.setdefault(model, []).append(fpath)

    if not by_model:
        print(Y("Aucun modele reconnu dans les fichiers. Verifiez le dossier '{}'.".format(args.dir)))
        return

    print()
    print(B("=============================================="))
    print(B("  ANALYSE DERIVE NPU - LPDDR5X / Strix Point"))
    print(B("=============================================="))
    print()

    for model in sorted(by_model):
        fpaths = by_model[model]
        w_nom  = w_eff_for_model(model)
        ceil   = BW_EFF_TARGET / w_nom
        print(C("Modele : {}".format(model)) +
              "  (W_eff nom={:.3f} GB, plafond={:.2f} t/s)".format(w_nom, ceil))
        print("-" * 54)

        all_results = []
        for fpath in fpaths:
            res = analyze_file(fpath, model, w_nom, ceil)
            all_results.extend(res)

        print_summary(all_results, model)
        print()
        print("=" * 54)
        print()


if __name__ == "__main__":
    main()
