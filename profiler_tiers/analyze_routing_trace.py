# Analyseur de trace routing (GGML_ROUTING_TRACE CSV) — 2026-09-21
# =====================================================================
# Entrée : CSV produit par le hook mul_mat_id du fork llama-cpp-turboquant
#   layer,src0_name,n_tokens,top_k,n_expert,ids   (ids = top-k par token, ';' séparateur tokens)
# Sortie : distribution MEASURED pour remplacer le skew ASSUMED du profiler :
#   - fréquence experts par couche (top-32/64/128 coverage)
#   - overlap inter-tokens (Jaccard moyen des sets consécutifs)
#   - skew empirique : fit hit(résidence) = (res/N)^skew sur la courbe de coverage réelle
#   - cache statique optimal : meilleurs experts par couche pour hit max @ résidence donnée
#
# Usage : py analyze_routing_trace.py <trace.csv> [--res 45 64 101 143] [--topk 8]
import csv, sys, argparse, math
from collections import Counter, defaultdict

def load(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        header = next(r)
        assert header[:5] == ["layer", "src0_name", "n_tokens", "top_k", "n_expert"], header
        for row in r:
            if not row or len(row) < 6:
                continue
            layer = int(row[0])
            ids_tok = [tuple(int(x) for x in tok.split()) for tok in row[5].split(";") if tok.strip()]
            rows.append((layer, row[1], ids_tok))
    return rows

def coverage_curve(freq_sorted, n_exp):
    """coverage[k] = proba qu'un expert tiré soit dans les k plus fréquents (loi empirique)."""
    total = sum(freq_sorted)
    cum, out = 0, []
    for i, v in enumerate(freq_sorted):
        cum += v
        out.append(cum / total if total else 0.0)
    return out  # index k-1 -> coverage des k premiers

def fit_skew(res_list, hit_list, n_exp):
    """fit log(hit) = skew * log(res/N) + log(topk*tokens_sim) — régression sur points coverage."""
    xs = [math.log(r / n_exp) for r in hit_list if r > 0]
    ys = [math.log(h) for r, h in zip(hit_list, hit_list)]
    # fit direct sur (res/N)^skew = hit : skew = somme(x*y)/somme(x²) avec y=log(hit)
    num = sum(x * math.log(h) for x, h in zip(xs, [h for _, h in zip(hit_list, hit_list)]))
    den = sum(x * x for x in xs)
    return (num / den) if den else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--res", type=int, nargs="+", default=[32, 45, 64, 101, 143])
    ap.add_argument("--decode-only", action="store_true", help="ignore les segments prefill (n_tokens>1)")
    args = ap.parse_args()

    rows = load(args.trace)
    # dédup : gate et down d'une même couche routent identiquement → garder 1 nœud/couche/segment
    seen = {}
    segments = []  # (layer, ids_tok) dans l'ordre, prefill (n>1) et decode (n==1) mélangés
    for layer, name, ids_tok in rows:
        key = (layer, len(ids_tok))
        if key in seen and seen[key] == ids_tok:
            continue
        seen[key] = ids_tok
        segments.append((layer, ids_tok))

    n_exp = max((max(tok) + 1 for _, toks in segments for tok in toks), default=0)
    topk = len(segments[0][1][0]) if segments else 0
    print(f"trace: {len(rows)} lignes brutes -> {len(segments)} segments uniques | n_exp>={n_exp} top_k={topk}")

    # séparer prefill (batch) vs decode (1 token)
    prefill = [(l, t) for l, t in segments if len(t) > 1]
    decode = [(l, t) for l, t in segments if len(t) == 1]
    print(f"  prefill: {len(prefill)} couches-segments | decode: {len(decode)} couches-tokens")
    if args.decode_only and decode:
        use = decode
    else:
        use = segments

    # fréquences globales + par couche
    freq = Counter()
    freq_layer = defaultdict(Counter)
    n_draws = 0
    sets_seq = []  # pour overlap
    for layer, toks in use:
        for tok in toks:
            for e in tok:
                freq[e] += 1
                freq_layer[layer][e] += 1
                n_draws += 1
            sets_seq.append(frozenset(tok))

    total = sum(freq.values())
    print(f"\n=== DISTRIBUTION GLOBALE ({n_draws} tirages, {len(freq)} experts distincts vus) ===")
    top = freq.most_common()
    cov = coverage_curve([v for _, v in top], n_exp)
    for k in (8, 16, 32, 64, 128):
        if k <= len(cov):
            print(f"  top-{k:>3}: coverage {cov[k-1]*100:5.1f}%")

    # overlap inter-tokens consécutifs (Jaccard)
    if len(sets_seq) > 1:
        js = [len(a & b) / len(a | b) for a, b in zip(sets_seq, sets_seq[1:])]
        print(f"\n=== OVERLAP INTER-TOKENS (Jaccard moyen) === {sum(js)/len(js)*100:.1f}%  "
              f"(médian {sorted(js)[len(js)//2]*100:.1f}%, n={len(js)})")

    # skew empirique par couche : coverage des N résidents les plus fréquents
    print("\n=== SKEW EMPIRIQUE PAR COUCHE (coverage @ résidence) ===")
    print(f"  {'couche':>6} {'@32':>6} {'@45':>6} {'@64':>6} {'@101':>6} {'@143':>6}")
    covs_all = []
    for layer in sorted(freq_layer):
        fl = freq_layer[layer].most_common()
        cl = coverage_curve([v for _, v in fl], n_exp)
        covs_all.append(cl)
        def c(r):
            return cl[min(r, len(cl)) - 1] * 100 if cl else 0.0
        print(f"  {layer:>6} {c(32):6.1f} {c(45):6.1f} {c(64):6.1f} {c(101):6.1f} {c(143):6.1f}")

    # skew global moyen (moyenne des coverage par couche) → équivalent skew du modèle (res/N)^skew
    if covs_all:
        n_layers = len(covs_all)
        print(f"\n=== FIT SKEW GLOBAL (moyenne {n_layers} couches, vs (res/{n_exp})^skew) ===")
        # calibrated skew : hit(res) = (res/N)^skew avec hit = coverage moyenne
        # skew = log(hit)/log(res/N) par point, médiane
        import statistics
        skews = []
        for r in args.res:
            hits = [cl[min(r, len(cl)) - 1] for cl in covs_all if len(cl) >= 1]
            h = sum(hits) / len(hits)
            if h > 0 and r < n_exp:
                skews.append(math.log(h) / math.log(r / n_exp))
        if skews:
            print(f"  skew MEASURED (médiane) = {statistics.median(skews):.3f}  [points: "
                  + ", ".join(f"{s:.2f}" for s in skews) + "]")
            print(f"  (le profiler utilisait 0.85 ASSUMED / 0.6 en scénario localité forte)")

if __name__ == "__main__":
    main()
