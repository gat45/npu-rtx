# Étape 5 — Benchmark cross-tier A/B/C (comme governor bench)
# A = RTX seul | B = RTX+XDNA2 | C = RTX+XDNA2+CPU
# Métrique : perf/compute + VoI + contention

CONFIGS = {
    "A_ RTX_only": {"tiers": ["RTX"], "expected_toks": 8.9},
    "B_ RTX_XDNA2": {"tiers": ["RTX","XDNA2"], "expected_toks": 10.5},
    "C_ RTX_XDNA2_CPU": {"tiers": ["RTX","XDNA2","CPU"], "expected_toks": 9.8},
}

def run_benchmark(seed=42):
    for name, cfg in CONFIGS.items():
        # Appeler place_expert + exécution simulée / réelle
        # Produire bench.json + comparison.json dans xdna2/bench/
        pass
