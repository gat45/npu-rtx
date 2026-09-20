# static/ — Static Oracle (Phase A) — Qwen3.8-Flash-Next
# Créé dans npu-rtx/ (jamais dans profiler_v3). Sources en lecture seule.
# Principe : calculer AVANT exécution ce que profiler-v3 validera (Phase B micro, C runtime).

## Fichiers (6 — tous exécutables)
| Fichier | Rôle | Statut |
|---|---|---|
| `model_parser.py` | config.json → dims canoniques + shapes experts | ✅ |
| `quant_size_engine.py` | taille par format (GGML+NVFP4+INT8, bpw effectif avec overhead) | ✅ |
| `expert_mapper.py` | mapping expert 2560×640 → tuiles mmul 8x8x8 XDNA2 + L1 64KB | ✅ |
| `conversion_matrix.py` | coût chemins A/B/C/D NVFP4→INT8 (premier test D2) | ✅ |
| `bytes_per_token.py` | octets actifs/token par phase + effet cache (hit 90%) | ✅ |
| `memory_planner.py` | budgets VRAM 6 composants + élimination (lower bound) | ✅ |

## Résultats clés (Static — à corriger par profiler-v3)

### 1. MoE actif/token SANS cache (routed + shared BF16)
| Format | GiB/token |
|---|---|
| Q2 | 1.162 |
| Q3 | 1.401 |
| **Q4/NVFP4** | **1.675** |
| Q6_K | 2.242 |
| INT8 | 2.637 |
| Q8_0 | 2.774 |
| BF16 | 4.834 |

### 2. PCIe/token AVEC cache (6 GiB Q4, 48 experts/couche, hit 90%)
```
miss PCIe = 0.124 GiB/token  vs  1.236 GiB sans cache
→ cache évite ~90% du trafic PCIe MoE (cohérent : preuve Haberstroh 0.31 GB vs 26 GB)
```

### 3. Élimination (borne pessimiste sans overlap — PCIe 20 GB/s placeholder)
| Format | 100% miss | max t/s | hit 90% | max t/s |
|---|---|---|---|---|
| Q2 | 62 ms | 16.0 | 3.9 ms | 258 |
| **Q4** | **90 ms** | **11.1** | **6.6 ms** | **151** |
| Q6_K | 120 ms | 8.3 | 9.7 ms | 103 |
| INT8 | 142 ms | 7.1 | 11.8 ms | 85 |
| BF16 | 260 ms | 3.9 | 23.6 ms | 42 |

**→ le cache est le multiplicateur dominant** : Q4 sans cache = 11 t/s max, avec hit 90% = 151.
Éliminer un candidat si max_tok/s < objectif OU si le budget cache dépasse VRAM_dispo.

### 4. Mapping XDNA2 (expert_mapper)
- gate/up : 25600 tuiles 8x8x8 (M320 K80) ; down : 25600 (M80 K320)
- **expert entier 1.64 MB > 64 KB L1 → sous-tuiles L1 requises** (le point de découpe D2)
- 640/2560 multiples de 8 → pas de pénalité d'alignement mmul

### 5. Conversion NVFP4→INT8 (conversion_matrix)
```
A: NVFP4→RTX  = 138 µs   (pas de conversion)
B: NVFP4→INT8→XDNA = 278 µs  ← la conversion DOUBLE le coût à froid
C: BF16→INT8→XDNA  = 984 µs
D: BF16→BFP16→XDNA = 985 µs
```
→ Premier test D2 : mesurer la conversion réelle + GEMM INT8, voir si le gain couvre
la conversion quand le cache précharge + overlap masque.

### 6. Budget VRAM composants (ctx 8K, placeholders)
```
VRAM 6.5 GiB dispo (8 - 1.5 WDDM)
GDN state ~0.05 GiB · QSA KV ~0.09 GiB · workspace 0.5 · PLE → host/SSD
dense + expert_cache à mapper (TODO)
```

## Règle
Chiffres = estimations statiques, borne pessimiste sans overlap. profiler-v3 (Phase B micro,
C runtime) remplace les placeholders (PCIe, DDR, conversion, GEMM réel) et corrige le modèle.