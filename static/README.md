# static/ — Static Oracle (Phase A) — Qwen3.8-Flash-Next
# Créé dans npu-rtx/ (jamais dans profiler_v3). Sources en lecture seule.
# Principe : calculer AVANT exécution ce que profiler-v3 validera (Phase B micro, C runtime).

## Fichiers
| Fichier | Rôle | Statut |
|---|---|---|
| `model_parser.py` | config.json → dims canoniques + shapes experts | ✅ |
| `quant_size_engine.py` | taille par format (GGML+NVFP4+INT8, bpw effectif avec overhead) | ✅ |
| `expert_mapper.py` | mapping expert 2560×640 → tuiles mmul 8x8x8 XDNA2 + L1 64KB | ✅ |
| `conversion_matrix.py` | coût chemins A/B/C/D NVFP4→INT8 (premier test D2) | ✅ |
| `bytes_per_token.py` | (à créer) bytes actifs/token par phase + cache | 🔴 |
| `memory_planner.py` | (à créer) budgets VRAM/RAM + candidats éliminés | 🔴 |

## Résultats clés (Static, à corriger par profiler-v3)
1. **Expert/layer = 4 915 200 params** = 2.64 MiB Q4/NVFP4, 3.85 Q6, 4.98 Q8, 9.38 BF16
2. **Mapping XDNA2** : gate/up = 25600 tuiles 8x8x8 (M320 K80), down = 25600 (M80 K320) ;
   un expert entier = 1.64 MB > 64 KB L1 → **sous-tuiles L1 requises** (le point de découpe D2)
3. **Conversion NVFP4→INT8** : A (RTX, pas de conv) = 138µs < B (INT8 XDNA) = 278µs **sans
   cache/overlap** → la conversion double le coût à froid ; avec cache + overlap à re-mesurer
4. 640 et 2560 multiples de 8 → pas de pénalité d'alignement mmul 8x8x8 (contrairement à 3584)

## Règle
Ces chiffres sont des **estimations statiques** (borne pessimiste sans overlap). Le premier
test D2 réel = mesurer la conversion NVFP4→INT8 (CPU/GPU/NPU) + GEMM INT8 NPU réel, puis
vérifier si le gain INT8 couvre le coût de conversion quand le cache précharge.