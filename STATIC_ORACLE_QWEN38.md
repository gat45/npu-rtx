# STATIC ORACLE — Analyse statique des poids Qwen3.8-Flash-Next (Phase A)
# Établi 2026-09-20 · Dossier npu-rtx/ · Sources : config.json + convert.log GGUF + checkpoints
# Principe : calculer AVANT exécution ce que profiler-v3 validera ensuite (Phase B micro, Phase C runtime).
# Static Oracle + Dynamic Oracle = D2 Oracle.

---

## 0. PRINCIPE EN 3 PHASES

```
Phase A STATIC  : config.json + poids + shapes + quant + topologie hardware
                  → memory map, expert map, bytes/token, cache capacity, lower bounds
Phase B MICRO   : mesurer SSD/DDR/PCIe/CUDA/XDNA/quant-dequant (profiler-v3)
Phase C RUNTIME : routing réel, cache réel, prefetch réel, kernel réel, critical path
D2 = Static Oracle + Dynamic Oracle (correction)
```

---

## 1. SOURCES VÉRIFIÉES (8 essentielles)

| Source | URL | Ce qu'elle donne |
|---|---|---|
| config.json officiel | https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json | 2560/640/512/top-10/48 layers/2KV/MTP/ngram |
| config.json unsloth | https://huggingface.co/unsloth/Qwen3.8-Flash-Next/blob/main/config.json | même config |
| convert.log GGUF officiel | https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF/blob/main/convert.log | shapes experts réelles + BF16→Q8_0 (1600→850 MiB) |
| dépôt Qwen | https://github.com/QwenLM/Qwen3.8-Flash-Next | architecture |
| paper | https://arxiv.org/abs/2608.30320 | architecture |
| Agention AP-GGUF | https://huggingface.co/agentionai/Qwen3.8-Flash-Next-AP-GGUF + README | mixed precision par groupe, KL/top-1 par bpw |
| Guile GGUF | https://huggingface.co/Guile/Qwen3.8-Flash-Next-GGUF | Q2_K..IQ3_XS tailles fichiers |
| Haberstroh 24 GB | https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang + sglang#37792 | 2.57 bpw, 0.31 GB PCIe/token, ~54-58 t/s |
| Starkweather NVFP4 | https://github.com/starkweatherdigital/qwen3.8-flash-next-nvfp4-recipe | experts NVFP4 67.95 GB, PLE NVFP4 28.8 GB |
| NVIDIA NVFP4 | https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4 | checkpoint officiel |
| tonyd2wild Spark | https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark | NVFP4+PLE disk+FP8 KV+MTP3, par workload |
| lukaLLM VRAM bench | https://github.com/lukaLLM/Qwen3.8-Flash-Next-VRAM-Benchmark | PLE GPU 1.95 vs CPU 108.5 t/s (55.6×) |
| llama.cpp cache | #28248 #20757 #27149 #27864 | cache persistant +84% |
| vumpt GGUF | https://huggingface.co/vumpt/Qwen3.8-Flash-Next-GGUF | Q4_K_M layout hétérogène |
| unsloth GGUF | https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF | UD-Q4_K_XL / UD-IQ4_XS |

---

## 2. DIMENSIONS RÉELLES (vérifiées convert.log / config.json)

```
hidden = 2560 · moe_intermediate = 640 · num_experts = 512 · top_k = 10 (+1 shared)
48 layers · 2 KV heads · head_dim 256 · MTP 1 layer · ngram 320 001 536 × 160
shapes experts : gate [2560,640,512] · up [2560,640,512] · down [640,2560,512]
```

## 3. TAILLE D'UN EXPERT / COUCHE (exacte)

```
gate = 2560×640 = 1 638 400 params
up   = 2560×640 = 1 638 400
down = 640×2560 = 1 638 400
total = 4 915 200 params/expert/layer
```

| Format | bpw effectif | Taille/expert |
|---|---|---|
| BF16 | 16 | 9.375 MiB |
| Q8_0 | ~8.5 | 4.98 MiB |
| Q6_K | ~6.56 | 3.85 MiB |
| Q5 | ~5.5 | 3.22 MiB |
| **Q4/NVFP4** | **~4.5 (avec overhead)** | **2.64 MiB** |
| Q3 | ~3.5 | 2.05 MiB |
| Q2 | ~2.63 | 1.54 MiB |

Vérif convert.log : expert BF16 1600 MiB → Q8_0 850 MiB = 53.125% (cohérent avec ~8.5 bpw/16).
Vérif NVFP4 : 67.95 GB / 120.8B experts ≈ 4.5 bpw avec scales.

## 4. TOUS LES EXPERTS (120.8B params)

```
48 × 512 × 3 × 2560 × 640 = 120 795 955 200
```
| Repr | Experts seuls |
|---|---|
| BF16 | 241.59 GB (225 GiB) |
| Q8_0 | ~128.35 GB |
| Q6_K | ~99.09 GB |
| Q5 | ~83.05 GB |
| **Q4/NVFP4** | **~67.95 GB** (confirmé par checkpoint Starkweather) |
| Q3 | ~52.85 GB |
| Q2 | ~39.64 GB |

## 5. BYTES ACTIFS / TOKEN (avant cache — le plancher de trafic)

```
10 routed × 48 layers × 4 915 200 = 2 359 296 000 params
shared × 48 × 4 915 200          =   235 929 600 params
total actif = 2.5952256B params
```
| Format | 10 routed | + shared | **Total actif/token** |
|---|---|---|---|
| BF16 | 4.395 GiB | 0.439 | **4.834 GiB** |
| Q2 | 0.721 | 0.439 | 1.160 GiB |
| Q3 | 0.961 | 0.439 | 1.401 GiB |
| **Q4** | **1.236** | **0.439** | **1.675 GiB** |
| Q5 | 1.511 | 0.439 | 1.950 GiB |
| Q6_K | 1.802 | 0.439 | 2.242 GiB |
| Q8_0 | 2.335 | 0.439 | 2.774 GiB |

**Note** : weight_bytes ≠ PCIe_bytes si cache ; ≠ DRAM_bytes si réutilisé. profiler-v3 mesure le réel.

## 6. CACHE EXPERT — SLOTS THÉORIQUES (AVANT workspace/KV/GDN/PLE/WDDM)

| Budget | Q2 | Q3 | Q4/NVFP4 | Q5 | Q6 | Q8 |
|---|---:|---:|---:|---:|---:|---:|
| 4 GiB | 2663 | 1997 | **1553** | 1271 | 1065 | 822 |
| 6 GiB | 3995 | 2996 | **2330** | 1907 | 1598 | 1234 |
| 8 GiB | 5326 | 3995 | **3107** | 2542 | 2130 | 1645 |

Experts/couche (8 GiB, Q4) : 3107/48 ≈ **64.7** ; Q6 : 44.4 → **Q4 = 1.46× plus de résidents**
→ second ordre : Q4 → plus de hit → moins de H2D → moins de PCIe (même si Q6 GEMM plus rapide).
**C'est exactement l'effet que D2 doit modéliser.**

## 7. WORKING SET DU ROUTING (résultat public Haberstroh)

```
32 experts → 37% du trafic · 64 → 53% · 128 → 73% · 171 → 82% · 256 → 93%
```
→ Avec 6 GiB Q4 ≈ 48 experts/couche, on couvre une bonne partie de la masse.
**Grille de recherche intelligente** (au lieu de tester 4/8/16/32/64/128/256 arbitraire) :
48 → 64 → 96 experts/couche, guidée par routing mass.

## 8. COÛT D'UN MISS (exemple illustratif, PAS mesure)

```
T_H2D ≈ expert_bytes / P(PCIe)     — à remplacer par profiler-v3 P(size, qdepth, contention)
Q4 ≈ 2.64 MiB / 20 GB/s ≈ 0.13 ms · Q6 ≈ 0.19 ms · Q8 ≈ 0.25 ms
```

## 9. PLE (calculable depuis shapes)

```
320 001 536 × 160 = 51.2B params
PLE_BF16 ≈ 95.4 GiB · PLE_FP8 ≈ 47.7 GiB · PLE_NVFP4 ≈ 23.8 GiB (théo)
PLE_NVFP4 réel rapporté ≈ 28.8 GB (Starkweather)  ← raw ≠ checkpoint réel
```
→ profiler-v3 doit stocker les DEUX (math vs réel).

## 10. LOWER BOUND MÉMOIRE (élimination avant benchmark)

```
T_memory_lower_bound = bytes/token / BW_effective
ex. PCIe 14 GB/s + candidat 2.0 GB/token → ≥ 143 ms/token → < 50 t/s IMPOSSIBLE → éliminer.
```
→ L'oracle doit faire : poids → bytes → lower bound → élimination.

## 11. DÉBIT MINIMUM PAR CONFIG (à comparer à BW réelles)

| Config | bytes actifs/token | à 20 GB/s PCIe | à 14 GB/s |
|---|---|---|---|
| Q2 | 1.16 GiB | ~62 ms | ~89 ms |
| Q4 | 1.675 GiB | ~90 ms | ~128 ms |
| Q6 | 2.242 GiB | ~120 ms | ~172 ms |
| Q8 | 2.774 GiB | ~149 ms | ~213 ms |

## 12. GRILLE DE CANDIDATS D2 (à estimer en coût mémoire avant exécution)

```
C0 Q4 uniform + cache Q4 + PLE SSD/RAM + GDN BF16 + KV BF16
C1 Q3 uniform + cache Q3 + PLE SSD/RAM + GDN BF16 + KV BF16
C2 Q4 hot / Q3 cold + PLE SSD/RAM + GDN BF16
C3 Q6 hot / Q4 warm / Q3 cold + PLE SSD/RAM
C4 NVFP4 hot + Q3 cold + PLE FP8
C5 NVFP4 experts + PLE NVFP4 + KV FP8 + GDN BF16
C6 mixed tensor precision + persistent cache + prefetch
C7 C6 + XDNA2 predictor
```

## 13. STATIC ANALYZER (à implémenter dans npu-rtx/static/)

```
static/
├── model_parser.py       # config.json → dims
├── tensor_mapper.py      # shapes/offsets
├── expert_analyzer.py    # tailles expert par format
├── quant_size_engine.py  # bytes par repr (GGML + NVFP4)
├── bytes_per_token.py    # routed + shared + PLE
└── memory_planner.py     # cache capacity + lower bounds + candidats éliminés
```
Sortie : `runs/<ts>/static_oracle.json` (section 19 : MODEL / ONE EXPERT / ACTIVE MOE / PLE /
candidats éliminés / Pareto statique).

## 14. CE QUE LES POIDS DONNENT (Static) vs profiler-v3 (Dynamic)

| Static (poids/config) | Dynamic (profiler-v3) |
|---|---|
| ✅ taille/params/shapes | ✅ vrai bandwidth/latency |
| ✅ bytes théoriques | ✅ vrai kernel cost |
| ✅ bytes/token | ✅ vrai dequant cost |
| ✅ taille expert | ✅ vraie contention |
| ✅ taille PLE | ✅ vrais cache hits |
| ✅ cache capacity | ✅ vrai routing |
| ✅ lower bounds | ✅ vrai overlap/stalls |
| ✅ budget par repr | ✅ élimination mesurée |
| ✅ espace de recherche | ✅ modèle corrigé |

## 15. CONCLUSION

**Avant de lancer le profiling complet : fabriquer l'analyseur statique** qui prend
config.json + safetensors/GGUF + metadata quant et produit automatiquement : taille par
tensor/expert/layer, bytes actifs/token, bytes par précision, working set théorique, experts
tenables par X GiB, estimation SSD/RAM/PCIe/VRAM, PLE/GDN/QSA séparés, lower-bound latency,
candidats éliminables, candidats Pareto.

Puis profiler-v3 corrige ces estimations avec les mesures réelles (Phase B/C).

---

*Sources §1 + calculs vérifiables depuis convert.log (1600→850 MiB = 53.125%),
config.json (2560/640/512/10/48), checkpoint NVFP4 (67.95 GB experts, 28.8 GB PLE).*