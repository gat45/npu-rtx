# ANGLES MORTS & SPÉCIFICITÉS MATÉRIELLES — RTX 5070 (SM120) + XDNA2 + Qwen3.8-Flash-Next
# Établi 2026-09-20 · Dossier npu-rtx/ · Croisement internet + corpus local
# ⚠️ Découvertes qui INVALIDENT des hypothèses du SPEC_SAQE.md / PLANS

---

## 1. DÉCOUVERTE MAJEURE — SM120 ≠ SM100 (la RTX 5070 n'est PAS un "mini B200")

**Source** : lna-lab/blackwell-geforce-nvfp4-gemm/docs/sm120-architecture.md (testé sur 7× RTX
PRO 6000, CUDA 13.0, CUTLASS 4.0) + chsasank/blackwell.

| Feature | SM100 (datacenter) | **SM120 (GeForce RTX 50)** | Impact |
|---|---|---|---|
| MMA instruction | tcgen05.mma (UMMA) | **mma.sync.aligned.kind::f8f6f4** | registre→registre, pas TMEM |
| Tensor Memory (TMEM) | 256 KB/SM | **AUCUN** | opérandes en registres = **register pressure = le goulot** |
| TMA multicast | Oui | **Désactivé** (cluster 1×1×1) | pas de multicast |
| Pipeline | TMEM + tcgen05 | **PipelineTmaAsync (comme SM90)** + ldmatrix→registres | étapes 2-4, budget SMEM 99 KB |
| Block scaling | via descripteurs TMEM | **natif registres** (mxf8f6f4.block_scale, scale UE8M0) | format SM100 mais livré par registres |

**Conséquences concrètes** :
1. **`sm_100a` code TRAPS sur RTX 50** — les kernels SM100 ne tournent pas sur SM120.
2. **Pas de TMEM** → l'opérande doit tenir en registres → les tiles NVFP4 sont limités par la
   pression registres. Tile sûr (mesuré) : ~(BLOCK_M×BLOCK_K×0.5)+(BLOCK_N×BLOCK_K×0.5) par
   opérande FP4 + buffers scale.
3. **ldmatrix explicite SMEM→registres avant CHAQUE MMA** (étape que SM100 n'a pas).
4. **FP8 scalaire = ÉMULÉ sur SM120** : `__nv_cvt_float_to_fp8()` produit un mauvais exposant
   (off by +6) → utiliser une conversion software `float_to_fp8_e4m3_sw()`. Le FP8 natif n'est
   que tensor-core MMA.

URLs :
- https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm/blob/main/docs/sm120-architecture.md
- https://github.com/chsasank/blackwell (SM100 vs SM120, "code compiled for sm_100a traps")

---

## 2. ANGLE MORT — Aucun stack prêt-à-l'emploi ne fait NVFP4 complet sur SM120

**Source** : chsasank/blackwell (table "SM120 NVFP4 Status").

| Plateforme | Statut NVFP4 sur SM120 | Limitation |
|---|---|---|
| **vLLM** | Partiel | **SM120 FP4 kernel detection fails (#31085) → fallback Marlin = -40-50% perf** |
| **TensorRT-LLM** | Partiel | **NVFP4 KV cache non shipped (#10241)** |
| **SGLang** | Bloqué | attention backends fail (triton SMEM overflow) |
| **CUTLASS custom** | Requis | **seul chemin NVFP4 weights + NVFP4 KV cache sur SM120/121** |

**Implication pour le D2 Planner** : si on veut du FP4/NVFP4 réel sur la 5070, il faut du
**CUTLASS custom** (ou FlashInfer SM120 cute backend). Le fallback Marlin = grosse perte. Le
SAQE doit donc savoir QUELLE lib fournit quel format par archi.

**TensorRT-LLM Hardware Support Matrix (sm120)** :
| Recipe | sm120 |
|---|---|
| NVFP4 | ✅ |
| MXFP4 | ✅ |
| FP8 (per tensor) | ✅ |
| **FP8 block scaling** | ❌ (dot) |
| **FP8 rowwise** | ❌ |
| FP8 KV cache | ✅ |
| **W4A8 / W4A16 AWQ/GPTQ** | ❌ (non listés sur sm120 ; seulement sm100/Hopper/Ada) |

⚠️ Donc sur la 5070 : NVFP4/MXFP4/FP8-pertensor/FP8-KV OK ; **FP8 block-scaling, W4A8, W4A16
non supportés**. Le "Q6/FP8 hot" du plan doit être re-qualifié.

URLs :
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md
- https://nvidia.github.io/TensorRT-LLM/supported-hardware.html
- https://github.com/chsasank/blackwell

---

## 3. ANGLE MORT — Qwen3.8-Flash-Next : les experts routed SONT DÉJÀ W4A4 NVFP4 (officiel NVIDIA)

**Sources** : nvidia/Qwen3.8-Flash-Next-NVFP4 (HF) + QwenLM/Qwen3.8-Flash-Next (README + ModelScope).

### Architecture officielle (dimensions RÉELLES)
```
125B main + 51B n-gram embeddings + 4B MTP · 6B activés/token
hidden = 2560 · vocab = 248320 (padded)
48 layers · layout = 12 × (3 × (GDN → MoE) → 1 × (QSA → MoE))
GDN : 48 heads V / 16 heads QK · head_dim 128
QSA : 24 heads Q / 2 heads KV · head_dim 256 · indexer MQA 4Q+1K · budget 512 blocs/2048 tokens
MoE : 512 experts · 10 routed + 1 shared · expert intermediate = 640
Gated Residual : 4 branches · bottleneck rank 320
ctx 262 144 natif → 1M
```

### Tailles expert (calculées depuis hidden=2560, interm=640)
```
gate : 640×2560 = 1.64M · up : 640×2560 = 1.64M · down : 2560×640 = 1.64M
par expert (3 tensors) ≈ 4.9M params
en Q4 (4 bpw)  ≈ 2.45 MB/expert
en NVFP4      ≈ 2.45 MB/expert (W4A4)
512 experts × 48 couches = 24 576 experts → ~60 GB en Q4 (poids experts dominants)
```

### Le checkpoint NVFP4 NVIDIA (LA référence pour notre cible)
- **Routed MoE experts = W4A4 NVFP4** (scales MSE-calibrées)
- Attention, shared experts, autres layers main = **BF16**
- MTP routed experts = **FP8 block-scaled 128×128**
- PLE n-gram = **per-tensor FP8**
- Résultat : **2.7× plus petit** que BF16 (-63%)

⚠️ **Implication massive** : le SAQE ne choisit pas "Q4 vs Q6" sur un GGUF Q4_K uniforme.
NVIDIA fournit un checkpoint où la précision est DÉJÀ mixée par composant (W4A4 experts,
BF16 attention, FP8 MTP, FP8 PLE). Notre moteur doit pouvoir ingérer/répliquer cette mixité,
et le "hot expert → Q6" du plan entre en conflit avec "experts = W4A4 natif" (passer un
expert W4A4 en BF16/FP8 = conversion coûteuse, pas une simple re-quant).

URLs :
- https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4
- https://www.modelscope.ai/models/Qwen/Qwen3.8-Flash-Next
- https://github.com/QwenLM/Qwen3.8-Flash-Next

---

## 4. XDNA2 — rappel des spécificités mesurées (corpus local) qui sont des angles morts du SPEC

| Spec | Valeur mesurée | Angle mort |
|---|---|---|
| Colonnes exposées | 4 (sur 8 ; 4 réservées firmware) | le planner suppose NPU_COLS_MAX=8 → **utiliser 4** |
| Compute réel | 51.3 TOPS INT8 peak / 38 eff / 9-12 GEMM | jamais le "50 TOPS" marketing |
| SRAM | 2 MB L1 + 4 MB L2 = 6 MB | cache experts SRAM = minuscule vs experts |
| BW DDR5 effective | 21.93 GB/s (24.5% de 89.6) | goulot NPU réel, à injecter dans BW model |
| Compute INT4 | **AUCUN** (déquant→INT8/BF16 avant kernel) | "Q4 sur NPU" = en réalité INT8/BF16 → double coût |
| MCDM dispatch | 3064 IOCTL/token, 81.32 ms run::wait | le NPU ne sera jamais rapide tant que ça reste |
| KDMA | "not supported on windows" | le DMA direct XRT ne marche pas → via vitis-ai-runtime2 |
| Persistent hw_context | -65% TTFT (2362 ms init) | PRÉREQUIS avant tout usage NPU en complément |
| Tiles | 8×4=32 AIE2P, aie2p 5 accum | — |

Sources : reference/SOSC_v4/CARTE_FONCTIONNELLE_XDNA2_FLM.md, reference/mesures_flm/

---

## 5. ANGLE MORT — La machine cible : 5070 8 GB LAPTOP (pas 12 GB desktop)

- RTX 5070 desktop = **12 GB GDDR7** (192-bit). RTX 5070 Laptop = **8 GB** (config possible).
- Notre machine : 8 GB (confirmé par RAPPORT_TIERS_HARDWARE et corpus).
- 672 GB/s GDDR7 = la BW de référence du cost model.
- **PCIe 4.0 ×16** sur le HX365 (les 16 lanes partagées avec NVMe selon topologie → à mesurer
  le lien réel RTX = x16 ou x8). La topologie exacte = angle mort (jamais mesurée sur cible).
- **Pas de GDS Windows** → chemin SSD→pinned RAM→H2D obligatoire pour le MVP.

---

## 6. ANGLES MORTS DU SPEC_SAQE.md (révélés par ce croisement)

1. **Le modèle "Q4→Q6 par expert" suppose un GGUF Q4 uniforme** — or le checkpoint NVFP4
   officiel est déjà mixé W4A4/BF16/FP8. Le SAQE doit traiter le **graphe de conversion entre
   familles** (NVFP4↔Q4_K↔FP8) avec leurs vrais coûts, pas seulement re-quant une même famille.
2. **Register pressure sur SM120** = le vrai goulot du tile NVFP4, PAS le trafic mémoire.
   Le planner doit connaître `max tile(précision, SM)` par archi (pas de TMEM).
3. **Le fallback Marlin sur vLLM (SM120)** = si on passe par vLLM, le FP4 n'est pas ce qu'on
   croit (-40-50%). → la lib/l'engine fait partie de la décision.
4. **FP8 block-scaling / W4A8 / W4A16 NON supportés sur sm120** (TensorRT-LLM matrix) → le
   plan "hot expert → FP8" doit être re-qualifié : FP8-per-tensor seulement, ou CUTLASS custom.
5. **Le NPU calcule INT8/BF16, pas Q4** → un expert "Q4 sur NPU" = déquant+recompute INT8.
   Le coût NPU du plan doit utiliser le chemin INT8.
6. **PLE = per-tensor FP8 officiel** (pas seulement "Q4 en RAM") → sa résidence dépend du
   support FP8 du device qui le lit.
7. **Pas de topologie PCIe mesurée** (x16 vs x8, partage NVMe) → la matrice 3×3 des transferts
   (point 26 RECHERCHE) devient obligatoire avant tout chiffre de coût PCIe.

---

## 7. MATRICE DE CAPABILITÉS (mise à jour — par device × famille)

### RTX 5070 (SM120, via stacks publics)
| Format | llama.cpp CUDA | FlashInfer | TensorRT-LLM | CUTLASS custom |
|---|---|---|---|---|
| Q2/Q3/Q4_K/Q5/Q6/Q8 | ✅ (MMQ) | — | ❌ | — |
| FP8 per-tensor | ❌ | ✅ | ✅ | ✅ |
| FP8 block-scale | ❌ | ✅ (SM89+) | ❌ (sm120) | ✅ |
| FP8 rowwise | ❌ | — | ❌ (sm120) | ✅ |
| FP4/NVFP4 | ❌ | ✅ (b12x SM120) | ✅ (NVFP4) | ✅ (obligatoire pour KV NVFP4) |
| MXFP4 | ❌ | ✅ | ✅ | ✅ |
| KV cache NVFP4 | ❌ | ❌ | ❌ (#10241) | ✅ (seul chemin) |

### XDNA2 (via XRT/FLM/IRON)
| Format | État |
|---|---|
| INT8 / BF16 | ✅ (compute natif) |
| Q4/Q4NX | ❌ compute (stockage seul, déquant→INT8/BF16) |
| FP8 | ❌ (pas documenté) |
| FP4/NVFP4 | ❌ |

---

## 8. IMPLICATION POUR LE D2 SAQE (conclusions actionnables)

1. **Le choix de précision doit être couplé au choix d'ENGINE** : Q4_K→llama.cpp MMQ,
   NVFP4→FlashInfer/CUTLASS, FP8→FlashInfer/TRT-LLM. Une matrice (format × engine × archi)
   remplace "Q4/Q5/Q6".
2. **"Hot expert → haute précision" doit rester DANS la famille du device** :
   - RTX hot → NVFP4 natif ou FP8-pertensor (pas Q6_K si on est en NVFP4 natif).
   - NPU hot → INT8/BF16 (pas Q4).
3. **Le cache expert garde le format du tenseur** (Q4_K sur le chemin GGML, NVFP4 sur le
   chemin CUDA natif) — jamais de conversion au bord du cache (leçon layout_conversion).
4. **La taille max du tile NVFP4 = contrainte register-pressure SM120**, à mesurer
   (pas de TMEM). C'est une entrée du cost model GEMM.
5. **Le PLE FP8 per-tensor** : résident en RAM/aux-GPU si device FP8, sinon BF16 re-quant.
6. **La topologie PCIe réelle (x16/x8) + la matrice de transferts 3×3** = premier benchmark
   obligatoire sur la machine cible (rien de crédible sans ça).
7. **Le NPU reste un contrôleur/prédicteur** (router GDN/QSA = 93-97% de prédiction possible),
   pas un moteur MoE : son chemin INT8/BF16 ne bat pas la RTX (ratio 7.4×).

## 9. URLS DE PREUVE (ce document)
- SM120 arch : https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm/blob/main/docs/sm120-architecture.md
- SM120 stack status : https://github.com/chsasank/blackwell
- TensorRT-LLM quant matrix : https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md
- TensorRT-LLM support : https://nvidia.github.io/TensorRT-LLM/supported-hardware.html
- Qwen3.8 NVFP4 NVIDIA : https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4
- Qwen3.8 model (ModelScope) : https://www.modelscope.ai/models/Qwen/Qwen3.8-Flash-Next
- Qwen3.8 repo : https://github.com/QwenLM/Qwen3.8-Flash-Next
- NVIDIA forum (tailles n-gram) : https://forums.developer.nvidia.com/t/qwen3-8-flash-next/381228
- FlashInfer : https://github.com/flashinfer-ai/flashinfer
- CUTLASS NVFP4 SM120 : https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu