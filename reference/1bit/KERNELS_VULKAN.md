# Référence — Kernels Vulkan 1bit (kernels/vulkan) — référence tier GPU du D2 Planner
# Source : https://github.com/1bit-MONSTER/1bit-MONSTER-scaffold-backup/tree/main/kernels/vulkan
# Fetched 2026-09-20 — fichiers .comp téléchargés dans reference/1bit/kernels-vulkan/

## Fichiers
| Kernel | Format | Rôle |
|--------|--------|------|
| `dmmv_tq2_bonsai.comp` | TQ2 (ternaire 2-bit, block 34B/128 poids) | GEMV Bonsai (Zaya) portable Vulkan |
| `dmmv_q1_bonsai.comp` | Q1 (binaire 1-bit) | GEMV binaire |
| `matmul_fp32.comp` | FP32 | matmul Zaya projections (tile 64×16) |
| `zaya_cca_attn.comp` + `.spv` | attention CCA Zaya | CCA attention (precompilé SPIR-V) |

## Faits techniques (utiles pour le tier RTX du planner)
1. **Portable Vulkan 1.2+** : AMD, NVIDIA, Intel, Apple — topologie subgroup-agnostic
   (fold cross-subgroup via shared memory), pas de wave32 requis. → nos experts MoE sur
   RTX 5070 peuvent utiliser EXACTEMENT ces kernels (pas de fork CUDA nécessaire).
2. **Format block TQ2** : `[fp16 d][qs 32B, codes 2-bit LSB-first, 4/byte]`, 128 poids/block,
   `value(code)=code-1 → {-1,0,+1,+2}`. `code==3 → 0` (réservé bonsai.h).
3. **`acc_mode` push-constant** : `y = out` (mode 0) ou `y += out` (mode 1) → **accumulation
   en-place = primitive de merge d'experts** (chaque expert écrit son GEMV, un kernel combine
   par accumulation sur le même buffer Y).
4. **Dépack manuel fp16** via `unpackHalf2x16` (pas de GL_EXT_shader_16bit_storage) → plus
   compatible mais léger coût de dépack. Leçons : portabilité > micro-opt quand target variée.
5. **matmul tile 64×16/workgroup, row-per-thread** : le pattern standard. `out[M]=in[K]@wt[M×K]^T`
   = GEMV decode (pas GEMM prefill) — exactement le cas des experts en decode.
6. **DMMV = dense-matrix × matrix-vector** : le format expert (weights denses par expert)
   → activations vecteur. K réduit = expert individuel.

## Implication D2 Planner
- **Le tier GPU (RTX) peut exécuter les experts MoE via Vulkan portable** avec les mêmes
  kernels que le NPU 1bit. Le planner ne dépend donc pas d'un backend GPU propriétaire.
- **`acc_mode` = la primitive de fusion des sorties experts** : chaque expert sur son tier
  (RTX ou XDNA2) écrit dans le même buffer Y via accumulation → le "weighted sum" de
  l'architecture (§6) est un simple `y +=` à la fin, pas une étape de collecte séparée.
- Le format Q4NX/TQ2 1bit stocké = bytes/token réduits (TQ2=4× vs INT8, cf performance.md)
  → quand le SSD→RAM→RTX charge un expert, moins de bytes à transférer = coût DMA réduit.
- Contraste : ces kernels sont écrits pour **Strix Halo iGPU** (wave32 absent) ; notre RTX 5070
  supporte wave32 → un variant wave32-pinned serait plus rapide (piste d'optimisation notée
  par 1bit comme "later optimization").