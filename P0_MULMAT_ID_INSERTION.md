# P0 — Dissection MUL_MAT_ID + point d'insertion D2 Planner (llama.cpp CUDA)
# Source disséquée : E:\oneplus\ab-wt\ggml\src\ggml-cuda\ (branche self-build-jz, PR #26501)
# Référence modèle : RFC #28248 (moe-expert-cache), PR #26563, vLLM #37190 (LFRU)
# Établi 2026-09-20

## CHEMIN COMPLET (expert_id → offset → bloc quantifié → déquant → MMA)

```
Router → topk_ids [n_tokens × n_expert_used]
   │
   ▼
ggml_cuda_mul_mat_id(ctx, dst)            ggml-cuda.cu:1903
   │
   ├─ [FAST PATH] si ne2 ≤ MMVQ_MAX_BATCH
   │    ├─ quantized → ggml_cuda_mul_mat_vec_q (mmvq)   ggml-cuda.cu:1922
   │    └─ else     → ggml_cuda_mul_mat_vec_f           ggml-cuda.cu:1927
   │
   ├─ [MMQ PATH] si ggml_cuda_should_use_mmq → ggml_cuda_mul_mat_q   ggml-cuda.cu:1934
   │    (kernels quantifiés mmq.cuh, déquant par bloc dans le kernel)
   │
   └─ [FALLBACK — nécessite sync, pas CUDA-graph-compatible]  ggml-cuda.cu:1944+
        1. cudaMemcpyAsync(ids → host) + cudaStreamSynchronize   L1972-1973  ← ⚠️ COÛT #1
        2. boucle CPU : ids_to_sorted / ids_from_sorted / tokens_per_expert  L1975-1988
           (pour chaque expert i02, chaque token, chaque iex → expert_to_use)
        3. cudaMemcpyAsync(ids triés → device) + synchronize       L1993-1994  ← ⚠️ COÛT #2
        4. get_rows_cuda(src1 → src1_sorted)                        L1999       ← réordonne activations
        5. POUR CHAQUE expert i02 :                                  L2007-2052
             src0_slice.data = src0->data + i02*nb02               L2017       ← ★ OFFSET EXPERT
             src0_slice.ne[2] = 1                                   L2013
             ggml_cuda_mul_mat(ctx, &src0_slice, ...)               L2047       ← kernel par expert
        6. get_rows_cuda(dst_sorted → dst)                          L2054       ← réordonne sorties
```

## POINTS D'INSERTION DU CACHE (le D2 Planner se branche ICI)

### ★ Point d'insertion PRINCIPAL — L2017 (`src0_slice.data = src0->data + i02*nb02`)
C'est l'endroit où l'expert i02 est adressé dans le tenseur de poids. **Un cache de slots
VRAM s'insère exactement ici** :
```
au lieu de : src0_slice.data = src0->data + i02*nb02        (lecture directe du tenseur complet)
faire :      src0_slice.data = slot_ptr(expert=i02)          (slot persistant dans le pool)
             si miss : memcpyAsync(CPU expert → slot) sur stream dédié + update mapping
```
- `nb02` = stride entre experts dans src0 (le tensor GGUF) → `expert_offset = i02 * nb02`
  = exactement "l'expert 17 couche 23 = ces plages d'octets" demandé.
- Le cache doit garder le **même layout quantifié** (Q4_K...) pour que les kernels MMQ
  continuent de fonctionner sans repack (leçon layout_conversion, RFC #28248 garde les slabs).

### Point d'insertion secondaire — L1972-1973 / L1993-1994 (les 2 synchronize)
Le fallback copie les ids CPU puis resynchronise. C'est le **coût le plus évitable** :
- `mm_ids_helper` (mmid.cu) fait déjà ce tri EN GPU sans sync (ids_src1/ids_dst/expert_bounds)
  → le chemin moderne (MMVQ/MMQ) ne fait pas ce fallback. Le D2 Planner doit **forcer le
  chemin GPU-only** (éviter le fallback sync) en s'assurant que les conditions des fast paths
  sont remplies (ne2 ≤ batch max, mmq applicable).

### Point d'insertion pour la prédiction — L1977-1987
La boucle CPU lit `expert_to_use` pour chaque token : c'est le moment où l'on connaît les
experts du prochain token. Un **predictor** s'y branche pour émettre `prefetch(expert_N+1)`
sur un stream CUDA dédié (pattern FATE/vLLM) pendant que le GEMM courant tourne.

## STRUCTURE DU KERNEL QUANTIFIÉ (mmq) — déquant fusionnée
- `mmq.cuh` : template `mul_mat_q<type, mmq_x>` — charge les blocs Q4_K/Q5_K/Q6_K depuis
  src0, les déquantifie dans les registres/shared memory, puis MMA.
- `mmq-load-tiles.cuh` / `mmq-vec-dot.cuh` : chargement tile + produit scalaire.
- `mmq-config-*.cuh` : configs par architecture (blackwell.cuh présent → la 5070 sm_120
  a son fichier de config).
- `topk-moe.cu` : le top-k MoE séparé (si utilisé).

## CE QUE ÇA CONFIRME (lié au plan PLAN_ADAPTIVE_RESIDENCY.md)

1. **Le layout quantifié est le contrat** : les kernels MMQ consomment directement Q4_K/Q5_K/
   Q6_K depuis le tenseur → un cache de slots qui garde le format GGUF = zéro repack, zéro
   conversion. (point 10 du plan)
2. **Le fallback sync est le pire chemin** (2× synchronize + boucle CPU) → le D2 Planner doit
   garantir le chemin MMVQ/MMQ. (point 14 plan)
3. **`expert_offset = i02 * nb02`** = la table "Layer Tensor Expert Shape QType Offset Bytes"
   de P0 est directement calculable depuis ggml_tensor (src0->nb[2] = nb02).
4. **Le cache s'insère SANS réécrire ggml** : un hook sur L2017 + un pool de slots (comme
   vLLM CachedWeightProvider) suffit. C'est exactement ce que font RFC #28248 (slabs) et
   PR #26563 (heatmap).

## FICHIERS CIBLES (pour l'implémentation)
| Fichier | Rôle | Ligne clé |
|---|---|---|
| ggml-cuda/ggml-cuda.cu | dispatch MUL_MAT_ID | 1903, 2017, 2047 |
| ggml-cuda/mmid.cu | tri experts GPU (sans sync) | 143-169 (launch) |
| ggml-cuda/mmq.cuh | kernels quantifiés (déquant fusionnée) | — |
| ggml-cuda/topk-moe.cu | top-k MoE | — |
| ggml-backend-sched (ggml/src) | input_cpy / graph splits | (partie scheduler, hors CUDA) |

## PROCHAINES ÉTAPES CONCRÈTES (P1)
1. Compiler/valider la dissection sur le build ab-wt (test-backend-ops MUL_MAT_ID).
2. Prototyper le hook slot cache sur L2017 (pool de slots + mapping expert→slot + miss H2D
   sur stream dédié) — réutiliser la logique de RFC #28248/PR #26563.
3. Mesurer sur la machine dev (GTX 1080) : hit rate, bytes H2D/token, tok/s → reproduire la
   courbe 4/8/16/32/64 slots (référence : 11.55→21.21 t/s @ 4090).
4. Sur machine cible (5070 + XDNA2) : idem + le NPU comme predictor (architecture B).