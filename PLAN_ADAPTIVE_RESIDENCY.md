# PLAN COMPLET — Adaptive Quantized Residency : Qwen3.8-Flash-Next sur RTX 5070 + NPU XDNA2
# Établi 2026-09-20 · Dossier npu-rtx/
# Convergence : llama.cpp cache + DynaExq (précision dynamique) + MoE-Infinity (résidence)
# + vLLM (prédiction) + KTransformers (hybrid CPU/GPU) + FlashInfer (fused quantized compute)

---

## 0. PRINCIPE CENTRAL (validé par la recherche)

> **Ne pas "quantifier à chaque token". Construire une représentation quantifiée qui change de
> précision pendant l'exécution, pilotée séparément de la résidence SSD→RAM→VRAM et de la
> déquantification.**

Deux caches distincts :
- **Cache de résidence** : Où est l'expert ? (SSD / RAM / VRAM)
- **Cache de représentation** : Sous quelle précision ? (Q4/Q5/Q6/Q8/BF16)

Exemple cible :
```
Expert 237 : SSD Q4 · RAM Q4 · VRAM Q6 (hot) · compute BF16 tile
Expert 418 : SSD Q4 · RAM Q4 · VRAM Q4 (cold) · compute BF16 tile
```

---

## 1. LISTE DES QUESTIONS (organisées par domaine) — avec réponses et preuves

### A. Quantification
| Question | Réponse / Preuve |
|---|---|
| Une requantification Q4→Q6/Q8 runtime est-elle plus rapide que charger une variante Q6 préexistante ? | **À mesurer (P0)**. DynaExq garde 2 versions (hi/lo) par expert et bascule async (arXiv:2511.15015). Les transitions sont découplées du compute. |
| À partir de quelle taille d'expert la quantification devient-elle prohibitive ? | DeepSeek-V2 : 160 experts × ~7M params → miss = 1-2 ms/expert (vLLM #37190). Le coût de quant doit être ≤ le gain PCIe. |
| Quel backend quantifie le plus vite : CPU, GPU, NPU ? | **À mesurer.** KTransformers fait de la quant online au chargement sur CPU (chemins spécifiques). Le NPU (51 TOPS INT8) pourrait requantifier moins cher que CPU (à tester, P5). |
| Peut-on créer plusieurs variantes au démarrage sans exploser la RAM ? | Oui : DynaExq ne matérialise que hi+lo, et l'éviction/transition est budget-feasible par construction (arXiv:2511.15015). |
| Peut-on dériver une variante basse précision par truncation depuis une haute ? | DynaExq : oui, la basse précision est une version dérivée ; scales partagées à étudier (Q4 vs Q6 partagent group-size). |
| Niveau de granularité ? | expert → tensor (gate/up/down) → row-block → quant-block. SliceMoE : cache au niveau bit-slice. MoEpic : split experts (bottom/full). |
| gate/up/down doivent-ils avoir la même précision ? | **Non.** down_exps est plus sensible (bug Q4_K/Q5_K corrigé par Q6_K sur ffn_down_exps — voir G). |

### B. Déquantification
| Question | Réponse / Preuve |
|---|---|
| Coût réel Q4_K→BF16 ? | Kernels ggml déquantent par bloc dans le kernel (dequantize_q4_K). Le buffer BF16 complet n'est PAS nécessaire (chemins MMQ CUDA natifs Q4_K/Q5_K/Q6_K). |
| Fusion au GEMM ? | FlashInfer / TensorRT-LLM : grouped GEMM + FP4/FP8/MXFP4 + quant activation en kernel. C'est la voie Kernel C (registre/shared → MMA). |
| Charger Q4 directement en shared/registers ? | Oui — c'est le pattern des kernels mmq CUDA et TurboQuant (apply_turbo_cuda_v2.py). |

### C. Cache
| Question | Réponse / Preuve |
|---|---|
| Combien d'experts dans 8 GB ? | Qwen3.8 : ~1.34 GB/layer d'experts (solarkyle/qwen38-flashnext-16gb). 64 slots = +84% (RFC #28248). Budget 8 GB → à calculer par taille expert. |
| Cache global vs par layer ? | vLLM : LFRU par layer (per-layer) pour éviter que les premières couches monopolisent (PR #37190). MoE-Infinity : "experts des premières couches bénéficient moins du prefetch → cache priorité couches tardives". |
| LRU suffit-il ? | Non — MoE-Infinity (LFU trace), vLLM (LFRU), FlashMoE (ML recence+freq). |
| Distance de réutilisation réelle ? | MoE-Infinity : ~50% des experts changent entre tokens consécutifs (FATE). Hot set stable par requête (tracé EAM). |
| Taille minimale utile ? | FATE : cache < working set = 0% hit = PIRE que vanilla (1.14 vs 6.4 t/s). Pool ≥ 1.44× working set. |

### D. Prédiction / Prefetch
| Question | Réponse / Preuve |
|---|---|
| Connaître l'expert suivant avant la fin du GEMM ? | Oui : prédiction inter-layer (router layer N → experts layer N+1). |
| Exploiter les logits du routeur avant le FFN ? | Oui : FATE (cross-layer + temporal), PreScope/LLaPor (layer-aware predictor), Pre-gated MoE. |
| Taux de prédiction atteignable ? | **93.03% (DeepSeek-V2 Lite), 94.69% (Qwen3-30B), 97.62% (Phi-mini-MoE)** — pre-attention prediction (source communauté, cf. analyse). FATE : 37998/76074 = ~50% du PCIe éliminé. |
| Coût d'un prefetch inutile ? | Un DMA spéculatif gaspillé = BW PCIe perdue. FATE : pool > working set → prefetch ratio 50%. À tolérer tant que T_DMA ≤ T_GEMM. |
| 2-3 experts spéculatifs ? | Oui (top-k candidates probables). La prédiction basse précision (HOBBIT) évite le miss coûteux. |

### E. PCIe / RAM / SSD
| Question | Réponse / Preuve |
|---|---|
| Goulot PCIe ou DDR ? | **PCIe (RTX↔DDR ~25 GB/s effective)** domine le coût H2D. DDR (21.93 GB/s NPU) domine le NPU. Deux goulots distincts. |
| Change entre prefill et decode ? | Oui : prefill = batch large (plus d'experts par itération), decode = 1 token, réutilisation temporelle forte. À mesurer. |
| H2D pageable vs pinned ? | **pinned** : évite la double copie. FATE : mmap ne peut pas être pinned via cudaHostRegister → hook expert pour gérer. |
| SSD peut-il alimenter assez vite ? | **Seulement page cache chaud** : 4.3 GB/token @ 7 GB/s = 634 ms (Qwen3-235B). N-gram/PLE : "hashing → il faut streamer ~1GB/token @ 7 GB/s ≈ 7 t/s max" (unsloth discussion). |
| Amplification de pages GGUF ? | Oui (interleaved layout). tinygiant : relayout experts contigus. **Sidecar GGUF** (anemll flashmoe-sidecar : extraire les experts dans un manifeste + banques brutes, bytes exacts préservés). |
| Faut-il un layout expert-contiguous ? | Recommandé si SSD streaming. Le sidecar anemll = solution existante. |

### F. NPU XDNA2
| Question | Réponse / Preuve |
|---|---|
| Le NPU apporte-t-il qq chose au chemin expert ? | **Comme prédicteur/contrôleur, pas comme moteur MoE.** RTX = compute (ratio 7.4× mesuré). NPU idle 99.73% pendant FLM. |
| Peut-il prédire assez tôt ? | Oui (router Qwen = GDN/QSA sur NPU) — l'architecture B (NPU contrôleur de trafic). |
| Requantification moins chère que CPU ? | 51 TOPS INT8 → probablement, à mesurer (P5). |
| Coût CPU↔NPU/XRT annule-t-il le gain ? | **Oui si MCDM dispatch reste** (81.32 ms/token, 3064 IOCTL). Persistent hw_context -65% TTFT d'abord. |
| Runtime séquence dynamique sans sync host ? | IRON/XDNA2 : séquences runtime dynamiques + DMA host-driven documentés (stack OSS). Le NPU peut préparer index/layout pendant que le RTX calcule. |

### G. Exactitude (le piège MUL_MAT_ID)
| Question | Réponse / Preuve |
|---|---|
| Q4_K fonctionne-t-il toujours avec MUL_MAT_ID ? | **NON.** Issues : #24591 (crash IDs dupliqués, mmid.cu), #21289 (invalid argument, fix CMAKE_CUDA_ARCHITECTURES), #13252 (FA+MLA mixte). down_exps Q4_K/Q5_K buggué → **Q6_K corrige** (analyse user). |
| Erreur locale sur down_exps vs gate_exps ? | down_exps plus sensible (même le bug CUDA). **Quantization Capability Matrix testée, pas supposée.** |
| Déterministe avec transitions de précision ? | DynaExq : "toujours la dernière version stable de chaque expert", transitions async versionnées → déterministe par construction. |

### H. GPUDirect Storage
| Question | Réponse / Preuve |
|---|---|
| SSD→VRAM direct ? | **NON pour le MVP Windows** : GDS non supporté Windows (NVIDIA doc). Chemin : SSD → pinned RAM → H2D → VRAM. GDS/cuFile = P10/P11 Linux. |

---

## 2. ARCHITECTURE CIBLE

```
                        Qwen3.8-Flash-Next
                                  │
                         ┌────────▼────────┐
                         │     ROUTER      │   ← GDN/QSA (NPU ou CPU)
                         └────────┬────────┘
                                  │
                         expert IDs + scores
                                  │
                    ┌─────────────▼─────────────┐
                    │        D2 PLANNER         │
                    │  résidence / précision /  │
                    │  prefetch / device / budget│
                    └──────┬─────────┬──────────┘
                           │         │
                  ┌────────▼───┐ ┌──▼────────┐
                  │  RESIDENCY │ │ PRECISION │
                  │  MANAGER   │ │  MANAGER  │
                  └──────┬─────┘ └────┬──────┘
                         │             │
              ┌──────────┴─────┐  ┌────┴─────────┐
              │                │  │              │
             SSD              RAM Q4/Q5/Q6      Q4/Q5/Q6/Q8
              │                │                  │
              └───────────────► VRAM CACHE ◄──────┘
                                  │
                           fused / tiled dequant
                                  │
                                  ▼
                              GEMM / MMA (RTX 5070)

   XDNA2 (séparé) : prediction / routing / prefetch planning / DMA scheduling / layout
```

## 3. PHASES P0 → P5 (ordre recommandé, avec preuves à reproduire)

### P0 — Reverse MUL_MAT_ID + mapping GGUF (la fondation)
Disséquer : router → topk_ids → MUL_MAT_ID → expert mapping → input_cpy → GPU pool → quant GEMM.
Construire la table :
```
Layer Tensor Expert Shape QType Offset Bytes Kernel
0     gate  17     …      Q4_K  …      …     …
```
Le système doit dire "l'expert 17 couche 23 = ces plages d'octets". C'est LE prérequis du
paging par tensor/tile. Refs : RFC #28248 (byte offset dans tenseur miroir),
anemll flashmoe-sidecar (extraction exacte des bytes GGUF).

### P1 — Cache VRAM persistant (reproduire la courbe llama.cpp)
Slots 4/8/16/32/64/128. Objectif : hit rate, bytes H2D/token, tok/s.
**Référence à reproduire** : 11.55 → 21.21 t/s avec 64 slots (RTX 4090, +84%) — à adapter 5070.
Refs : RFC #28248, PR #26563 (heatmap), PR #24524 (gist), vLLM #37190 (LFRU, `--moe-expert-cache-size`).

### P2 — DMA async + pinned RAM
- Pinned H2D sur stream CUDA dédié (FATE).
- Double buffering : `GEMM(A)` ‖ `DMA(B)`.
- Objectif : `T_DMA ≤ T_GEMM` → transfert invisible.
Ref : FATE (stream prefetch, 20× D2D vs PCIe).

### P3 — Prédiction
LRU → LFU → LFRU → LRU + predicted_next.
Niveaux : freq globale → P(next|current) → cross-layer → activation → experts suivants.
Cibles mesurées : 93-97% (pre-attention prediction, littérature).
Refs : MoE-Infinity (EAM trace), PreScope/LLaPor, MoEpic (expert split + prefetch).

### P4 — Précision dynamique (DynaExq-style)
Après cache OK : hotness, error_sensitivity, transfer_cost, compute_cost, reuse_distance →
choix Q4/Q5/Q6/Q8 par expert sous budget VRAM.
"La VRAM est un portefeuille de précision, pas tous-Q4."
Refs : DynaExq (arXiv:2511.15015 : +4.03 accuracy vs static, up to 2.73× throughput),
HOBBIT (arXiv:2411.01433 : cache-miss → version basse précision).

### P5 — XDNA2 planner + requant NPU
- CPU planner vs CPU + XDNA predictor (mesurer).
- Test si le NPU requantifie moins cher que le CPU (51 TOPS INT8).
- Persistent hw_context (éliminer les 2362 ms MCDM).

### P6 (après) — PLE manager séparé
Qwen3.8 PLE : 51B N-gram, lookup O(1), hash → illisible SSD (~7 t/s max). Le garder en RAM
(27 GB) ou aux GPU (aux-GPU vLLM #53908), PAS SSD. API identique : request_range/prefetch/promote/evict.
Refs : vLLM #53908, solarkyle/qwen38-flashnext-16gb (--ram-tensors, --no-prefetch), unsloth
discussion (hashing SSD ≈ 7 t/s).

### P7 (Linux) — SSD direct
GPUDirect Storage + cuFile (NVIDIA, Linux only). Pas le MVP Windows.

---

## 4. INSTRUMENTATION profiler-v3 (champs à ajouter)

```
timestamp layer expert_id tensor_id
storage_precision compute_precision
storage_location (SSD/RAM/VRAM)
cache_hit cache_miss
prefetch_predicted prefetch_correct
bytes_storage bytes_h2d bytes_dequant
t_read_us t_pcie_us t_prefetch_us t_quant_us t_dequant_us t_gemm_us t_sync_us queue_wait_us
eviction_reason hotness reuse_distance prediction_score
device (CPU/XDNA/RTX) kernel_id
critical_path_us hidden_transfer_us visible_transfer_us   ← un transfert 300µs masqué par 500µs de compute n'est PAS un problème
```

## 5. MODÈLE DE COÛT D2 PLANNER

```
Texpert = Tstorage + TPCIe + Tquant + Tdequant + Tcompute + Tsync + Tqueue
Tcritical = max(Tcompute, Ttransfer) + non-overlappable

score(représentation) =
  accuracy_penalty
  + λ1·memory_cost + λ2·transfer_cost + λ3·dequant_cost
  + λ4·eviction_risk + λ5·miss_probability
```

Le D2 Planner choisit simultanément : représentation + résidence + prefetch + device.

## 6. GATES DE VALIDATION (pas d'objectif tok/s arbitraire)

1. **Exactitude** : tokens identiques OU PPL stable (avant perf !)
2. **Mémoire** : VRAM < budget, RAM < budget
3. **Cache** : hit rate, miss rate, bytes transferred/token
4. **Recouvrement** : hidden DMA %
5. **Adaptivité** : gain à budget VRAM identique
6. **Robustesse** : 100+ prompts, plusieurs contextes, plusieurs sessions

## 7. CRITÈRES DE NON-RÉGRESSION (leçons à conserver)

- **rc=0 ≠ preuve** (fallback CPU silencieux) — vérifier n_backends, mirror, repack
- **Cache mal dimensionné = contre-productif** (FATE 0% hit = 1.14 < 6.4 t/s)
- **Break-even hit-rate ≈ 42%** — au-delà, désactiver le cache (self-disable PR #24524)
- **Layout conversion ≥ compute parfois** — coûter layout_conversion_cost
- **Ne pas patcher le synchronize du scheduler** (PR #17795/#20793 revertés)
- **down_exps Q4_K/Q5_K = bug CUDA** → Q6_K (Quantization Capability Matrix testée)

## 8. DÉPÔTS À DISSÉQUER (priorité)

| Dépôt | Ce qu'il apporte | URL |
|---|---|---|
| llama.cpp RFC #28248 | cache persistant +84% | https://github.com/ggml-org/llama.cpp/discussions/28248 |
| llama.cpp PR #26563 | heatmap, -ehs | https://github.com/ggml-org/llama.cpp/pull/26563 |
| vLLM #37190 / RFC #38256 | LFRU, CachedWeightProvider | https://github.com/vllm-project/vllm/pull/37190 |
| DynaExq | précision dynamique budget-constrained | https://arxiv.org/abs/2511.15015 |
| MoE-Infinity | EAM trace, sparsity-aware cache | https://github.com/EfficientMoE/MoE-Infinity |
| HOBBIT | miss → basse précision | https://arxiv.org/abs/2411.01433 |
| PreScope/LLaPor | prédiction cross-layer layer-aware | https://arxiv.org/html/2509.23638v1 |
| MoEpic | expert split + prefetch | https://arxiv.org/html/2509.08342v1 |
| anemll flashmoe-sidecar | sidecar GGUF experts contigus | https://github.com/Anemll/anemll-flash-llama.cpp |
| solarkyle/qwen38-flashnext-16gb | mesures réelles 16 GB (n-cpu-moe, PLE RAM) | https://github.com/solarkyle/qwen38-flashnext-16gb |
| vLLM #53908 | PLE aux-GPU | https://github.com/vllm-project/vllm/issues/53908 |
| KTransformers | hybrid CPU/GPU + quant online | (github.com/kvcache-ai/KTransformers) |
| FlashInfer | grouped GEMM FP4/FP8 | (github.com/flashinfer-ai/flashinfer) |
| FluxMoE | découplage résidence (3.0× vLLM) | https://arxiv.org/html/2604.02715v1 |

## 9. PROCHAINE ÉTAPE CONCRÈTE

**P0 : disséquer MUL_MAT_ID de llama.cpp** (qwen4exp/PR #27742) jusqu'au niveau
expert_id → offset → bloc Q4_K → déquant → MMA, puis concevoir le point d'insertion du
D2 Planner (prefetch + cache + fused dequant) SANS réécrire ggml. C'est le morceau le plus
rentable à reverse-engineer avant de toucher à XDNA2.

Fichiers cibles : `ggml/src/ggml-cuda/mul_mat_id.cu` (ou mmid.cu), `ggml/src/ggml-cuda/moe-cache.cu`
(vLLM/PR), `ggml/src/ggml-cuda/mmq.cuh` (kernels quantifiés), la partie input_cpy de
`ggml_backend_sched_compute_splits()`.