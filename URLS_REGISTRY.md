# REGISTRE DES SOURCES — Qwen3.8-Flash-Next / précision adaptative / paging / cache experts / RTX-CUDA / XDNA2
# Établi 2026-09-20 · Dossier npu-rtx/ · Version consolidée (mise à jour des docs précédents)
# Centrales vs complémentaires, ~50 références, 12 noyau dur à disséquer avant de coder.

---

## 1. QWEN3.8-FLASH-NEXT — MODÈLE ET ARCHITECTURE

| Source | URL |
|---|---|
| Dépôt officiel Qwen | https://github.com/QwenLM/Qwen3.8-Flash-Next |
| Poids HF (BF16) | https://huggingface.co/Qwen/Qwen3.8-Flash-Next |
| FP8 officiel | https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8 |
| NVFP4 NVIDIA | https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4 |
| ModelScope (specs complètes) | https://www.modelscope.ai/models/Qwen/Qwen3.8-Flash-Next |
| Papier architecture | "On the Design of Qwen3.8-Next Architecture" (Qwen Team blog/tech report) |
| Support llama.cpp issue | https://github.com/ggml-org/llama.cpp/issues/27741 |
| PR qwen4exp | https://github.com/ggml-org/llama.cpp/pull/27742 |
| Forum NVIDIA (tailles n-gram 51B) | https://forums.developer.nvidia.com/t/qwen3-8-flash-next/381228 |

**Spécifications vérifiées** (ModelScope + HF) :
- 125B main + 51B n-gram + 4B MTP · 6B activés/token
- hidden 2560 · vocab 248320 · ctx 262 144 natif → 1M
- 48 couches · 12×(3×(GDN→MoE)→1×(QSA→MoE))
- MoE : 512 experts · 10 routed + 1 shared · expert interm 640
- GDN 48V/16QK heads · QSA 24Q/2KV · indexer MQA 4Q+1K · budget 512 blocs

## 2. GGUF / QUANTIFICATIONS DISPONIBLES

| Source | URL |
|---|---|
| Unsloth GGUF | https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF |
| UD-Q4_K_XL | (sur le repo unsloth ci-dessus) |
| UD-IQ4_XS | (idem) |
| vumpt Q4_K_M | https://huggingface.co/vumpt/Qwen3.8-Flash-Next-GGUF (README quantization layout) |

**⚠️ Le README vumpt prouve le point clé** : le modèle N'EST PAS uniformément Q4_K_M —
fallback Q5_0/Q8_0 quand la forme ne convient pas, embedding/output Q6_K, **PLE en Q5_0**
(layout 160 colonnes). Le modèle est déjà hétérogène par tenseur.

## 3. LLAMA.CPP — CACHE EXPERTS

| Source | URL | Rôle |
|---|---|---|
| RFC --moe-expert-cache | https://github.com/ggml-org/llama.cpp/discussions/28248 | **Noyau dur** : pool persistant de slots, +84% (11.55→21.21 t/s @ 4090, 64 slots), hit 90-95% |
| Two-tier GPU+RAM | https://github.com/ggml-org/llama.cpp/issues/20757 | **Noyau dur** : Tier1 GPU slots, Tier2 pinned RAM, Tier3 SSD/mmap ; expert_offset = first_id*expert_size |
| Ancienne RFC CUDA | https://github.com/ggml-org/llama.cpp/discussions/24528 | cache expert CUDA historique |
| SSD expert streaming | https://github.com/ggml-org/llama.cpp/discussions/27149 | **Noyau dur** : lectures expert-aware, préfetch, layout contigu |
| PLE SSD | https://github.com/ggml-org/llama.cpp/discussions/27864 | **Noyau dur** : PLE Qwen4 offload, lectures lignes à la demande |
| mmap vs direct I/O | https://github.com/ggml-org/llama.cpp/discussions/18758 | amplification de pages, warm vs cold |

## 4. LLAMA.CPP — CODE EXACT DU CHEMIN QUANTIFIÉ

| Source | URL |
|---|---|
| tools/quantize/README.md | https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md |
| common/arg.cpp (--cpu-moe) | https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp |
| dequantize.cuh | https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/dequantize.cuh |
| quantize.cu | https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/quantize.cu |
| mmq.cu | https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu |
| CUDA backend refactor | https://github.com/ggml-org/llama.cpp/discussions/22975 |
| README | https://github.com/ggml-org/llama.cpp |

**Dissection faite localement** (P0_MULMAT_ID_INSERTION.md) : L2017 `src0_slice.data =
src0->data + i02*nb02` = le hook du cache ; les 2 synchronize du fallback = coût évitable ;
mmid.cu fait le tri GPU sans sync.

## 5. LLAMA.CPP — PROBLÈMES SPÉCIFIQUES QWEN3.8 / MOE

| Source | URL | Danger |
|---|---|---|
| **ggml #1506 K-quants mul_mat_id** | https://github.com/ggml-org/llama.cpp/issues/1506 | **⚠️ À garder ABSOLUMENT** : Q4_K/Q5_K/Q6_K incorrects sur mul_mat_id, Q8_0 fonctionnait → change la stratégie de l'oracle |
| #27911 concurrent batching | https://github.com/ggml-org/llama.cpp/issues/27911 | crash rms_norm_fused en batch concurrent |
| #28655 dynamic routing | https://github.com/ggml-org/llama.cpp/discussions/28655 | top-p/min-p dynamique |
| #28588 RAM saturée | https://github.com/ggml-org/llama.cpp/discussions/28588 | Qwen3.8 RAM |

## 6. DYNAEXQ — PRÉCISION DYNAMIQUE

| Source | URL |
|---|---|
| Dépôt | https://github.com/DynaQuant/DynaExQ (ou référencé arXiv) |
| Papier | https://arxiv.org/abs/2511.15015 |

**Preuve** : hot→haute précision, cold→basse, 2 versions hi/lo, transitions async versionnées,
pools sans fragmentation, budget HBM feasible. Révisé sept. 2026. +4.03 accuracy vs static,
2.73× throughput (arXiv:2511.15015).

## 7. MOE-INFINITY

| Source | URL |
|---|---|
| Dépôt actuel | https://github.com/EfficientMoE/MoE-Infinity |
| Ancien | https://github.com/TorchMoE/MoE-Infinity |
| configuration.md | https://github.com/EfficientMoE/MoE-Infinity/blob/main/docs/configuration.md |
| **adaptive expert precision** | https://github.com/EfficientMoE/MoE-Infinity/blob/main/docs/adaptive_expert_precision.md |
| Papier | https://arxiv.org/abs/2401.14361 |

**Preuve** : ExpertResidencyManager sépare checkpoint/representation cache/dtype execution ;
prefetch fenêtre = BW mesurée × temps de calcul par couche (pas BW idéale) ; policies prefill
et decode séparées ; 3.1-16.7×.

## 8. VLLM

| Source | URL |
|---|---|
| RFC incremental MoE offload | https://github.com/vllm-project/vllm/issues/38256 |
| PR #37190 (impl) | https://github.com/vllm-project/vllm/pull/37190 |

**Preuve** : GPU slots + CPU pinned + **LFRU** + cross-layer prediction + async H2D ;
CachedWeightProvider. LFRU évite que les premières couches monopolisent.

## 9. KTRANSFORMERS

| Source | URL |
|---|---|
| Dépôt | https://github.com/kvcache-ai/KTransformers |
| CPU-GPU Expert Scheduling | (docs du repo) |
| AMX.md (online quant) | https://github.com/kvcache-ai/KTransformers/blob/main/ktransformers/ops/README.md (AMX INT4/INT8) |
| KT-Kernel README | https://github.com/kvcache-ai/KTransformers/blob/main/ktransformers/optims/README.md |

**Preuve AMX** : quant online BF16→INT4/INT8 AU CHARGEMENT (le "Niveau B" de requant runtime).

## 10. FLASHINFER

| Source | URL |
|---|---|
| Dépôt | https://github.com/flashinfer-ai/flashinfer |
| Unified MoE API | https://docs.flashinfer.ai/api/fused_moe.html |
| MoE design docs | https://docs.flashinfer.ai/ (gemm + fused_moe) |
| Papier | https://arxiv.org/pdf/2501.01005 |

**Preuve** : config **weight × activation × output** (exactement le modèle W×A du SAQE) ;
kernels BF16/FP8/FP4/MXFP4/NVFP4 distincts ; grouped GEMM SM120 (b12x cute_dsl) ; benchmarks
en débit mémoire effectif (octets réels / temps kernel) ; FP8 GEMM SM89+.

## 11. TENSORRT-LLM

| Source | URL |
|---|---|
| Dépôt | https://github.com/NVIDIA/TensorRT-LLM |
| Quantization | https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md |
| fused MoE quantization.py | https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/plugins/fused_moe_quantization/quantization.py |
| quantization/mode.py | https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/quantization/mode.py |

**Preuve SM120** (matrice) : NVFP4✅ MXFP4✅ FP8-pertensor✅ FP8-KV✅ ; **FP8 block-scale,
FP8 rowwise, W4A8, W4A16 ❌ sur sm120** (seulement sm100/Hopper/Ada). W4A16/W4A8/W8A8/FP8
avec scales per-channel/per-token documentés.

## 12. CUDNN — GROUPED GEMM + FP4

| Source | URL |
|---|---|
| Grouped GEMM + GLU | (docs cuDNN — API grouped_gemm / MoE) |

**Preuve** : contraintes M/N tile, **alignement 16 B**, certaines dims alignées à 256, scale-vector
size, FP4 layout, padding → la géométrie du kernel compte, pas que les bytes.

## 13. HOBBIT

| Source | URL |
|---|---|
| Papier | https://arxiv.org/abs/2411.01433 |

**Preuve** : cache-miss → **version basse précision** (réduit le coût de chargement), prefetch
adaptatif, cache multidimensionnel. Directement lié à l'idée expert substitution.

## 14. PRÉDICTION EXPERTS

| Source | URL |
|---|---|
| Pre-Attention Expert Prediction | https://arxiv.org/abs/2511.10676 |

**Preuve** : prédiction légère avant l'attention, taux élevés sur plusieurs MoE (93-97%).

## 15. FLASHMOE (SSD)

| Source | URL |
|---|---|
| Papier | https://arxiv.org/abs/2601.17063 |

**Preuve** : SSD + cache ML (recency+frequency), +51% hit vs LRU/LFU, 2.6×.

## 16. DAMP — ÉTATS GDN/KDA

| Source | URL |
|---|---|
| DAMP (Decay-Aware Recurrent-State Quant) | (arXiv — à rechercher par titre exact) |

**Preuve** : **weights ≠ seuls objets quantifiables** — les états récurrents GDN/KDA sont
quantifiables en précision mixte (énergie d'erreur + persistance des canaux). Nouvel angle.

## 17. ROUTING

| Source | URL |
|---|---|
| Counterfactual Routing | https://arxiv.org/abs/2605.07260 |
| Efficient Quantization MoE | https://arxiv.org/abs/2604.06515 |

**Preuve** : la route standard n'est pas toujours la meilleure (contre-factuelle) ; la
quantification doit considérer le routing ; hotness ≠ sensitivity (variation router +
variance intra-neurone).

## 18. AWQ

| Source | URL |
|---|---|
| AWQ | https://arxiv.org/abs/2306.00978 |

**Preuve** : importance ≠ uniforme ; activation-aware → canaux saillants ; l'unité de quant
optimale peut être channel-group, pas l'expert.

## 19. SLICEMOE / DIGEST AMD

| Source | URL |
|---|---|
| llm-paper-radar digest | https://github.com/chu-tianxiang/llm-paper-radar (digest 2025-12-15) |

**Preuve** : cache à précision/bit-slice variable (SliceMoE) — granularité au-delà de l'expert.

## 20. NVIDIA CUDA — BRIQUES

| Source | URL |
|---|---|
| CUDA Best Practices | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide |
| Async Execution | https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution |
| CUDA C++ Intro | https://docs.nvidia.com/cuda/cuda-c-programming-guide |
| Best Practices archive 12.6 | https://docs.nvidia.com/cuda/archive/12.6.0/pdf/CUDA_C_Best_Practices_Guide.pdf |

**Preuves** : pinned = ~12 GB/s PCIe x16 Gen3 ; cudaMemcpyAsync + streams = overlap ; batching
petits transferts ≫ transferts séparés.

## 21. NVIDIA BLACKWELL

| Source | URL |
|---|---|
| Blackwell Tuning Guide | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ (Blackwell section) / tuning guide |

**Preuves** (cross-réf BLINDSPOTS_HW.md) : CC 12.0, 48 warps/SM max, 128 KB SMEM/SM, **pas de
TMEM** sur SM120, mma.sync f8f6f4, register pressure = goulot NVFP4.

## 22. GPUDIRECT STORAGE

| Source | URL |
|---|---|
| GDS Design Guide | https://docs.nvidia.com/gpudirect-storage/design-guide/index.html |
| GDS Overview | https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html |
| GPUDirect | https://developer.nvidia.com/gpudirect |

**Preuves** : SSD→GPU direct vs SSD→RAM→PCIe→GPU ; exigences IO explicite + pinned ;
**non supporté Windows** → MVP = SSD→pinned RAM→H2D.

## 23. MICROSOFT / WDDM

| Source | URL |
|---|---|
| Process Residency Budgets | https://learn.microsoft.com/en-us/windows-hardware/drivers/display/process-residency-budgets |
| Driver Residency WDDM 2.0 | https://learn.microsoft.com/en-us/windows-hardware/drivers/display/driver-residency-in-wddm-2-0 |

**Preuves** : **VRAM annoncée ≠ budget garanti** ; budgets de résidence variables sous pression
mémoire ; allocations non résidentes illégales → guard rail dynamique obligatoire.

## 24. AMD / XDNA2 / IRON

| Source | URL |
|---|---|
| mlir-aie Programming Guide | https://github.com/Xilinx/mlir-aie/tree/main/docs (programming_guide.md) |
| Runtime Data Movement / DMA | https://xilinx.github.io/mlir-aie/programming_guide/runtime-sequences.html |
| IRON configuration | (mlir-aie docs) |
| DMA transpose examples | (mlir-aie examples) |
| ROADMAP.md | https://github.com/Xilinx/mlir-aie/blob/main/ROADMAP.md |
| Releases | https://github.com/Xilinx/mlir-aie/releases |

**Preuves** : runtime sequences dynamiques, tailles/strides/offsets DMA dynamiques, contrôle
host-driven → correspond exactement au NPU contrôleur/prédicteur du D2.

## 24bis. AMD PROFILING / TÉLÉMÉTRIE (profiler-v3 integration — voir PROFILER_V3_AMD_INTEGRATION.md)

| Source | URL | Rôle |
|---|---|---|
| AMD AI Analyzer 1.8 | https://ryzenai.docs.amd.com/en/main/ai_analyzer.html | partition CPU/NPU + timeline layer (⚠️ BF16 only) — validation NPU |
| AMD XDNA Driver | https://github.com/amd/xdna-driver | npu_perf_trace.sh, telemetry, SDT events |
| **npu_perf_trace.sh** | https://github.com/amd/xdna-driver/blob/main/scripts/npu_perf_trace.sh | **Cible de dissection n°1** : XRT SDT + amdxdna_trace + perf → timestamps |
| **amdxdna telemetry UAPI** | https://github.com/amd/xdna-driver/blob/main/src/include/uapi/drm_local/amdxdna_accel.h | QUERY_TELEMETRY/SENSORS/HW_CONTEXTS/CLOCK_METADATA |
| AIE/XDNA doc (MERT counters) | https://www.kernel.org/doc/html/latest/accel/amdxdna/amdnpu.html | L1/DMA/DeepSleep counters |
| telemetry vs XRT issue #1447 | https://github.com/amd/xdna-driver/issues/1447 | incompatibilités driver mainline / SHIM XRT |
| **xdna-top** | https://github.com/Glabby000/xdna-top | **Cible n°2** : monitoring NPU+iGPU, record/compare/baseline, compteurs réels submissions/completions |
| ryzenai-lab | https://github.com/ryzenai-lab | benchmarks NPU/CPU/power prefill/decode |
| xdna-engine | https://github.com/ryanhnr/xdna-engine | kernels AIE Rust/XRT, latence int8 |
| xdna-engine README | https://github.com/ryanhnr/xdna-engine/blob/main/README.md | mesures CPU vs NPU, WER, build time |
| **IRON performance guide** | https://xilinx.github.io/mlir-aie/programming_guide/performance.html | **Cible n°4** : DMA / performance |

## 24ter. AMD/XILINX — IRON / MLIR-AIE (pages directement exploitables)

| Source | URL |
|---|---|
| MLIR-AIE / IRON | https://github.com/Xilinx/mlir-aie |
| IRON Programming Guide | https://xilinx.github.io/mlir-aie/programming_guide/ |
| IRON API | https://xilinx.github.io/mlir-aie/api/ |
| IRON compilation stages | https://xilinx.github.io/mlir-aie/programming_guide/compilation.html |
| getting started | https://xilinx.github.io/mlir-aie/programming_guide/getting_started.html |
| exemples complets | https://xilinx.github.io/mlir-aie/programming_guide/examples.html |
| mini tutorial | https://xilinx.github.io/mlir-aie/programming_guide/tutorial.html |
| **performance / DMA** | https://xilinx.github.io/mlir-aie/programming_guide/performance.html |
| ryzen-npu-linux (XDNA1 vs XDNA2, HX PRO 370) | https://github.com/ryzen-npu-linux (doc FR, validation XDNA2) |

**Les 4 URLs à disséquer en premier pour profiler-v3** :
1. `npu_perf_trace.sh` (XRT SDT + tracepoints driver + perf)
2. `amdxdna telemetry UAPI` (compteurs NPU : clock/power/col_util/DMA/L1/DeepSleep)
3. `xdna-top` (soumissions/completions réelles + iGPU, philosophie "mesurer, ne pas inventer")
4. `IRON performance guide` (DMA / débits / transferts)

→ à croiser avec les mesures profiler-v3 : events XRT + tracepoints driver + DMA + compteurs NPU +
timestamps, puis corrélation dispatch/compute/DMA/gaps/sync.

## 25. TES PROJETS

| Source | URL |
|---|---|
| adaptive-xdna-runtime | https://github.com/Tagman45/adaptive-xdna-runtime |
| jarvix-memory | https://github.com/gat45/jarvix-memory |

## 26. FORKS / EXPÉRIENCES QWEN FLASH

| Source | URL |
|---|---|
| GenerelSchwerz/llama.cpp | https://github.com/GenerelSchwerz/llama.cpp |
| benchmark gist | (gist lié) |
| démo vidéo | (lien démo) |

## 27. NOYAU DUR (12 URLs à disséquer AVANT de coder)

1. llama.cpp #28248 — cache expert persistant
2. llama.cpp #20757 — cache GPU/RAM
3. llama.cpp #27149 — SSD expert streaming
4. llama.cpp #27864 — PLE paging
5. DynaExQ (arXiv:2511.15015)
6. MoE-Infinity adaptive precision
7. vLLM #38256 — prédiction/cache
8. HOBBIT (arXiv:2411.01433)
9. FlashMoE (arXiv:2601.17063)
10. FlashInfer Unified MoE API
11. TensorRT-LLM Quantization
12. llama.cpp dequantize.cuh + mmq.cu

**Convergence** : représentation d'un expert = f(mémoire, transport, kernel, réutilisation,
précision) — JAMAIS "Q4_K_M partout". DynaExQ = précision · llama.cpp/vLLM/MoE-Infinity =
résidence/prefetch · FlashInfer/TensorRT-LLM = calcul quantifié.

---

## CORRÉLATION BLINDSPOTS → SOURCES (réponses avec preuve)

| Angle mort (BLINDSPOTS) | Source de preuve |
|---|---|
| Q4_K_M ≠ NVFP4 | TensorRT-LLM quant matrix (§11), FlashInfer docs (§10), BLINDSPOTS_HW §1-2 |
| SM120 ≠ SM100 (pas TMEM) | lna-lab sm120-architecture + chsasank/blackwell (BLINDSPOTS_HW §1) |
| Aucun stack NVFP4 complet SM120 | chsasank/blackwell table (§2) |
| Qwen3.8 experts = W4A4 NVFP4 officiel | nvidia/Qwen3.8-Flash-Next-NVFP4 (§1) |
| MUL_MAT_ID K-quants bug | ggml #1506 (§5) — **oracle : tester format×kernel×shape** |
| KV/état GDN = objets quantifiables | DAMP (§16) |
| PLE ≠ MoE (lookup, shards, hash) | llama.cpp #27864 (§3), unsloth discussion, vLLM #53908 |
| Hotness ≠ sensitivity | arXiv:2605.07260, arXiv:2604.06515 (§17), AWQ (§18) |
| Online quant au chargement | KTransformers AMX (§9) |
| Fenêtre prefetch = BW mesurée | MoE-Infinity adaptive precision (§7) |
| Prédiction avant attention | arXiv:2511.10676 (§14) |
| SSD amplification pages | llama.cpp #27149/#18758 (§3) |
| WDDM budget résidence variable | Microsoft WDDM (§23) |
| Pinned = rare | CUDA Best Practices (§20) |
| Blackwell register pressure | Blackwell Tuning Guide (§21) |
| cuDNN FP4 constraints | cuDNN grouped GEMM (§12) |
| Cache miss → basse précision | HOBBIT (§13) |
| SSD cache ML recency+freq | FlashMoE (§15) |
| Fragmentation pools | DynaExQ (§6) |