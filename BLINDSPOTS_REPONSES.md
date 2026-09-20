# RÉPONSES AUX 66 ANGLES MORTS — avec preuves et URLs
# Établi 2026-09-20 · Dossier npu-rtx/ · Corrélé à URLS_REGISTRY.md, BLINDSPOTS_HW.md, SPEC_SAQE.md

## LES 10 ANGLES MORTS LES PLUS DANGEREUX (P0) — réponses

### P0-1. Le routing change avec la quantification des activations
**Question** : Quantifier le hidden state avant le router (BF16→FP8/INT8) peut changer le
top-k → boucle de rétroaction.
**Réponse** : Le planner doit mesurer `route_identity`, `route_overlap`, `Jaccard(top-k)`,
`KL(router_probs)` avant/après quant des ACTIVATIONS. La route standard n'est pas toujours la
meilleure (contre-factuelle) → ne jamais quantifier les activations du router au MVP.
**Preuve** : arXiv:2605.07260 (counterfactual routing), arXiv:2604.06515 (efficient quant MoE),
AWQ arXiv:2306.00978 (canaux sensibles).

### P0-2. PLE traité comme un simple poids
**Question** : Le PLE est une table lookup, pas un MoE.
**Réponse** : PLE = 320 001 536 lignes × width 160 = 51.2B params, 128 shards physiques, n-grams
en structure lookup → **ExpertPager ≠ PLEPager**. Petit accès dispersé + hash → illisible SSD
(~7 t/s max, "hashing → streamer ~1GB/token"). Budget DDR/SSD/RAM/PCIe partagé.
**Preuve** : llama.cpp #27864 (PLE SSD offload), unsloth Qwen3.8 discussion (hashing ≈ 7 t/s),
vLLM #53908 (PLE aux-GPU), vumpt README (PLE en Q5_0), nvidia NVFP4 (PLE per-tensor FP8).

### P0-3. Kernel/shape compatibility
**Question** : Un format valide ne signifie pas un kernel valide.
**Réponse** : Clé de compatibilité = `format × backend × kernel × shape × commit`. Q4_K/Q5_K/
Q6_K incorrects sur mul_mat_id, Q8_0 OK → **oracle doit tester**, pas supposer. cuDNN : alignement
16 B, certaines dims à 256. FlashInfer : grouped GEMM SM120.
**Preuve** : ggml #1506 (K-quants mul_mat_id), #21289 (arch CUDA), #24591 (IDs dupliqués), cuDNN
grouped GEMM, BLINDSPOTS_HW §2/§7.

### P0-4. VRAM budget réel Windows/WDDM
**Question** : nvidia-smi free ≠ budget utilisable.
**Réponse** : WDDM = budgets de résidence variables sous pression mémoire ; allocations non
résidentes illégales. Guard rail dynamique obligatoire (VRAM announced ≠ usable ≠ safely
reservable). Strix Halo : spill WDDM → freeze machine.
**Preuve** : Microsoft Process Residency Budgets + Driver Residency WDDM 2.0 (§23 URLS).

### P0-5. GDN state + QSA KV (zones mémoire oubliées)
**Question** : On ne quantifie pas que des poids.
**Réponse** : GDN a un état récurrent réutilisé token à token (36 couches) ; QSA = 12 couches
KV. Mesurer : bytes/token, update bandwidth, dequant cost, state error, long-context drift.
KV FP8/NVFP4 (QSA) peut doubler le pool utile sur SM120.
**Preuve** : DAMP (decay-aware recurrent-state quant), nvidia NVFP4 (KV FP8), corpus FLM
(état SSM [32,128,128]).

### P0-6. Batch / prefill / decode
**Question** : Le coût d'un expert dépend de M.
**Réponse** : Decode M≈1 (latence/PCIe/launch dominants), prefill M≫1 (GEMM dominé). La
quantification peut changer de rang entre phases. MoE-Infinity a des politiques prefill/decode
séparées + fenêtre de prefetch = BW mesurée × temps calcul.
**Preuve** : MoE-Infinity configuration.md + adaptive_expert_precision (§7 URLS).

### P0-7. Contention DDR/PCIe
**Question** : BW théorique ≠ BW disponible.
**Réponse** : BW_available(t) = BW_idle − KV − PLE − CPU − H2D − iGPU. Le planner travaille en
BW_available(resource, t), pas en BW_theoretical. MoE-Infinity calcule la fenêtre depuis BW
mesurée.
**Preuve** : MoE-Infinity (§7), corpus FLM (BW 21.93 GB/s, contention W_eff +60%).

### P0-8. Requantification depuis quant
**Question** : Q4→Q6 ≠ BF16→Q6.
**Réponse** : `--allow-requantize` peut dégrader fortement. Graphe de conversion NON réversible :
Q3→Q6 ne recrée pas l'info perdue. Pour permettre la promotion, conserver un **master BF16**
(ou représentation haute) → coût master_storage.
**Preuve** : llama.cpp tools/quantize/README.md (§4 URLS).

### P1-9. Fragmentation + versioning
**Question** : Q3/Q4/Q5/Q6/Q8 mélangés fragmentent la VRAM.
**Réponse** : fixed-size slots ou segregated size classes ; transitions async versionnées ;
attendre la fin des lecteurs avant libération (ExpertHandle).
**Preuve** : DynaExQ (pools sans fragmentation, allocations déterministes) arXiv:2511.15015.

### P1-10. Prediction/prefetch pollution
**Question** : Un mauvais prefetch peut être pire que rien.
**Réponse** : `prefetch_value = P(use)×saved_latency − transfer_cost − eviction_cost`. Mesurer
l'impact de la victime (mauvaise eviction → miss suivant). Admission conservatrice au cold start.
**Preuve** : MoE-Infinity (admission conservatrice), FATE (pool > working set sinon 0% hit =
contre-productif), PreScope/LLaPor (coût global cross-layer).

---

## LES 56 AUTRES ANGLES MORTS — réponses courtes avec preuves

11. **6 classes d'objets à quantifier** (backbone/expert/shared/GDN/QSA/GR/états/KV/PLE/MTP) →
3 moteurs : Weight/State/Memory-Residency Quantization. Preuve : DAMP, nvidia NVFP4 (mix
W4A4/BF16/FP8).

12. **État GDN** → précision mixte par énergie d'erreur + persistance canal. Preuve : DAMP.

13. **KV QSA** → KV precision/residency/bandwidth/pool pressure séparés ; KV FP8/NVFP4 SM120.
Preuve : nvidia NVFP4 (KV FP8), llama.cpp #27864.

14. **PLE ≠ MoE** → PLEPager séparé (lookup, shards, hash). Preuve : vLLM #53908, #27864.

15. **Amplification de pages** → mesurer useful_bytes vs physical_read_bytes (amplification).
Preuve : #27864, #18758, anemll flashmoe-sidecar.

16. **Layout GGUF > quantification** → le modèle est déjà hétérogène (vumpt). Preuve : vumpt
README, nvidia NVFP4.

17. **Géométrie du kernel** → shape/stride/alignment/tile/kernel_compat/padding. Preuve : cuDNN
grouped GEMM, ggml #1506.

18. **Format dispo ≠ kernel performant** → clé format×backend×kernel×shape×commit. Preuve :
ggml #1506, #21289, #24591.

19. **Concurrence** → -np 2/4 peut casser (crash rms_norm_fused). Preuve : llama.cpp #27911.

20. **Batch change le coût de quant** → indexer par phase/batch/ubatch/seq count. Preuve :
MoE-Infinity prefill/decode.

21. **Union des experts du batch** → bytes/token = f(union(E_req1,E_req2,...)) ; mesurer
unique_experts_per_step, freq, reuse within batch / across layers. Preuve : vLLM #38256.

22. **Router sensible à la quant des activations** → route_identity/Jaccard/KL ; ne pas quant
les activations du router au MVP. Preuve : arXiv:2605.07260.

23. **Hotness ≠ sensitivity** → 2 axes : activation_hotness + quality_sensitivity + transfer_cost.
Preuve : arXiv:2604.06515 (variation router + variance intra-neurone), AWQ.

24. **imatrix globale insuffisante** → tester global/layer/expert/tensor/workload imatrix ; coût
de collecte CPU-side. Preuve : llama.cpp imatrix (tools).

25. **Requant depuis quant** → parent de représentation obligatoire ; `--allow-requantize`
dégrade. Preuve : tools/quantize/README.

26. **Graphe de conversion non réversible** → master BF16 pour la promotion. Preuve :
tools/quantize/README.

27. **Multi-représentations** → canonical_storage vs materialized_cache_variants ; variante
créée si amortie. Preuve : DynaExQ (2 versions hi/lo).

28. **Amortissement conversion** → conversion_cost/expected_reuse. Preuve : DynaExQ, MoE-Infinity.

29. **Prefetch inutile** → prefetch_value = P(use)×saved − transfer − eviction. Preuve :
MoE-Infinity (fenêtre BW mesurée).

30. **Prédiction modifie le cache** → mesurer prediction accuracy + victim impact. Preuve :
FATE (victim), vLLM #38256.

31. **LRU insuffisant** → victim_score = future_value − reload_cost − precision_recovery_cost.
Preuve : vLLM LFRU (#38256/#37190), MoE-Infinity LFU.

32. **Fragmentation** → fixed-size slots / size classes. Preuve : DynaExQ.

33. **Sécurité de l'éviction** → versioning + drain des lecteurs avant libération. Preuve :
DynaExQ.

34. **CUDA Graphs** → cache dynamique vs graph statique (IDs/adresses variables) ; vérifier
capture/replay/address stability. Preuve : MoE-Infinity (CUDA graph capture), CUDA prog guide.

35. **WDDM** → guard rail dynamique. Preuve : Microsoft WDDM (§23).

36. **Pinned rare** → expert_cache_RAM ≠ pinned_staging_RAM ; budget séparé. Preuve : CUDA
Best Practices (pinned = ressource rare).

37. **Contention DDR** → BW_idle/CPU/XDNA/PLE/H2D mesurés. Preuve : MoE-Infinity, corpus FLM.

38. **XDNA2 DMA/sync limits** → NPU_DMA_channels, BD_available, sync_us, runtime_issue_us,
L1_capacity, DDR_pressure. Preuve : mlir-aie runtime sequences (§24 URLS), corpus FLM (3064
IOCTL).

39. **NPU chemin complet ≠ kernel seul** → device_compute + transfer + host_submit + wait
séparés (30 µs kernel → 140 µs chemin). Preuve : corpus FLM (décomposition token : CPU 86.8%,
dispatch 41%).

40. **PCIe = f(size, direction, concurrency)** → pas une constante 25 GB/s. Preuve : CUDA
Best Practices (batching petits transferts).

41. **SSD request size** → profiler 4K→64MB à différents queue depths. Preuve : GDS design
guide (alignment/fallback).

42. **Filesystem page cache** → mesurer cold/warm/direct (3 mesures). Preuve : llama.cpp
#18758 (mmap vs direct I/O).

43. **Température** → EWMA + variance + confidence (SSD 6.5→3.8 GB/s). Preuve : corpus FLM
(thermal >70°C), BLINDSPOTS_HW.

44. **P-state / power / clocks** → benchmark enregistre clock/temp/power/load. Preuve : CUDA
Best Practices, Blackwell tuning.

45. **Accuracy multi-métrique** → PPL/KL/token agreement + coding/reasoning/long-ctx/tool use.
Preuve : nvidia NVFP4 eval (GPQA/HLE/etc.), Qwen tech report.

46. **Erreur autoregressive** → first_divergence_token, route_divergence, logit_KL_over_time.
Preuve : corpus FLM (CPU-Q4==NPU-Q4 8/8, 64-token golden).

47. **Transition en plein flux** → mesurer dynamic transition vs static equivalent. Preuve :
DynaExQ (transitions versionnées).

48. **Hystérésis** → promote_threshold < demote_threshold + minimum_residency + cooldown.
Preuve : DynaExQ (controller smoothing).

49. **Expert fréquent ≠ prochain** → prédiction conditionnelle (layer, prev expert, phase,
request). Preuve : vLLM cross-layer (#38256), PreScope/LLaPor.

50. **Top-k réductible** → hors MVP (modifie le calcul). Preuve : llama.cpp #28655.

51. **Shared expert** → toujours actif → traiter comme dense-hot tensor. Preuve : specs Qwen
(10 routed + 1 shared).

52. **MTP** → commit/rollback des caches/états/KV. Preuve : Qwen MTP (4B), NVIDIA Dynamo MTP3,
llama.cpp spec (MTP régressions dans corpus FLM).

53. **Multimodal** → vision encoder/projection/image tokens dans le budget. Preuve : Qwen
multimodal (NIM doc), nvidia NVFP4 (vision).

54. **GGUF ≠ checkpoint BF16** → vérifier naming/fusion/transpose/layout/scale. Preuve :
vumpt README, P0_MULMAT_ID_INSERTION.

55. **Format/hardware** → Q4_K vs NVFP4 = 2 candidats technologiques. Preuve : BLINDSPOTS_HW,
TensorRT-LLM matrix.

56. **Déquant vs GEMM vs fused** → T_dequant + T_GEMM ≠ T_fused. Preuve : FlashInfer (fused
MoE, mono_moe), TensorRT-LLM fused MoE.

57. **Activation quantization** → choix (W_precision, A_precision). Preuve : FlashInfer
(weight×activation×output), TensorRT-LLM W4A16/W4A8.

58. **Scales/metadata** → payload vs metadata bytes ; à 2 bits la déquant est limitée par
autre chose. Preuve : Q4NX format (scales BF16 per group), vumpt.

59. **Unité de quant optimale** → expert × tensor × block (canaux saillants AWQ). Preuve : AWQ.

60. **Économiser bytes ≠ gagner du temps** → min bytes ≠ objectif (decode plus cher). Preuve :
NVIDIA Blackwell tuning (coalescing, occupancy).

61. **Occupation GPU** → enregistrer occupancy, registers/thread, SMEM, warps, L2 hit. Preuve :
Blackwell tuning guide (48 warps/SM, 128 KB SMEM).

62. **Cache L2** → modèle SSD→RAM→PCIe→VRAM→L2→L1/shared→register. Preuve : CUDA (L2 access
windows).

63. **Launch overhead** → T_launch/T_submit/T_event/T_sync séparés (M=1). Preuve : corpus FLM
(3064 dispatch × 67 µs), CUDA Best Practices.

64. **JIT cold/warm** → compile ≠ run ; JIT cold/warm/cache hit/load. Preuve : IRON/MLIR-AIE
runtime sequences.

65. **Startup vs steady-state** → TTFT/cold start vs TPOT. Preuve : corpus FLM (2362 ms init,
persistent hw_context -65%).

66. **Long contexte** → hot set 8K ≠ 128K ; profiler context_length/hotness/cache hit. Preuve :
Qwen 262K natif, llamaperf (dégradation >90k ctx).

---

## STRUCTURE FINALE D2 PLANNER (mise à jour)

```
D2 PLANNER
 ├ PRECISION (W_precision, A_precision, state_precision, KV_precision)
 ├ RESIDENCY (SSD/RAM/VRAM, cache/prefetch, eviction)
 ├ DEVICE (CPU/XDNA/RTX, kernels, backend)
 ├ MODEL AWARENESS (MoE routing/shared | PLE lookup/shards | GDN/QSA/MTP recurrent+KV+spec)
 ├ RESOURCE MODEL (SSD DDR PCIe VRAM NPU — BW_available(t))
 ├ KERNEL MODEL (dequant GEMM launch — format×kernel×shape×commit)
 └ CRITICAL PATH → DECISION (representation, activation, residency, prefetch, device, tile, victim)

T_critical = critical_path(storage, conversion, transfer, cache, dequant, compute, sync, launch, contention)
contraintes : VRAM RAM pinned SSD_BW DDR_BW PCIe_BW NPU_DMA/BD GPU_workspace KV quality thermal
```

## MODÈLE DE DONNÉES (mise à jour)

```
ExpertVariant { model_id layer_id expert_id tensor_id
  weight_format activation_format
  bytes_storage bytes_cache bytes_compute
  storage_offset shard_id source_variant
  quant_cost dequant_cost compute_cost
  quality_score quality_risk
  backend kernel_id shape alignment
  hotness sensitivity reuse_distance prediction_probability
  location generation }

ResourceState { ssd_bw ddr_bw pcie_h2d_bw pcie_d2h_bw vram_bw
  cpu_quant_bw cpu_dequant_bw xdna_dma_bw xdna_issue_us xdna_sync_us
  gpu_compute gpu_occupancy
  kv_usage expert_cache_usage workspace_usage
  temperature clocks confidence }
```

## RENOMMAGE CONCEPTUEL

"Adaptive Quantization Engine" → **D2 Adaptive Precision & Dataflow Planner**
(variables simultanées : Precision, Residency, Routing, Prefetch, Bandwidth, Device, Kernel,
Quality — la quantification n'est qu'une décision).