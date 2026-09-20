# SPEC — D2 System-Aware Quantization Engine (SAQE)
# "La quantification ne doit pas optimiser le nombre de bits ;
#  elle doit optimiser le temps de parcours de toute la chaîne de données."
# Établi 2026-09-20 · Dossier npu-rtx/ · Sources : RECHERCHE_NPU_RTX_URLS.md + P0_MULMAT_ID_INSERTION.md

## 0. PRINCIPE (validé par la recherche)

Abandonner `Q4 = petit = meilleur`. Le choix Q2/Q3/Q4/Q5/Q6/Q8/IQ/FP4/FP8 dépend du chemin
complet : stockage (SSD) → transport (DDR/PCIe) → calcul (déquant + GEMM), avec chevauchement.

**Preuve clé (chemin critique, pas somme)** :
- Expert Q4 = 1.0 GB, Q6 = 1.5 GB.
- Scénario A (SSD lent 5, PCIe 20) : DMA Q4 0.25 ms / Q6 0.38 ms ; GEMM Q4 0.80 / Q6 0.45.
  Avec overlap : Q4 ≈ max(0.25, 0.80) = 0.80 ms ; Q6 ≈ max(0.38, 0.45) = 0.45 ms → **Q6 gagne**.
- Scénario B (PCIe 1.35 ms pour Q6) : Q4 ≈ 0.90 ; Q6 ≈ 1.35 → **Q4 gagne**.
⇒ Le choix change avec le chemin ET le débit réel. Le planner doit être path-aware.

**Preuves externes** :
- **MoE-Infinity** : "unified_transfer_scheduler" + "expert_prefetcher" + "expert_predictor" ;
  fenêtre de prefetch calculée à partir de la BW de transfert mesurée et du temps de calcul par
  couche (pas de BW idéale supposée). URL : https://github.com/EfficientMoE/MoE-Infinity
  (memory/expert_prefetcher.py, memory/unified_transfer_scheduler.py)
- **CUDA Best Practices** : pinned memory = BW max H2D (12 GB/s PCIe x16 Gen3) ; cudaMemcpyAsync
  + streams = overlap transfer/compute ; "batching many small transfers into one larger transfer
  performs significantly better" (coût fixe par transfert). URL :
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide (sections 10.1.1, 10.1.2)
- **FlashInfer** : kernels BF16/FP8/FP4/MXFP4/NVFP4, grouped GEMM (Blackwell SM120 cute backend),
  MoE FP8/FP4 block-scale, benchmarks calculant le débit mémoire effectif depuis octets lus +
  temps kernel. URL : https://github.com/flashinfer-ai/flashinfer + https://docs.flashinfer.ai
- **DynaExq** : précision = ressource runtime, budget HBM feasible, transitions async versionnées.
  URL : https://arxiv.org/abs/2511.15015

---

## 1. L'ESPACE DE REPRÉSENTATION (pas "un modèle = une quantification")

### 1.1 Familles (matrice, pas liste)
| Famille | bpw effectif | Intérêt |
|---|---|---|
| IQ1 | ~1.5 | cold/extrême |
| IQ2 / Q2_K | ~2.0-3.0 | très faible trafic, cold |
| IQ3 / Q3_K | ~3.0-4.0 | zone intéressante |
| Q4 / IQ4 | ~4.2-4.9 | compromis |
| Q5 | ~5.5-5.7 | warm |
| Q6_K | ~6.56 | hot / zones sensibles |
| Q8_0 | ~8.5 | haute fidélité |
| FP8 (E4M3/E5M2) | 8 | GPU modernes, W8A8 |
| FP4/NVFP4/MXFP4 | 4 | **Blackwell (SM120) — PAS interchangeable avec Q4_K** |

⚠️ **Q4_K_M ≠ NVFP4** : Q4_K_M = GGML/GGUF (quantification poids) ; NVFP4 = Tensor Core CUDA
moderne (format orienté calcul). Même bits ≠ même débit/déquant/GEMM. FlashInfer expose des
kernels DISTINCTS pour BF16/FP8/FP4 (docs.flashinfer.ai/api/gemm.html) ; TensorRT-LLM distingue
W4A16/W4A8/W8A8/FP8/NVFP4 avec scaling per-channel/per-token/block.

### 1.2 La décision devient un tuple
```
expert.state = {
    storage_representation,   // SSD :  IQ3_XXS / Q4_K
    host_representation,      // RAM  :  Q4_K / Q5
    device_representation,    // VRAM :  Q6_K / Q8 / NVFP4
    compute_representation,   // tile :  BF16 / FP8 / FP4
    activation_precision,     // BF16 / FP8 / INT8 / FP4 (W4A16 / W8A8 ...)
    device,                   // CPU / XDNA2 / RTX
    location,                 // SSD / RAM / VRAM
    tile_size,                // 64/128/256/512
    prefetch_time,
}
```

### 1.3 Exemple
```
Expert 237 : SSD IQ3_XXS → RAM Q4_K → VRAM Q6_K → compute BF16 tile
Expert 418 : SSD Q4_K   → RAM Q4_K → VRAM Q4_K → compute BF16 tile
```

---

## 2. LES RESSOURCES À MESURER (Hardware Throughput Matrix)

### 2.1 Microbenchmarks (Level 1) — mesurés SUR LA MACHINE, pas copiés
| Ressource | Mesure | Preuve |
|---|---|---|
| SSD read | GB/s + latence | CUDA : batching petits transferts (guide §10.1.2) |
| RAM memcpy | GB/s | — |
| RAM→RTX | GB/s + latence | pinned ~12 GB/s PCIe x16 Gen3 (CUDA guide) |
| RTX VRAM | GB/s effectif | à mesurer (GDDR7 672 théorique) |
| CPU quant | GB/s entrée | KTransformers online quant au chargement |
| CPU dequant | GB/s | — |
| CUDA dequant | GB/s | FlashInfer benchmarks (octets réels / temps kernel) |
| CUDA GEMM Q4/Q6/Q8 | TFLOP/s effectif | FlashInfer grouped GEMM |
| CUDA GEMM FP4/NVFP4 | TFLOP/s (SM120) | FlashInfer b12x cute_dsl (SM120/SM121) |
| XDNA2 DMA | GB/s | CARTE_FONCTIONNELLE : 21.93 GB/s effective |
| XDNA2 kernel | µs | mesures FLM |
| XDNA2↔DDR | GB/s | — |
| sync CPU↔CUDA | µs | à mesurer |
| sync XRT | µs | CARTE : 67 µs/dispatch ×3064 |

### 2.2 Contention = BW_available(resource, t), PAS BW_theoretical
```
BW_available(DDR, t) = 89.6 - KV - PLE - CPU - autres transferts  (à l'instant t)
```
**Preuve** : MoE-Infinity calcule la fenêtre de prefetch depuis la BW mesurée ET le temps de
calcul par couche → un scheduler sensible à la contention, pas une BW idéale.

### 2.3 Débit effectif (jamais théorique)
- PCIe théorique 32 GB/s → effectif 21 (à mesurer)
- SSD théorique 7 GB/s → random 2.1
- DDR peak X → sous contention Y
- VRAM peak → effective de CE kernel

---

## 3. MODÈLE MATHÉMATIQUE

### 3.1 Coût par candidat (DAG, pas somme)
```
Cost(candidate) = T_storage + T_conversion + T_transfer + T_dequant + T_gemm + T_sync + T_eviction
CriticalPath = max(IO, transfer, compute, conversion) + non-overlappable
```

### 3.2 Graphe de conversion (chaque arc a un benchmark)
```
IQ2 → Q3 → Q4 → Q5 → Q6 → Q8
         ↘    ↘    ↘
          FP4 (NVFP4)  FP8
Chaque arc : conversion_rate, latency, CPU_cost, DDR_bytes, cache_bytes, quality_delta
```
Q3→Q6 n'est PAS gratuit : le planner compare
`SSD Q3→PCIe→RTX` vs `SSD Q3→CPU requant Q6→PCIe→RTX` vs `SSD Q6→PCIe→RTX`.

### 3.3 Optimisation globale
```
minimize  T_critical_path
s.t.      VRAM ≤ V ; RAM ≤ R ; SSD_BW ≤ S ; DDR_BW ≤ D ; PCIe_BW ≤ P ;
          XDNA_resources ≤ limit ; GPU_mem_BW ≤ limit ; accuracy_loss ≤ Q
variables : precision(expert) location(expert) device(expert)
            prefetch(expert) cache_slot(expert) tile_size(tensor)
            activation_precision(expert)
```

---

## 4. LE PLANNER (3+2 niveaux)

```
                  D2 PLANNER
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   PRECISION      RESIDENCY        DEVICE
   Q1..Q8/IQ/    SSD/RAM/VRAM     CPU/XDNA2/RTX
   FP4/FP8
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                ACTIVATION MODE (BF16/FP8/INT8/FP4)
                       │
                       ▼
                  BANDWIDTH MODEL (BW_available(res, t))
                       │
                       ▼
                   CRITICAL PATH
                       │
                       ▼
                  chosen strategy (Qtype, location, device, slot, prefetch, tile, A_precision)
```

### Candidat (structure)
```
Candidate {
    expert_id, tensor_id, precision,
    storage_bytes, ram_bytes, vram_bytes,
    ssd_bw_req, ddr_bw_req, pcie_bw_req, vram_bw_req,
    quant_cost, dequant_cost, compute_cost,
    cache_value, accuracy_cost
}
```

---

## 5. POINT D'INSERTION llama.cpp (rappel P0, vérifié)

- **L2017 ggml-cuda.cu** : `src0_slice.data = src0->data + i02*nb02` → le hook slot-cache
  (remplacer par slot_ptr(expert), miss = H2D sur stream dédié).
- **L1972-1994** : les 2 synchronize du fallback → forcer le chemin MMVQ/MMQ (GPU-only).
- **L1977-1987** : le point où les experts du token sont connus → brancher le predictor.
- Le cache garde le **même layout quantifié** (contrat des kernels MMQ) → zéro repack.

---

## 6. PROFILER-v4 (simulateur de machine, 4 niveaux)

```
LEVEL 1 microbench : SSD / DDR / PCIe / CUDA / XDNA
LEVEL 2 kernel bench : Q4 / Q5 / Q6 / Q8 / FP8 / NVFP4 (GEMM + dequant)
LEVEL 3 expert trace : routing / cache / DMA / dequant / GEMM
LEVEL 4 system model : critical path + BW contention
```

### Compteurs par expert/tensor (ajoutés à profiler-v3)
```
timestamp layer expert_id tensor_id
storage_precision compute_precision activation_precision
storage_location (SSD/RAM/VRAM)
cache_hit cache_miss
prefetch_predicted prefetch_correct
bytes_storage bytes_h2d bytes_dequant bytes_conversion
t_read_us t_pcie_us t_prefetch_us t_quant_us t_conversion_us
t_dequant_us t_gemm_us t_sync_us t_queue_wait_us t_eviction_us
eviction_reason hotness reuse_distance prediction_score
device kernel_id
critical_path_us hidden_transfer_us visible_transfer_us
```

### Table fondamentale par expert (le livrable central)
```
Quant | SSD B | RAM B | PCIe B | VRAM fp | dequant | GEMM | slots | accuracy | end-to-end
Q3    | ...   | ...   | ...    | ...     | ...     | ...  | ...   | ...      | ...
Q4    | ...   | ...   | ...    | ...     | ...     | ...  | ...   | ...      | ...
Q6    | ...   | ...   | ...    | ...     | ...     | ...  | ...   | ...      | ...
Q8    | ...   | ...   | ...    | ...     | ...     | ...  | ...   | ...      | ...
FP8   | ...   | ...   | ...    | ...     | ...     | ...  | ...   | ...      | ...
NVFP4 | ...   | ...   | ...    | ...     | ...     | ...  | ...   | ...      | ...
→ argmin(end_to_end_time) par expert, par couche, par workload (prefill vs decode), par état cache
```

---

## 7. PREFILL ≠ DECODE (deux modèles séparés)

- **Prefill** : batch élevé → GEMM énorme → transferts mieux amortis → le compute domine.
- **Decode** : petits lots → latence / PCIe / launch dominants → la précision/transfert importe.
⇒ "Q4 optimal en decode" ≠ "Q4 optimal en prefill". Le profiler produit DEUX modèles.

---

## 8. PLE (Qwen3.8) = problème séparé, même budget BW

```
                  Qwen Flash
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
         PLE                        MoE
          │                         │
   SSD/RAM bandwidth        expert bandwidth
          │                         │
          └────────────┬────────────┘
                       ▼
                 DDR contention
                       │
                 PCIe contention
                       │
                       ▼
                     RTX
```
PLE = 51B N-gram, lookup O(1), hash → illisible SSD (~7 t/s max). Le garder en RAM ou aux-GPU
(vLLM #53908). Même API de résidence : request_range/prefetch/promote/evict. (Sources :
vLLM #53908, solarkyle/qwen38-flashnext-16gb --ram-tensors, unsloth discussion.)

---

## 9. INTÉGRATION XDNA2 (le NPU dans le modèle)

Le planner compare 3 chemins complets par expert :
```
CPU : T_quant_CPU + T_transfer
XDNA: T_DMA + T_kernel + T_sync
RTX : T_H2D + T_dequant + T_GEMM
→ choisir le plus petit critical path, pas le meilleur TOPS
```
Le NPU peut : prédiction du routing (93-97% démontrés), requantification (51 TOPS INT8), layout
transform pendant DMA, index prep. PAS le moteur MoE (ratio 7.4× RTX/NPU mesuré).

---

## 10. ORACLE (ne pas profiler 18 × 48 × 512 × 3 × 3)

L'espace (18 formats × 48 couches × 512 experts × 3 tensors × 3 devices) explose.
1. microbenchmark (Level 1) → ThroughputModel
2. le planner PRÉDIT `Q3 = x ms, Q4 = y ms...` SANS exécuter tous les candidats.
3. calibration périodique (rouge → recalcule).

---

## 11. EXPERIENCES Q4/Q5/Q6/Q8/FP8/NVFP4 (grille Qwen Flash)

IQ2_XXS · IQ2_XS · Q2_K · IQ3_XXS · IQ3_S · Q3_K · Q3_K_M · Q3_K_L · Q4_K_S · Q4_K_M ·
IQ4_XS · IQ4_NL · Q5_K_S · Q5_K_M · Q6_K · Q8_0 · FP8 · FP4/NVFP4/MXFP4 (RTX compatible).

Politique cible :
```
hot expert → Q6/Q8/FP8      warm → Q4/Q5      cold → Q3/IQ2      very cold → IQ1
hot tensor → Q8             medium → Q6       cold → Q3
```

## 12. GATES DE VALIDATION (rappels PLAN_ADAPTIVE_RESIDENCY.md)
1. exactitude (tokens identiques OU PPL stable) AVANT perf
2. mémoire (VRAM/RAM < budget)
3. cache (hit/miss/bytes H2D par token)
4. recouvrement (hidden DMA %)
5. adaptivité (gain à budget VRAM identique)
6. robustesse (100+ prompts, contextes, sessions)

## 13. FICHIERS / LIVRABLES liés
- `RECHERCHE_NPU_RTX_URLS.md` — 27 points creusés + preuves/URLs
- `P0_MULMAT_ID_INSERTION.md` — dissection MUL_MAT_ID + points d'insertion (L2017 etc.)
- `PLAN_ADAPTIVE_RESIDENCY.md` — architecture cache hiérarchique + phases P0-P7
- `DOC_EXHAUSTIVE_5070_NPU_BUS.md` — bus mémoire 5070+NPU (21.93 GB/s, MCDM goulot)
- `PLAN_EXECUTION_5070_NPU.md` — plan exécution 4 phases
- `place_expert.py` / `cost_contention_patch.py` / `benchmark_cross_tier.py` — code D2

## 14. URLS DE PREUVE (supplémentaires pour ce doc)
- CUDA Best Practices (pinned, overlap, batching) : https://docs.nvidia.com/cuda/cuda-c-best-practices-guide
- NVIDIA overlap transfers : https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc
- FlashInfer (FP4/FP8/NVFP4/grouped GEMM SM120) : https://github.com/flashinfer-ai/flashinfer
- FlashInfer docs gemm : https://docs.flashinfer.ai/api/gemm.html
- FlashInfer fused_moe : https://docs.flashinfer.ai/api/fused_moe.html
- FlashInfer paper (achieved bandwidth util) : https://arxiv.org/pdf/2501.01005
- CUTLASS NVFP4 grouped GEMM SM120 : https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu
- MoE-Infinity : https://github.com/EfficientMoE/MoE-Infinity + https://arxiv.org/abs/2401.14361
- DynaExq : https://arxiv.org/abs/2511.15015
- TensorRT-LLM (W4A16/W8A8/FP8/NVFP4) : https://github.com/NVIDIA/TensorRT-LLM
- KTransformers : https://github.com/kvcache-ai/KTransformers

## 15. RENOMMAGE + MODÈLE DE DONNÉES (mise à jour 2026-09-20 — voir BLINDSPOTS_REPONSES.md)

Concept : **D2 Adaptive Precision & Dataflow Planner** (pas "Adaptive Quantization Engine").

### Structure finale
```
D2 PLANNER
 ├ PRECISION (W, A, state, KV precision)
 ├ RESIDENCY (SSD/RAM/VRAM, cache/prefetch, eviction)
 ├ DEVICE (CPU/XDNA/RTX, kernels, backend)
 ├ MODEL AWARENESS (MoE routing/shared | PLE lookup/shards | GDN/QSA/MTP recurrent+KV+spec)
 ├ RESOURCE MODEL (SSD DDR PCIe VRAM NPU — BW_available(t))
 ├ KERNEL MODEL (dequant GEMM launch — format×kernel×shape×commit)
 └ CRITICAL PATH → DECISION
```

### ExpertVariant / ResourceState
(voir BLINDSPOTS_REPONSES.md §modèle de données — champs complets expert_variant et
resource_state, dont hotness, sensitivity, reuse_distance, prediction_probability, location,
generation, et côté ressource : xdna_dma_bw, xdna_issue_us, xdna_sync_us, gpu_occupancy,
kv_usage, temperature, clocks, confidence.)

### Docs liées
- URLS_REGISTRY.md : ~50 sources classées, 12 noyau dur
- BLINDSPOTS_REPONSES.md : 66 angles morts → réponses + preuves/URLs
- BLINDSPOTS_HW.md : spécificités SM120/XDNA2/Qwen Flash
- P0_MULMAT_ID_INSERTION.md : hooks llama.cpp (L2017)