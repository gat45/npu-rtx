# MATRICE EXPÉRIMENTALE P0→P5 — D2 System-Aware Adaptive Precision & Residency Planner
# Établi 2026-09-20 · Dossier npu-rtx/ · Répond aux 50 questions + 12 non-prouvées
# Chaque ligne : question · benchmark · compteurs · variable · résultat attendu · critère de validation
# Légende statut : ✅ prouvé (code/mesure) · 🟡 prouvé ailleurs · 🔴 à mesurer sur Qwen3.8+5070+XDNA2

---

## P0 — RÉFÉRENCES ET FONDATION (pas encore de variable de précision)

### P0.1 — Matrice des transferts 3×3 (la base de TOUT le modèle de coût)
| Champ | Valeur |
|---|---|
| Question | Quelles sont les vraies BW/latences CPU-DDR / NPU-DDR / RTX-DDR / NPU-RTX sur la machine cible ? |
| Benchmark | microbench 1K/4K/16K/64K/256K/1M/4M/16M/64M/256M/1G, directions TX/RX, queue depth 1-16 |
| Compteurs | latency_ms, GB/s effectif, cpu_util%, overlap_possible, sync_us, jitter |
| Variable | taille × direction × concurrency |
| Attendu | courbe BW(size) ; PCIe ~25 GB/s effective (pas 32), DDR NPU 21.93 GB/s (mesuré), VRAM 672 (théo) |
| Critère | matrice remplie sur la machine cible = prérequis ; 🔴 rien de crédible avant |

### P0.2 — Hardware Throughput Matrix (débits réels des kernels)
| Champ | Valeur |
|---|---|
| Question | Quel débit effectif par kernel (Q4/Q6/Q8/FP8/NVFP4) sur RTX 5070, CPU, XDNA2 ? |
| Benchmark | FlashInfer benchmark framework (octets réels / temps kernel, CUPTI) + llama-bench + mesures XRT |
| Compteurs | dequant_GBs, gemm_TFLOPS, fused_TFLOPS, occupancy, registers/thread, SMEM, warps |
| Variable | format × kernel × shape (M=1 decode vs M≫ prefill) |
| Attendu | — |
| Critère | table remplie ; **format×kernel×shape×arch** comme clé (FlashInfer API unifiée = preuve qu'aucun backend n'est universel) |

### P0.3 — Dissection MUL_MAT_ID (déjà fait, P0_MULMAT_ID_INSERTION.md)
| Champ | Valeur |
|---|---|
| Question | Où brancher le cache sans réécrire ggml ? |
| Réponse | **L2017 ggml-cuda.cu** `src0_slice.data = src0->data + i02*nb02` = offset expert = hook slot-cache ; 2 synchronize fallback = évitables ; mmid.cu tri GPU sans sync |
| Critère | ✅ démontré par dissection du code local (ab-wt) |

### P0.4 — Table Offset/Bytes des experts (P0 du plan original)
| Champ | Valeur |
|---|---|
| Question | "Expert 17 couche 23 = exactement ces plages d'octets" |
| Benchmark | lecture GGUF + ggml_tensor (nb[2] = nb02) |
| Compteurs | layer, tensor, expert, shape, QType, offset, bytes |
| Variable | — |
| Attendu | Qwen3.8 : expert ≈ 4.9M params (~2.45 MB Q4 / NVFP4), 512×48=24 576 experts ≈ 60 GB |
| Critère | table générée automatiquement |

---

## P1 — CACHE PERSISTANT (reproduire la preuve llama.cpp, première variable)

### P1.1 — Courbe slots (le +84% à reproduire)
| Champ | Valeur |
|---|---|
| Question | Combien de slots dans 8 GB et quel gain ? |
| Benchmark | llama-bench, slots 4/8/16/32/64/128, Qwen3.8-Flash-Next UD-Q4_K_XL, -ngl 99 --n-cpu-moe |
| Compteurs | hit_rate, miss_rate, bytes_H2D/token, tok/s, VRAM_used |
| Variable | N slots |
| Attendu | **Référence RTX 4090 : 11.55→21.21 t/s avec 64 slots (+84%), hit 90-95%, +34% codegen** (RFC #28248) — à adapter 5070 (VRAM 8 GB, kernels différents) |
| Critère | hit ≥ 90% ET tok/s > baseline sans cache ; **cache < working set = contre-productif (FATE 0% hit = 1.14 vs 6.4 t/s)** |
| URLs | llama.cpp #28248, #20757, PR #26563, FATE, vLLM #37190 |

### P1.2 — LRU vs LFU vs LFRU vs ML vs D2 oracle
| Champ | Valeur |
|---|---|
| Question | Quelle politique d'éviction gagne sur Qwen3.8 ? |
| Benchmark | mêmes traces de routing, 5 politiques |
| Compteurs | hit_rate, bytes_H2D/token, tok/s, evictions, victim_reload_cost |
| Variable | politique (LRU baseline / LFU / LFRU / ML recency+freq / D2) |
| Attendu | LFRU > LRU (vLLM #38256 : évite que les 1res couches monopolisent) ; FlashMoE ML +51% hit vs LRU/LFU (arXiv:2601.17063) |
| Critère | hit_rate et tok/s supérieurs, écart significatif (n≥3 runs) |

### P1.3 — Cache global vs par couche vs hybride
| Champ | Valeur |
|---|---|
| Question | Un pool global ou des pools par couche ? |
| Benchmark | 3 configs, même traces |
| Compteurs | hit, thrash, bytes_H2D |
| Variable | global / per-layer / hybrid |
| Attendu | 🔴 aucun gagnant universel (DynaExQ = per-layer résident ; vLLM = global) |
| Critère | mesuré sur Qwen3.8 |

---

## P2 — DMA ASYNC + PINNED (masquer le transfert)

### P2.1 — Pinned vs pageable
| Champ | Valeur |
|---|---|
| Question | Le H2D pinned est-il requis pour l'overlap ? |
| Benchmark | cudaMemcpy vs cudaMemcpyAsync, pageable vs pinned, 1M/16M/64M |
| Compteurs | GB/s, overlap%, cpu_util |
| Variable | pinned × async × taille |
| Attendu | **✅ CUDA : pinned = BW max H2D (~12 GB/s PCIe x16 Gen3), async + streams = overlap** (CUDA Best Practices §10.1) |
| Critère | pinned requis pour l'overlap (CUDA : async exige pinned) |

### P2.2 — Double buffering (GEMM(A) ‖ DMA(B))
| Champ | Valeur |
|---|---|
| Question | T_DMA ≤ T_GEMM atteignable ? |
| Benchmark | 2 streams, slots A/B |
| Compteurs | hidden_transfer_us, visible_transfer_us, critical_path_us |
| Variable | préchargement slot suivant |
| Attendu | transfert masqué si T_DMA ≤ T_GEMM ; **un transfert 300µs masqué par 500µs de compute n'est PAS un problème** |
| Critère | visible_transfer ≈ 0 |

### P2.3 — Budget pinned séparé
| Champ | Valeur |
|---|---|
| Question | Combien de pinned sans dégrader le système ? |
| Benchmark | 0/1/2/4/8 GB pinned |
| Compteurs | RAM libre, latence système, GB/s H2D |
| Variable | quantité pinned |
| Attendu | **❌ ne pas tout pinner : NVIDIA = ressource rare, surusage dégrade le système** (CUDA BP) ; expert_cache_RAM ≠ pinned_staging_RAM |
| Critère | 2 budgets séparés dans ResourceState |

---

## P3 — PRÉDICTION / PREFETCH (3e variable)

### P3.1 — Prédictibilité du routing Qwen3.8
| Champ | Valeur |
|---|---|
| Question | P(E_l+1 | E_l) et taux de prédiction cross-layer sur Qwen3.8 ? |
| Benchmark | traces de routing, predictor LLaPor-style / FATE cross-layer + temporal |
| Compteurs | prediction_accuracy, P(use|predicted), false_positive_rate, prefetch_correct |
| Variable | predicteur 0/1/2/3/4 (fréq → transition → cross-layer → activation → hybride) |
| Attendu | **Références autres modèles : 93.03% DeepSeek-V2 Lite, 94.69% Qwen3-30B, 97.62% Phi-mini-MoE** (arXiv:2511.10676) ; FATE ~50% du PCIe éliminé |
| Critère | 🔴 à mesurer sur Qwen3.8 ; P(use) ≥ 0.8 pour activer le prefetch |

### P3.2 — Coût d'un faux positif (prefetch pollution)
| Champ | Valeur |
|---|---|
| Question | Un mauvais prefetch est-il pire que rien ? |
| Benchmark | predicteur avec taux de faux positifs 0/10/30/50% |
| Compteurs | prefetch_value = P(use)×saved_latency − transfer_cost − eviction_cost |
| Variable | taux de faux positifs |
| Attendu | **✅ risque réel** : victime éjectée → miss suivant (FATE, MoE-Infinity admission conservatrice) |
| Critère | prefetch activé seulement si valeur > 0 |

### P3.3 — Nombre d'experts spéculatifs
| Champ | Valeur |
|---|---|
| Question | 1/2/4/8 experts préchargés spéculatifs ? |
| Benchmark | decode_expert_prefetch_top_k 1/2/4/8 (vLLM #38256) |
| Compteurs | tok/s, hit, bytes_H2D |
| Variable | top_k prefetch |
| Attendu | 🔴 optimal inconnu ; MoE-Infinity borne la fenêtre par bytes inflight |
| Critère | benchmark |

---

## P4 — PRÉCISION DYNAMIQUE (DynaExQ-style, 4e variable)

### P4.1 — Q4→Q6 online vs Q6 stocké
| Champ | Valeur |
|---|---|
| Question | La requant runtime est-elle plus rapide que charger la variante ? |
| Benchmark | BF16→Q6 vs Q4→Q6 vs Q6 stocké, même expert |
| Compteurs | quant_GBs, transfer_saved, dequant_saved, quality_delta |
| Variable | chemin de conversion |
| Attendu | **🔴 non démontré ; ⚠️ --allow-requantize dégrade la qualité vs direct 16/32-bit** (llama.cpp tools/quantize/README) ; KTransformers = online BF16→INT4/INT8 au chargement (pas par miss) |
| Critère | T_requant+transfer < T_stocké+transfer ET quality_delta acceptable — sinon stocker la variante |

### P4.2 — Hotness vs sensitivity (la bonne paire de variables)
| Champ | Valeur |
|---|---|
| Question | Un expert fréquent doit-il être haute précision ? |
| Benchmark | calibration : hotness, sensitivity (variation router + variance intra-neurone), reload_cost par expert |
| Compteurs | activation_hotness, quality_sensitivity, transfer_cost, value = hotness×sensitivity×reload_cost |
| Variable | budget haute précision par couche |
| Attendu | **❌ hot ≠ important** : littérature MoE 2026 montre que fréquence ≠ sensibilité (arXiv:2604.06515) ; AWQ : canaux saillants (arXiv:2306.00978) |
| Critère | PPL/KL après allocation budget |

### P4.3 — Granularité expert vs tensor vs tile
| Champ | Valeur |
|---|---|
| Question | Faut-il descendre sous l'expert ? |
| Benchmark | allocation par expert / par tensor / par block |
| Compteurs | PPL, tok/s, VRAM |
| Variable | granularité |
| Attendu | **🟡 expert/tensor démontré (DynaExQ, GGUF mixte vumpt) ; tile 🔴 non prouvé Qwen3.8** |
| Critère | 🟡 benchmark |

### P4.4 — Hystérésis + versioning
| Champ | Valeur |
|---|---|
| Question | Comment éviter l'oscillation Q4↔Q6 ? |
| Benchmark | transitions avec promote_threshold < demote_threshold + minimum_residency + cooldown ; ExpertHandle versionnés |
| Compteurs | oscillations, transition_atomic (drain lecteurs avant libération) |
| Variable | hystérésis, cooldown |
| Attendu | **✅ DynaExQ : pools sans fragmentation + transitions async versionnées + déterminisme** (arXiv:2511.15015) |
| Critère | zéro oscillation ; déterministe |

---

## P5 — INTÉGRATION COMPLÈTE (Qwen3.8 + 5070 + XDNA2 + PLE + GDN + QSA)

### P5.1 — Qwen3.8 complet (les 6 classes d'objets)
| Champ | Valeur |
|---|---|
| Question | Quel budget VRAM pour poids + KV QSA + GDN state + workspace + activations + PLE + expert cache ? |
| Benchmark | Qwen3.8 UD-Q4_K_XL / NVFP4 NVIDIA, ctx 8K/32K/128K |
| Compteurs | vram_usage par composant, tok/s, TTFT |
| Variable | répartition budget |
| Attendu | **⚠️ budget = VRAM − dense − GDN_state − QSA_KV − workspace − activations − PLE** ; KV QSA FP8 = +1.889× pool, NVFP4 = +3.059× (patch SM120) ; GDN state = 15 GB @ batch256 (DAMP sur Qwen3.6 — 🟡 pas Qwen3.8) |
| Critère | 6 compteurs VRAM séparés dans ResourceState |

### P5.2 — PLE manager séparé
| Champ | Valeur |
|---|---|
| Question | PLE = lookup, pas GEMM : comment le résider ? |
| Benchmark | PLE résident BF16 / Q5_0 / NVFP4 / aux-GPU / SSD à la demande |
| Compteurs | lookup_us, bytes_read, amplification, RAM_usage |
| Variable | résidence PLE |
| Attendu | **PLE = 320 001 536 lignes × 160 = 51.2B params, 128 shards** ; hashing → illisible SSD (~7 t/s) ; llama.cpp #27864 (lectures lignes à la demande) ; vLLM #53908 (aux-GPU) |
| Critère | PLEPager séparé de ExpertPager, budget partagé DDR/SSD/RAM/PCIe |

### P5.3 — MUL_MAT_ID : la matrice de compatibilité (le piège #1506)
| Champ | Valeur |
|---|---|
| Question | Quels formats marchent sur mul_mat_id CUDA, par shape ? |
| Benchmark | test-backend-ops MUL_MAT_ID, formats Q4_K/Q5_K/Q6_K/Q8_0/IQ/NVFP4, shapes expert |
| Compteurs | correctness (bit-exact vs CPU), CUDA errors |
| Variable | format × shape × commit |
| Attendu | **⚠️ ggml #1506 : Q4_K/Q5_K/Q6_K incorrects sur mul_mat_id, Q8_0 OK → l'oracle doit TESTER** ; #21289 (arch CUDA 520), #24591 (IDs dupliqués) |
| Critère | matrice de capacités générée par test, jamais supposée |

### P5.4 — XDNA2 : calcul vs contrôleur (la décision finale)
| Champ | Valeur |
|---|---|
| Question | Le NPU calcule-t-il ou contrôle-t-il ? |
| Benchmark | A : RTX experts seul · B : XDNA predictors + RTX experts · C : XDNA experts partiels + RTX |
| Compteurs | tok/s, T_submit+T_DMA+T_compute+T_sync XDNA, fenêtre prefetch |
| Variable | rôle XDNA |
| Attendu | **baseline : XDNA = planner/prefetch, RTX = GEMM** (ratio 7.4× RTX/NPU mesuré) ; IRON runtime sequences dynamiques = contrôle crédible |
| Critère | B > A ET C > A sinon garder contrôleur |

### P5.5 — MTP (hors MVP, après cache stable)
| Champ | Valeur |
|---|---|
| Question | MTP3 change-t-il l'optimum cache/précision ? |
| Benchmark | MTP off/1/2/3, mêmes traces |
| Compteurs | commit/rollback cache+KV+GDN, tok/s |
| Variable | n MTP |
| Attendu | **⚠️ MTP = draft/verify/rollback → caches et états doivent gérer commit/rollback** (Qwen MTP 4B, NVIDIA Dynamo MTP3) |
| Critère | comparer après P4 stable |

### P5.6 — Quantification des états GDN (angle DAMP)
| Champ | Valeur |
|---|---|
| Question | Le state GDN peut-il être BF16/FP8/INT8/mixte ? |
| Benchmark | state GDN en BF16/FP8/INT8/mixte par énergie d'erreur + persistance canal |
| Compteurs | bytes/token, update_bw, dequant_cost, state_error, long-context_drift |
| Variable | précision du state |
| Attendu | **🟡 DAMP (decay-aware) : states GDN/KDA = memory-BW bound, quantifiables** (Qwen3.6 batch256 : 15 GB states, 24.3% latence — pas Qwen3.8) |
| Critère | drift ≤ seuil sur ctx 128K |

---

## LES 12 NON-PROUVÉES — SYNTHÈSE (réponse directe avec état)

| Question | État | Expérience qui la tranche |
|---|---|---|
| Q4→Q6 online > Q6 stocké ? | 🔴 | P4.1 |
| Quel format gagne sur 5070 (Q4_K/IQ4/NVFP4/FP8) ? | 🔴 | P0.2 + P1.1 |
| Quel format en decode M=1 ? | 🔴 | P0.2 (M=1 colonne) |
| Quel format en prefill ? | 🔴 | P0.2 (M≫ colonne) |
| Meilleure granularité expert/tensor/tile ? | 🔴 | P4.3 |
| Quelle politique cache sur Qwen3.8 ? | 🔴 | P1.2 |
| Taux prédiction cross-layer Qwen3.8 ? | 🔴 | P3.1 |
| XDNA utile au calcul ou scheduling ? | 🔴 | P5.4 |
| XDNA quantifie mieux que CPU/GPU ? | 🔴 | P0.2 (XDNA colonne) |
| PLE + expert cache combinés ? | 🔴 | P5.2 |
| Précision GDN state viable Qwen3.8 ? | 🔴 | P5.6 |
| D2 bat le cache llama.cpp actuel ? | 🔴 | P5 complet vs baseline #28248 |

## CRITÈRES DE VALIDATION TRANSVERSAUX (gates)
1. exactitude : tokens identiques OU PPL stable, AVANT perf (P4.1/P4.4)
2. mémoire : VRAM < budget DYNAMIQUE (WDDM : budget résidence variable — Microsoft)
3. cache : hit/miss/bytes H2D par token
4. recouvrement : hidden DMA % (P2.2)
5. adaptivité : gain à budget VRAM identique
6. robustesse : 100+ prompts, ctx 8K-128K, batch 1/2/4/8, MTP off/1/2/3
7. format×kernel×shape×commit : matrice de compatibilité testée (P5.3, piège #1506)

## RENOMMAGE FINAL
**D2 — System-Aware Adaptive Precision & Residency Planner**
- Niveau 1 QUANTIZATION (Q1-Q8/IQ/INT/FP4/FP8/NVFP4)
- Niveau 2 RESIDENCY (SSD/RAM/PINNED/VRAM/L2/shared/register)
- Niveau 3 DATAFLOW (prefetch/prediction/DMA/PCIe/contention/overlap)
- Niveau 4 COMPUTE (CPU/XDNA2/RTX/kernel/GEMM/dequant)
- **La quantification est une variable du dataflow, pas une propriété fixe du modèle.**

## SOURCES (liens dans URLS_REGISTRY.md)
llama.cpp #28248 · #20757 · #27149 · #27864 · #18758 · ggml #1506 · DynaExQ (2511.15015) ·
MoE-Infinity (2401.14361) · vLLM #38256 · HOBBIT (2411.01433) · FlashMoE (2601.17063) ·
FlashInfer · TensorRT-LLM · DAMP · IRON/MLIR-AIE · CUDA Best Practices · NVIDIA Blackwell ·
Microsoft WDDM · AWQ (2306.00978) · arXiv:2605.07260 · arXiv:2604.06515 · arXiv:2511.10676