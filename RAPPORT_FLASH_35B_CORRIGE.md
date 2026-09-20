# RAPPORT FLASH 35B — CORRECTIONS (66 angles morts) + CIBLE MATÉRIELLE
# Établi 2026-09-20 · Dossier npu-rtx/ · Remplace/complète RAPPORT_FLASH_QWEN36_35B.md
# ⚠️ CIBLE MATÉRIELLE : Ryzen 9 HX 365 (XDNA2) + RTX 5070 8 GB.
# La GTX 1080 (détectée sur la machine dev) = UNIQUEMENT machine de test logique,
# PAS la cible. Ne jamais extrapoler 1080 → 5070 (sm_61 → sm_120).

---

## 0. CIBLE MATÉRIELLE (verrouillée)

| Composant | Cible réelle | Machine dev (test logique uniquement) |
|---|---|---|
| GPU | **RTX 5070 8 GB (Blackwell sm_120, 384 GB/s)** | GTX 1080 (Pascal sm_61) |
| NPU | **XDNA2 (Ryzen 9 HX 365, INT8 P0 / BFP16 P0bis)** | aucun |
| RAM | 32 GB DDR5 | 32 GB (7.5 dispo) |
| Budget VRAM | **8 - 1.5 WDDM ≈ 6.5 GiB** | idem |

⚠️ Les mesures de la machine dev (DDR 49.6, SSD 12.3 GB/s) sont des ORDRES DE GRANDEUR.
Les valeurs de la cible (PCIe 5070, NPU XDNA2, DDR) doivent être MESURÉES sur la machine
cible via hw_discovery + profiler-v3. Provenance obligatoire (MEASURED/DERIVED/ASSUMED/UNKNOWN).

---

## 1. CORRECTION DE NOMENCLATURE (angle mort n°0)

- Sources officielles actuelles : **Qwen3.5-35B-A3B** (config `qwen3_5_moe`).
- Discussions llama.cpp utilisent aussi **Qwen3.6-35B-A3B** (branches expérimentales).
- ⚠️ **Verrouiller le checkpoint EXACT testé avant de figer le Static Oracle.**
  Deux modèles proches mais différents (hidden/layers/vision peuvent différer).

## 2. LES 12 CORRECTIONS MAJEURES (angles morts 1-12)

| # | Angle mort | Correction |
|---|---|---|
| 1 | "Q4 = taille unique" | ❌. GGUF réels : Q4_K_M 22.29 GB · Q4_K_S 21.49 · Q4_0 20.84 · IQ4_XS 19.70 · Q3_K_XL 18.22. **Extraire du GGUF réel** : tensor name/type/dims/offset/size/quant/alignment → expert_bytes[layer][id] |
| 2 | Modèle multimodal | config a `vision_config` (depth 27, hidden 1152, interm 4304). **Deux profils : Q35-TEXT et Q35-MM**. Ne pas attribuer au MoE la mémoire de l'encodeur visuel |
| 3 | "3B actifs" ≠ bytes déplacés | `active_parameters` = compteur computationnel. La bonne mesure = **bytes actually read** par layer/expert/tensor/token |
| 4 | Shared expert | 8 routed + 1 shared. **shared = ALWAYS résident** (jamais routé normal) → test "shared pinned vs evictable" (P0-18) |
| 5 | 40 couches ≠ identiques | config : linear×3 + full_attention répété 10× → 30 GDN + 10 Attention. `layer_type[layer_id]` obligatoire |
| 6 | État DeltaNet | 16 key heads / 32 value heads / head_dim 128. Mesurer **DeltaNet state ≠ Attention KV** séparément. PAS de formule KV Transformer × 40 |
| 7 | Contexte 262K | ≠ gratuit avec 6.5 GiB. Tester 4K→256K, mesurer VRAM/RAM/state/KV/workspace/TTFT/decode |
| 8 | Cache ≠ forcément LRU | Comparer LRU/LFU/LFU-aging/SLRU/freq+recency/oracle |
| 9 | Localité inter-token | Mesurer `reuse_distance` par expert (ex. expert 37 : 12843 usages, mean 3.7 tokens) |
| 10 | Cache peut empirer | ❌ "cache = toujours mieux". Tests publics montrent des régressions. **D2 doit pouvoir décider CACHE=OFF** |
| 11 | Cache vole la VRAM du compute | `cache_budget = VRAM_budget − weights − KV/state − workspace_peak − runtime` PAS "6.5 − weights" |
| 12 | SSD ≠ aléatoire idéal | Distinguer seq/random/qdepth/4K/64K/1MB/multi-MB + cold/page-cache/warm |

## 3. CORRECTIONS 13-20 (I/O et mémoire)

| # | Angle mort | Correction |
|---|---|---|
| 13 | mmap | **Variable expérimentale T13 mmap ON / T14 mmap OFF** (retours publics : 11→1 tok/s sur 397B avec mmap mal configuré) |
| 14 | Page cache | Distinguer SSD physical I/O vs page cache hit (cold/warm/drop-cache) |
| 15-16 | Prefetch | prefetch ≠ toujours bon : mesurer prefetch_accuracy, prefetch_waste_bytes, prefetch_latency ; tester P0 none/P1 next-layer/P2 next-token/P3 predicted/P4 co-occurrence |
| 17 | Pageable vs pinned | Comparer pageable/pinned/registered + sync/async H2D |
| 18 | PCIe théorique | Mesurer H2D/D2H réels + latence petit/gros transfert (4K→64M) |
| 19 | Petits transferts | T_transfer = latency + bytes/BW (GB/s insuffisant pour experts de quelques MiB) |
| 20 | DMA ≠ PCIe | Séparer app memcpy / CUDA H2D / DMA / PCIe transaction ; profiler host_submit/DMA_start/DMA_end/GPU_visible |

## 4. CORRECTIONS 21-30 (pipeline, routing, KV)

| # | Angle mort | Correction |
|---|---|---|
| 21 | Synchronisation | overlap_ratio = métrique obligatoire (cudaStreamSync/event/host callback sérialisent) |
| 22 | CPU routing | Mesurer routing_us/topk_us/expert extraction/scheduling_us (peut dominer le transfert) |
| 23-24 | top-k / dupliqués | Mesurer **unique_routed_experts_per_layer** (pas 8 × même taille) |
| 25 | shared+routed fusion | Mesurer shared_compute / routed_compute / fusion |
| 26 | batch | batch 1/2/4/8 (transfert change entre decode/prefill) |
| 27 | decode vs prefill | **Static Oracle decode ≠ prefill** (prefill touche plus d'experts, éviction du hot set — règle spéciale requise) |
| 28 | longueur prompt | 128/512/2K/8K/32K/128K/256K : PP/TG/VRAM/cache hit |
| 29 | KV quantifié | Intégrer KV Q8/Q6/Q5/Q4 au planner (KV mémoire ↓ → expert cache ↑) |
| 30 | cache vs KV = même VRAM | D2 résout **cache experts = X, KV = Y** simultanément (pas indépendants) |

## 5. CORRECTIONS 31-45 (hardware + conversions)

| # | Angle mort | Correction |
|---|---|---|
| 31 | RTX : compute vs transfert | Comparer effective token latency (compute+movement+scheduler), pas TOPS |
| 32 | XDNA2 ≠ accélérateur MoE auto | Séparer **XDNA2 hardware capability** de **OGA model/runtime capability** ; pas utiliser OGA comme preuve kernel MoE |
| 33 | XDNA2 à mesurer au niveau DMA | xrt-smi → hw_discovery ; XRT synthetic ≠ your kernel measured |
| 34 | 64 KiB L1 | Distinguer capacity / usable / DMA burst / alignment / bank conflicts / double buffering |
| 35 | Conversion NVFP4→INT8 | Convert once (cached) vs convert every use. **Cache multi-format : expert Q4 + expert INT8** |
| 36 | Cache multi-format | SSD Q4 → RAM Q4 → VRAM INT8 → XDNA INT8 (storage→transport→compute) |
| 37 | Double cache | L0 VRAM + L1 RAM + L2 SSD (architecture GPU+RAM public, SSD = ta zone d'extension) |
| 38 | SSD comme L3 | RFC publique : "No SSD tier, No prefetch" = limitations → **ton extension** |
| 39 | Granularité cache | whole expert / tensor / matrix / block / tile (P2) |
| 40 | Compression SSD | Q4 vs Q3 vs IQ4 + décompression vs plus de SSD traffic |
| 41 | GGUF pas idéal Flash | Envisager expert_index.bin + expert_NNNNN.bin (offset/length/quant/checksum) |
| 42 | Filesystem | NTFS/ReFS/ext4, direct vs buffered I/O, filesystem cache ≠ Flash cache |
| 43 | Intégrité experts | requested_expert/loaded_expert/slot/generation (checksum/version) |
| 44 | Correction numérique | logits max diff, KL, perplexity, token agreement (Q4→RTX vs Q4→INT8→XDNA2) |
| 45 | Benchmark reproductible | model hash, GGUF hash, llama.cpp commit, driver, CUDA, XRT, firmware, OS, governor, clocks, NPU pmode, prompt, seed, temp, ctx, batch |

## 6. CORRECTIONS 46-57 (oracle + cache)

| # | Angle mort | Correction |
|---|---|---|
| 46 | Static Oracle ne "prédit" pas le cache | Produire theoretical traffic PUIS measured cache trace ; 90% vient de la trace, pas du benchmark précédent |
| 47 | 90ms→6.6ms ≠ preuve 35B | Conserver MEASURED/DERIVED/ASSUMED dans chaque sortie |
| 48 | hit rate insuffisant | Ajouter **byte_hit_rate** = hit_bytes/requested_bytes (2 caches à 90% hit ≠ mêmes perfs) |
| 49 | hit-rate par couche | layer 0:97%... layer 39:71% → planner peut cacher couches 0-20 seulement |
| 50 | cache par expert vs par couche | `cache[layer][expert]` |
| 51 | Planner déplace les experts | Expert A→VRAM, B→RAM, C→SSD, D→XDNA2 staging (freq/reuse/size/conversion/compute) |
| 52 | Coût du planner | replanning_interval + planner_overhead_us (ne pas replan chaque token) |
| 53 | Warm-up | cold-start/warm-up/steady-state séparés (6.6ms peut cacher 500ms warmup) |
| 54 | Changement de tâche | hot set varie : Python/math/code/conversation/raisonnement |
| 55 | Prefill pollution | **prefill cache policy ≠ decode cache policy** (gros prompt expulse le hot set decode) |
| 56 | Shared pinned | Test P0-18 : shared pinned vs evictable |
| 57 | MTP | config `mtp_num_hidden_layers = 1` → profiler MTP ON/OFF |

## 7. CORRECTIONS 58-66 (modèle complet + règle n°1)

| # | Angle mort | Correction |
|---|---|---|
| 58 | Vision | TEXT / IMAGE / VIDEO = 3 profils distincts |
| 59 | Validation publique | llama.cpp RFC : CPU-resident experts + persistent GPU cache + ID remapping + VRAM budget + hit-rate, **MAIS "No SSD tier, No prefetch, No imatrix pinning" = ta zone d'extension** |
| 60 | Nouveau schéma | GGUF/index → SSD/L3 → RAM/L2 → RTX VRAM L1 / XDNA2 local → COMPUTE |
| 61 | Métriques profiler-v3 | + expert_access_count, expert_unique_count, byte_hit_rate, count_hit_rate, reuse_distance, cooccurrence, SSD/RAM/PCIe bytes+us, conversion_us, RTX/XDNA compute, routing/scheduler/sync, prefetch requested/useful/wasted, VRAM/RAM peak, KV, GDN state, workspace, cold/warm/steady latency |
| 62 | 20 tests P0 (au lieu de 10) | P0-01..P0-20 (voir §8) |
| 63 | Questions finales | Voir liste (architecture/cache/SSD/PCIe/GPU/XDNA2/D2) |
| 64 | **Règle n°1** | **Ne jamais optimiser le nombre de paramètres. Optimiser le chemin des octets.** (expert_id → tensor → bytes → location → transfer → conversion → compute → reuse) |
| 65 | Corrections Static Oracle | + source_type(measured/derived/assumed/unknown), confidence, measurement_method, model_hash, tensor_hash, expert[layer][id]{bytes,offset,quant,tensor_count}, cache_model{hit,byte_hit,reuse,policy}, movement{SSD,RAM,PCIe,DMA} |
| 66 | D2 complet | T(plan)=T_route+T_SSD+T_RAM+T_PCIe+T_DMA+T_conversion+T_compute+T_sync−T_overlap SOUS VRAM/RAM/SSD_BW/PCIe_BW/L1/workspace |

## 8. LES 20 TESTS P0 (format de trace IDENTIQUE pour chacun)

| Test | Variable |
|---|---|
| P0-01 | RTX dense |
| P0-02 | RTX MoE |
| P0-03 | 35B cold (cache 0) |
| P0-04 | 35B warm |
| P0-05 | cache 25% |
| P0-06 | cache 50% |
| P0-07 | cache 75% |
| P0-08 | cache 90% |
| P0-09 | cache 95% |
| P0-10 | cache 99% |
| P0-11 | LRU vs LFU |
| P0-12 | mmap ON/OFF |
| P0-13 | pageable/pinned |
| P0-14 | prefetch OFF/ON |
| P0-15 | prefetch accuracy |
| P0-16 | Q4/Q6/Q8 |
| P0-17 | KV Q4/Q8 |
| P0-18 | shared pinned/evict |
| P0-19 | RTX vs XDNA2 |
| P0-20 | RTX + XDNA2 |

## 9. ÉTAT — CE QUI EST CORRIGÉ / À CORRIGER DANS LE CODE

| Fichier | Correction à faire |
|---|---|
| `models/qwen36_35b_a3b/config.json` | ⚠️ verrouiller Qwen3.5 vs Qwen3.6 (checkpoint exact) ; + vision_config (Q35-MM) |
| `static/static_oracle_35b.py` | + source_type/confidence/model_hash sur chaque sortie ; + byte_hit_rate ; + layer_type[layer] |
| `oracle/feasibility.py` | + cache_budget = VRAM−weights−KV−workspace (pas 6.5−weights) ; + contrainte L1 usable |
| `oracle/d2_planner.py` | + T(plan) complet avec T_overlap ; + CACHE=OFF possible ; + planner_overhead |
| `collectors/hw_discovery.py` | cible = Ryzen 9 HX 365 + 5070 8GB (le 1080 = dev seulement) ; + xrt-smi (P0-19) |
| `static/conversion_matrix.py` | + convert once (cached) vs convert every use ; + cache multi-format |

## 10. SOURCES CLÉS (voir URLS_REGISTRY §29)
- Qwen3.5-35B-A3B officiel : huggingface.co/Qwen/Qwen3.5-35B-A3B (+ config.json)
- Qwen3.6-35B-A3B : huggingface.co/Qwen/Qwen3.6-35B-A3B (branches expérimentales llama.cpp)
- GGUF Qwen3.6 : Infatoshi, bartowski, notanoption1, 0xSero (tailles Q4/Q3 réelles)
- llama.cpp RFC : persistent expert cache, GPU expert cache, two-tier GPU+RAM (#28248/#20757)
- AMD OGA Hybrid + XRT/NPU management (flux officiel NPU+iGPU, limites)