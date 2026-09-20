# RÉFÉRENCE RÉSULTATS PUBLICS Qwen3.8-Flash-Next + PLAN DE REPRODUCTION
# Établi 2026-09-20 · Dossier npu-rtx/ · Croisement des résultats réellement publiés
# ⚠️ profiler-v3/V5 = SNAPDRAGON (HTP/FastRPC). Cible ici = XDNA2 + RTX 5070.
# → V5 = socle logique (réutiliser), MAIS les collecteurs/benchmarks sont à refaire pour
#   XDNA2/RTX (jamais dans profiler_v3, toujours dans npu-rtx/).

---

## 0. CONCLUSION PRINCIPALE

Qwen3.8-Flash-Next fonctionne déjà de ~2.6 bpw à ~5 bpw, mais les **meilleurs résultats**
viennent de modifications SIMULTANÉES du working set, du PLE, du cache, du KV/GDN et du
pipeline — **pas d'un choix Q4/Q6**. Certains résultats sont CONTRADICTOIRES selon le
matériel → profiler-v3 doit mesurer la machine, pas copier une recette.

---

## 1. CARTE DES RÉSULTATS PUBLICS (ce que chaque projet PROUVE)

| Projet | Matériel | Représentation | RAM/VRAM | Résultat | Preuve |
|---|---|---|---|---|---|
| HaberstrohSystems (SGLang) | RTX PRO 4000 24 GB + 32 GB RAM | **2.572 bpw** (2-bit experts g128, INT8 dense, 16-bit router/norm) | 24+32 GB | **54-58 tok/s**, prefill 2271 t/s | 🔴 → 🟢 : possible TRÈS bas en bits si runtime+paging adaptés |
| llama.cpp expert cache | RTX 4090 24 GB | UD-Q3_K_XL | experts CPU + cache VRAM | **11.55 → 21.21 tok/s (+84%)** | ✅ cache expert persistant = levier démontré |
| lukaLLM | RTX PRO 6000 96 GB RAM / GPU 8 GB | GGUF | **PLE host** | **~36 tok/s avec 8 GB GPU** | ✅ gros levier = NE PAS mettre PLE sur GPU |
| Weschera | DGX Spark 121 GB | UD-Q4_K_XL | ~95 GB | 47-48 tok/s code | 🟡 MTP ≈ 2× code, 0% prose, négatif si draft long |
| NVIDIA/Tony | DGX Spark 128 GB | NVFP4 officiel | PLE sur NVMe, FP8 KV, MTP3 | **43.9 tok/s médian** | ✅ NVFP4 officiel + PLE disk + FP8 KV + MTP3 = une recette complète |
| Starkweather | DGX Spark 121 GB | NVFP4 custom (PLE aussi NVFP4) | 109 GB | single-GPU | ✅ PLE BF16 102 GB → NVFP4 28.8 GB (E2M1 + scale FP8/16) |
| MiaAI-Lab | DGX Spark GB10 | FP8 KV + GDN BF16 | working-set adapté | **+8.5% decode 8 streams ; KV ~1.8-1.9× pool** | 🟡 KV/state = leviers majeurs ; prefill FP8 -6.1% = trade-off |
| Agention/AP | GGUF | mixed precision tensorielle | 49-77 GB GPU | 3.69-5.46 bpw | 🟡 plus de précision/Go par allocation mixte |
| AtomicChat | GGUF/imatrix | IQ4_XS→Q5_K_M | 45.8→56.1 GB | Q4_K_M KLD 0.0842 top1 89.49 | 🟡 précision↔taille NON linéaire ; calibration/mixed compte |

---

## 2. LES 6 RÉSULTATS LES PLUS PRÉCIEUX (pour ton projet)

### 2.1 — 2.57 bpw sur 24 GB + 32 GB RAM (Haberstroh) ⭐
- 2-bit experts (group 128), INT8 dense, 16-bit router/norm → **2.572 bpw**
- 262K ctx, loader 2-bit + kernel Triton 2-bit + expert streaming + **PLE mmap NVMe** + VMM +
  cache élastique
- 101/421/1701/6821/10001 ctx → 56.2/54.3/56.8/55.7/54.5 tok/s ; prompt ~258K : prefill 1560,
  decode 52.3
- **Remplacement du chargement de TOUTE la couche expert (26 GB PCIe) par 0.31 GB/token**
  (seuls les 10 experts sélectionnés par couche/token)
→ Preuve : quantification + sélectivité routing + paging DOIVENT être optimisés ensemble.

### 2.2 — Distribution de routing extraite (le working set n'est pas top-k)
```
32 premiers experts → 37% du trafic
64                 → 53%
128                → 73%
171                → 82%
256                → 93%
184                → working set GPU
```
→ **`cache_size ≠ top_k`** ; cache_size = f(routing_mass, reuse_distance, precision,
device_memory, phase). Commencer à **2-4× top-k** (recommandation llama.cpp #28248).

### 2.3 — PLE GPU peut TUER le decode (lukaLLM) ⭐
```
PLE GPU      → decode 1.95 tok/s
PLE CPU/RAM  → decode 108.5 tok/s   (55.6× plus lent si PLE sur GPU)
```
→ Le gros bloc mémoire n'est PAS forcément celui à mettre dans le GPU. PLE = lookup sparse,
experts = GEMM → **PLE → host/SSD, experts → VRAM working set**.

### 2.4 — PLE : 3 stratégies réelles + quantification indépendante
- **PLE BF16 → host/disk** (lukaLLM)
- **PLE FP8 → host/disk** (tonyd2wild : 16 rows/token depuis NVMe, buffer GPU fixe)
- **PLE NVFP4 → 28.8 GB** (Starkweather : E2M1 + scale FP8/16, modèle total 109.18 GB)
→ Le planner choisit PLE precision INDÉPENDAMMENT de expert precision. 16 rows × 160 = 2.7 KB/
token ≈ 3 MB/s @ 36 tok/s, réponses NVMe < 100 µs → **PLE sur SSD faisable, experts sur SSD non**
(mêmes propriétés).

### 2.5 — FP8 KV : mémoire plus petite ≠ kernel plus rapide (MiaAI)
- KV pool FP8 = 1.8-1.9× tokens ; decode +8.5% (8 streams) MAIS prefill -6.1% (32K)
- **FP8→FP32→BF16 matérialisé** = bottleneck → déplacer les scales hors du chemin de déquant :
  QSA kernel 2.984 → 1.772 ms (**-40.6%**)
- **GDN state ~0.23 GB/séquence** lu/écrit à chaque step ; FP32→BF16 = +6.8% decode 1 stream,
  +8.5% 8 streams (step 141.4→130.4 ms)
→ La manière dont le kernel déquantifie est une VARIABLE critique, pas juste "FP8 = plus petit".

### 2.6 — MTP : dépendance extrême au workload (Weschera)
```
sans MTP           code 24.5 · thinking 24.6 · prose 23.0
draft-n-max=4      code 47.1 · thinking 36.9 · prose 24.2
draft-n-max=8      code 41.3 · thinking 36.7 · prose 14.6  ← pire que sans MTP
```
→ D2 doit connaître draft_acceptance, tokens_per_step, draft_cost, target_cost — pas MTP on/off.

---

## 3. PROBLÈMES DE CORRECTION DÉTECTÉS (profiler-v3 doit aussi valider)

1. **llama.cpp expert cache : mapping en retard d'un ubatch** → 20-30% de lectures de mauvais
   poids avant correctif (mapping = vrai buffer ancré au bon split). → profiler-v3 doit tracer
   expert_id_requested vs expert_id_loaded vs slot_generation.
2. **SGLang : BF16 Q + FP8 K/V** exige des dtypes correspondants sur le chemin QSA.
3. **SGLang : QSA concurrent** → illegal memory access à 8 requêtes (cause non confirmée).
4. **CUDA graph GB10 TP2 : corruption silencieuse** QSA/NEXTN.
5. **Tool calling : boucle sur token 0** avec thinking + qwen3_coder.
→ **mesurer performance + correctness + determinism ensemble.**

---

## 4. CUDA VMM + ELASIC CACHE (la solution Haberstroh = exactement le D2 cache)
- CUDA VMM + elastic expert cache : working set croît/rétrécit par morceaux de **4 MiB**,
  **adresses virtuelles stables** (préserve CUDA graphs)
- **physical residency ≠ virtual address** : le planner modifie la résidence en gardant les
  adresses logiques constantes (compatible CUDA graph replay — le point #31 des angles morts)
- `SGLANG_MOE_ELASTIC_CTL` : pilotage direct du nombre d'experts GPU sous pression VRAM
→ prototype réel de "VRAM pressure → shrink experts → free memory → grow experts".

**Manque encore** : pas de tier SSD expert, pas de imatrix-guided pinning, cache mono-
accélérateur → là où D2 va plus loin (SSD+RAM+VRAM+precision+prefetch+NPU+RTX).

---

## 5. COMPARAISON NVIDIA NVFP4 vs NVFP4 CUSTOM
- NVIDIA officiel : ~124-133 GB (PLE FP8 sur NVMe)
- Starkweather custom : **109.18 GB** (PLE aussi NVFP4 28.8 GB ; experts NVFP4 67.95 ;
  MTP mixed 1.42 ; reste BF16 10.98)
→ **NVFP4 ≠ taille fixe** : la composition des tenseurs compte.

---

## 6. CE QUE ÇA CHANGE DANS LE D2 PLANNER (mise à jour)

```
D2
├── choose weight representation (par tensor/expert)
├── choose activation representation
├── choose PLE representation         ← indépendant des experts (3 stratégies prouvées)
├── choose GDN state representation   ← FP32→BF16 = +6.8-8.5% (prouvé)
├── choose QSA KV representation      ← FP8 : +8.5% decode, -6.1% prefill (trade-off)
├── choose residency (SSD/RAM/pinned/VRAM)
├── choose expert working set (routing_mass, PAS top-k)
├── choose prefetch
├── choose eviction
├── choose kernel / tile / device (CPU/XDNA/RTX)
└── choose MTP policy (par workload)
critère : MIN critical path SOUS RAM/VRAM/DDR/PCIe/SSD/accuracy/stability
```

## 7. UNITÉ D'ANALYSE
```
(layer, tensor, expert, representation, activation, residency, device, kernel, phase, batch)
ex. L23 / down_exps / E417 / IQ4_NL / RAM→RTX / M=1 / CUDA kernel X
```
C'est cette granularité qui découvre "ce tensor doit être Q6 sur RTX mais Q4 sur SSD".

## 8. 4 CLASSES D'OPTIMISATION
- **A réduire les bytes** : Q2/Q3/Q4/IQ/NVFP4
- **B réduire les bytes réellement transférés** : expert selection, cache, prefetch, PLE locality
- **C réduire le coût des bytes** : layout, coalescing, fused dequant, DMA, kernel
- **D éviter les bytes** : cache hit, resident expert, PLE host, GDN compressé, KV compressé
→ **La classe D peut être plus importante que Q4→Q3** (preuve : 0.31 GB vs 26 GB PCIe).

## 9. MESURES PAR TOKEN / RESSOURCE / KERNEL / ÉVÉNEMENT
- Par token : active_experts, unique_experts, expert_bytes, PLE_bytes, GDN_state_bytes,
  QSA_KV_bytes, MTP_bytes
- Par ressource : SSD read, DDR read/write, PCIe H2D, VRAM read/write, NPU DMA
- Par kernel : dequant, GEMM, QSA, GDN, PLE, MTP, router
- Par événement : cache_hit/miss, prefetch_hit/false, promotion/demotion/eviction, sync,
  queue_wait

## 10. RÉFÉRENCES (corpus de reproduction)
| Projet | Rôle |
|---|---|
| HaberstrohSystems | 2.57 bpw / 24 GB / streaming 0.31 GB — **reproduire en premier** |
| llama.cpp #28248 | cache persistant (+84%) |
| lukaLLM | PLE host / 8 GB GPU |
| Weschera | MTP DGX Spark |
| tonyd2wild | NVFP4 officiel + PLE disk + FP8 KV + MTP3 |
| Starkweather | NVFP4 custom 109 GB (PLE quantifié) |
| MiaAI-Lab | FP8 KV + GDN BF16 |
| Agention | AP mixed precision GGUF |
| AtomicChat | imatrix / calibration |
| NVIDIA | Qwen3.8-Flash-Next-NVFP4 officiel |
| NVIDIA Dynamo | serving recipe |
| SGLang | Qwen3.8 roadmap (FP8 KV, PLE offload, HiCache, PD disaggregation) |

---

## 11. PLAN DE REPRODUCTION (l'action la plus rentable)

> **Ne pas tester 30 quants au hasard. Reproduire quelques configurations publiques
> représentatives, extraire leurs traces expert/PLE/KV/GDN/DMA/PCIe, puis entraîner l'oracle
> D2 sur ces données AVANT la recherche adaptative.**

### Étape R1 — Configs publiques à reproduire (sur machine cible 5070+XDNA2)
| Config | Objectif de mesure |
|---|---|
| **R1a : streaming sélectif (Haberstroh-style)** | 0.31 GB PCIe/token vs 26 GB ; routing mass ; working set 2-4× top-k |
| **R1b : cache persistant (llama.cpp #28248)** | courbe 0/16/32/48/64 slots ; +84% à reproduire ; mapping ubatch bug check |
| **R1c : PLE host/disk (lukaLLM)** | PLE GPU vs CPU/RAM vs SSD ; amplification ; lookup latency |
| **R1d : KV FP8 + GDN BF16 (MiaAI)** | KV pool 1.8-1.9× ; decode vs prefill trade-off ; scales hors chemin dequant |
| **R1e : NVFP4 + MTP (NVIDIA/Tony/Weschera)** | PLE disk + FP8 KV + MTP3 ; MTP par workload |

### Étape R2 — Extraction de traces (collecteurs npu-rtx/)
- expert/PLE/KV/GDN/DMA/PCIe par token → format JSONL profiler-v3 (adaptateur)
- routing mass, reuse distance, working set 80/90/95%
- mapping correctness (expert_id_requested vs loaded vs slot_generation)

### Étape R3 — Entraînement oracle D2
- Sur les traces R1+R2 : calibrer throughput model, cache model, prediction model,
  conversion model, thermal model
- Puis recherche adaptative guidée (pas de grid aveugle)

---

## 12. LIENS
- URLS_REGISTRY.md (sources officielles)
- PLAN_CAMPAGNE_PARETO.md (structure A→X, 10 tests)
- MATRICE_EXPERIMENTALE_P0_P5.md (benchmark/compteurs/critères)
- PROFILER_V3_AMD_INTEGRATION.md (adaptateurs AMD)
- BLINDSPOTS_HW.md (SM120/XDNA2)
- ⚠️ profiler-v3/V5 = Snapdragon : réutiliser la LOGIQUE (5 objets, résidence, quality gate),
  refaire les collecteurs pour XDNA2/RTX dans npu-rtx/.