# RAPPORT XDNA2 — D2 Planner / MoE tiering (Qwen Flash → RTX / XDNA2 / RAM / SSD)

Créé : 2026-09-20 — dossier `geniex_harness/xdna2/`
Source : synthèse utilisateur + recherches AMD / communauté llama.cpp / Strix Halo + 5 briques centrales.

---

## 1. Idée centrale (le changement de paradigme)

Au lieu de : VRAM → RAM → CPU → SSD (CPU = destination normale du débordement),
proposer : SSD → RAM ──┬→ RTX (VRAM HOT)  \
                       ├→ XDNA2 NPU (WARM/HOT)  \
                       └→ CPU (dernier recours)

Le NPU XDNA2 devient un **tier d'exécution supplémentaire**, pas un simple coprocesseur. Pour Qwen Flash (MoE ultra-sparse, experts nombreux, mémoire VRAM 8 Go insuffisante), le planner doit décider **où placer chaque expert pour le token suivant**, pas seulement s'il est en cache.

Exemple de décision par expert :
| Expert | Prédiction | Placement | Justification |
|--------|-----------|-----------|---------------|
| E17 | très fréquent | RTX VRAM | hot, petit, MUL_MAT rapide |
| E43 | fréquent | XDNA2 | débordement VRAM, DMA acceptable |
| E81 | moyen | RAM → XDNA2 | staging + prefetch |
| E142 | faible | SSD → RAM | cold, chargement au besoin |
| E311 | imprévisible | SSD | évité, coût DMA > gain |

---

## 2. Les 5 briques centrales de l'architecture D2 Planner

| Brique | Rôle dans le planner | Référence / URL citée |
|--------|---------------------|------------------------|
| **SSD-LLaMA** | Streaming experts depuis SSD vers RAM/VRAM/NPU ; modélisation du coût énergie/transfert | paper SSD-LLaMA (trillion-parameter MoE depuis SSD) |
| **QwFNfer** | Cache 3 niveaux (VRAM / RAM / SSD) + prédiction d'experts pour Qwen3.8-Flash-Next ; détermine quand précharger | QwFNfer (Qwen3.8-Flash-Next / cache 3 niveaux) |
| **hetero-llm-scheduler** | Placement dynamique CPU/GPU/NPU ; arbitration contention mémoire partagée | hetero-llm-scheduler (CPU/GPU/NPU) |
| **MoE CPU-GPU Collaborative Inference** | Collaboration CPU (routage/split) + GPU (compute experts) ; base du split planner | MoE CPU-GPU Collaborative Inference |
| **LLM.xpu** | NPU+iGPU hétérogène (Strix Halo / XDNA2 + iGPU) ; mémoire unifiée mais bande passante limitée ; valide que NPU et GPU peuvent coexister avec contention | LLM.xpu (NPU+iGPU) |

---

## 3. Ce que AMD documente vraiment

- **XDNA = dataflow NPU** : mémoires locales sur les tuiles + transferts DMA depuis mémoire hôte (host DDR → NPU local). Ce n'est pas une mémoire fixe isolée ; les poids peuvent être alimentés depuis RAM via DMA.
- **Ryzen AI Software 1.8** : support LLM sur NPU (Linux : XRT + amdxdna).
- **Pas de flux officiel hybride Linux NPU+GPU** : AMD ne fournit pas actuellement de mode « prends GGUF arbitraire et répartis automatiquement sur NPU/GPU ». Le flux Linux reste contraint (modèles préquantifiés, flux NPU dédié, pas hybrid dans LLM officel).
- **Driver** : transferts DMA explicites documentés entre host DDR et NPU local memory → `DDR → DMA → NPU local → AI Engine`.

### 3bis. CORRECTION — AMD documente désormais un hybride NPU+iGPU officiel (OGA 1.8)

Le RAPPORT initial disait « pas de flux officiel hybride ». Depuis Ryzen AI 1.8.0 (OGA 0.14.0), AMD fournit officiellement un **mode Hybrid OGA** : NPU+iGPU partitionnés (prefill/decode). Détail : `reference/AMD_OGA_HYBRID_OFFICIEL.md`.

Nuance importante :
- **Officiel AMD** = partition statique par phase (prefill→NPU, decode→iGPU) sur modèles ONNX pré-optimisés, pas sur GGUF arbitraire. Support Strix Point + Krackan Point (notre Ryzen AI 9 365 = Strix Point ✅).
- **Notre idée** = partition dynamique par expert MoE (RTX/XDNA2/RAM/SSD) = au-delà de l'officiel.
- `hybrid_opt_free_after_prefill` libère le NPU pendant le decode → slot idéal pour absorber les experts MoE en overflow pendant que le GPU décode.

### 3ter. UAPI driver amdxdna — primitives du planner (source : header officiel)

`reference/AMDXDNA_DRIVER_UAPI.md` — points clés :
- **BO types** : `SHARE` (user↔device), `DEV_HEAP` (heap device), `DEV` (alloc depuis heap), `CMD`. Pas de limite de taille BO imposée par le driver (limite = memlock Linux).
- **Sync explicite** : `SYNC_DIRECT_TO_DEVICE` / `SYNC_DIRECT_FROM_DEVICE` → primitive "charger expert → XDNA2".
- **QoS hints** : `dma_bandwidth`, `latency`, `frame_exec_time`, `priority`, `user_start_col` → le planner peut signaler au driver sa BW DMA attendue et réserver des colonnes.
- **Télémétrie temps réel** : sensors power/column_utilization/temperature, resource info (tops_curr, task_curr), AIE load, migrations/preemptions → alimentent directement `cost_model` et `contention`.
- **Contrainte plateforme** : header Linux ; notre machine est Windows (XRT Windows / ryzenai). À vérifier quel sous-ensemble est exposé sous Windows.

### 3quater. REVERSE FLM (1bit) — faits durs XDNA2 prouvés par désassemblage complet

`reference/1bit/` (README, FLM_SECRETS, NPU_ISA, Q4NX_FORMAT, NPU_GEMM_FIX) — FastFlowLM v0.9.24, 22 .so désassemblés, 1 390 symboles. Points qui changent le D2 Planner :

- **Q4NX : I8 = FAUX, c'est du BF16 en byte-pairs** `[lo,hi]`, 0.625 bytes/element (group 32, scale+zp bf16). **W = (q − zp) · scale** (déterminé bit-exact vs runtime).
- **NPU compute = INT8/BF16 uniquement** : Q4 = stockage seulement. Un expert "Q4" coûte en réalité INT8 côté NPU (leçon de conversion à l'init, BO persistant).
- **Limite firmware : 8 colonnes HARDCODEN** (`CREATE_HWCTX` rejette >8, testé 9/10/12/16/40 ; kernel permet 40, firmware RSA-4096 valide). 31 TFLOPS = plafond pratique NPU. `XDNA2_COLS_ACTIVE=4` sur Strix Point (mesuré fastflow).
- **ISA NPU (5 opcodes)** : 0x00 WRITE, 0x01 BLOCKWRITE (DMA N-dim), 0x03 MASKWRITE (sync token), 0x80 WAIT (TCT), 0x81 DDR_PATCH (patch adresses). RTP 0x1000-0x1010 = M/K/N/act/bias, kick-off 0x1f0a0.
- **Stack** : `npu_sequence → npu_app (ELF) → XRT → xrt::bo/xrt::run/runlist → AIE`.
- **Init XRT** : `xclbin → device.register_xclbin → hw_context → module → ext::kernel → bo → run → runlist.execute → wait`.
- **`xrt::bo::sync(XCL_BO_SYNC_BO_TO_DEVICE, size, 0)` / FROM_DEVICE** = équivalent Windows/C++ de `SYNC_DIRECT_TO/FROM_DEVICE` Linux. **group_id > 0 OBLIGATOIRE** sinon no-op silencieux (NPU_GEMM_FIX.md — écho de "rc=0 ≠ preuve").
- **runlist = batch d'ops** : `runlist.add(run) → execute → wait` → le planner groupe les experts du même token en 1 runlist.
- **Checkpoint/restore KV** : `bytes::sync_from_device()` / `sync_to_device()` → copier le KV cache NPU→host→NPU = primitive "reprendre le contexte" quand un expert change de backend.
- **MHA 6 variantes** (d64/d128/d256 × Q2/Q3/Q4) : attention sur NPU (QK^T, online softmax, PV). Le planner connaît le coût attention par (head_dim, quant).
- **MOE OFFICIEL FLM (Qwen3.6)** : `_send_router_w_and_share_exp_gate()`, `send_manual_expert_up_gate_q41()`, `send_manual_expert_down_gate_q41()`, `setup_expert_up/down_gate_q41()`, `gen_dequant_mm()`. → **AMD fait DÉJÀ des experts MoE sur NPU** (Qwen3.6 35B), par layer. Notre idée = étendre aux experts en overflow RTX (dense ≠ MoE-only).

### 3quinquies. 1bit-MONSTER — moteur ouvert NPU (Strix Halo)

`reference/1BIT_MONSTER.md` + `1BIT_REVERSE_XDNA2.md` :
- Engine C++26 model-agnostic (NPU/GPU/CPU auto-routé), format 1BP pour poids NPU.
- **Qwen3-0.6B sur NPU : 82-91.5 t/s decode, 2W, 46 tok/s/W** (25× plus efficace que GPU), prefill 1591 t/s (chunk 8192).
- GTT dma-buf zero-copy : **56 GB/s** mesurés host↔NPU.
- **8+ hw_contexts concurrents** vérifiés → co-exécution NPU (attention) + GPU (decode) réaliste.
- 31 TFLOPS plafond 8-col ; INT8 = chemin retenu (évite double-quant Q4NX→BFP16→HW).

### 3sexies. Benchmarks calibrés pour le planner (1bit performance.md, Strix Halo vs notre machine)

`reference/1bit/docs/performance.md` — références numériques à injecter dans `cost_model` :

| Mesure | Valeur | Backend | Note pour le planner |
|--------|-------:|---------|----------------------|
| NPU HW raw (xrt-smi validate) | **51 TOPS INT8, 50µs lat, ~75k op/s** | XDNA2 Max+ 395 | plafond NPU de référence |
| **FLM MoE 35B (Qwen3.6-A3B)** | **11.66 tok/s decode @1k** (→8.82 @32k) | FLM v0.9.46 NPU | **référence MoE NPU réelle** — cible de notre overflow |
| Native MoE 35B (1bit, naive) | ~0.94 tok/s (layer ELF + top-8 experts batched en 2 runlist) | 1bit npu | preuve runlist=2 submits pour 8 experts |
| Qwen3-0.6B FLM on-box | decode 73.58 tok/s, prefill ~430 tok/s | FLM | densité = référence |
| Qwen3-0.6B native 1bit | decode 79-91 tok/s, prefill 655 tok/s @256 | 1bit runlist | double-buffered runlist = +7% vs FLM |
| Qwen3.6-35B-A3B via llama.cpp Vulkan | **75.65 tok/s** (RTX-like) | GPU | **GPU bat le NPU 6.5× sur MoE** → le NPU n'est pas pour la vitesse brute |
| Qwen3-0.6B 8 concurrents | 82-85 t/s | NPU | KV cache persiste multi-turn |
| DDR savings | TQ2=4×, TQ1=4.9× vs INT8 | formats | **réduire bytes/token = levier orthogonal au placement** |
| runlist whole-layer | 1 submit/token → 94 tok/s byte-identical (0.6B) | 1bit | **grouper les experts d'un token en 1 runlist = levier majeur** |
| Géométrie packing | G=K/128, CH=H/16, tile 5120 B | Q4NX | coût DMA par expert = n_chunks × 5120 |

**Conséquence clé pour le D2 Planner** : sur Strix Halo, le GPU (llama.cpp Vulkan 75.65 tok/s MoE) bat le NPU (FLM 11.66 tok/s MoE) d'un facteur ~6.5. Notre machine a un RTX 5070 (encore plus rapide que l'iGPU). Donc :
- **Le NPU n'est JAMAIS le backend principal pour le MoE dense-sparse** — c'est le tier d'overflow/éco, pas le tier de performance.
- Le planner doit traiter le NPU comme "acceptable si GPU saturé/VRAM pleine ET économie énergie", pas comme concurrent du RTX.
- `hybrid_opt_free_after_prefill` : le NPU libéré après prefill = coût marginal ~0 pour absorber des experts → candidat naturel pour l'overflow pendant le decode GPU (comme établi §3bis).

### 3septies. Stack NPU AMD open-source (amd-oss-knowledge / ROCm/FastFlowLM)

`reference/AMD_OSS_NPU_STACK.md` — point stratégique : **le backend XDNA2 est reconstruisible en OSS**, pas seulement via FLM propriétaire :
- **IRON** (Apache-2.0) : Python structural close-to-metal, 28 opérateurs (GEMM/MHA/RMSNorm/RoPE/softmax bf16), Llama 3.2 1B e2e, tests=benchmarks CSV.
- **Triton-XDNA** (MIT) : `@triton.jit` → MLIR-AIR → aircc → xclbin, **Windows natif**, matmul à parité kernel manuscrit.
- **FastFlowLM = ROCm/FastFlowLM** (Apache-2.0) : acquis AMD 2026, NPU-first, ctx 256k, `libxrt_driver_xdna.so.2`.
- **HRX2 llama.cpp lane** : true zero-DMA-copy decode +10-49% mais prefill régressé (taxe cache-coherency GTT) → le cost model doit calibrer GTT vs host_coherent, pas supposer.
- **Hybride prefill/decode OSS** : HIP prefill 1227-1313 tok/s + HRX decode ~80-87 tok/s = même pattern qu'AMD hybrid.
- Stack : Triton/IRON → MLIR-AIR → MLIR-AIE (substrat, pas AIR) → Peano/LLVM-AIE → XRT → NPU.

---

## 4. Ce que la communauté a déjà construit (première brique)

- **Ian-Pratt/OllamaAMDNPU** : backend ggml XDNA2 pour llama.cpp. Expose MUL_MAT vers NPU via XRT + `.xclbin`, quantifie poids, met en cache poids quantifiés, découpe matrices en tiles, laisse ops non supportées au CPU.
- **Issue llama.cpp #21725** : demande d'intégration backend XDNA (ggml XDNA, XRT, kernels compilés, MUL_MAT sur NPU, GGUF).
- **Résultat intermédiaire** : on a déjà la chaîne `llama.cpp → op supportée → XDNA2` + `op non supportée → CPU`. L'objectif du planner est de passer de cela à `Router MoE → D2 Planner → RTX / XDNA2 / CPU` avec choix dynamique.

---

## 5. Pièges mesurés par la communauté (critiques pour le planner)

### a) Bande passante mémoire partagée (le vrai goulot)
- NPU et iGPU utilisent **le même sous-système mémoire** (Strix Halo, mémoire unifiée).
- Mesures : iGPU seul ~226 GB/s read ; CPU seul ~108 GB/s ; CPU+iGPU → CPU tombe vers ~17 GB/s (contention sévère).
- **Conséquence planner** : `RTX + NPU = toujours mieux` est faux. Le planner doit optimiser la **contention**, pas seulement le nombre d'ops envoyées au NPU.

### b) Performance brute NPU vs GPU sur LLM
- Gemma 4 E4B : NPU ~12,1 tok/s @ ~22 W ; iGPU ~57,3 tok/s @ ~72 W ; énergie par token ~1,1 J similaire hors idle.
- **Implication** : XDNA2 n'est pas un remplacement de GPU pour la vitesse brute, mais un **tier supplémentaire efficace énergétiquement** pour absorber le débordement.

### c) NPU + GPU simultanés sont possibles
- Test communautaire : NPU fait tâche de fond pendant qu'iGPU fait inférence principale (Strix Halo). Confirme que co-exécution est réalisable, mais la synchronisation et la contention doivent être gérées par le planner.

### d) Strix Halo / mémoire unifiée invalide l'hypothèse « VRAM limitée = problème »
- La mémoire unifiée permet d'adresser beaucoup plus que la VRAM physique via RAM, mais avec **bande passante limitée**. Donc le problème réel n'est pas « pas assez de VRAM » mais « pas assez de BW pour tout faire sur GPU ».
- **Le NPU apporte alors une autre unité de calcul efficace**, pas nécessairement une nouvelle capacité mémoire massive.

---

## 6. Architecture proposée du planner (3 niveaux de décision)

### Niveau 1 — Où se trouve l'expert ?
- SSD (cold, stockage) → RAM cache → XDNA2 buffers → RTX VRAM

### Niveau 2 — Qui l'exécute ?
- RTX (MUL_MAT rapide, grandes matrices)
- XDNA2 (opérations supportées via ggml-XDNA, tiles, DMA)
- CPU (opérations non supportées par XDNA / fallback)

### Nivau 3 — Quand le charger ?
- Maintenant (token actuel, expert hot)
- Token suivant (prefetch prédictif)
- +N tokens (staging)
- Jamais / SSD (trop rare, coût DMA > gain)

### Flux de décision pour un token :
```
Router (Qwen Flash top-k) → experts demandés (E17, E43, E81, E142)
  → Planner (D2) choisit emplacement par expert selon prédiction + capacité
  → Execution : RTX (E17) | XDNA2 (E43 via DMA) | RAM→XDNA2 (E81 prefetch) | SSD→RAM (E142)
  → Combine → output
```

---

## 7. Questions techniques à résoudre (variables du planner)

| Question | Pourquoi c'est déterminant | Source / indice |
|----------|---------------------------|-----------------|
| Combien d'experts Qwen Flash tiennent simultanément dans mémoire XDNA2 ? | Détermine la capacité du tier NPU ; si trop petit, c'est seulement un cache partiel | Captation mémoire locale NPU + DMA host (pas documentée par AMD en GB pour XDNA2 exact) |
| Quel format de quantification XDNA2 accepte pour experts ? | Q4_0 ? Q4_K_M ? FP16 ? Influence taille par expert et qualité | Backend ggml XDNA (OllamaAMDNPU) utilise quantification ; format GGUF doit être compatible XRT / .xclbin |
| Coût DMA host→NPU vs CPU fallback ? | Si DMA > CPU, le planner doit choisir CPU même si NPU est disponible | Mesures BW mémoire partis (CPU+iGPU contention) ; pas encore de benchmark DMA spécifique NPU seul |
| Synchronisation RTX/NPU / temps de frontière ? | Le planner doit éviter le hang lors du switch backend par op (comme le bug pass-4.5 / backend_switch) | Observation K5 (100 % splits = backend_switch) ; fix topo sort déjà appliqué pour réduire splits |
| Format de poids pour tiles ? | Les matrices trop grandes sont découpées ; le planner doit savoir si un expert doit être tilé ou gardé entier | Documentation OllamaAMDNPU (décomposition matrices trop grandes) |

---

## 8. Liens et références utilisées (tous consultés / cités)

### Projets / repos cités
- `Tagman45/gglm-xdna2`
- `Tagman45/adaptive-xdna-runtime`
- `gat45/profiler-v3`
- `gat45/jarvix-memory`
- `llama.cpp` / discussion #28043 (Snapdragon/HTP hétérogène)
- `llama.cpp` issue #21725 (XDNA backend)
- `Ian-Pratt/OllamaAMDNPU`
- `hetero-llm-scheduler`
- `LLM.xpu`
- `MoE CPU-GPU Collaborative Inference`
- `SSD-LLaMA`
- `FlashMoE` (SSD + cache prédictif, fine-grained / CUDA, Apple Silicon, ANEMLL)
- `QwFNfer`
- `Moe-slices`
- `1bit-MONSTER`
- `XDNA2 / NPU communautaire`
- Reddit : Qwen3 0.6B sur XDNA2 ; grands MoE sur NPU ; FLM/NPU Strix Halo ; xdna-top monitoring NPU/iGPU

### Références scientifiques / stockage
- FlashMoE paper — cache SSD piloté par ML
- SSD-LLaMA paper — trillion-parameter MoE depuis SSD
- SSD offloading / coût énergétique

### Documentation AMD / driver
- AMD Ryzen AI Software (doc 1.8, LLM support NPU)
- AMD XDNA Driver (DMA host DDR → NPU local)
- `amdxdna` Linux, XRT

---

## 9. Ce que le planner ne doit PAS faire (leçons du repo)

D'après le travail existant sur `profiler_v3`, `governor/`, `device_artifacts_20260906/` :

- **Ne pas confondre succès d'exécution (rc=0) avec preuve d'engagement** : un fallback CPU silencieux donne des chiffres plausibles mais faux. Pour le planner : vérifier que `n_backends=3`, `dev=HTP0` (ou XDNA2 identifié), `mirror: 0`, `repack > 0`.
- **Pré-engagement exigé** : la preuve d'engagement pour XDNA2 doit montrer que le backend est bien celui exécuté (pas un fallback). Pour le planner : ajouter un compteur `npu_executed_ops` vs `cpu_fallback_ops`.
- **Ne pas ignorer la contention mémoire** : sur Strix Halo / mémoire unifiée, envoyer trop d'experts sur NPU peut ralentir le GPU par contention BW. Le planner doit avoir un modèle de coût `U(a)=E[ΔV]+λ·Î−α·C−β·R` (comme dans `governor/policy.py`) avec `C` = coût DMA/contention, pas seulement coût calcul.
- **Ne pas écrire dans `memory.json`** côté planner : mémoire governor dans `governor_state/provenance.json` ; planner doit écrire ses décisions dans son propre état (`adaptive-xdna-runtime` / `profiler-v3`).

---

## 10. Actions immédiates suggérées (plan du dossier xdna2)

1. **Documenter capacité XDNA2 réelle** : mesurer mémoire locale disponible, taille max d'un expert en Q4_0, temps DMA unitaire.
2. **Implémenter choix placement** dans `adaptive-xdna-runtime` / D2 adapter : fonction `place_expert(expert_id, freq, size, thermal_state) → tier`.
3. **Ajouter métrique contention** au `cost_model` : BW utilisée par RTX + NPU + CPU ; prédire contention avant placement.
4. **Valider avec llama.cpp + OllamaAMDNPU** : faire tourner Qwen Flash (ou un sous-ensemble) avec le backend XDNA et mesurer `tok/s` côté NPU vs CPU fallback.
5. **Créer le benchmark cross-tier** : A (RTX seul) / B (RTX+XDNA2) / C (RTX+XDNA2+CPU) avec même batch d'experts ; comparer `perf/compute` et `VoI` comme dans `governor/benchmark.py`.

---

*Fichier : `E:\oneplus\geniex_harness\xdna2\RAPPORT_XDNA2_D2_PLANNER.md`*  
*Références : AGENTS.md (§architecture, §benchmarks, §protocole canonique), docs/TRAVAUX_ET_FONCTIONNEMENT.md (§11 MTP, §AXE-7, §AXE-8), bench_results/RAPPORT_*.md, governor/*.py.*
