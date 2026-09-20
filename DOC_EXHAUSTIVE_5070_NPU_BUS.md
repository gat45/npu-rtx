# DOCUMENTATION EXHAUSTIVE — Co-exécution RTX 5070 + NPU XDNA2 sur le bus mémoire
# Version : 2026-09-20 · Dossier : geniex_harness/xdna2/
# Portée : machine réelle (Ryzen AI 9 365 / RTX 5070 8 GB / 32 GB RAM)
# Sources : bench_results/ (geniex_harness), D:\fastflow compagnon, D:\vrac tour\SOSC_v4,
#           D:\vrac tour\mesures_flm, D:\vrac tour\apply_turbo_cuda_v2.py,
#           C:\Users\videl\Desktop\lama-tensorRT 1050-5070, AMD officiel (OGA/UAPI/XDNA)

---

## 0. OBJECTIF

Faire travailler le **RTX 5070 (8 GB Blackwell)** et le **NPU XDNA2 (Ryzen AI 9 365)**
ensemble, coordonnés par le **D2 Planner**, en exploitant intelligemment le **bus mémoire**
partagé. Objectif final : quand la VRAM 8 GB est pleine (modèle MoE type Qwen Flash), les
experts qui débordent ne tombent pas sur CPU par défaut — ils sont absorbés par le NPU (tier
d'overflow/éco) pendant que la 5070 décode, avec RAM comme staging et SSD comme stockage froid.

```
                     Qwen Flash / MoE
                           │
                     Router / Planner
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          RTX 5070      XDNA2 NPU      CPU
         VRAM HOT     overflow WARM   dernier recours
              │            │            │
              └───────┬────┴───────┬────┘
                      ▼            ▼
                    RAM cache    SSD COLD
                 (staging)     (stockage)
```

---

## 1. MATÉRIEL RÉEL (machine cible)

| Composant | Spec | Rôle dans l'architecture |
|---|---|---|
| **NPU** | AMD Ryzen AI 9 365 (Strix Point) — XDNA2, PCI 0x17F0 rev 0x10 (NPU4, aie2p), 8 cols × 4 rows = 32 tiles AIE2P, **4 colonnes exposées à XRT** (4 réservées firmware), 1.8 GHz, 51.3 TOPS INT8 peak / 38 eff / 9-12 TOPS GEMM | Tier d'overflow / éco |
| **SRAM NPU** | 2 MB L1 (64 KB × 32 tiles) + 4 MB L2 (512 KB × 8 MemTiles) = **6 MB total** | Cache poids SRAM (FLM y garde ~1.6 GB de W_eff 2.87 GB) |
| **GPU** | NVIDIA RTX 5070 **8 GB GDDR7** (Blackwell sm_120, 384 GB/s) | Tier HOT principal (53.83 t/s mesuré Qwen3.5-9B) |
| **RAM** | 32 GB DDR5 (~89.6 GB/s peak, 21.93 GB/s effectif mesuré NPU) | Staging / cache chaud / KV |
| **SSD** | NVMe (page file actif, 92.4% RAM utilisée) | Stockage froid experts |

**Point décisif** : sur cette machine le dGPU (5070, GDDR7 384 GB/s) et le NPU (DDR5 ~89.6)
**ne partagent PAS le même contrôleur mémoire** — contrairement à Strix Halo (mémoire unifiée
226 GB/s). La contention "bus" existe donc surtout **côté NPU/CPU/iGPU sur la DDR5**, pas entre
la 5070 et le NPU. C'est un avantage structurel pour la co-exécution, MAIS le NPU reste limité
par sa bande passante DDR5 (21.93 GB/s effective).

---

## 2. MESURES RÉELLES DU BUS MÉMOIRE (corpus local)

### 2.1 Bande passante DDR5 mesurée (NPU FLM, 21 juin 2026)

| Valeur | Mesure |
|---|---|
| BW nominale LPDDR5X | 89.6 GB/s (peak) / ~128 GB/s (littérature TileFuse) |
| **BW effective decode** | **21.93 GB/s = 24.5% du peak** |
| BW effective prefill | ~30 GB/s |
| η (efficacité DDR) | 0.724 (perte 27.6%) |

**Causes des 24.5% d'efficacité** :
- Scheduling XRT idle pendant compute : ×3.7
- iGPU 880M partage le bus : **-3.5 GB/s** (framebuffer 2560×1600 @ 180 Hz)
- Column mapping idle tiles : **-40%**
- Page file actif (92.4% RAM utilisée) : -10%

### 2.2 Contention mémoire = le vrai goulot (Qwen3.5-9B)

| État | W_eff (poids rechargés/token) | TPS decode |
|---|---|---|
| **En contention** (18-19 juin : navigateur, GPU, OS) | 4.59-4.93 GB (**+60%**) | **4.27-4.65 t/s** |
| **Propre** (21 juin) | 2.86-2.92 GB (-2%) | **7.06-7.43 t/s** |

**Conclusion** : l'inflation du W_eff sous contention (+60%) est le symptôme direct du goulot
bande passante DDR5. Le coût d'un token est `T_token = (W_eff + c·KV) / BW_eff`. Tout levier
qui **réduit les octets lus/token** (quantification, cache SRAM, experts en cache) se traduit
directement en t/s.

### 2.3 Modèle roofline calibré (flm_bottleneck_analyzer.py, validé R²=0.96)

```
TPS = BW_eff × η / (W_eff + c × KV_bytes)
TPS = 21.93 × 0.7309 / (3.80 + 0.28 × KV)

Temps par token : step_ms(ctx) = 82.3 + 0.00175 × ctx   (ms)
  A = 82.3 ms fixe (poids + dispatch + wakeup)
  B = 0.00175 ms/token (scaling KV)
```

KV par modèle (octets/token) : llama3.2:1b=16384 · qwen3.5:4b=30000 · qwen3.5:9b=30000 ·
deepseek-r1:8b=65536 (GQA standard).

**W_eff calibrés** : qwen3.5:9b = 3.83 GB (standard) / 2.87 GB (text-cust, grâce au cache SRAM
NPU ~1.6 GB) · qwen3.5:4b = 1.75 GB · deepseek = 5.74 GB (MoE).

### 2.4 Décomposition temporelle d'un token NPU (FLM, mesurée)

| Phase | Durée | % temps | Détail |
|---|---|---|---|
| decode loop CPU (0x310D0) | 60 ms | 54% | 32 layers orchestration |
| ├ gen_*_seq (32 layers) | ~15 ms | 13% | production npu_sequence* CPU |
| ├ dispatches XRT | ~45 ms | 41% | 3064 appels × 67 µs |
| bo::sync + DMA | 20 ms | 18% | sync WRITE/READ poids |
| **NPU compute (matmul)** | **5 ms** | **4.5%** | 639 instructions GEMM/chaîne |
| Wakeup/sampling | 26 ms | 23% | next-token decoding |
| **Total token** | **111 ms** | 100% | 9 TPS théorique, 7.6 mesuré |

**CPU total 86 ms (86.8%) · NPU idle 99.73%** (138/51300 GOPS actifs) · **DMA idle pendant
compute** (pas de prefetch du layer suivant).

---

## 3. LE NPU XDNA2 — ce qu'il peut réellement faire

### 3.1 Capacités (SOSC_v4 / CARTE_FONCTIONNELLE, confirmées Ghidra + mesures)

- 32 tiles AIE2P, 4 colonnes exposées à XRT (partition 0), 4 réservées firmware
- 51.3 TOPS INT8 peak, 38 eff (paper AMD), 9-12 TOPS GEMM réels
- SRAM 6 MB total (2 L1 + 4 L2)
- Horloge kernel 1.8 GHz

### 3.2 Les 4 goulots (avec preuves)

**Goulot #1 — MCDM SHIM dispatch (96.5% du temps NPU)**
- `run::wait` = 81.32 ms/token = 96.5% du temps NPU ; 537 gaps > 111 ms = 80% du temps token
- 3064 dispatch × 67 µs = 205 ms cumulé/token (scheduling pur) ; 50 IOCTL/token
- **Cause racine : Windows MCDM n'a PAS `force_cmdlist`** (contrairement à Linux amdxdna.ko).
  Chaque commande = 1 IOCTL D3DKMT séparé, pas de batching automatique.
- **"KDMA not supported on windows"** (2× dans xrt_coreutil.dll, #1431 run::start / #4133
  runlist::execute) : le chemin XRT public (scheduler ERT) ne peut PAS faire le DMA direct sur
  Windows → FLM contourne via `vitis-ai-runtime2.dll → RadeonML_ipu.dll → D3DKMT → ipustack.sys`.
- Bypass estimés : proxy driver IOCTL interceptor +393% · direct PCIe NPU driver +500% ·
  patch amdxdna.sys +300%.

**Goulot #2 — BW DDR5 saturée (96.7% utilisée)**
- 21.93/89.6 GB/s = 24.5% du peak. Solutions : ROCmFP4 (-25% BW → +25% TPS) · XQuant
  (7.9→~4 GB → +100% TPS) · désactiver iGPU (+12% TPS).

**Goulot #3 — XRT init overhead fixe (2362 ms/requête)**
- 2362 ms = reconstruction hw_context + remapping BO à CHAQUE requête HTTP (zéro compute).
  Décomposition : hw_context 800 ms · kernel 500 ms · bo::bo ×N 600 ms · runlist 462 ms.
- **Solution : persistent hw_context (mode stateful) → TTFT -65%** (1420 ms prefill pur).
  Blocage : AMD/FLM doit réécrire le serveur.

**Goulot #4 — Proxy XRT inactif** (v25b, 502 forwarders) : GetProcAddress NULL pour
xrtBOAlloc, 8 hypothèses écartées, cause probable = conflit loader Windows.

### 3.3 Découvertes XRT critiques (reverse Ghidra)

- **xrtBOAlloc IGNORE le paramètre size** (rdx jamais utilisé sur 12 appels) : le driver
  décide la taille d'allocation lui-même. Le proxy marche par timing/état initial, pas par
  correction de size.
- **xrt::bo réel = 152 B** (au lieu de 48 B documentés), run = 80 B, hw_context = 136 B,
  device = 208 B. (Preuve : MOV ECX,size dans .text — 0x98 ×19, 0x50 ×19.)
- **Q4NX = I8 (INT8), PAS INT4** : fichier 7.9 GB au lieu de 4.5 GB. Gain = réduction BW
  uniquement (dequant → BF16 avant compute). Pas de compute INT4 natif.
- **Pas de cache BO** : allocation/free à chaque token via create_bo_buffer ; transfert
  DDR→SRAM AIE via `npu_dma_memcpy_nd` (5514 B, plus grosse fonction).
- **KV cache absent entre requêtes** (GEN2/GEN3 = même perf que GEN1).
- Pipeline XRT/token : 3064 appels de bo::bo/set_arg/add/sync (35-400 µs), 32 runlist::execute
  + wait (~3000 µs chacun), 68 bo::~bo.

---

## 4. LE RTX 5070 — le tier HOT mesuré

### 4.1 Performance mesurée (CUDA Blackwell, dossier lama-tensorRT 1050-5070)

| Modèle | Backend | Decode | Prefill |
|---|---|---|---|
| **Qwen3.5-9B** | **RTX 5070 CUDA** | **53.83 t/s** | **2029 t/s** |
| Qwen3.5-9B | FLM NPU XDNA2 | 7.2-7.8 t/s | 46 t/s |
| Qwen3.6-35B-A3B (MoE) | llama.cpp Vulkan (iGPU) | 75.65 t/s | 1105 pp512 |

**Ratio GPU/NPU ≈ 7.4× sur Qwen3.5-9B.** La 5070 est donc le backend principal incontesté ;
le NPU ne peut que servir de **tier d'appoint** (overflow / économie d'énergie), jamais de
concurrent.

### 4.2 Format de build 5070

- `build_rtx_5070_blackwell.ps1` : TensorRT (`trtllm-build`), dtype float16, **quantization
  NVFP4** pour les couches sûres + FP16 pour les clusters d'outliers (per-layer precision,
  ex. `blk.0.ssm_conv1d.weight:fp16`).
- `ggml-cuda.dll` (32 MB) compilé CUDA Blackwell présent dans le dossier.
- `d2_rtx_gguf_profiler.py` / `d2_layer_profiler.py` / `d2_profiler.py` : profilage GGUF par
  couche sur CUDA (alimente le planner avec les coûts réels par op).
- `apply_turbo_cuda_v2.py` : patch CUDA pour llama-cpp-turboquant — kernels **turbo3** (3-bit,
  centroïdes ±0.1907, norm caching FA) et **turbo4** (4-bit) avec mixed K/V (turbo K + q8_0 V).
  → moins d'octets/token = moins de trafic bus = levier direct sur le goulot BW.

### 4.3 Échelle matérielle (RAPPORT_TIERS_HARDWARE_2026-09-13.md)

- 8 GB VRAM = même cap que la GTX 1080 ; ce qui sépare les cartes = **bande passante**
  (384 vs 320 GB/s sur Laptop, 672 sur Desktop) et tensor cores.
- Débit ∝ 1/(taille des poids lus/token) : Qwen9B IQ4NL = 5.0 GB → 32.9 t/s → BW effective
  observée 165 GB/s (51% du théorique).
- Modèles réalistes sur 8 GB : T1 (≤4B), T2 (Bonsai 27B 1-bit 3.8 GB ✅, ternaire 7.17-7.59 ⚠️).
  T3+ (Qwen3.6-27B 12.36 GB) = offload VRAM+RAM, lent.

---

## 5. CO-EXÉCUTION GPU + NPU — preuves et patterns

### 5.1 Double-flux indépendant : VALIDÉ (Snapdragon, geniex_harness 2026-09-05)

`bench_results/CONCURRENCE_GPU_NPU_QWEN35_9B_MMAP_20260905.md` :
- Deux process (GPUOpenCL + HTP0) sur le **même fichier GGUF via mmap** → pages physiques
  partagées (pas de duplication : MemAvailable min 1.57 GB au lieu de ~0 pour 2×4.94 GB).
- Résultat 9B : GPU 6.87 (+2.4%), NPU 6.63 (-1.6%) → **débit agrégat 13.50 t/s, contention
  quasi nulle**.
- **Pénalité inversement corrélée à la taille du modèle** : petits (≤2B) = contention BW
  dominante (-14 à -30%) ; gros (≥9B) = compute domine, contention marginale.

### 5.2 Split séquentiel de couches : PIRE que pur (pas de chevauchement)

`RESULTATS_GPU_NPU_ORDER_QWEN17B_20260905.md` : sur 1.7B, GPU-first 18.62 / NPU-first 18.54
pires que GPU pur 29.59, NPU pur 39.86, CPU pur 46.60. Cause : `ggml_backend_synchronize()`
bloquant après chaque split (`ggml-backend.cpp` ~1893) court-circuite les événements async.
Le split séquentiel n'est PAS de la co-exécution.

### 5.3 Infra async : PR #26501 (fusionnée 2026-09-01) — la base

- Backend Hexagon async + fences cross-device + événements + buffers non-host.
- **Déjà intégrée dans la branche `origin/self-build-jz`** du worktree `E:\oneplus\ab-wt`
  (4 commits de merge + PR #27785 + 94 commits d'avance sur notre branche).
- Notre travail sur `ggml-backend.cpp`/`ggml-alloc.c` (pass4.5, diagnostic gallocr) touche une
  zone stable (89 lignes de diff) → rebasable proprement.

### 5.4 Littérature convergente (3 papiers, aucun ne patche le scheduler générique)

| Papier | Matériel | Mécanisme | Gain |
|---|---|---|---|
| APEX | GPU CUDA T4/A10 | batch pré-attention unifié + sync CPU différée | +84-96% |
| HeteroInfer | Snapdragon 8 Gen3 | **partition du tenseur de poids par LIGNES entre GPU+NPU** sur la MÊME op + sync prédictive (sommeil calibré + poll flag mémoire partagée) au lieu de synchronize() | 1.34-6.02× ; 60 GB/s (96% pic) |
| HeteroMosaic | AMD Ryzen AI (XDNA) | graphe restructuré en micro-batches + split GEMM M/K/N entre devices + flux HIP indépendants + signalisation host | 2.05× vs llama.cpp, -45.3% énergie |
| HeRo | multi | `CL_MEM_USE_HOST_PTR` zero-copy CPU/GPU/NPU | — |

HeteroMosaic déclare explicitement : *"llama.cpp supports micro-batching, but this remains
primarily a memory-management mechanism within a selected backend path, not a mechanism for
exposing cross-accelerator overlap."*

---

## 6. ARCHITECTURE D2 PLANNER (coordination 5070 + NPU + bus)

### 6.1 Modèle de décision (place_expert.py — coût argmin par expert)

```
for expert in routed_experts:
    cost_rtx  = GPU_compute + VRAM_transfer + sync          (BW_RTX 384 GB/s)
    cost_xdna = SSD + RAM + DMA + NPU_compute + sync         (BW_GTT/BO, 8 cols)
    cost_cpu  = DDR_read + compute                           (BW_DDR 89.6)
    cost_ssd  = SSD_io + staging
    placement = argmin(cost_rtx, cost_xdna, cost_cpu, cost_ssd)
    + pénalité contention mémoire (BW partagée)
    + contraintes physiques (VRAM 8 GB pleine → RTX inf ; NPU cols=0 → XDNA inf ;
      thermal > 70°C → RTX inf)
```

### 6.2 Primitives XRT Windows utilisables (prouvées par reverse FLM)

| Primitive | Usage |
|---|---|
| `bo::sync(XCL_BO_SYNC_BO_TO_DEVICE / FROM_DEVICE, size, 0)` | charger expert → NPU / récupérer résultat |
| `group_id` (>0 obligatoire sinon no-op silencieux) | opcode=0, instr=1, ninstr=2, host BOs à partir de 3 |
| `runlist::add(run) → execute → wait` | **grouper les experts d'un token en 1 batch** |
| `hw_context` (8+ concurrents vérifiés) | co-exécution attention NPU + decode GPU |
| `bytes::sync_from/to_device()` | checkpoint/restore KV (reprendre contexte si expert change de backend) |
| `xrt::bo::flags` HOST/CACHE/P2P (0x17 optimal, 0x37 +5.4% execbuf, 0x1F SVM destructif) | allocation |

### 6.3 Trois niveaux de décision du planner

- **Niveau 1 — où est l'expert ?** SSD / RAM / XDNA2 buffers / RTX VRAM
- **Niveau 2 — qui l'exécute ?** RTX (Vulkan/CUDA kernels turbo3/4, dmmv) / XDNA2 (runlist
  NPU, GEMV INT8/BF16) / CPU (fallback)
- **Niveau 3 — quand le charger ?** maintenant / token suivant (prefetch) / +N tokens (staging)
  / jamais (SSD)

### 6.4 Le merge d'experts

- Côté GPU : accumulation en-place via `acc_mode` dans le kernel (y = out ou y += out) →
  weighted sum direct sur le buffer commun.
- Côté NPU : `send_manual_expert_up_gate_q41()` / `down_gate_q41()` (Qwen3.6 MoE) → chaque
  expert écrit son GEMV, combine par accumulation.

### 6.5 Contraintes de co-exécution (leçons mesurées)

1. **Le NPU est idle 99.73% pendant son propre decode FLM** → en le branchant en overflow
   pendant que la 5070 décode, on exploite un hardware quasi gratuit.
2. **Le vrai goulot NPU = MCDM dispatch (3064 IOCTL/token), pas le compute** → toute
   co-exécution doit réduire les IOCTL (batch, runlist, persistent hw_context) sinon le NPU
   ne dépassera jamais ~7-8 t/s.
3. **Persistent hw_context = -65% TTFT** — prérequis avant usage NPU en complément.
4. **Ne pas activer zero-copy aveuglément** : le chemin GTT/host-coherent a une taxe de
   cache-coherency (leçon HRX2) → distinguer host_coherent vs GTT dans le cost model.
5. **Contention DDR5** : iGPU 880M, page file, colonnes idle → si la DDR5 est saturée,
   envoyer des experts au NPU peut ralentir le CPU et l'iGPU (mais pas la 5070, qui a son
   propre GDDR7).

---

## 7. PLAN D'ACTION (5070 + NPU + bus mémoire)

### Voie A — Double-flux indépendant (rapide, faible risque)
1. Build llama.cpp Windows avec CUDA (ggml-cuda Blackwell, dll déjà compilée) + backend XDNA.
2. Deux process, même GGUF mmap → co-exécution vérifiée (timestamps + ps).
3. Mesure contention réelle sur cette machine (dGPU séparé = attendu meilleur que le bus
   partagé Snapdragon).

### Voie B — Co-exécution sur un seul flux (HeteroInfer-style, chantier)
1. Base `origin/self-build-jz` (PR #26501 async déjà intégrée).
2. Partition par LIGNES de MUL_MAT entre CUDA (5070) et XDNA2, calcul simultané.
3. Sync légère : flag mémoire partagée (CUDA managed memory / Vulkan shared buffer) + poll,
   isolé dans un nouveau chemin de code (pas le scheduler générique).
4. Validation correction AVANT perf : diff token-par-token, seed fixe, temp=0.
5. Profiler par shape : ratio de split optimal GPU/NPU par MUL_MAT (d2_rtx_gguf_profiler).

### Levier bus mémoire concret (le plus rentable)
- **Étendre le cache SRAM NPU aux experts MoE fréquents** : FLM garde déjà ~1.6 GB de poids en
  SRAM (W_eff 2.87 au lieu de 3.83 GB). Si le planner place les experts fréquents en SRAM NPU,
  le bus DDR5 se libère pour la 5070 → gain sur les deux tiers simultanément.
- **Réduire les octets/token** (quantification turbo3/turbo4, NVFP4, TQ2/Q1) : débit ∝
  1/(poids lus/token) → moins de trafic bus = plus de t/s partout.
- **Persistent hw_context** pour éliminer les 2362 ms d'init NPU par requête.

### Risques / pièges (leçons AGENTS.md à conserver)
- **rc=0 ≠ preuve** : un fallback CPU silencieux donne des chiffres plausibles mais faux →
  toujours vérifier `n_backends`, `mirror=0`, `repack>0`, `dev` identifié.
- **Ne pas écrire dans memory.json** côté planner ; mémoire dans governor_state/provenance.json.
- **Thermique** : >70°C = throttling ; hystérésis 60/50°C.
- **Upstream a échoué 2×** à patcher le synchronize du scheduler (PR #17795, #20793 revertés)
  → ne pas patcher naïvement cette ligne.

---

## 8. RÉFÉRENCES LOCALES (dossier xdna2)

- `RAPPORT_XDNA2_D2_PLANNER.md` — architecture + corrections AMD hybrid + reverse FLM + benchmarks
- `OBJECTIF_5070_NPU_BUS.md` — synthèse ciblée 5070+NPU+bus
- `SPEC_RYZEN9_HX365_5070_8GB.md` — adaptation machine réelle
- `place_expert.py` — modèle coût argmin + primitives XRT (v3)
- `cost_contention_patch.py` — overlay contention (à scinder host_coherent/GTT)
- `benchmark_cross_tier.py`, `validate_llama_xdna.py` — squelettes A/B/C + validation
- `reference/` :
  - `SOSC_v4/` : CARTE_FONCTIONNELLE_XDNA2_FLM.md (54 KB reverse complet), 19_FLM_INTERNAL.md,
    resultats_xdna2.json, SOSC_v4_PUBLICATION_READY.md, MAP_COMPLETE.md, D2_SOSC_CROSS.md
  - `mesures_flm/` : logs live Qwen3.5-9B + flm_bottleneck_analyzer.py (roofline calibré)
  - `apply_turbo_cuda_v2.py` : kernels CUDA turbo3/turbo4 pour 5070
  - `AMD_OGA_HYBRID_OFFICIEL.md`, `AMDXDNA_DRIVER_UAPI.md`, `AMD_OSS_NPU_STACK.md`
  - `GATMANNES_RAPPORT_XDNA2.md`, `xdna2-repo/` (ggml-xdna Windows)
  - `fastflow/`, `d2-quant-planner/`, `reference/1bit/` (RE FLM FastFlowLM + kernels Vulkan
    portables — fichiers bruts de recherche externe)
- Sources non modifiées : `geniex_harness` (governor, profiler_v3, AGENTS.md) intactes.

---

*Document généré le 2026-09-20 — consolidation exhaustive de l'état des lieux local.
Références de première main : bench_results/, SOSC_v4, mesures_flm, lama-tensorRT 1050-5070,
AMD OGA/UAPI/XDNA officiels.*