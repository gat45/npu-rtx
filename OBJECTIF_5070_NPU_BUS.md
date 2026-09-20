# OBJECTIF — Faire fonctionner RTX 5070 + NPU XDNA2 ensemble sur le bus mémoire
# Consolidation 2026-09-20 — sources locales (sans clonage) + recherches git

## CONTEXTE MATÉRIEL RÉEL (machine cible)
- CPU/NPU : AMD Ryzen AI 9 365 (Strix Point) — NPU XDNA2, DRM amdxdna, XRT Windows/Linux
- GPU : **RTX 5070 8 GB (Blackwell sm_120, 384 GB/s GDDR7)** — projet déjà présent :
  `C:\Users\videl\Desktop\lama-tensorRT 1050-5070` (ggml-cuda.dll 32MB compilé CUDA Blackwell)
- RAM 32 GB DDR5 (bus mémoire partagé CPU/NPU/iGPU)
- Fait critique : le NPU et le GPU **ne partagent PAS le même contrôleur mémoire** sur cette
  machine (discret dGPU vs NPU sur DDR) → la contention n'est PAS le problème Strix Halo
  (226 GB/s unifiés). C'est un avantage pour la co-exécution.

## CE QUI EST DÉJÀ PROUVÉ LOCALEMENT (git geniex_harness + dossier 5070)

### 1. Co-exécution GPU+NPU sur bus mémoire : VALIDÉE (device Snapdragon, 2026-09-05)
`bench_results/CONCURRENCE_GPU_NPU_QWEN35_9B_MMAP_20260905.md` :
- 2 process (GPUOpenCL + HTP0) sur le MÊME fichier GGUF via `mmap` partagé → pages physiques
  partagées, pas de duplication (MemAvailable descendu à 1.57GB au lieu de ~0 pour 2×4.94GB).
- Résultat 9B : GPU 6.87 (+2.4%), NPU 6.63 (-1.6%) → **débit agrégat 13.50 t/s, contention quasi nulle**.
- **Pénalité inversement corrélée à la taille du modèle** : petits modèles (≤2B) = contention
  BW dominante (-14 à -30%) ; gros modèles (≥9B) = compute domine, contention marginale.

### 2. Split séquentiel GPU+NPU = PIRE que pur (pas de chevauchement)
`RESULTATS_GPU_NPU_ORDER_QWEN17B_20260905.md` : sur 1.7B, GPU-first 18.62 / NPU-first 18.54
tous deux pires que GPU pur 29.59 et NPU pur 39.86, CPU pur 46.60. Le split par couches
séquentiel n'est PAS de la co-exécution (synchronise bloquant après chaque split).

### 3. Cause racine dans le code + l'infra async existe déjà
`PLAN_COEXECUTION_GPU_NPU_RECHERCHE_COMPLETE_20260905.md` :
- `ggml-backend.cpp` : `ggml_backend_synchronize(split_backend)` force un blocage synchrone
  après CHAQUE split (~1893), court-circuitant les événements async.
- Upstream a échoué 2× à patcher cette ligne (PR #17795 reverté, #20793 reverté via #25138).
- **PR #26501 (fusionnée 2026-09-01)** : backend Hexagon async + fences cross-device +
  événements + buffers non-host → l'infrastructure manquante existe.
- **Branche `origin/self-build-jz` dans `E:\oneplus\ab-wt` contient DÉJÀ PR #26501** (4 commits
  de merge) + PR #27785 + 94 commits d'avance → la base pour la co-exécution réelle.

### 4. Papiers convergents (APEX, HeteroInfer, HeteroMosaic)
- **HeteroInfer** (Snapdragon 8 Gen3) : partition du tenseur de poids par LIGNES entre GPU+NPU
  sur la MÊME opération + sync prédictive (sommeil calibré + poll flag mémoire partagée) au lieu
  de synchronize() → 1.34x-6.02x, 60 GB/s mesurés (96% pic).
- **HeteroMosaic** (AMD Ryzen AI / XDNA) : graphe restructuré en micro-batches + split GEMM par
  M/K/N entre devices + flux HIP indépendants → 2.05x vs llama.cpp, -45.3% énergie. Déclare
  explicitement que llama.cpp micro-batching ≠ cross-accelerator overlap.
- **APEX** : batch unifié pré-attention, sync différée CPU/GPU → +84-96%.
- **HeRo** : `CL_MEM_USE_HOST_PTR` pour mapper buffers NPU dans espace GPU (zero-copy 3-processeur).

## LE DÉFI SPÉCIFIQUE 5070 (pas Snapdragon)

Le corpus local est Android (OpenCL + Hexagon, mmap). Pour la 5070 il faut transposer :

| Élément Snapdragon | Transposition RTX 5070 |
|---|---|
| backend GPUOpenCL | **ggml-cuda** (dll déjà compilée Blackwell) ou Vulkan (kernels 1bit) |
| backend HTP (Hexagon NPU) | **XDNA2 via XRT/amdxdna** (xrt::bo, runlist) ou ggml-xdna (fork) |
| mmap partagé GGUF | mmap Windows aussi possible (llama-mmap.cpp) |
| flags sync mémoire partagée | **CUDA P2P / pinned memory / managed memory** ou Vulkan buffer partagé |
| split par lignes MUL_MAT | possible en ggml (tensor_split) MAIS sync bloquant reste |

## PLAN D'ACTION 5070 + XDNA2 (2 voies complémentaires)

### Voie A — Double-flux indépendant (déjà prouvé conceptuellement) — rapide
1. Build llama.cpp Windows avec **CUDA (ggml-cuda) + backend XDNA** (ggml-xdna du fork 1bit/xdna2-)
   → `--list-devices` montre CUDA0 (5070) + XDNA2.
2. Deux process, même GGUF mmap → co-exécution vérifiée (comme le rapport Snapdragon 9B).
3. Mesure : débit agrégat, contention réelle sur cette machine (dGPU séparé = attendu meilleur
   que le bus partagé Snapdragon).

### Voie B — Co-exécution sur un seul flux (HeteroInfer-style) — chantier
1. **Base `origin/self-build-jz`** (PR #26501 async déjà intégrée) au lieu de réinventer.
2. Implémenter partition par LIGNES de MUL_MAT entre CUDA (5070) et XDNA2, calcul simultané.
3. **Sync légère** : flag mémoire partagée (CUDA managed memory / Vulkan shared buffer) + poll
   au lieu de `ggml_backend_synchronize()` — isolé dans un nouveau chemin de code (pas le
   scheduler générique, pas de risque de régression type #17795/#20793).
4. Validation correction AVANT perf : diff token-par-token, seed fixe, temp=0.
5. Profiler par shape : trouver le ratio de split optimal GPU/NPU par MUL_MAT (d2_rtx_gguf_profiler
   du dossier 5070 peut le faire côté CUDA).

### Contrainte bus mémoire (le vrai sujet)
- **dGPU 5070 (GDDR7 384 GB/s) + NPU (DDR5 ~89 GB/s)** ne partagent pas le même bus → la
  contention "RAM" est uniquement côté NPU/CPU/iGPU. Le gain théorique de Voie B = 384 GB/s
  (GPU) + ~89 GB/s (NPU DDR) au lieu de 384 seul (Laptop), mais le NPU est ~7× plus lent en BW → son
  apport est limité aux experts/ops où le GPU est saturé (decode memory-bound).
- **Réalité chiffrée** (RAPPORT §3sexies) : MoE 35B = GPU 75.65 tok/s vs NPU 11.66 → le NPU
  n'apporte un gain que si le GPU est saturé (8 GB VRAM pleine) ET que les experts overflow
  tiennent sur NPU (XDNA2_COLS_ACTIVE=4, ~11-16 t/s).
- **Conclusion** : l'intérêt principal du NPU ici = tier d'overflow pour la VRAM 8 GB pleine,
  PAS un co-processeur qui débloque la 5070. Co-exécution = double-flux indépendant (Voie A)
  rentable quand 2 requêtes arrivent, ou experts overflow (planner D2).

## FICHIERS / DÉPÔTS UTILES TROUVÉS (sans clonage)
- `E:\oneplus\ab-wt` : worktree llama.cpp, branche `self-build-jz` (PR #26501 async, 94 commits)
- `C:\Users\videl\Desktop\lama-tensorRT 1050-5070` : projet D2 5070 complet (ggml-cuda.dll
  Blackwell, d2_rtx_gguf_profiler.py, plans sm_120 NVFP4, RAPPORT_TIERS_HARDWARE_2026-09-13.md)
- `bench_results/CONCURRENCE_GPU_NPU_*.md`, `PLAN_COEXECUTION_*.md`, `RESULTATS_GPU_NPU_ORDER_*.md`
- `reference/1bit/kernels-vulkan/` : kernels Vulkan portables (dmmv_tq2/q1/matmul) pour tier GPU
- `reference/1bit/amd-oss/` : stack IRON/Triton-XDNA/MLIR-AIR pour compiler backend XDNA2
- `D:\vrac tour\mesures_flm/` → `reference/mesures_flm/` : mesures live FLM Qwen3.5-9B
  (7.05-7.42 t/s, w_eff 2.86-2.92 GB) + flm_bottleneck_analyzer.py (roofline calibré BW_eff 21.93 GB/s)
- `D:\vrac tour\SOSC_v4/` → `reference/SOSC_v4/` : CARTE_FONCTIONNELLE_XDNA2_FLM.md (54KB, reverse
  FLM complet) + 19_FLM_INTERNAL.md + resultats_xdna2.json + SOSC_v4_PUBLICATION_READY.md
- `D:\vrac tour\apply_turbo_cuda_v2.py` → `reference/apply_turbo_cuda_v2.py` : patch CUDA turbo3/turbo4
  pour llama-cpp-turboquant (kernels k_set_rows, mixed K/V, norm caching) — tier GPU 5070

## CORPUS SOSC_v4 — REVERSE FLM COMPLET (le référentiel bus mémoire XDNA2)
`reference/SOSC_v4/CARTE_FONCTIONNELLE_XDNA2_FLM.md` (55 découvertes, Ghidra 14318 fns) :

### Données mesurées clés (machine cible = Ryzen AI 9 365 Strix Point)
- **NPU XDNA2** : PCI 0x17F0 rev 0x10 = NPU4 (aie2p), 8 cols × 4 rows = 32 tiles AIE2P, **4 colonnes
  exposées à XRT** (4 réservées firmware), 1.8 GHz, **51.3 TOPS INT8 peak / 38 eff / 9-12 GEMM**.
- **SRAM NPU** : 2 MB L1 (64KB × 32 tiles) + 4 MB L2 (512KB × 8 MemTiles) = 6 MB total.
- **BW DDR5 effective decode = 21.93 GB/s = 24.5% du peak 89.6** (le vrai plafond du NPU).
  Causes : iGPU 880M -3.5 GB/s (framebuffer 2560×1600@180Hz) · column-mapping idle tiles -40% ·
  page file -10% (92.4% RAM utilisée) · overhead XRT ×3.7.
- **Contention RAM = inflation W_eff +60%** : Qwen3.5-9B en contention W_eff 4.59-4.93 GB → TPS
  7.43→4.27-4.65 ; propre W_eff 2.86-2.92 GB → TPS 7.06-7.43. **Le bus mémoire EST le goulot.**
- **Goulot #1 = MCDM SHIM dispatch** : run::wait 81.32 ms/token = 96.5% du temps NPU, 3064 dispatch
  × 67µs = 205ms/token, 50 IOCTL/token. Cause : **Windows MCDM n'a PAS force_cmdlist** (Linux oui) →
  chaque commande = 1 IOCTL séparé. Bypass estimés : proxy driver +393%, direct PCIe +500%.
- **"KDMA not supported on windows"** (2× dans xrt_coreutil) : le chemin XRT public (ERT) ne peut
  PAS faire le DMA direct sur Windows → FLM passe par **vitis-ai-runtime2.dll → RadeonML_ipu.dll →
  D3DKMT → ipustack.sys** (MCDM), pas par le scheduler ERT.
- **xrtBOAlloc IGNORE le paramètre size** (rdx jamais utilisé, 12 appels analysés) : le driver
  décide la taille lui-même — le proxy marche par timing/état initial, pas par correction de size.
- **Q4NX = I8 (INT8), PAS INT4** : 7.9 GB au lieu de 4.5. Gain = réduction BW uniquement
  (dequant→BF16 avant compute). Pas de compute INT4 natif.
- **Décomposition temporelle/token** : CPU 86ms (86.8%), NPU idle 99.73% (138/51300 GOPS actifs),
  DMA idle pendant compute (pas de prefetch layer suivant).
- **Bench Qwen3.5-9B : FLM NPU 7.2-7.8 t/s vs RTX 5070 CUDA 53.83 t/s, prefill 2029 t/s** →
  GPU ×7.4 le NPU, mesuré localement.

### Opportunité pour l'objectif 5070+NPU (synthèse)
1. **Le NPU est idle 99.73% du temps de token FLM** → si on l'utilise comme tier d'overflow
   PENDANT que la 5070 décode, on exploite un hardware quasi gratuit.
2. **Le vrai problème du NPU = MCDM dispatch, pas le compute** → toute co-exécution 5070+NPU
   doit viser à réduire les 3064 IOCTL/token (batching, runlist, persistent hw_context) sinon
   le NPU ne fera jamais mieux que ~7-8 t/s quoi qu'on y mette.
3. **Persistent hw_context = -65% TTFT** (goulot H6, 2362ms init/requête) — indispensable avant
   tout usage NPU en complément du GPU.
4. **apply_turbo_cuda_v2.py** : le pont quantization→kernel GPU. Turbo3/turbo4 (3/4-bit centroïdes)
   sur CUDA 5070 = moins de bytes/token → le coût DMA du bus baisse (leçon §2 TIERS_HARDWARE :
   débit ∝ 1/taille des poids lus/token).
5. **w_eff 2.87 GB (9B-cust) avec cache SRAM** : FLM garde ~1.6 GB en SRAM NPU → seul le résidu
   passe par le bus. Si on étend ce cache aux experts MoE fréquents, le bus DDR5 se libère pour
   la 5070. C'est LE levier bus mémoire concret.

## DÉCISION REQUISE
- **A court terme** : Voie A (double-flux indépendant 5070+XDNA2 via build CUDA+ggml-xdna) —
  faible risque, gain débit agrégat quand 2 requêtes.
- **A moyen terme** : Voie B (partition par lignes HeteroInfer-style sur base self-build-jz) —
  chantier plusieurs jours, risque isolé au nouveau chemin.
- La vraie valeur du NPU sur CETTE machine = **tier d'overflow experts quand VRAM 8GB pleine**
  (planner D2), pas compétiteur de la 5070 en vitesse brute.