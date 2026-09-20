# RECHERCHE EXHAUSTIVE — NPU XDNA2 + RTX : chaque point creusé, questions, preuves, URLs
# Établi 2026-09-20 · Dossier npu-rtx/
# Répond point par point à l'analyse "faire tourner NPU + RTX + CPU" avec preuves (mesures
# locales) + sources URLs. Ne parle PAS du projet 1bit (remplacé par sources AMD/communauté).

---

## POINT 1 — Le vrai problème = le mouvement des données (pas les TOPS)

**Question** : Pourquoi les TOPS ne suffisent-ils pas à comparer NPU et RTX ?
**Réponse** : Pour un LLM en decode, l'intensité arithmétique ≈ 1 FLOP/octet → le débit est
proportionnel à `1 / (poids lus/token)`, pas aux TOPS. La bande passante mémoire domine.

**Preuve locale** : `RAPPORT_TIERS_HARDWARE_2026-09-13.md` — Qwen9B IQ4NL 5.0 GB → 32.9 t/s →
BW effective 165 GB/s (51% du théorique 320). Le débit suit la lecture des poids.

**Preuve AMD officielle** : GEMM NPU XDNA2 mesuré par AMD (IRON) :
- XDNA (Phoenix) : **6.76 TOPS INT8 / 3.14 TFLOPS BF16**
- XDNA2 (Krackan) : **38.05 TOPS INT8 / 14.71 TFLOPS BF16**
- Source : arXiv:2512.13282 "Striking the Balance: GEMM Performance Optimization Across
  Generations of Ryzen AI NPUs" (DOI 10.1145/3748173.3779551)
- URL : https://arxiv.org/html/2512.13282

**Preuve locale bus** : `CARTE_FONCTIONNELLE_XDNA2_FLM.md` — Qwen3.5-9B decode : NPU idle
99.73% (138/51300 GOPS actifs), CPU 86.8% du token, DMA idle pendant compute. Le compute NPU
n'est que 4.5% du temps de token.

**Conclusion** : `T_token = BW_eff / (W_eff + c·KV)`. Réduire les octets/token est le levier
dominant, pas augmenter les TOPS.

---

## POINT 2 — Il y a 4 bandes passantes distinctes

**Question** : Comment séparer les 4 liaisons mémoire et laquelle est catastrophique ?
**Réponse** :

| Liaison | BW (cible HX365/5070) | Coût relatif |
|---|---|---|
| A. DDR ↔ NPU | 21.93 GB/s effective (24.5% de 89.6) | Goulot NPU dominant |
| B. DDR ↔ iGPU | ~30 GB/s (partage bus) | Contention |
| C. VRAM ↔ RTX | 672 GB/s (GDDR7) | Très rapide (local) |
| D. DDR ↔ RTX (PCIe) | ~25 GB/s effective (PCIe 4.0 ×16) | **Critique — le pire** |

**Preuve locale** : CARTE_FONCTIONNELLE — BW DDR5 effective 21.93 GB/s (causes : iGPU -3.5,
column-mapping -40%, page file -10%, overhead XRT ×3.7). Le kernel doc Linux confirme :
"Each column also has dedicated DMA engines to move data between host DDR and memory tile"
et "Strix Point = 4×8 topology" (URL : https://www.kernel.org/doc/html/latest/accel/amdxdna/amdnpu.html).

**Preuve communautaire (FATE)** : PCIe 4.0 ×16 effectif ~25 GB/s vs D2D GPU ~500 GB/s →
20× par octet. URL : https://github.com/ongunm/llama-moe-cache/blob/main/FATE_RESULTS_QWEN3.md

**Conclusion** : le lien DDR↔RTX (PCIe ~25 GB/s) est le plus lent → il faut MINIMISER les
transferts CPU↔GPU, pas les optimiser. Tout expert transféré à chaque token = perte.

---

## POINT 3 — Le NPU est utile malgré ses 50 TOPS

**Question** : À quoi sert le NPU si la RTX a beaucoup plus de TOPS ?
**Réponse** : Le NPU partage la DDR avec CPU/iGPU, a ses propres DMA engines colonne-par-
colonne (kernel doc), et FastFlowLM garde ~1.6 GB de poids en SRAM (W_eff 2.87 au lieu de
3.83). Le NPU est utile pour : (1) les petits tensors fréquents, (2) l'orchestration/prédiction
(router, prefetch), (3) les opérations où la frontière DDR↔PCIe serait plus chère que le
compute local.

**Preuve** : CARTE_FONCTIONNELLE §9.2 — "W_eff (poids decode) = 2.87-3.83 GB chargé layer par
layer" → le cache SRAM NPU réduit le W_eff. Le NPU idle 99.73% pendant FLM = capacité de
compute inutilisée que le planner peut exploiter.

---

## POINT 4 — Ne pas utiliser la RTX comme "deuxième NPU" (split par couches)

**Question** : Pourquoi le split par couches (Layer1 NPU, Layer2 RTX...) est une erreur ?
**Réponse** : Chaque frontière layer = transfert NPU→DDR→PCIe→VRAM→RTX→PCIe→DDR→NPU. Le
pipeline devient communication-bound (chaque token traverse PCIe 2× par frontière).

**Preuve locale** : `RESULTATS_GPU_NPU_ORDER_QWEN17B_20260905.md` — split séquentiel GPU/HTP
0.8/0.2 = 18.6 t/s, pire que GPU pur 29.59, NPU pur 39.86, CPU pur 46.60. Le split séquentiel
n'est PAS de la co-exécution (synchronize bloquant après chaque split).

**Preuve source** : `PLAN_COEXECUTION_GPU_NPU_RECHERCHE_COMPLETE_20260905.md` — la cause racine
est `ggml_backend_synchronize(split_backend)` (~1893) qui bloque après chaque split ; PR #17795
et #20793 revertés par #25138 (upstream a échoué 2×). **PR #26501** (fusionnée 2026-09-01)
apporte l'infra async (fences, événements) — dans `origin/self-build-jz` du worktree
`E:\oneplus\ab-wt`.

**Conclusion** : partitionner par RÉGIONS ou par experts (frontières larges), jamais par layer.

---

## POINT 5 — Partitionner par régions / MoE

**Question** : Quelle partition est optimale pour un MoE ?
**Réponse** : `NPU = dense/attention/gating/routing` + `RTX = experts` (grosses GEMM) est le
schéma logique : les experts = beaucoup de poids peu utilisés, la RTX excelle dessus, et le
NPU gère l'orchestration sans traverser PCIe à chaque token.

**Preuve** : Esonhjz benchmarks (RTX 5070 Ti 16GB, Qwen3.6-35B-A3B) : attention GPU + experts
CPU (`-ncmoe 999`) = modèle MoE 35B utilise MOINS de VRAM qu'un dense 8B (3876 vs 7635 MiB) et
va plus vite. URL : https://github.com/esonhjz/llama-cpp-moe-vram-benchmarks
OpenClaw : "experts en RAM + attention GPU = pourquoi on fait tourner des 35B sur 12-16 GB".
URL : https://openclawdc.com/blog/llama-cpp-moe-offload-flags-explained

---

## POINT 6 — Le problème PCIe : les experts ne doivent PAS être re-transférés

**Question** : Comment éviter le transfert PCIe répété des experts ?
**Réponse** : **Cache VRAM persistant d'experts** (slots fixes, les poids restent résidents ;
seules les activations circulent). C'est la pièce fondamentale.

**Preuves mesurées (4 implémentations indépendantes)** :

1. **llama.cpp Discussion #28248 (RFC officiel, +84%)** :
   - `--moe-expert-cache N (-mec N)` : pool persistant de N slots experts en VRAM pour tout
     expert offloadé en RAM.
   - RTX 4090, Qwen3.8-Flash-Next : 11.55 → 21.21 t/s (**+84%**) avec 64 slots, hit 90-95%.
   - URL : https://github.com/ggml-org/llama.cpp/discussions/28248

2. **FATE (llama-moe-cache, ongunm)** :
   - Qwen3-30B-A3B Q4_K_M (18.6 GB) sur RTX 4070 Ti 12GB : **hit rate 99.50%**, D2D ~500 GB/s
     vs PCIe H2D ~25 GB/s (20×), prefetch cross-layer + temporal, pool 1663 slots × 1.2 MB.
   - **1.91×** speedup. Seulement ~50% des experts changent entre tokens consécutifs.
   - URLs : https://github.com/ongunm/llama-moe-cache (FATE_RESULTS_QWEN3.md,
     FATE_IMPLEMENTATION_REPORT.md)

3. **PR #24524 / gist dekoza ("118B MoE on one RTX 3090")** :
   - Laguna-S-2.1 118B-A8B UD-Q4_K_XL, 40 layers experts CPU-resident : **+62.6% decode warm**,
     +40-50% à 16k-98k depth, self-disable à ~192k (hit rate < break-even).
   - Préférence placement : "laisser TOUS les experts sur CPU + cache dynamique" mesure plus
     vite que tout placement statique partiel (754B: 19.2 vs 14.0 t/s ; 397B: 33.3 vs 28.2).
   - URL : https://gist.github.com/dekoza/e6b4a69989b3d5bd0f904ce204f1646c

4. **PR #26563 (miltos22)** : heatmap experts + hot experts en GPU. Qwen3.6-35B-A3B Q2_M 8GB
   VRAM : **1.7-2.1×** (56 vs 33, 36 vs 17 tok/s) autofit. Off par défaut (`-ehs N`).
   URL : https://github.com/ggml-org/llama.cpp/pull/26563

**Break-even cache-hit ≈ 42%** (fork RTX 2060 12GB, Qwen35B A3B : 19→26 t/s à 62% hit).
URL : https://theneuralfeed.com/article/experts-first-llama-cpp/9SXuDKah

**Conclusion** : le premier gros gain n'est PAS la déquantification, c'est **éviter de
re-transférer le même expert** via PCIe. Cible : hit rate > 90% (slots ≥ working set + prefetch
prédictif).

---

## POINT 7 — Qwen3.8-Flash-Next : l'architecture idéale pour le cache expert

**Question** : Pourquoi Qwen Flash est-il le cas parfait ?
**Réponse** : 48 layers × 512 experts/layer, top-10 routed + 1 shared → seulement ~2% des
poids sont actifs par token. Le problème n'est pas "faire tenir 120 GB dans 8 GB" mais "faire
arriver les ~11 experts au bon endroit avant le GEMM".

**Preuves** :
- GitHub QwenLM/Qwen3.8-Flash-Next : "GDN + QSA hybrid, Gated DeltaNet compresses history,
  Qwen Sparse Attention selects important context at micro-block granularity". URL :
  http://github.com/QwenLM/Qwen3.8-Flash-Next
- llamaperf : Qwen3.8-Flash-Next sur 2×3090 : 41 t/s decode avec "expert cache + MTP, cache
  hit 90-92%, host RAM 73 GB pinned + 28 GB PLE". URL : https://llamaperf.com/
- qwen3.8-flash-next-16gb (hocestnonsatis) : 16 GB avec UD-IQ1_S + experts/PLE sur CPU/SSD mmap,
  llama.cpp PR #27742 Vulkan. URL : https://github.com/hocestnonsatis/qwen3.8-flash-next-16gb

**Conclusion** : Qwen Flash coche TOUTES les cases : experts petits (~MB), routing parcimonieux,
corrélation temporelle du routing exploitable → le cache + prefetch y fonctionnent au maximum.

---

## POINT 8 — JIT-dequant fusionnée : pas de buffer FP16 géant

**Question** : Faut-il déquantifier l'expert complet en FP16 avant GEMM ?
**Réponse** : NON. Les kernels CUDA (Q4_K/Q5_K/Q6_K) déquantifient par bloc dans le kernel
(`dequantize_q4_K()`), MMQ lit les blocs quantifiés et reconstruit les valeurs pendant le
compute. Garder Q4/Q5/Q6 partout (SSD→RAM→VRAM), déquantifier par tuile dans shared/register.

**Preuves** :
- ggml-cuda : chemins Q4_K/Q5_K/Q6_K natifs dans les kernels MMQ.
- CARTE_FONCTIONNELLE §7 : Q4NX = I8 (pas INT4) ; la déquant se fait avant compute → le format
  stocké doit rester compact (Q4/Q5/Q6), jamais FP16 en transit.
- TurboQuant (llama.cpp) : turbo3/turbo4 (3/4-bit) en VRAM KV et poids — mesuré sur ROCm 3×
  7900XTX : turbo3 KV = -63.7% mémoire, PPL +0.45%. URL :
  https://github.com/mkadrlik/llama-cpp-vulkan-rocm/blob/main/benchmarks/README.md

**Conclusion** : `Q4 en VRAM → kernel déquant tile → MMA`, jamais `Q4 → FP16 4× → GEMM`.

---

## POINT 9 — Quantification multiple simultanée (hot/cold)

**Question** : Un expert peut-il exister en plusieurs formats selon son état ?
**Réponse** : Oui — hot tiles en Q8/Q6, cold en Q4. Le cache VRAM garde Q4/Q5, le buffer de
dequant est uniquement les tiles calculées. Ça multiplie la capacité en 8 GB.

**Preuves** :
- apply_turbo_cuda_v2.py : kernels CUDA turbo3/turbo4 (3/4-bit centroïdes ±0.1907), mixed K/V
  (turbo K + q8_0 V), norm caching FA → preuve que plusieurs formats coexistent par tensor.
- CARTE_FONCTIONNELLE : Qwen3.5-9B Q4NX = 475 tenseurs dont BF16 (178) + I8 (249) + F32 (48) →
  formats mixtes déjà présents dans un modèle.

**Conclusion** : le planner choisit `format(expert, état)` : hot→Q6/Q8 en VRAM, warm→Q4,
cold→Q4 sur SSD. Quantification = propriété de résidence, pas propriété fixe du modèle.

---

## POINT 10 — Cache quantifié en VRAM (slots Q4/Q5, pas FP16)

**Question** : Le cache d'experts doit-il être FP16 ou quantifié ?
**Réponse** : Quantifié (Q4/Q5). 64 experts × Q4 tiennent dans un working set bien plus grand
que FP16. Dequant buffer = seulement les tiles calculées.

**Preuves** :
- Discussion #28248 : le cache garde des "slabs d'experts dans le format du modèle", utilisés
  via MUL_MAT_ID — pas de copie FP16. URL : https://github.com/ggml-org/llama.cpp/discussions/28248
- FATE : pool de slots garde les experts en Q4_K_M (format du GGUF), D2D entre slots.
- PR #26563 : "the GPU will act as caching" avec heatmap, formats inchangés.

---

## POINT 11 — Piège : tous les tenseurs ne supportent pas tous les formats

**Question** : Peut-on tout mettre en Q4_K ?
**Réponse** : NON. Il faut une **Quantization Capability Matrix** testée, pas supposée. Des
bugs documentés existent : certaines combinaisons Q4_K/Q5_K dans MUL_MAT_ID produisent des
sorties incorrectes ; passer `ffn_down_exps` en Q6_K corrige.

**Preuves (issues llama.cpp)** :
- **#24591** (ouverte) : crash `MUL_MAT_ID` CUDA "illegal memory access" avec IDs d'experts
  non-uniques (DeepSeek-V4-Flash-162B) — le kernel mmid.cu ne gère pas les IDs dupliqués
  (`nex_prev`/`it_compact` non initialisés). URL : https://github.com/ggml-org/llama.cpp/issues/24591
- **#21289** : `MUL_MAT_ID failed CUDA error: invalid argument` — fix = `CMAKE_CUDA_ARCHITECTURES`
  correct (520 vs 86). URL : https://github.com/ggml-org/llama.cpp/issues/21289
- **#13252** : `MUL_MAT failed` avec FA + MLA DeepSeek-V3 en mixte CPU+GPU, fixé par PR #13306.
  URL : https://github.com/ggml-org/llama.cpp/issues/13252

**Conclusion** : le planner doit vérifier `backend × tensor × quant × shape × accuracy` et
générer automatiquement la matrice de capacités par test (pas par hypothèse).

---

## POINT 12 — Prédiction, pas juste LRU (le D2 Planner dépasse le cache)

**Question** : Comment aller au-delà d'un cache LRU ?
**Réponse** : Apprendre `P(expert_j | expert_i)` (corrélation inter-layer et temporelle). FATE :
prédiction cross-layer (layer N → N+1) + temporelle (token t → t+1) ; 37998 prefetch / 76074
acces = ~50% du trafic PCIe éliminé. La corrélation du routing entre tokens consécutifs est
confirmée (HOBBIT Fig. 3.2, cité dans FATE).

**Preuves** : FATE_IMPLEMENTATION_REPORT.md — le prefetcher (thread worker + stream CUDA dédié)
soumet des `cudaMemcpyAsync` sur le stream de prefetch. URL :
https://github.com/ongunm/llama-moe-cache/blob/main/FATE_IMPLEMENTATION_REPORT.md
Discussion #28248 : "issues async H2D copies". FlashMoE paper : cache ML ~113KB per layer
(recency+frequency) → +51% hit-rate vs LRU/LFU, 2.6×. URL : https://arxiv.org/abs/2601.17063

---

## POINT 13 — Double buffering : masquer PCIe derrière GEMM

**Question** : Comment rendre le transfert invisible ?
**Réponse** : Pipelining : pendant `GEMM(A)`, faire `DMA(B)`, puis `GEMM(B)` + `DMA(C)`. Objectif
`T_DMA ≤ T_GEMM`. Slots A/B (voire triple pour experts : batch courant / transfert / résultats).

**Preuves** :
- PR #26563 : "asynchronous H2D copies", self-contained. FATE : prefetch stream dédié.
- Local : `CONCURRENCE_GPU_NPU_QWEN35_9B_MMAP` — 2 flux indépendants → 13.5 t/s agrégat sans
  pénalité sur 9B (les process se chevauchent nativement).

---

## POINT 14 — Synchronisation asynchrone (le goulot n°1 Windows)

**Question** : Pourquoi le NPU est-il limité à ~7-8 t/s alors que son compute est 5 ms ?
**Réponse** : **MCDM SHIM dispatch** : `run::wait` = 81.32 ms/token = 96.5% du temps NPU,
3064 dispatch × 67 µs = 205 ms/token, 50 IOCTL/token. Windows n'a PAS `force_cmdlist`
(contrairement à Linux amdxdna.ko) → chaque commande = 1 IOCTL séparé. Et **"KDMA not
supported on windows"** dans xrt_coreutil → le chemin XRT public ne peut pas faire le DMA
direct ; FLM contourne via vitis-ai-runtime2/RadeonML/D3DKMT/ipustack.

**Preuves** : CARTE_FONCTIONNELLE §14 (Goulot #1) : bypass estimés proxy driver +393%, direct
PCIe +500%. `run::wait` = 81.32 ms/token. Décomposition token : CPU 86 ms (86.8%), NPU compute
5 ms (4.5%).

**Conclusion** : toute co-exécution NPU+RTX doit (1) batch les commandes NPU (runlist), (2)
utiliser un hw_context persistant (goulot H6 = 2362 ms/requête → -65% TTFT), (3) idéalement
contourner MCDM (proxy IOCTL / chemin vitis-ai-runtime2).

---

## POINT 15 — La VRAM et le PCIe doivent être QUERY à l'exécution

**Question** : Peut-on coder "RTX 5070 = 8 GB" en dur ?
**Réponse** : NON. La 5070 desktop = 12 GB, laptop = 8 ou 12 GB. Le planner doit `query VRAM
libre / BW / topologie PCIe` au démarrage.

**Preuves** :
- RAPPORT_TIERS_HARDWARE : "GTX 1080 et RTX 5070 ont la même VRAM (8 GB) — ce qui les sépare
  c'est la bande passante et les tensor cores". La 5070 laptop = 8 GB ici.
- Framework Laptop 16 : modules RTX 5070 Laptop 8/12 GB documentés (référence utilisateur).
- llamaperf : RTX 3090 (24 GB) est la carte de référence communautaire pour MoE.

---

## POINT 16 — Layout conversion = pire que la quantification

**Question** : Quel est le coût caché le plus dangereux ?
**Réponse** : Les conversions de layout (transpose/reshape/repack/quantize) aux frontières
NPU↔RTX peuvent coûter plus que le compute (ex. compute 0.3 ms, repack 0.8 ms).

**Preuves** :
- Local : Q4NX a un layout propriétaire tile (32×256 = 5120 B/block, scales BF16, Q4 nibbles
  repackés) — CARTE_FONCTIONNELLE §7.3. Passer de Q4NX→GGUF Q4_K → CUDA = 2 repacks.
- FATE : le cache évite le repack en gardant le format du modèle et en copiant par offset
  byte dans un tenseur miroir.
- Leçon AGENTS.md : les conversions/récopies coûtent (K5 : 100% splits = backend_switch).

**Conclusion** : le planner doit avoir `layout_conversion_cost` par paire (source,dest).

---

## POINT 17 — JIT-QUANT (quantifier les activations avant PCIe)

**Question** : Comment réduire le trafic PCIe des activations ?
**Réponse** : Quantifier les activations côté NPU avant PCIe (FP16→Q8 = ÷2, →INT4 = ÷4) puis
déquant côté RTX. Perte numérique à mesurer.

**Preuves** :
- CARTE_FONCTIONNELLE : activation [1×4096] FP16 = 8 KB (petite frontière OK) mais [4096×4096]
  FP16 = 32 MB (grosse frontière = choisir le split différemment).
- Leçon locale : la déquant INT4 est un mythe de gain compute (CARTE §3) → JIT-quant sert au
  TRANSFERT, pas au compute.

---

## POINT 18 — Le profiler doit mesurer ~30 compteurs

**Question** : Que mesurer pour alimenter le cost model ?
**Réponse** : Les compteurs de l'analyse : compute (npu/gpu/cpu µs), transfert
(ddr↔npu, ddr↔gpu), PCIe (tx/rx bytes+µs), mémoire (dram/vram read/write, L2 hits), runtime
(xrt submit/wait, cuda launch/sync), graphe (kernel/fusion/boundary/conversion), pipeline
(overlap, idle npu/gpu, dma overlap).

**Preuves locales** :
- `mesures_flm/flm_bottleneck_analyzer.py` : roofline calibré R²=0.96 (BW_eff 21.93, η 0.7309,
  W_eff, KV bytes/token) — le modèle de coût existe déjà.
- CARTE_FONCTIONNELLE §6.3 : xrt_trace.csv (6144 lignes) : bo::sync 62.9%, run::wait 18%,
  runlist::wait 18%, bo::~bo 1.1% → 12.6 s.
- d2_rtx_gguf_profiler.py (dossier 5070) : profilage GGUF par couche CUDA.

---

## POINT 19 — Le seuil de rentabilité de l'offload (break-even)

**Question** : Quand ne PAS offloader sur RTX/NPU ?
**Réponse** : `GPU_OFFLOAD rentable ⟺ GPU_compute_gain > transfer_cost + sync_cost +
conversion_cost`. Exemple : gain GPU 200 µs mais PCIe 140 + sync 40 + conversion 50 = 230 µs
→ refuser.

**Preuves** :
- Discussion #28248 : break-even hit-rate ~42% (mesuré fork RTX 2060). PR #24524 : self-disable
  à ~192k ctx quand hit rate < break-even.
- FATE : à 0% hit (cache trop petit, 108 slots < working set) → 1.14 t/s = PIRE que vanilla
  6.4 t/s : un cache mal dimensionné est contre-productif. URL :
  https://github.com/ongunm/llama-moe-cache/blob/main/FATE_IMPLEMENTATION_REPORT.md

---

## POINT 20 — 4 architectures à tester

**Question** : Quelles configs comparer ?
**Réponse** : A = NPU-only, B = RTX-only, C = static hybrid (régions fixes), D = adaptive
hybrid (D2 Planner). D est l'objectif.

**Preuves** :
- Local : NPU-only Qwen3.5-9B = 7.2-7.8 t/s (FLM), RTX-only = 53.83 t/s (CUDA), → ratio 7.4×.
- HeteroMosaic (AMD Ryzen AI) : jusqu'à 1.73× vs iGPU, 1.78× vs NPU, 2.05× vs llama.cpp,
  -45.3% énergie (MICRO 2026). URL : https://arxiv.org/abs/2607.12839
- AMD OGA hybrid : NPU+iGPU (prefill NPU / token phase GPU) = preuve que la partition NPU/GPU
  est viable mais pas NPU+RTX. URL : https://ryzenai.docs.amd.com/en/main/hybrid_oga.html

---

## POINT 21 — Le bus mémoire = ressource du scheduler

**Question** : Le scheduler doit-il considérer le bus comme une ressource ?
**Réponse** : Oui. Resource vector : NPU_compute, GPU_compute, CPU_compute, DDR_bw, VRAM_bw,
PCIe_bw, NPU_SRAM, VRAM, DRAM, DMA, CUDA_stream, XRT_queue. Chaque op consomme plusieurs
ressources simultanément.

**Preuves** :
- HeteroMosaic : "trace-guided critical-interval co-optimization under memory contention, DVFS,
  device variation, NPU runtime overheads" — la contention mémoire est un paramètre explicite.
- Local : contention DDR = W_eff +60% → TPS 7.43→4.27 (mesuré). Le planner doit prédire cette
  inflation.

---

## POINT 22 — SSD streaming : le flash (preuve 2.8 GB/s vs 377 MB/s)

**Question** : Le streaming depuis SSD est-il rentable ?
**Réponse** : Oui si (1) page cache chaud (répétition), (2) I/O async chevauché avec compute.
Mesure réelle : un MoE 73.5 GB a atteint **~2.8 GB/s en lecture asynchrone** vs ~377 MB/s en
paging mmap. Réduire les octets ne suffit pas si les copies/évictions coûtent trop.

**Preuves** :
- FlashMoE paper (arXiv:2601.17063) : +51% hit-rate vs LRU/LFU, 2.6× speedup, cache ML 113KB,
  "expert load SSD ~3ms vs FFN 158µs → chevauchable".
- cecil-the-coder/llama-cpp-moe-flash : budget I/O Qwen3-235B Q2_K = 4.3 GB/token → 634 ms @
  7 GB/s cold / 148 ms @ 30 GB/s warm ; "flash-moe async prefetch : 2.3 vs 4.1 t/s (lecture
  disque chaque token) — plus lent que cache GPU buffers". URL :
  https://github.com/cecil-the-coder/llama-cpp-moe-flash/blob/main/README.md
- SSD energy : arXiv:2508.06978 — le SSD coûte plus d'énergie/bit que DRAM. URL :
  https://arxiv.org/abs/2508.06978

**Conclusion** : streaming SSD = dernier recours (cold), jamais pour les experts fréquents.
RAM/VRAM cache d'abord.

---

## POINT 23 — Le découpage fin : expert → tensor → block (pas "expert" entier)

**Question** : L'unité de scheduling doit-elle être l'expert ?
**Réponse** : NON. Un expert = gate + up + down, chacun en blocs quantifiés. Scheduler par
`expert → tensor → row block → quant block`. Le cache peut garder des SLABS d'experts (offsets
byte dans le tenseur miroir) — pas nécessairement l'expert complet.

**Preuve** : Discussion #28248 : "copies by byte offset into a GPU tensor that mirrors the full
CPU tensor layout (expert_offset = first_id * expert_size) — no slot remapping". FATE :
1 slot = 1 tensor expert (gate/up/down séparés).

---

## POINT 24 — Priorité des problèmes (PCIe et DDR d'abord)

**Question** : Par quoi commencer ?
**Réponse** : Critiques : PCIe/transfert RTX↔DDR, DDR partagé, sync NPU↔GPU, VRAM limitée,
poids résidents, layout/quant conversion, DMA overlap. Élevés : SRAM NPU, kernel launch, XRT
overhead, CUDA sync, orchestration CPU, power/thermal. Faibles : fragmentation, NUMA, compile
cache, scheduling dynamique.

**Preuve** : le document d'analyse (37 points priorisés) + preuves mesurées locales (PCIe =
goulot cache, DDR = goulot NPU).

---

## POINT 25 — OGA hybrid AMD = référence à disséquer (mais pas NPU+RTX)

**Question** : AMD fournit-il déjà le hybride NPU+GPU que je veux ?
**Réponse** : NON. AMD fournit OGA hybrid = NPU+iGPU (même mémoire, iGPU pas RTX). Mais c'est
la preuve que la partition NPU/GPU par phase est viable, et `hybrid_opt_free_after_prefill`
libère le NPU pendant le decode → slot pour l'overflow experts pendant que la 5070 décode.

**Preuves** :
- https://ryzenai.docs.amd.com/en/main/hybrid_oga.html (OGA 1.8 : hybrid NPU+iGPU, prefill/decode,
  `hybrid_opt_free_after_prefill: 1`, Strix/Krackan Point)
- https://www.amd.com/zh-cn/developer/resources/technical-articles/model-pipelining-on-npu-and-gpu-using-ryzen-ai-software.html
  (pipelines NPU+iGPU : 179.65 s → 16.57 s en répartissant modèles)
- HeteroMosaic (arXiv:2607.12839) : roofline hétérogène pour décider QUAND combiner iGPU+NPU.

---

## POINT 26 — Ce qu'il faut MESURER maintenant (matrice de transferts 3×3)

**Question** : Quelle est la prochaine action concrète ?
**Réponse** : Mesurer la matrice de transferts sur la machine cible :
```
              CPU/DDR       NPU        RTX/VRAM
CPU/DDR          —           ?           ?
NPU              ?           —           ?
RTX              ?           ?           —
```
Pour chaque paire : latence, GB/s, tailles 1K→1GB, sync cost, CPU overhead, overlap possible.
Le couple NPU↔RTX (via PCIe+DDR) est le plus critique.

**Preuves** : le chemin réel RTX→NPU = RTX→PCIe→DDR→NPU (2 sauts lents). C'est LE coût à
mesurer avant de valider l'architecture. Local : `npu_tracer_v2.py` (fastflow) + `d2_rtx_gguf_profiler`
peuvent alimenter cette matrice.

---

## POINT 27 — Le NPU = contrôleur/prédicteur de trafic (pas 2e GPU)

**Question** : Quel est le meilleur rôle du NPU dans l'architecture ?
**Réponse** : Le NPU comme **coprocesseur de planification** (router, GDN/QSA, prefetch,
prédiction d'experts) plutôt que moteur de calcul MoE. Ça évite la frontière DDR↔PCIe
supplémentaire à chaque expert, et ça utilise le NPU idle 99.73%.

**Preuves** : CARTE §1 : FLM passe 54% du token en CPU orchestration + dispatches. Le NPU idle
99.73%. L'architecture recommandée (analyse) : NPU = Router/GDN/QSA/prédiction, RTX = MoE GEMM.

---

## CONCLUSION SYNTHÈSE (recommandation)

1. **Cache VRAM persistant d'experts quantifiés (Q4/Q5) + prefetch prédictif = le gain le plus
   sûr et immédiat** (+84% démontré sur 4090, +62.6% sur 3090/118B, 1.91× FATE) — brancher ça
   sur la 5070 d'abord.
2. **Ne jamais split par layer** (preuve locale 18.6 < 29.59/39.86) ; partitionner par régions
   ou par experts.
3. **Le goulot NPU Windows = MCDM dispatch** → contourner/batch avant d'espérer quoi que ce
   soit du NPU en co-exécution.
4. **Le NPU = contrôleur + tier d'overflow éco**, jamais concurrent de la RTX (ratio 7.4× mesuré).
5. **Mesurer la matrice 3×3 des transferts** sur la machine cible avant tout design du planner.
6. **Utiliser les briques existantes** : llama.cpp `--moe-expert-cache`/`-mec` (RFC #28248),
   FATE/PR #26563, `--n-cpu-moe`, PR #26501 (async Hexagon), HeteroMosaic (roofline hétérogène).

---

## ANNEXE — URLs de référence complète

### llama.cpp (MoE expert cache / MUL_MAT_ID)
- RFC #28248 (+84% decode, -mec) : https://github.com/ggml-org/llama.cpp/discussions/28248
- PR #26563 (miltos22, heatmap, -ehs) : https://github.com/ggml-org/llama.cpp/pull/26563
- Issue #20757 (two-tier cache, la base concept) : https://github.com/ggml-org/llama.cpp/issues/20757
- PR #24524 / gist 118B-RTX3090 (+62.6%) : https://gist.github.com/dekoza/e6b4a69989b3d5bd0f904ce204f1646c
- PR #26501 (backend Hexagon async, fences) : (fusionné 2026-09-01, à chercher dans le repo)
- Issue #24591 (MUL_MAT_ID IDs dupliqués crash) : https://github.com/ggml-org/llama.cpp/issues/24591
- Issue #21289 (MUL_MAT_ID invalid argument, arch) : https://github.com/ggml-org/llama.cpp/issues/21289
- Issue #13252 (MUL_MAT FA+MLA mixte) : https://github.com/ggml-org/llama.cpp/issues/13252

### FATE / moe-cache (prédiction + slots)
- FATE results Qwen3 : https://github.com/ongunm/llama-moe-cache/blob/main/FATE_RESULTS_QWEN3.md
- FATE implementation report : https://github.com/ongunm/llama-moe-cache/blob/main/FATE_IMPLEMENTATION_REPORT.md
- Qwen3.6-35B sur RTX 3060 (worked example) : https://github.com/maxta85/llama-cpp-moe-cache/blob/main/docs/qwen-3.6-35b-a3b-rtx3060.md

### MoE VRAM benchmarks / flags
- esonhjz (5070 Ti, -ncmoe) : https://github.com/esonhjz/llama-cpp-moe-vram-benchmarks
- OpenClaw MoE offload flags : https://openclawdc.com/blog/llama-cpp-moe-offload-flags-explained
- ik_llama.cpp hybrid CPU/GPU (--cpu-moe, -ot regex) : https://ikawrakow-ik_llama-cpp.mintlify.app/inference/hybrid-cpu-gpu
- cecil-the-coder llama-cpp-moe-flash (I/O budget, slots remap) : https://github.com/cecil-the-coder/llama-cpp-moe-flash
- MoE-OLDHW (GTX 1060 15-30 t/s) : https://github.com/BlackRainLabs/ResearchPapers/blob/main/MoE-LLM-Research/MoE-OLDHW.md
- TurboQuant KV cache benchmarks : https://github.com/mkadrlik/llama-cpp-vulkan-rocm/blob/main/benchmarks/README.md

### Qwen3.8-Flash-Next
- Repo modèle : http://github.com/QwenLM/Qwen3.8-Flash-Next
- 16GB recipe (UD-IQ1_S, experts+PLE CPU/SSD) : https://github.com/hocestnonsatis/qwen3.8-flash-next-16gb
- llamaperf (reports communautaires) : https://llamaperf.com/

### Papers / systèmes hétérogènes
- HeteroMosaic (MICRO 2026, AMD Ryzen AI iGPU+NPU) : https://arxiv.org/abs/2607.12839
- FlashMoE SSD (cache ML Belady) : https://arxiv.org/abs/2601.17063
- FlashMoE distribué (single kernel, 8×H100) : https://arxiv.org/abs/2506.04667
- SSD offloading énergie : https://arxiv.org/abs/2508.06978
- GEMM XDNA/XDNA2 (AMD, IRON) : https://arxiv.org/html/2512.13282
- HeteroInfer (Snapdragon GPU+NPU) : (cité par HeteroMosaic, arXiv:2501.14794)

### AMD officiel
- OGA Hybrid (NPU+iGPU) : https://ryzenai.docs.amd.com/en/main/hybrid_oga.html
- Model Pipelining NPU+GPU : https://www.amd.com/zh-cn/developer/resources/technical-articles/model-pipelining-on-npu-and-gpu-using-ryzen-ai-software.html
- Kernel doc amdxdna (topologie, DMA colonnes) : https://www.kernel.org/doc/html/latest/accel/amdxdna/amdnpu.html
- Ryzen AI Software : https://www.amd.com/en/developer/resources/ryzen-ai-software.html

### Corpus local (mesures, preuves mesurées)
- `bench_results/CONCURRENCE_GPU_NPU_QWEN35_9B_MMAP_20260905.md` (13.5 t/s agrégat mmap)
- `bench_results/RESULTATS_GPU_NPU_ORDER_QWEN17B_20260905.md` (split layer = pire)
- `bench_results/PLAN_COEXECUTION_GPU_NPU_RECHERCHE_COMPLETE_20260905.md` (PR #26501, cause racine)
- `reference/SOSC_v4/CARTE_FONCTIONNELLE_XDNA2_FLM.md` (BW 21.93, MCDM goulot, idle 99.73%)
- `reference/mesures_flm/flm_bottleneck_analyzer.py` (roofline R²=0.96)
- `reference/apply_turbo_cuda_v2.py` (kernels turbo3/turbo4)
- `C:\Users\videl\Desktop\lama-tensorRT 1050-5070\RAPPORT_TIERS_HARDWARE_2026-09-13.md` (5070 53.83 t/s)