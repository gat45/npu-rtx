# PLAN D'EXÉCUTION — Campagne Pareto mémoire↔précision↔perf (Qwen3.8-Flash-Next)
# Établi 2026-09-20 · Dossier npu-rtx/ · Basé sur MATRICE_EXPERIMENTALE_P0_P5.md + SPEC_SAQE.md
# + PROFILER_V3_AMD_INTEGRATION.md · Objectif : frontière de Pareto réelle, PAS un benchmark simpliste.
# Règle : sources en LECTURE SEULE ; toute création/modif dans npu-rtx/ (projet D2).

---

## 0. PRINCIPE (règles absolues, rappel)

1. **Mesurer avant de conclure** — jamais de valeur théorique si mesurable.
2. **Ne jamais optimiser les bits seuls** — chemin complet (storage+conversion+transfer+
   contention+cache+dequant+kernel+launch/sync+accuracy).
3. **Matrice format×tensor×backend×arch×shape×kernel** (SUPPORTED/UNSUPPORTED/
   SUPPORTED_BUT_SLOW/SUPPORTED_BUT_INCORRECT/UNTESTED/UNVERIFIED).
4. **Ne pas modifier le routing** dans le benchmark principal.
5. **profiler-v3 = instrument de référence** : NE PAS le réécrire, ajouter des collecteurs.
6. Résultat = frontière de Pareto, pas "le meilleur" arbitraire.

---

## 1. INSPECTION PROFILER-V3 (+ V5) — état des lieux (fait, lecture seule)

### Modules existants (socle à réutiliser, NE PAS dupliquer)
| Module | Ce qu'il fait | Réutiliser pour |
|---|---|---|
| `predict_from_hf_v5.py` (⭐ CŒUR V5) | `classify_tensor_family`, `pick_best_htp_format`, `analyze_residency_feasibility`, `_simulate_requant_error`, `attach_level1_quality_gate`, `run_level1_weight_error`, `render_extended_format_grid` — plan par couche/expert avec résidence SSD/RAM/HTP + quality gate | manifest Qwen, compat matrice, résidence |
| `profiler.py` | plan quant sûr + `bw_effective_decomposed()` + `simulate_l3_bounds()` + `apply_device_regime()` + rapport + JSONL live-trace + `--emit-tensor-type-file` | baseline, memory map, bandwidth model |
| `quant_formats.py` | catalogue GGML + chemins backend (couvre F32/F16/BF16, Q4/Q5/Q8, K-quants, IQ, TQ, MXFP4, NVFP4, Q1_0, Q2_0) + "unknown ≠ impossible" | matrice format×backend |
| `expert_profile.py` | `normalize_access_event()` + `aggregate_expert_access(cache_capacity)` + `fuse_htp_cost()` + `assess_htp_cache_value()` | cache metrics, routing |
| `expert_cache_metrics.py` | `assess_htp_cache_value(min_net_saved_time)` | coût cache |
| `expert_cache_replay.py` | rejeu des accès | LRU/LFU/replay |
| `predictor.py` | prédiction | prefetch/predictor |
| `parse_hexagon_profile.py` / `parse_opencl_profile.py` | parsing traces HTP/OpenCL | kernel timing NPU/iGPU |
| `profile_model.py` | profilage modèle | manifest |
| `capability_db.py` | base capacités | matrice compat |
| `tensor_sources.py` | lecture offsets tenseurs | manifest offsets |
| `quality_gate.py` | gates qualité | validation |
| `adaptive_lever.py` | leviers adaptatifs | validation |

### Le V5 (doc RAPPORT_FORMATS_QUANTIFICATION_ET_PROFILER_V5) distingue 5 objets
1. **conteneur** (GGUF / Safetensors)
2. **format stocké** (Q4_0, MXFP4, F8_E4M3, ...)
3. **recette de quantification** (GPTQ/AWQ/NF4/PTQ-QAT, sym/affine, granularité scales)
4. **layout exécuté** après repack backend
5. **types de calcul** des activations/produits/accumulations
→ corrige 2 erreurs : SafeTensors FP8/FP4 compté BF16 ; durée ARGSORT 215µs utilisée comme coût
d'un dispatch FastRPC. **Une op logique ≠ kernel ≠ OPBATCH ≠ RPC.**

### Exemple de sortie V5 (RAPPORT_PROFILAGE_V5_QWEN38_27B_MTP_IQ2M)
- 866 tenseurs analysés (header GGUF seul, aucune valeur de poids lue = PROXY trafic/taille)
- Budget poids 9 GiB · fichier source OVER_BUDGET (-2.282 GiB) · plan HTP OVER_BUDGET (-5.780)
- Verdict résidence : BLOCKED_DENSE_STREAMING (dense = relu à chaque token, pas cold)
- Conversions par couche : attn_qkv Q4_K → Q4_0 (TAILLE EGALE, tenseur récurrent non-HTP →
  repli CPU à chaque token — motif calibré) ; ffn_down/gate/up Q8_0 → Q4_0 (ÉCONOMIE)
- Candidats HTP limités : IQ4_NL, MXFP4, Q4_0, Q4_1, Q8_0 (PAS MXFP2/W2/W1/ternaire)

### Ce qu'il MANQUE (à créer dans npu-rtx/, PAS dans profiler_v3)
1. Collecteurs AMD (npu_perf_trace, amdxdna telemetry, xdna-top)
2. Collecteur NV GPU (CUDA timing/Nsight, occupancy)
3. Adaptateur de format unifié vers le JSONL profiler-v3
4. Modèle Qwen3.8-Flash-Next (manifest complet — le V5 a profité Qwen3.8-27B, pas Flash-Next 125B)
5. PLE/GDN/QSA/KV séparés (le V5 traite dense MoE, pas PLE/state)
6. Frontière de Pareto + oracle D2 multi-device (V5 = HTP-centric ; ici RTX+XDNA2)

---

## 2. HARDWARE DISCOVERY (à exécuter sur la machine cible — 5070 + XDNA2)

Fichier cible : `npu-rtx/collectors/hw_discovery.py` (créé dans le dossier projet).

Détecter : CPU (modèle/cores/fréq) · RAM (total/dispo/BW) · NUMA · SSD (type/fs/cap/temp/seq/random)
· PCIe (gen/link/lanes/H2D/D2H) · GPU (modèle/arch/CC/VRAM_total/usable/available/BW/clocks/temp/
power) · NPU (modèle/cols/clock/power/util/DMA) · XRT/amdxdna/firmware/compiler/CUDA · OS.

Windows : `VRAM_TOTAL ≠ VRAM_VISIBLE ≠ VRAM_RESIDENCY_BUDGET ≠ VRAM_SAFE_RESERVE` (WDDM).
Sortie : `npu-rtx/runs/<ts>/hw.json`.

---

## 3. MANIFEST QWEN3.8 (candidat prioritaire)

Fichier cible : `npu-rtx/models/qwen38_flash_next/manifest.json` (créé dans le dossier projet).

Hypothèse initiale (à VÉRIFIER contre le checkpoint réellement profilé — jarvix-memory,
tensor_sources.py, predict_from_hf_v5.py peuvent aider) :
```
layers 48 · 36 GDN · 12 QSA · 512 routed experts · top-10 + 1 shared
hidden 2560 · expert_interm 640 · vocab 248320 · ctx 262144 (→1M)
GDN_state [..] · QSA_KV [..] · PLE 320 001 536 × 160 = 51.2B
MTP 4B · layout 12×(3×(GDN→MoE)→1×(QSA→MoE))
```
Classes de tenseurs : DENSE / MOE_ROUTED / MOE_SHARED / GDN / QSA / GDN_STATE / QSA_KV /
PLE / MTP / EMBEDDING / LM_HEAD / NORMALIZATION / ROUTER.
Sortie : `runs/<ts>/manifest.json` + vérif offsets via `tensor_sources.py`.

---

## 4. FORMATS À TESTER (grille minimale)

GGML/GGUF : IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, Q2_K, Q3_K, Q4_K, Q5_K,
Q6_K, Q8_0, IQ4_NL, IQ4_XS, Q4_0, Q5_0.
GPU-native (UNIQUEMENT si backend+hw compatibles) : FP8, NVFP4, MXFP4, W4A16, W4A8, INT8.
Références : ggml Tensor-Encoding-Schemes, TensorRT-LLM quantization, FlashInfer MoE API.

---

## 5. ARCHITECTURE À CONSTRUIRE (dans npu-rtx/, profiler_v3 en lecture seule)

```
npu-rtx/
├── collectors/
│   ├── hw_discovery.py          # §2
│   ├── amd_npu.py               # npu_perf_trace.sh + amdxdna telemetry + xdna-top events
│   ├── nv_gpu.py                # CUDA timing / Nsight / occupancy / Blackwell tuning
│   ├── ssd_ddr_pcie.py          # microbench 1K→1G, queue depth
│   └── quant_bench.py           # quant/dequant/requant par taille (4K→1G)
├── correlation/
│   ├── unify_events.py          # format unifié → JSONL profiler-v3 (adaptateur)
│   ├── dag_timeline.py          # SSD→RAM→H2D→VRAM→dequant→GEMM + branches (PLE/GDN)
│   └── critical_path.py         # visible/hidden/critical path
├── models/
│   └── qwen38_flash_next/manifest.json
├── experiments/
│   ├── cache_sweep.py           # slots 0/4/8/16/32/64/128 × LRU/LFU/LFRU/ML/D2
│   ├── prefetch_sweep.py        # off/exact/top-1/2/4/8 + predictors
│   ├── precision_sweep.py       # uniform/tensor-mixed/expert-mixed/dynamic/backend-mixed
│   ├── ple_paging.py            # resident/fs-cache/mmap/pread + amplification
│   ├── gdn_qsa_state.py         # state BF16/FP8/INT8/mixed + KV QSA
│   ├── xdna_roles.py            # A compute / B quant / C scheduling
│   └── full_matrix.py           # test1→test10 (voir §55 mission)
├── oracle/
│   └── d2_planner.py            # prédiction critical path + confidence + Pareto
├── report/
│   └── build_report.py          # structure A→X (§58 mission)
└── runs/<ts>/                   # toutes les sorties
```

---

## 6. MATRICE DE COMPATIBILITÉ (avant tout benchmark)

`npu-rtx/collectors/compat_matrix.py` — construit :
```
format × tensor × backend × arch × shape × kernel → SUPPORTED/UNSUPPORTED/
SUPPORTED_BUT_SLOW/SUPPORTED_BUT_INCORRECT/UNTESTED/UNVERIFIED
```
- Sources : quant_formats.py (GGML), TensorRT-LLM matrix (SM120 : NVFP4✅ MXFP4✅ FP8-pertensor✅
  FP8-KV✅ ; **FP8 block/rowwise, W4A8, W4A16 ❌**), FlashInfer (b12x SM120), llama.cpp MMQ.
- ⚠️ ggml #1506 : Q4_K/Q5_K/Q6_K incorrects sur mul_mat_id, Q8_0 OK → tester, pas supposer.
- Sortie : `runs/<ts>/compat_matrix.json`.

---

## 7. MESURES OBLIGATOIRES PAR EXPÉRIENCE (adaptateur JSONL profiler-v3)

Format unifié (créé par correlation/unify_events.py) :
```json
{ "timestamp_ns":0, "phase":"decode", "layer":12, "expert_id":237, "tensor":"gate_exps",
  "weight_precision":"Q4_K", "activation_precision":"BF16", "storage":"RAM",
  "compute_device":"RTX", "cache_hit":true, "prefetch":false,
  "bytes_storage":0, "bytes_pcie":0, "t_quant_ns":0, "t_dequant_ns":0,
  "t_gemm_ns":0, "t_sync_ns":0, "critical_path_ns":0 }
```
Respecter le format JSONL natif profiler-v3 (adaptateur, pas de duplication).

---

## 8. LES 10 TESTS MINIMAUX (§55 mission) — ordre

| Test | Objet | Fichier |
|---|---|---|
| 1 | dense small model → valider profiler | experiments/full_matrix.py --test 1 |
| 2 | small MoE → valider expert cache | --test 2 |
| 3 | Qwen3.8 cache disabled | --test 3 |
| 4 | Qwen3.8 cache enabled | --test 4 |
| 5 | Qwen3.8 prefetch | --test 5 |
| 6 | Qwen3.8 adaptive precision | --test 6 |
| 7 | Qwen3.8 PLE paging | --test 7 |
| 8 | Qwen3.8 GDN/QSA state | --test 8 |
| 9 | Qwen3.8 RTX-only | --test 9 |
| 10 | Qwen3.8 RTX + XDNA2 | --test 10 |

---

## 9. SORTIES OBLIGATOIRES (§45 mission)

`runs/<ts>/` : report.md · results.json · runs.jsonl · pareto.csv · memory.csv ·
bandwidth.csv · kernel.csv · routing.csv · cache.csv · quantization.csv · xdna.csv ·
(timeline.html/json, plots/ si possible).

---

## 10. TABLEAU FINAL (§46) + FRONTIÈRE DE PARETO (§47) + CONFIDENCE (§51)

Table : Model | W | A | PLE | GDN state | QSA KV | Cache | Prefetch | Device |
RAM peak | VRAM peak | SSD | DDR | PCIe | TPOT | tok/s | TTFT | route agreement | PPL | stability.

Pareto : mémoire/précision · mémoire/perf · précision/perf · globale (RAM, VRAM, precision,
latency, throughput). Configs dominées identifiées.

Chaque estimation D2 : prediction + confidence + sample_count + variance + source_measurement
(⚠️ ne pas choisir Q3 à forte variance juste parce que la moyenne est basse).

---

## 11. VALIDATION CROISÉE (§54)

Chaque résultat important : profiler-v3 ↔ AMD XRT/perf ↔ npu_perf_trace ↔ xdna-top ↔ CUDA
timing/Nsight. Différences expliquées = chaîne de preuve.

---

## 12. D2 PLANNER (oracle, §49-52)

`oracle/d2_planner.py` :
- candidate → predicted critical path / RAM / VRAM / BW / quality risk / confidence
- choisit : W_precision, A_precision, residency, cache slot, prefetch, device, kernel, tile
- online learning : prediction → actual → model correction (throughput/cache/prediction/
  conversion/thermal)
- objective : minimize(critical_path, RAM, VRAM, SSD, DDR, PCIe traffic) SOUS contraintes
  (quality ≥ seuil, budgets) — objectifs séparés pour Pareto, PAS un score fusionné.

---

## 13. MÉTHODE DE CONCLUSION (§58 — structure A→X du rapport final)

A Hardware détecté · B Versions · C Config référence · D Baseline profiler-v3 · E Bottlenecks ·
F Memory map · G Bandwidth map · H Quantization matrix · I Kernel matrix · J Cache matrix ·
K Prefetch matrix · L Routing sensitivity · M PLE · N GDN state · O QSA KV · P XDNA2 ·
Q RTX · R Pareto frontier · S Best low-RAM · T Best precision · U Best throughput ·
V Best balanced · W Unresolved · X Reproduction commands.

---

## 14. ÉTAPE 0 IMMÉDIATE (à faire maintenant)

1. Créer le squelette `npu-rtx/` (collectors/correlation/models/experiments/oracle/report/runs)
2. `collectors/hw_discovery.py` (squelette exécutable sur la machine dev GTX 1080, extensible cible)
3. `collectors/compat_matrix.py` (construit depuis quant_formats.py + TensorRT-LLM matrix)
4. `correlation/unify_events.py` (adaptateur → format JSONL profiler-v3)
5. `models/qwen38_flash_next/manifest.json` (template à vérifier contre checkpoint réel)
6. `oracle/d2_planner.py` (squelette : candidate → cost → confidence → Pareto)

Puis exécution des 10 tests minimaux (sur dev d'abord, puis machine cible).

## 15. SOURCES (liens dans URLS_REGISTRY.md)
llama.cpp #28248/#20757/#27149/#27864 · ggml #1506 · Tensor-Encoding-Schemes · dequantize.cuh ·
mmq.cu · Qwen3.8-Flash-Next · NVIDIA-NeMo coverage · arXiv:2608.30320 (paper) ·
MoE-Infinity config+adaptive-precision · DynaExQ (2511.15015) · vLLM #38256 · HOBBIT ·
FlashMoE · FlashInfer MoE API · TensorRT-LLM quant+mode.py · CUDA gpus/blackwell-tuning/
best-practices/async · GDS · WDDM residency · amd/xdna-driver (npu_perf_trace, telemetry,
amdnpu.rst, aie2_pci.c) · AI Analyzer · mlir-aie (programming guide, iron.md, iron_configuration,
roadmap, #3460) · xdna-top

## 16. ⚠️ V5 = SNAPDRAGON + RÉSULTATS PUBLICS (mise à jour critique 2026-09-20)

### profiler-v3/V5 est conçu pour Snapdragon (HTP/FastRPC), PAS XDNA2/5070
- V5 = SOCLE LOGIQUE à réutiliser (5 objets, résidence SSD/RAM/HTP, quality gate, catalogue
  MXFP4/NVFP4/TQ) — MAIS ses collecteurs/benchmarks sont HTP-centric.
- → pour XDNA2 + RTX 5070 : **refaire les collecteurs dans npu-rtx/** (amd_npu.py, nv_gpu.py,
  ssd_ddr_pcie.py, quant_bench.py), jamais dans profiler_v3.

### Résultats publics Qwen3.8 (voir REFERENCE_RESULTATS_PUBLICS_QWEN38.md)
- 2.57 bpw sur 24 GB + 32 GB RAM (Haberstroh) : **0.31 GB PCIe/token vs 26 GB** (routing-aware
  streaming) — la preuve n°1 que quantification + sélectivité + paging doivent être couplés
- **working set ≠ top-k** : 64 experts = 53% du trafic, 256 = 93% → cache_size = f(routing_mass),
  commencer à 2-4× top-k
- **PLE GPU = 55.6× plus lent que PLE CPU/RAM** (lukaLLM) → PLE → host/SSD, experts → VRAM
- **PLE quantifiable indépendamment** : BF16/FP8/NVFP4 (Starkweather : 102→28.8 GB)
- **FP8 KV : +8.5% decode mais -6.1% prefill** ; dequant fusionnée = -40.6% kernel (MiaAI)
- **GDN state** : ~0.23 GB/séquence, FP32→BF16 = +6.8-8.5%
- **MTP** : 2× code, 0% prose, négatif si draft long → par workload
- Bugs correctness : mapping ubatch (20-30% mauvais poids), QSA dtypes, CUDA graph corruption

### Plan de reproduction (l'action la plus rentable)
R1 : reproduire 5 configs publiques (streaming sélectif / cache #28248 / PLE host / KV FP8+GDN
BF16 / NVFP4+MTP) sur la machine cible → R2 : extraire traces expert/PLE/KV/GDN/DMA/PCIe en
JSONL profiler-v3 → R3 : entraîner l'oracle D2 AVANT la recherche adaptative.