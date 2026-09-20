# PLAN D'EXÉCUTION — 5070 + NPU XDNA2 + bus mémoire (D2 Planner)
# Établi 2026-09-20 — basé sur DOC_EXHAUSTIVE_5070_NPU_BUS.md
# ⚠️ CONTRAINTE HARDWARE DÉCOUVERTE : la machine de DEV actuelle = GTX 1080 (sm_61, Pascal).
#    La machine CIBLE (Ryzen AI 9 365 + RTX 5070 + NPU XDNA2) est SÉPARÉE et INDISPONIBLE
#    (données copiées : mesures_flm, SOSC_v4, lama-tensorRT 1050-5070 = artefacts uniquement).

## STATUT HARDWARE
- Dev (cette machine) : GTX 1080 8GB Pascal · llama.cpp CUDA 12 build présent (dlls) · AUCUN
  NPU XDNA2 · AUCUN modèle gguf réel (que des vocabs) · flm.exe absent
- Cible (indisponible) : Ryzen AI 9 365 (XDNA2) + RTX 5070 8GB + FLM v0.9.43 (mesures 06/2026)
- ⇒ Phase 1 = faisable ICI (code, simulation, validation logique). Phase 2/3 = sur machine cible.

---

## PHASE 1 — Socle logiciel (faisable maintenant, machine dev) [PRIORITÉ]

### 1.1 Câbler le planner D2 (sans toucher les sources)
- `xdna2/planner_core.py` : module unique qui importe place_expert + cost_contention, expose
  `PlanResult = {expert: tier, couts: {...}, contraintes: [...]}`.
- `xdna2/adapter_bridge.py` : pont conforme D2Adapter (contract governor/adapter.py) —
  inclusion dynamique, AUCUNE modification de governor/adapter.py.
- Vérif : `python -c "import planner_core"` + un cas de test unitaire.

### 1.2 Cost model contention v2 (leçon HRX2)
- Scinder BW en host_coherent vs GTT dans cost_contention_patch.py.
- Injecter les constantes calibrées FLM réelles (BW_eff 21.93, η 0.731, W_eff 2.87/3.83,
  KV 30000) dans le modèle de coût NPU.
- Ajouter la pénalité DDR5 (iGPU, page file, colonnes idle) au coût XDNA.

### 1.3 Benchmark cross-tier SIMULÉ (produit concret)
- `xdna2/benchmark_cross_tier.py` → génère bench.json + comparison.json.
- Configs : A = RTX seul · B = RTX+XDNA2 · C = RTX+XDNA2+CPU, sur distributions d'experts
  synthétiques (fréquence Pareto, tailles type Qwen Flash / MoE 35B A3B).
- Métriques : perf/compute, VoI, contention, tok/s estimés (calibré sur 53.83 GPU / 7.4 NPU).

### 1.4 README lancement (machine cible)
- `xdna2/README_DEPLOIEMENT.md` : commandes exactes à exécuter sur la machine cible
  (nvidia-smi, xrt-smi, flm.exe, llama-bench CUDA, runlist, mmap double-flux).

---

## PHASE 2 — Validation kernels GPU (dev machine GTX 1080, représentation du tier GPU)
- Compiler/tester `apply_turbo_cuda_v2.py` sur le build CUDA local (GTX 1080 sm_61 → valide
  la logique turbo3/turbo4, pas la perf Blackwell).
- Vérifier les kernels Vulkan 1bit (dmmv_tq2/q1/matmul) si un loader Vulkan simple dispo —
  sinon report sur machine cible.
- Mesurer le coût du bus GDDR7 (384 GB/s théorique vs ~165 GB/s effective = 51%, déjà mesuré
  sur la 1080 : 32.9 t/s pour Qwen9B IQ4NL).

---

## PHASE 3 — Machine cible (Ryzen AI 9 365 + RTX 5070) [bloqué hardware]

### 3.1 Mesures bus mémoire réelles
- `nvidia-smi` 5070 (VRAM libre, BW, thermique) + `xrt-smi validate` (NPU sain) + `xrt-smi
  configure --pmode performance`.
- Rejouer le roofline calibré (flm_bottleneck_analyzer.py) sur qwen3.5:9b → confirmer
  BW_eff 21.93, W_eff 2.87, η 0.731 sur la machine réelle.
- Mesurer la contention : Qwen3.5-9B en FLM pendant que la 5070 décode en parallèle
  (mmap partagé, même GGUF) → comparer à CONCURRENCE_GPU_NPU_9B_MMAP (13.5 t/s agrégat).

### 3.2 Double-flux 5070 + NPU (Voie A)
- Build llama.cpp Windows : CUDA Blackwell (ggml-cuda.dll déjà compilée) + backend XDNA
  (ggml-xdna du fork). `--list-devices` doit montrer CUDA0 (5070) + XDNA2.
- 2 process, même GGUF mmap → co-exécution vérifiée (timestamps + ps) → débit agrégat.
- Mesurer la contention réelle : dGPU séparé du bus DDR5 → attendu meilleur que Snapdragon.

### 3.3 Co-exécution un seul flux (Voie B, chantier)
- Base `origin/self-build-jz` (PR #26501 async) — worktree E:\oneplus\ab-wt à synchroniser.
- Partition par LIGNES MUL_MAT entre CUDA (5070) et XDNA2 + sync légère (flag mémoire
  partagée + poll) isolée du scheduler générique.
- Validation correction avant perf : diff token-par-token, seed fixe, temp=0.
- Profiler par shape (d2_rtx_gguf_profiler) → ratio split optimal GPU/NPU.

### 3.4 Levier bus mémoire (le plus rentable)
- Évaluer l'extension du cache SRAM NPU aux experts MoE fréquents (W_eff 2.87→ plus bas).
- Quantification turbo3/turbo4/NVFP4/TQ2 pour réduire octets/token (débit ∝ 1/poids lus).
- Persistent hw_context → -65% TTFT (éliminer les 2362 ms d'init).

---

## PHASE 4 — Consolidation
- Mettre à jour DOC_EXHAUSTIVE + RAPPORT avec les valeurs MESURÉES (remplacer les estimations).
- Cloner/copier les repos externes manquants (gglm-xdna2, adaptive-xdna-runtime, QwFNfer,
  LLM.xpu, hetero-llm-scheduler) dans reference/ sans les modifier.
- Générer le dashboard comparaison A/B/C avec les résultats réels.

---

## RÉPARTITION / RISQUES
| Phase | Machine | Bloqué par | Risque |
|---|---|---|---|
| 1.1-1.4 | dev (GTX 1080) | rien | faible (pure logique) |
| 2 | dev | loader Vulkan simple | faible-moyen |
| 3.1-3.4 | cible (5070+XDNA2) | **machine indisponible** | moyen (chantier Voie B) |
| 4 | les deux | dépend des phases | faible |

## ORDRE DE LANCEMENT
1. **Maintenant** : Phase 1.1 → 1.3 (planner_core + contention v2 + benchmark simulé = livrables
   concrets exécutables ici).
2. **Dès que possible** : Phase 2 (kernels GPU sur 1080) si loader Vulkan dispo.
3. **Sur machine cible** : Phase 3 (double-flux, Voie B, levier SRAM).
4. **Continu** : Phase 4 (consolidation docs).

Prochaine action immédiate : Phase 1.1 (planner_core.py + adapter_bridge.py) + 1.3 (benchmark
simulé → bench.json).