# PLAN xdna2 — Liste des manques consolidée + plan d'exécution
# Mis à jour : 2026-09-20 (synthèse MANQUANT.md, état réel du dossier)

## ÉTAT ACTUEL (ce qui est FAIT)
- RAPPORT_XDNA2_D2_PLANNER.md : architecture complète + corrections AMD hybrid + reverse FLM + benchmarks calibrés
- SPEC_RYZEN9_HX365_5070_8GB.md : adaptation machine réelle (HX365 + 5070 8GB)
- place_expert.py v3 : modèle coût argmin par expert (RTX/XDNA2/CPU/SSD) + primitives XRT (sync, group_id, runlist, MoE q41) + constantes calibrées
- cost_contention_patch.py : overlay contention (BW partagée)
- benchmark_cross_tier.py / validate_llama_xdna.py : squelettes
- reference/ : 1bit (RE FLM + docs + kernels Vulkan + amd-oss), fastflow, d2-quant-planner, GaTmaNnes, AMD OGA/UAPI
- Sources (governor, profiler_v3, AGENTS.md) NON modifiées

## MANQUES (18) — statut

### A. DONNÉES / MESURE (bloquant device)
| # | Manque | Statut |
|---|--------|--------|
| 1 | Mesure XDNA2/HX365 réelle (memoire locale, DMA, BW effective) | 🔴 à faire sur device |
| 9 | Calibration HX365 locale (thermal, profile NPU, logs device) | 🔴 à faire sur device |
| 5 | Build + validation xdna2 (ggml-xdna / FLM / IRON) | 🔴 à faire sur device |
| 13 | Référence MoE NPU réelle (Qwen3.6-35B FLM, SEGFAULT upstream) | 🟡 à surveiller (ROCm/FastFlowLM open) |

### B. CODE / INTÉGRATION (faisable maintenant, sans device)
| # | Manque | Statut |
|---|--------|--------|
| 2 | Brancher place_expert.py dans governor/adapter.py (D2Adapter) | 🟡 script prêt, non câblé |
| 3 | Injecter cost_contention_patch.py dans governor/cost_model.py | 🟡 overlay prêt, non appliqué |
| 7 | Relier d2-quant-planner au pipeline (import spectral/ILP dans place_expert) | 🟡 fichiers copiés, non intégrés |
| 8/14 | Structurer les données fastflow dans le RAPPORT | 🟢 fait (§3quater) |
| 16 | Distinguer host_coherent vs GTT dans cost_contention (leçon HRX2) | 🔴 à faire |
| 18 | Tester kernels Vulkan sur RTX 5070 + brancher acc_mode dans le merge | 🔴 à faire |
| 12 | Mapping API→code (OGA/UAPI) dans place_expert | 🟡 partiel (v3) |

### C. OUTILS / VÉRIF (faisable maintenant)
| # | Manque | Statut |
|---|--------|--------|
| 4 | Benchmark A/B/C réel : bench.json + comparison.json | 🔴 skeleton seulement |
| 15 | Tester Triton-XDNA (Windows natif) pour compiler kernel expert → xclbin | 🔴 à faire |
| 17 | Déployer GitNexus (skills Claude → opencode) pour trace/impact | 🟡 documenté, non déployé |
| 11 | Confirmer sensors/télémétrie + QoS sous Windows | 🟡 partiel (reverse prouve l'essentiel) |

### D. REPOS / DOC
| # | Manque | Statut |
|---|--------|--------|
| 6 | Cloner gglm-xdna2, adaptive-xdna-runtime, QwFNfer, LLM.xpu, hetero-llm-scheduler | 🔴 liens seulement |
| 10 | README lancement STEP1→5 sur machine cible | 🔴 absent |

---

## PLAN D'EXÉCUTION (priorisé)

### PHASE 1 — Verrouiller le socle (faisable maintenant, sans device)
1. **Câbler l'adapter** : overlay `xdna2/adapter_bridge.py` qui importe place_expert + cost_contention
   et expose `place_expert(expert_id, freq, size, hidden, state) → tier` conforme au contrat D2Adapter
   (sans toucher governor/adapter.py — inclusion dynamique).
2. **Cost model contention v2** : split host_coherent vs GTT (leçon HRX2) → 2 BW, pas 1.
3. **Benchmark cross-tier simulé** : remplir benchmark_cross_tier.py pour produire
   bench.json + comparison.json à partir du modèle coût (RTX seul / RTX+XDNA2 / +CPU) sur
   des distributions d'experts synthétiques (fréquence Pareto, tailles variées).
4. **README lancement** : doc STEP1→5 avec les commandes exactes + prérequis (amdxdna/XRT Windows,
   modèle Qwen Flash, état des références).
5. **Vérif kernels Vulkan** : compile+exécute dmmv_tq2/dmmv_q1/matmul_fp32 sous Vulkan (si GPU dispo)
   → mesure tok/s experts + valide acc_mode (merge).

### PHASE 2 — Pipeline placement réel (device ou simulation avancée)
6. **Intégrer d2-quant-planner** : utiliser alpha_spectral_scanner + d2_compiler pour choisir
   la quant par expert (INT8 vs Q4/Q1) selon target tier → alimenter place_expert (size_bytes).
7. **MoE sur RTX + XDNA2** : structurer le "second moteur d'experts" — runlist côté NPU (grouper
   les experts du token) + Vulkan acc_mode côté RTX ; mesurer le coût réel DMA vs compute.
8. **Métrique VoI/contention** : injecter dans le cost model la télémétrie (power/temp/col_util)
   quand dispo, sinon simulée (distribution).

### PHASE 3 — Device (blocage hardware)
9. **Mesure XDNA2/HX365** (MANQUANT 1/9) : via xrt-smi --pmode performance + npu_tracer + profil
   GGML_HEXAGON/CL → produire xdna2_cap.json (memoire locale, DMA µs, BW effective, colonnes actives).
10. **Validation ggml-xdna / FLM / IRON** (MANQUANT 5) : Qwen Flash subset sur NPU, comparer
    CPU-Q4 == NPU-Q4 (leçon "rc=0 ≠ preuve"), 64-token golden.
11. **Référence MoE NPU** (MANQUANT 13) : dès que ROCm/FastFlowLM ouvre le MoE, mesurer
    expert up/down/gate q41 ; sinon benchmark kernel expert isolé.

### PHASE 4 — Consolidation
12. **Cloner les 5 repos externes** (MANQUANT 6) : gglm-xdna2, adaptive-xdna-runtime, QwFNfer,
    LLM.xpu, hetero-llm-scheduler → copier dans reference/ (pas modifier).
13. **GitNexus local** (MANQUANT 17) : adapter les skills en opencode, indexer geniex_harness,
    lancer trace(place_expert → cost_model) + impact avant chaque modif.
14. **Mise à jour RAPPORT/SPEC** avec résultats mesurés (remplacer les constantes estimées
    BW_GTT=56, NPU_TOPS=31, etc. par les valeurs mesurées).

---

## PRIORITÉS IMMÉDIATES (si tu veux que je continue maintenant)
- **P1** : Phase 1 items 1-4 (câblage adapter + contention v2 + benchmark simulé + README) — tout faisable sans device
- **P2** : Phase 1 item 5 + Phase 2 items 6-8 (kernels Vulkan + d2-quant + MoE merge)
- **P3** : Phase 3-4 (device obligatoire ou consolidation)

Veux-tu que je lance la Phase 1 maintenant ?