MANQUANT xdna2 / D2 Plan (audit rapide)

1. Mesure réelle : pas de mesure XDNA2/HX365 (mémoire locale, DMA, BW ~30 GB/s non vérifié ici) ; measure_xdna2_capacity.md = template uniquement
2. Intégration adapter : place_expert.py non branché dans governor/adapter.py ; pas d'appel depuis D2Adapter
3. Cost model : cost_contention_patch.py = overlay, pas injecté dans governor/cost_model.py (source non modifié, patch non appliqué)
4. Benchmark A/B/C : benchmark_cross_tier.py = skeleton ; pas de bench.json / comparison.json produits
5. Validation XDNA2 : validate_llama_xdna.py = commande type, mais pas de build xdna2-forensics exécuté ni de log real
6. Repos externes non clonés : Tagman45/gglm-xdna2, adaptive-xdna-runtime, QwFNfer, LLM.xpu, hetero-llm-scheduler (seuls liens dans reference/)
7. d2-quant-planner : fichiers copiés mais pas reliés au pipeline (pas d'import dans place_expert / pas de fusion spectral)
8. Fastflow analyses : 4 fichiers copiés mais pas intégrés en données structurées dans RAPPORT (seul résumé manuel dans SPEC)
9. Hardware spécifique HX 365 : SPEC adapté au papier (HX 370 / 5070) mais pas calibré avec logs réels du device local (pas de profile NPU, pas de thermal log intégré)
10. README / guide lancement : pas de doc utilisateur xdna2 expliquant comment exécuter STEP1→5 sur machine cible
11. UAPI Windows : PARTIELLEMENT RÉSOLU — le reverse FLM (reference/1bit/) prouve que le stack XRT Windows expose xrt::bo::sync(TO/FROM_DEVICE), group_id (obligatoire >0), runlist, hw_context (8+ concurrents), checkpoint KV sync_to/from. Reste : confirmer sous Windows l'accès aux sensors/télémétrie (power/temp/col_util) + QoS hints de l'UAPI Linux
12. Fichiers de référence OGA_HYBRID + UAPI créés mais pas encore exploités : pas de mapping API→code dans place_expert/cost_contention — place_expert v3 intègre maintenant sync_to/from_device, group_id, runlist, MoE q41 (partiel)
13. Qwen3.6-35B MoE : le reverse FLM montre que le moteur AMD gère les experts MoE mais SEGFAULT upstream sur load_weights (bug binaire) → pas de référence de mesure pour le MoE NPU sur cette machine ; à surveiller — NOTA : FastFlowLM est maintenant ROCm/FastFlowLM (Apache-2.0), si le code MoE s'ouvre on peut dériver les primitives sans reverse
14. Fastflow analyses : 4 fichiers copiés mais pas intégrés en données structurées dans RAPPORT (seul résumé manuel dans SPEC) — RAPPORT §3quater intègre maintenant FLM_SECRETS/NPU_ISA/Q4NX/NPU_GEMM_FIX
15. Stack OSS (IRON/Triton-XDNA/MLIR-AIR/Peano) documenté mais pas testé sur notre machine Windows : à vérifier si Triton-XDNA (Windows natif) compile un kernel expert MoE → xclbin directement (levier #2 de AMD_OSS_NPU_STACK.md)
16. Leçon HRX2 (zero-copy GTT vs cache-coherency) non intégrée : cost_contention_patch.py suppose une BW unique ; il faudrait distinguer host_coherent vs GTT pour le coût DMA (XCL_BO_SYNC_TO/FROM)
17. Outil GitNexus documenté (reference/GITNEXUS.md) mais pas déployé localement : `trace`/`impact` cross-repo aideraient le câblage adapter (point 2) et l'audit "sources non modifiées" (detect_changes) ; skills sont au format Claude → à adapter opencode
18. Kernels Vulkan 1bit téléchargés (reference/1bit/kernels-vulkan/) mais pas encore testés sur le tier RTX 5070 : à valider le `dmmv_tq2`/`dmmv_q1`/`matmul_fp32` sous Vulkan NVIDIA (wave32 présent → variant wave32-pinned = piste perf) ; le pattern `acc_mode` (accumulation experts) n'est pas encore branché dans place_expert/merge
