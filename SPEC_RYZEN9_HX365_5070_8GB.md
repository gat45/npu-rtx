# ADAPTATION — Ryzen 9 HX 365 + RTX 5070 8 GB + 32 GB RAM + SSD
# (machine réelle du projet, pas Strix Halo générique)

## Hardware spécifié
- CPU/NPU : AMD Ryzen 9 HX 365 (Dragon Range) → XDNA2 NPU intégré, ~16 TOPS, mémoire unifiée avec CPU/iGPU
- GPU : NVIDIA RTX 5070 8 GB VRAM (discret, pas unifié)
- RAM : 32 GB DDR5 (staging / cache chaud)
- Stockage : SSD NVMe (cold / experts aberraunts)
- Thermal : hystérésis 60/50 °C (comme dans AGENTS.md protocole)

## Différence clé vs Strix Halo / iGPU unifié
- Ici le NPU XDNA2 est **séparé du GPU RTX** : pas de contention BW directe NPU↔GPU sur même contrôleur mémoire, MAIS le CPU et NPU partagent la RAM 32 GB (DMA host→NPU passe par DDR5).
- Le RTX 5070 8 GB est le **tier HOT principal** (mieux que iGPU Strix Halo en raw tok/s).
- Le XDNA2 devient **tier WARM/SECONDARY** pour experts qui débordent de 8 GB VRAM, pas pour remplacer le GPU.
- CPU reste fallback pour ops non supportées par XDNA2 (comme dans OllamaAMDNPU).

## Adaptation du planner (tierre local) — données du repo GaTmaNnes (Ryzen AI 9 HX 370 / XDNA2)

| Tier | Capacité estimée | Rôle dans Qwen Flash MoE |
|------|------------------|--------------------------|
| RTX 5070 8 GB | ~8 GB VRAM (poids Q4_0 9B ≈ 5-6 GB + KV + buffers) | HOT : experts fréquents, MUL_MAT grandes matrices |
| RAM 32 GB | 32 GB (cache + staging) | WARM : experts moyens, prefetch, routing |
| XDNA2 NPU | Mémoire locale inconnue (probablement 8-16 MB par tuile, total < 100 MB utile) + DMA depuis RAM | WARM/HOT secondaire : petits experts, ops supportées (GGML_OP_MUL_MAT), tiles |
| SSD | Très grand | COLD : experts rares / imprévisibles |

## Questions adaptées pour cette machine
1. Combien d'experts Qwen Flash (Q4_0) tiennent dans 8 GB VRAM ? → mesure avec profiler-v3 / ggml-backend
2. Quels experts peuvent être réduits en tiles sur XDNA2 sans dépasser BW DDR5 ? → benchmark DMA unitaire
3. Quand le planner doit-il basculer RTX → XDNA2 au lieu de RTX → RAM → CPU ? → seuil température + BW
4. Impact thermal : RTX 5070 chauffe → throttle → planner doit réduire tier RTX si T>70°C (hystérésis)

## Sources copiées dans reference/ (sans modification des originaux)
- AGENTS.md (extrait XDNA/protocole)
- profiler_v3/FONCTIONS.md (xdna2-forensics / llm-upstream)
- DOC_QUANTISATION_*.md (SSD-LLaMA ref)
- ARCHITECTURE_ET_INTERCONNEXIONS_*.md (profiler-v3↔HTP/DMA)

## Données copiées de fastflow/ (sources D:\fastflow compagnon, non modifiées)
- ANALYSE_0x1F_CORRIGEE.md : flag 0x1F dégrade decode 7.80→6.95 t/s, prefill 46.3→8.9 t/s (éviter sur planner)
- ANALYSE_DEFINITIVE_XDNA2.md : bottleneck = tiling GEMM (pas RAM) ; INT4 = déquantifié en INT8/BF16 avant kernel ; CP starvation (pas fréquence) ; mapping colonnes rigide (static slicing)
- ANALYSE_FINALE_XDNA2.md : hidden_size 3584 (Qwen 3.5 9B) = problème de tiling vs 4096 (DeepSeek-R1 8B) ; tile_util critique
- ANALYSE_FLAGS_XRT.md : flags XRT à calibrer selon driver (voir doc copié)
- Impact planner : éviter 3584 si possible, préférer 2048/4096 ; activer `--pmode performance`; ne pas croire INT4 = speed ; BW_eff réel ~30 GB/s (pas limite)

## Références copiées (sources non modifiées)
- `reference/fastflow/` : 4 analyses XDNA2 (0x1F, definitive, finale, XRT flags)
- `reference/d2-quant-planner/` : `build_rtx_5070_blackwell.ps1`, `d2_compiler.py`, `alpha_spectral_scanner.py`, `app.py` (layer-wise quant planner pour RTX 5070 / Blackwell, spectral + ILP)
- `reference/GATMANNES_RAPPORT_XDNA2.md` : synthèse repo GaTmaNnes (tile_util, INT4 dequant, hybrid RTX+NPU)

## Impact d2-quant-planner sur le planner
- Utiliser build xdna2-forensics (NDK r27c) pour créer backend ggml XDNA sur Linux/WSL
- Mesurer capacité réelle XDNA2 via `amdxdna` + `profiler-v3`
- Exécuter `place_expert` avec seuils adaptés à 8 GB VRAM (plus restrictif que 24 GB)
- Benchmark A/B/C : RTX seul vs RTX+XDNA2 vs RTX+XDNA2+CPU sur même batch experts
