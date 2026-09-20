# Référence externe — GaTmaNnes / RAPPORT-TECHNIQUE-Optimisation-de-l-inf-rence-LLM-sur-NPU-XDNA2
# URL : https://github.com/GaTmaNnes/RAPPORT-TECHNIQUE-Optimisation-de-l-inf-rence-LLM-sur-NPU-XDNA2
# Copié dans xdna2/reference/ — sources d'origine NON modifiées

## Dépôts pertinents trouvés sur le profil
- snapdragon-d2-planner
- d2-quant-planner  (layer-wise quant planner pour llama.cpp / GGUF)
- xdna2-  (C++, backend XDNA2)
- RAPPORT-TECHNIQUE... (ce repo-ci)

## Conclusions techniques clés (extrait du README du repo)
1. HW cible : AMD Ryzen AI 9 HX 370 (NPU XDNA2, 32 tiles AIE2) — proche de HX 365 (même famille Dragon Range, XDNA2)
2. Bottleneck = tiling / alignement, pas BW mémoire. hidden_size multiple de 2048 = OK (tile_util=1.0) ; 3584 = ~0.50 → -30% TPS attendus (Qwen3.5-9B 7.68 t/s vs 10.75 pour 4096)
3. INT4 n'améliore PAS TPS sur kernels AIE2 actuels (déquantifié vers INT8/BF16 avant kernel). G_tps(INT4)=0. G_storage=1.34. Préférer INT8 pour TPS (G_tps=0.693, risque ×5 moins)
4. Hybrid : NPU pour couches alignées ; GPU (RTX) pour lm_head / couches non alignées + validation Hardware Fitness
5. `--pmode performance` = +44% TPS, zéro coût (firmware)
6. BW_eff réelle mesurée = ~30 GB/s (vs ~55-60 GB/s théorique LPDDR5X) — important pour le planner (coût DMA)
7. Problèmes listés P1-P7 : mapping HF→ONNX, INT4 natif futur, KV cache dynamique, calibration BW, cols actives (XDNA2_COLS_ACTIVE=4 pas 32), pmode, crash >20B
8. Seuils λ D2 corrigés : λ_indiff(INT4)=1.675 ; λ_indiff(INT8)=4.621

## Impact pour notre SPEC_RYZEN9_HX365_5070_8GB.md
- Adopter hidden_size multiple 2048/4096 dans le planner (éviter 3584 si possible)
- Préférer INT8 pour speed, INT4 seulement si RAM contrainte
- RX 5070 8GB = tier HOT / fall-back pour lm_head et non-aligné
- BW_eff ~30 GB/s → contention moins sévère que 226 GB/s unifié, mais DMA coût réel à caler
- `--pmode performance` à activer systématiquement (+44%)
- Crash >20B : planner doit rejeter modèles ≥20B sur NPU (seuil de sécurité)
