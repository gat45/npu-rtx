# Référence — Reverse XDNA2 exposé par 1bit-MONSTER (docs/journey.md + architecture.md)
# Source : https://github.com/1bit-MONSTER/1bit-MONSTER · ~600h de RE open-source
# Fetched 2026-09-20 — faits "PROVEN" de leur journey (machine Ryzen AI Max+ 395 Strix Halo)

## Ce que 1bit a reverse (tout est public)
- **Q4NX format entièrement reverse** : dtype I8 = FAUX. Les données sont du **BF16 stocké en byte-pairs** `[lo,hi]` little-endian. Pas de dequant par groupe (absmax per-group = WRONG). Shape [256,5120] I8 = [256,2560] BF16.
- **Limite firmware : 8 colonnes HARDCODEN** — `DRM_IOCTL_AMDXDNA_CREATE_HWCTX` rejette EINVAL pour column_width > 8 (testé 9/10/12/16/40). Le kernel permet 40 (`aie2_max_col`), le firmware valide indépendamment (npu_7.sbin 1.1.2.65, RSA-4096, section chiffrée). **31.0 TFLOPS = plafond pratique NPU**.
- **Architecture XRT : runlist** — FLM fait TOUT via `xrt::runlist` (weight-DMA ops 2 BOs + compute ops 3 BOs). `runlist::execute()` charge les poids atomiquement, puis `run::start()` drive le compute. Poids en tile SRAM (weight-stationary), pas depuis DDR BO.
- **NPU2 : 8+ hw_contexts concurrents** vérifiés (firmware 1.1.2.65) → on peut avoir plusieurs xclbins/kernels vivants en parallèle (base de "experts sur NPU pendant que GPU calcule").
- **bo.sync bug XRT** : `sync(dir, 0, size)` traite sz=0 comme flag → crash ; utiliser `sync(dir, sz, 0)`. (Écho exact du `SYNC_DIRECT_TO_DEVICE` Linux mais via API C++ XRT.)
- **INT8 = chemin retenu** : weights Q4NX → float32 → requant symétrique INT8 (per-tensor max/127), uploadé une fois en BO persistants. Activations quantifiées on-the-fly (scale 5/127 validée). Évite la double-quant Q4NX→BFP16→hardware.
- **GTT dma-buf zero-copy : 56 GB/s mesurés** (host↔NPU via GTT). Référence pour le coût DMA du planner.
- **Perf NPU Qwen3 0.6B (Strix Halo)** : 82-91.5 t/s decode, TTFT 0.48s, ~2W, 46 tok/s/W (25× plus efficace que GPU), prefill 1591 t/s en chunk 8192, KV ~15k tokens.

## Points qui changent le D2 Planner
1. **Capacité NPU = 8 colonnes / 4×4 tiles (ou 8×4)** — PAS 32/40. Le planner doit modéliser ce plafond de colonnes, pas une VRAM imaginaire. Ex. Qwen3.5-9B hidden 3584 → ~50% tiles idle (static slicing).
2. **W4/INT4 = stockage seulement** : le compute NPU se fait en INT8/BF16. Un expert Q4 sur NPU → en pratique INT8 → le planner doit comparer coût INT8, pas Q4.
3. **Poids en SRAM tile (weight-stationary)** : charger un expert = DMA poids → tile SRAM. Le coût "DMA + SRAM" est LA variable, pas "envoyer une matrice sur un device".
4. **runlist = primitive de batch** : soumettre plusieurs ops d'un coup (runlist + wait une fois) → le planner peut grouper les experts du même token.
5. **hw_context multiples** → co-exécution NPU (attention sur un contexte) + GPU (decode) réaliste, comme AMD hybrid mais à granularité op.

## Limites / pièges pour notre adaptation
- Mesures 1bit sur **Strix Halo (Max+ 395, LPDDR5x 128GB unifié)** = NPU peut être plus gros. Notre **Strix Point (Ryzen AI 9 365)** a un NPU XDNA2 de même famille mais capacités/mémoire différentes (à mesurer, MANQUANT #1).
- Le 8-col limit firmware est vérifié sur Strix Halo ; à reconfirmer sur Strix Point (nombre de colonnes expose via QUERY_AIE_METADATA).
- Format 1BP (proprio 1bit) ≠ Q4NX ≠ GGUF : le planner doit gérer la conversion poids par target (comme d2-quant-planner).