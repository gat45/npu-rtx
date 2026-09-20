# Référence — amd-oss-knowledge (1bit-MONSTER) — stack NPU open-source AMD
# Source : https://github.com/1bit-MONSTER/amd-oss-knowledge (snapshot ~2026-08-31)
# Fetched 2026-09-20 — fichiers copiés dans reference/1bit/amd-oss/ (README, ANALYSIS, INDEX, DESCENT, structure)

## La chaîne open-source AMD NPU (vérifiée bit-exact à chaque hop)
```
PyTorch (Brevitas) → QONNX → FINN (FPGA)     [chaîne FPGA, maturité mais ralentie]
  ── et côté NPU ──
Triton kernels / IRON Python API              (programming models)
   → MLIR-AIR                                  (scheduling spatial: tiling/placement/buffering/sync)
   → MLIR-AIE                                  (substrat partagé — PAS AIR ; IRON bypass AIR)
   → LLVM-AIE / Peano                          (codegen ISA AIE)
   → XRT (libxaie) / HSA                       (dispatch hardware)
   → Ryzen AI NPU (AIE2/AIE2P : Phoenix/Hawk/Strix/Krackan)
```

## Le verdict stratégique pour le D2 Planner
**Le backend XDNA2 n'a PAS besoin du FLM propriétaire** : AMD pousse un stack OSS complet
(IRON Apache-2.0, Triton-XDNA MIT, MLIR-AIR MIT, MLIR-AIE, Peano/LLVM-AIE) qui produit
xclbin/elf/pdi. Deux chemins de programmation sur UN substrat (comme CUDA/PTX) :
- **IRON** : structural Python close-to-metal (ObjectFIFOs, per-column Workers, 28 opérateurs :
  GEMM/MHA/RMSNorm/RoPE/softmax bf16, kernels AIE C++ aie2/aie2p, Llama 3.2 1B e2e, CI sur
  Phoenix (NPU1) + Krackan (NPU2)).
- **Triton-XDNA** : `@triton.jit` → TTIR → MLIR-AIR → aircc → xclbin/elf/pdi ; Windows natif ;
  matmul généré à parité avec kernels manuscrits (>90% des configs ≥90% du baseline).

## FastFlowLM = ROCm/FastFlowLM (Apache-2.0) — acquis AMD 2026
- NPU-first, contextes jusqu'à 256k, xclbins + plugin driver XDNA (`libxrt_driver_xdna.so.2`),
  libs par modèle (llama_npu, qwen_npu).
- HRX (amdxdna) release pin : tag `flm-hrx-amdxdna-v2026.07.30`.
- Chemin déterministe corr=1.0 = pipeline 1BP fixed-point ; le float 32-token corr≈0.99997
  = bruit d'ordre de sommation f32 amplifié par récurrence (~10⁵×), PAS de la perte de quant.

## HRX2 llama.cpp lane (recherche 1bit, docs/research/hybrid-prefill-decode.md)
- Q4NX (format tile engine) tourne sur gfx1151 via le fork llama.cpp HRX2.
- Round 25i : TRUE zero-DMA-copy → decode +10-49% sur la liste, MAIS pp32 prefill régressé
  121.5 → 27.8 (3B). Attribution débattue (CPU attention F32 mms → GTT, NPU lit lentement =
  taxe cache-coherency).
- **Hybride préfill/decode documenté** : fast prefill lane = HIP (1227-1313 tok/s) ;
  warm decode = HRX (~80-87 tok/s). → exactement le pattern AMD hybrid, en OSS.
- Leçon planner : zero-DMA-copy ≠ toujours gagnant (taxe cache-coherency sur prefill) → le
  cost model doit mesurer GTT vs host_coherent, pas supposer.

## Télémétrie / outillage
- Omnitrace (déprécié) → **rocprofiler-systems** ; **Omnistat** actif (per-GPU Instinct telemetry,
  recouvre aussi NPU via rocprofiler-sdk). SlabPool KV cache, paged attention dans PACE (CPU).
- NPUEval : 102 problèmes de codegen AIE kernel, scorés sur hardware réel (compile Peano/chess →
  xclbin → XRT → correctness + vpu_cycles + speedup jusqu'à ~300×). Outil de validation d'un
  kernel expert sur notre NPU.

## Conséquences concrètes pour xdna2/
1. **Reconstruire le backend = faisable en OSS** (IRON/Triton-XDNA/MLIR-AIR) au lieu de
   dépendre de xclbins FLM fermés. C'est le "model-driven compiler" du RAPPORT §3 (1bit
   l'a fait : MODELE → Analysis → IR → Codegen → xclbin → Validation → Runtime).
2. **IRON = outil de mesure** : ses tests = benchmarks (CSVReporter) ; 28 opérateurs couvrent
   attention + RMSNorm + RoPE → coût par op mesurable pour le planner.
3. **Triton-XDNA Windows natif** : notre machine Windows peut compiler des kernels experts
   MoE via ce chemin (plus léger que la toolchain Linux complète).
4. **Leçon HRX2** : ne pas activer zero-copy aveuglément (coût cache-coherency). Le coût DMA
   du planner (XCL_BO_SYNC_TO/FROM) doit être calibré en mode host_coherent vs GTT.
5. **ROCm/FastFlowLM open** : si AMD ouvre le code MoE (qwen3_6_moe_npu), on peut dériver
   les primitives expert (up/down/gate q41) sans reverse. À surveiller (MANQUANT #13).