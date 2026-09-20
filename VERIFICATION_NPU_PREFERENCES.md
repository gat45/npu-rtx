# VERIFICATION — CE QUE LE NPU XDNA2 AIME RÉELLEMENT (reverse + mesures)
# Établi 2026-09-20 · Dossier npu-rtx/ · Sources : reference/1bit/amd-oss/DESCENT.md (mesures
# réelles on-device), reference/1bit/FLM_SECRETS.md, NPU_ISA, ANALYSE_DEFINITIVE (fastflow), AMD docs
# Réponse : OUI, INT8 = chemin de premier rang ; BFP16 = chemin float natif ; BF16 = ÉMULÉ (¼ débit).
# ⚠️ valeurs = preuves de principe sur matériel XDNA2 réel (HX370/Max+395), à REPRODUIRE sur HX365.

---

## 1. VERDICT (ce que le NPU aime)

```
XDNA2 (AIE2P)
   │
   ├── INT8  → chemin natif massif, accumulation INT32, bit-exact ✓
   ├── BFP16 → chemin float natif optimisé (8x8x8 BFP16 MMUL)
   └── BF16  → ÉMULÉ via BFP16 (~¼ du débit BFP16 natif) ⚠️ NE PAS UTILISER en P0
```

| Chemin | Débit mesuré | Exactitude | Verdict |
|---|---|---|---|
| **INT8→INT32** | **6.65-8.69 TOPS** (2048³, tile 64³) | bit-exact vs numpy int32 (0 bad) | **⭐ P0 — le plus solide** |
| BFP16 | 4.64 TFLOPS (2048³, tile 64×32×64) | bf16 noise (~0.02) | P0 bis |
| BF16 direct | ~2.65 TFLOPS (2048³) ; émulé ~¼ BFP16 | — | ⚠️ éviter en P0 |

## 2. PREUVES MESURÉES (DESCENT.md, XDNA2 réel)

### 2.1 INT8 GEMM (bit-exact, 0 bad elements)
```
1024×512×1024 : ~0.33 ms → 3.24 TOPS
2048³         : 2.095-2.46 ms → 7.0-8.20 TOPS (best 8.69, mean 8.49)
4096³         : 17.5 ms → 7.84-7.90 TOPS
```
- Bottleneck : **pipeline-bound** (objdump : 16 vmac vs ~100 nops + 24 vlda + 17 vst/loop)
- Software-pipelined k-loop (prefetch ping/pong) : +5-8% (7.6 → 8.0-8.3)
- 2x4 register-blocked : RÉGRESSION (register pressure + j-loop) → revert
- tile_n=128 : L1 overflow (C tile 32KB + double-buffer) → pas testable
- **Peak théorique 8x8x8 int8 vmac = 512 MACs/cyc/tile ≈ 40+ TOPS** (50 marketing = 512×~1.5GHz)
- Kernel-only tuning sature à ~8 TOPS ; 50 TOPS-class = design-level rewrite (B/A en L1, double
  ObjectFifos, C accum L1)

### 2.2 Géométrie de tuile domine (leçon profiler-v3)
```
INT8 2048³ tile 32³ : 1.97 TOPS
INT8 2048³ tile 64³ : 6.65 TOPS   ← 3.38× par simple changement de tuile
contrainte : > 64 KB core-local → placement impossible
```

### 2.3 BFP16 vs BF16
- `emulate_bf16_mmul_with_bfp16=True` (DÉFAUT) : BF16 routé via le datapath BFP16
- BF16 direct nettement moins performant sur AIE2P (~¼ du BFP16 natif)
- 8x8x8 BFP16 = mode MMUL natif XDNA2

### 2.4 Scaling colonnes (linear, kernel-bound)
```
8 cols → 8.71 TOPS total (272 Gops/s/core)
4 cols → 4.91 TOPS (307/core)
→ scaling linéaire en cœurs, PAS array-DMA-bound
```

## 3. L'AVANTAGE ARCHITECTURAL INT8 (au-delà du débit)

À INT8 : A bytes = M×K, B bytes = K×N.
À BF16 : ×2.
→ Même L1 (64 KB), **INT8 charge ~2× plus d'éléments** → tile plus grand + plus de réutilisation
+ moins de DMA. C'est un avantage de dataflow spatial, pas juste de stockage. Exactement le
genre de calcul que l'analyseur statique (STATIC_ORACLE) doit produire.

## 4. LES PIÈGES DU CHEMIN (bugs réels documentés — profiler-v3 doit tracer)

1. **group_id=0 → no-op silencieux** (ERT COMPLETED, NPU ne touche pas les BOs) — le bug qui
   a "faké" des "wrong results". Fix : kernel.group_id(3+i) pour les host groups 0x10000/0x20000.
   → profiler-v3 doit vérifier **n_backends/mirror/repack** (leçon AGENTS.md rc=0≠preuve).
2. **group-0 BOs à grosses shapes → WEDGE NPU** (IO_PAGE_FAULT, aie2_set_cmd_timeout, PCI
   remove/rescan requis).
3. **3ème hw_context distinct → garbage silencieux** (ctx0/1 exact, ctx2 1,047,753 wrong,
   déterministe 3/3 si tensors créés entre runs). **LLama prefill 32 layers = 13 xclbins ≈ 480
   context switches** → risque réel. Decode (ELF unique) = propre.
4. **Driver SANS barrière de données** : aie2_sync_bo via mailbox firmware ; completion =
   AIE2_STATUS_SUCCESS → ERT COMPLETED ; le driver ne voit PAS la corruption.
5. Variance process-to-process 7.7-8.6 TOPS (même binaire) : SoC clock/thermal.

## 5. CONCLUSION D2 (pour Qwen Flash → XDNA2)

```
Qwen storage (NVFP4/Q4...)
   │
   ├──► RTX 5070 : NVFP4/FP8 (chemin CUDA natif)
   │
   └──► XDNA2 : INT8 (chemin natif P0) ou BFP16 (P0 bis)
            └── conversion NVFP4→INT8 à benchmarker (A/B/C/D)
```
- **INT8-XDNA2 = voie P0** : reverse au niveau mmul<...> + Peano assembly + L1/DMA
- **BFP16-XDNA2 = voie P0 bis**
- **BF16-XDNA2 = à éviter** (émulé ¼)
- Point à trouver ensuite : mapping d'un expert 2560×640 vers les géométries 4×8×8 / 8×8×8,
  et coût réel de conversion depuis le format de stockage.

## 6. SOURCES (reverse)
- reference/1bit/amd-oss/DESCENT.md : INT8 6.65-8.69 TOPS, BFP16 4.64, bug group_id, pipeline-bound
- reference/1bit/FLM_SECRETS.md : RTP, ISA, tiles, MHA variants
- reference/1bit/NPU_ISA.md : opcodes NPU, tile layout (8 shim + 4 mem + 16 compute)
- reference/fastflow/ANALYSE_DEFINITIVE_XDNA2.md : tiling 3584 vs 4096, INT4 déquant, CP starvation
- AMD docs : xrt-smi validate --run gemm (INT8 full-array), aie_api MMUL modes, IRON objdump guide