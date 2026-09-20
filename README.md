# npu-rtx/ — D2 System-Aware Adaptive Precision & Residency Planner
# Dossier projet (jamais modifie profiler_v3 / sources). Tout ce qui est cree ici.
# Objectif : frontiere de Pareto memoire<->precision<->perf pour Qwen3.8-Flash-Next
# sur RTX 5070 + XDNA2, avec profiler-v3 comme instrument de reference (lecture seule).

---

## ARBORESCENCE

```
npu-rtx/
├── static/                  # Static Oracle (Phase A - calculs avant execution)
│   ├── model_parser.py        # config.json -> dims + shapes experts          ✅
│   ├── quant_size_engine.py   # taille par format (bpw effectif avec overhead) ✅
│   ├── expert_mapper.py       # mapping expert 2560x640 -> tuiles mmul XDNA2   ✅
│   ├── conversion_matrix.py   # cout chemins A/B/C/D NVFP4->INT8               ✅
│   ├── bytes_per_token.py     # octets actifs/token + effet cache             ✅
│   └── memory_planner.py      # budgets VRAM + elimination lower-bound        ✅
├── collectors/               # Phase B - mesures reelles
│   ├── hw_discovery.py         # HardwareProfile avec provenance MEASURED/    ✅
│   │                           #   DERIVED/ASSUMED/UNKNOWN                    ✅
│   ├── ssd_ddr_pcie.py         # microbench 1K->64M (DDR 49.6, SSD 12.3 GB/s) ✅
│   └── quant_bench.py          # conversion NVFP4->INT8 (Python = minimum)    ✅
├── correlation/
│   └── unify_events.py         # adaptateur -> JSONL profiler-v3              ✅
├── oracle/                    # Decision
│   ├── feasibility.py          # filtre : rejette plans impossibles (VRAM/L1/ ✅
│   │                           #   PCIe/workspace) avant Pareto
│   └── d2_planner.py           # Static+Dynamic+feasibility -> front Pareto   ✅
├── runs/<ts>/                 # sorties (hw.json, ...)
├── *.md                       # 26 docs (plan, matrices, blindspots, sources)
└── reference/                 # corpus local lecture seule
```

## RESULTATS CLES OBTENUS

### Static Oracle (deja exploitable)
| Resultat | Valeur | Consequence D2 |
|---|---|---|
| Cache 90% | 90 -> 6.6 ms/token | cache = variable de 1er ordre |
| PCIe MoE | 1.236 -> 0.124 GiB/token | optimiser bytes/token, pas FLOPS |
| Q4 sans cache | <= 11.1 t/s | toute mesure > = cache/prefetch requis |
| NVFP4->INT8 | 138 -> 278 us | conversion dans le cout de placement |
| VRAM utilisable | 6.5 GiB (8 - 1.5 WDDM) | hard constraint, pas penalite |
| L1 XDNA2 | sous-tuiles <= 64 KiB | contrainte de plan/kernel |

### HardwareProfile (source de verite, provenance explicite)
- Machine dev : GTX 1080 (8GB, sm_61), 32 GB RAM (6.9 dispo), CPU AMD 12 cores, driver 581.80
- Microbench : DDR memcpy 49.6 GB/s, SSD seq 12.3 GB/s (reels)
- UNKNOWN a mesurer sur machine cible : PCIe gen/lanes, NPU tile geometry, XRT/telemetry, H2D/D2H

### Feasibility filter (rejette avant Pareto)
```
FEASIBLE  C0_q4_rtx          Q4    RTX
FEASIBLE  C0_q4_xdna_int8    Q4    XDNA2-INT8
REJECT    C5_bf16_rtx        BF16  RTX    (VRAM 7.5 > 6.5)
FEASIBLE  C1_q2_xdna         Q2    XDNA2
REJECT    bad_l1             Q4    XDNA2  (L1 200KB > 64KB)
```

### D2 Planner (Pareto - lambdas ASSUMED a calibrer)
- Front actuel : C1_q2_xdna non-domine (lambdas = ASSUMED, confidence 0.5)
- **Calibration profiler-v3 requise (Phase C) avant toute conclusion**

## PIPELINE

```
Static Oracle (bornes)
      ↓
feasibility filter (rejette impossible)
      ↓
Dynamic measurements (profiler-v3 : PCIe/DDR/GEMM/conversion reels)
      ↓
calibration (lambdas)
      ↓
cost model (score plan)
      ↓
front Pareto
      ↓
D2 Planner (choix)
```

## PROCHAINE ETAPE
- Phase C : les 10 tests minimaux (full_matrix.py) pour calibrer les lambdas
  avec les mesures profiler-v3 (jamais ASSUMED comme MEASURED)
- Machine cible (5070 + XDNA2) : remplir les UNKNOWN du HardwareProfile

## REGLE D'OR
Le Static Oracle ne transforme JAMAIS silencieusement ASSUMED en MEASURED.
Chaque prediction porte provenance + confidence + samples + variance.