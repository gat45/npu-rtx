# npu-rtx/ — D2 System-Aware Adaptive Precision & Residency Planner
# Dossier projet (jamais modifie profiler_v3 / sources). Tout ce qui est cree ici.
# ⚠️ CIBLE PRINCIPALE : Qwen3.6/3.5-35B-A3B (40 layers, 256 experts, top-8 + 1 shared, ~35B/3B actifs).
# Materiel cible : Ryzen 9 HX 365 (XDNA2) + RTX 5070 8 GB. GTX 1080 = machine dev test logique.
# Objectif : frontiere de Pareto memoire<->precision<->perf, avec profiler-v3 (lecture seule).

## STATUS (honnete — ne pas confondre "le planner calcule" et "le systeme execute")

| Composant | Etat |
|---|---|
| STATIC ORACLE | ✅ OPERATIONAL (model_parser, quant, expert_mapper, bytes/token, memory, lower bounds) |
| HARDWARE MODEL | 🟡 PARTIAL (dev = GTX 1080 ; cible 5070+XDNA2 a mesurer) |
| TARGET MEASUREMENTS | 🔴 NOT COMPLETE (PCIe/H2D/NPU tile/XRT sur machine cible) |
| RUNTIME | 🟡 SQUELETTE (residency_manager + kernel_registry + full_matrix P0-P9) |
| DYNAMIC CALIBRATION | 🟡 SQUELETTE (online_calibration + prediction_error) |
| D2 DECISION | 🔴 EXPERIMENTAL (lambdas ASSUMED, confidence 0.5) |
| END-TO-END | 🔴 NOT VALIDATED (P0-P9 sur machine cible requis) |

## ARBORESCENCE (nouvelle — runtime/kernels/feedback ajoutes)

```
npu-rtx/
├── static/                  # Static Oracle (Phase A)
│   ├── model_parser.py        # config.json -> dims + shapes experts
│   ├── quant_size_engine.py   # taille par format
│   ├── expert_mapper.py       # mapping expert -> tuiles mmul XDNA2
│   ├── conversion_matrix.py   # cout chemins A/B/C/D NVFP4->INT8
│   ├── bytes_per_token.py     # octets actifs/token + cache (parametre par config)
│   ├── memory_planner.py      # budgets VRAM + elimination
│   └── static_oracle_35b.py   # Static Oracle 35B-A3B (cible)
├── collectors/               # Phase B - mesures
│   ├── hw_discovery.py         # HardwareProfile + provenance (cible=HX365+5070)
│   ├── ssd_ddr_pcie.py         # microbench
│   └── quant_bench.py          # conversion
├── kernels/
│   └── kernel_registry.py      # KernelCapabilityDB (sm120 + xdna2)  [TROU 12]
├── runtime/
│   └── residency_manager.py    # ExpertResidencyManager (promote/evict/pin) [TROU 24]
├── feedback/
│   └── online_calibration.py   # boucle prediction->actual->confidence [TROU 19]
├── experiments/
│   └── full_matrix.py          # 10 tests P0-P9 (meme format de trace)
├── correlation/
│   └── unify_events.py         # adaptateur -> JSONL profiler-v3
├── oracle/
│   ├── feasibility.py          # filtre contraintes dures (cache_budget = VRAM-w-KV-ws)
│   └── d2_planner.py           # T(plan) avec T_overlap + Pareto
├── models/qwen36_35b_a3b/config.json
├── runs/<ts>/
└── *.md                       # docs (rapports, matrices, blindspots)
```

## RESULTATS CLES OBTENUS

### Cible 35B-A3B (Static Oracle, 40 layers / top-8 / 3.146M params par expert)
| Resultat | Valeur | Consequence D2 |
|---|---|---|
| **DENSE backbone / token (FINDING)** | **4.375 GiB = 85.2% du trafic** (vs 0.761 MoE) | **le cache expert seul n'optimise que ~15%** — ajouter DENSE cache |
| Expert Q4 | 1.69 MiB (3.146M params) | unite de cache |
| MoE actif/token Q4 | 0.762 GiB (0.527 routed + 0.234 shared BF16) | plancher trafic MoE |
| Cache MoE 90% | 0.762 -> 0.053 GiB/token PCIe (x14) | cache = variable de 1er ordre |
| Lower bound Q4 @20GB/s | cold ~40 ms/token · hit90 ~4 ms | elimination |
| NVFP4->INT8 | conversion double le cout a froid (138->278us) | conversion dans placement |
| VRAM utilisable | 6.5 GiB (8 - 1.5 WDDM) | hard constraint |
| RTX 5070 Laptop | **8GB GDDR7, 384 GB/s** (PAS 672 desktop) | BW corrigee |
| Ryzen AI 9 365 | **Strix Point** (PAS Dragon Range), NPU 50 TOPS marketing | spec corrigee |
| L1 XDNA2 | sous-tuiles <= 64 KiB (dims 2048/512 multiples de 8) | contrainte de plan |

### Pareto actuel (35B, lambdas ASSUMED - calibration requise)
```
C1_q2_xdna : Q2 XDNA2 · latency 7.9ms · pcie 0.074 GiB/t · vram 3.0 GiB  (non-domine)
```

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