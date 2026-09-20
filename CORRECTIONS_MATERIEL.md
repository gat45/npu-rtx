# CORRECTIONS MATÉRIELLES — à appliquer partout dans npu-rtx
# Établi 2026-09-20 · Source : NVIDIA + AMD officielles + analyse utilisateur

## 1. RTX 5070 : 384 GB/s (Laptop), PAS 672 (Desktop)
```
RTX 5070 desktop : 12 GB GDDR7 · 672 GB/s
RTX 5070 Laptop  :  8 GB GDDR7 · 384 GB/s   ← NOTRE MACHINE
SM = 120 · CC 12.0
```
- Impact : `bytes / 672 GB/s` sous-estime le coût mémoire GPU de 672/384 = **1.75×**.
- Corrections dans : DOC_EXHAUSTIVE_5070_NPU_BUS, OBJECTIF_5070_NPU_BUS, BLINDSPOTS_HW,
  RAPPORT_FLASH_35B_CORRIGE, RECHERCHE_NPU_RTX_URLS, PLAN_EXECUTION_5070_NPU, place_expert.py (BW_RTX).
- ⚠️ La GTX 1080 (machine dev) = 320 GB/s → la 5070 Laptop (384) n'est qu'1.2× plus rapide en BW.

## 2. Ryzen AI 9 365 = Strix Point, PAS Dragon Range
```
Ryzen AI 9 365 : Strix Point · 4 Zen 5 + 6 Zen 5c · 10C/20T
XDNA2 NPU · peak marketing = 50 TOPS (spec AMD)
PCIe 4.0 · 16 lanes · 2 canaux mémoire · DDR5-5600
```
- ⚠️ NE PAS utiliser 50 TOPS comme perf réelle : `NPU_TOPS_MARKETING = 50` ≠ `NPU_TOPS_EFFECTIVE = MEASURED`.
- Mesures internes : 6.65-8.69 TOPS INT8 kernel-only, 21.93 GB/s DDR effectif, orchestration dominante.

## 3. Règle de provenance (rappel)
```
MEASURED : benchmark réel (profiler-v3 / hw_discovery cible)
DERIVED  : calculé depuis une mesure
ASSUMED  : hypothèse documentée
UNKNOWN  : à mesurer (ne pas inventer)
A (constructeur) ≠ B (documentation logicielle) ≠ C (mesure machine)
```

## Fichiers à corriger (fait : SPEC_RYZEN9_HX365_5070_8GB.md)
- [x] SPEC_RYZEN9_HX365_5070_8GB.md (384 GB/s, Strix Point, 50 TOPS)
- [ ] DOC_EXHAUSTIVE_5070_NPU_BUS.md
- [ ] OBJECTIF_5070_NPU_BUS.md
- [ ] BLINDSPOTS_HW.md
- [ ] RAPPORT_FLASH_35B_CORRIGE.md
- [ ] RECHERCHE_NPU_RTX_URLS.md
- [ ] PLAN_EXECUTION_5070_NPU.md
- [ ] place_expert.py (BW_RTX 672→384e9)