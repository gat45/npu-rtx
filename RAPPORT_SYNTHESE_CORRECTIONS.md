# RAPPORT SYNTHÈSE — corrections matérielles + 10 blocages + tests P0.1-P0.10
# Établi 2026-09-20 · Dossier npu-rtx/ · Croise : docs locales + sources publiques
# (NVIDIA, AMD, vLLM/FlashInfer, Qwen) + jarvix-memory + profiler-v3.

---

## 1. CORRECTIONS MATÉRIELLES (appliquées — voir CORRECTIONS_MATERIEL.md)

| Correction | Avant | Après (source) |
|---|---|---|
| RTX 5070 BW | 672 GB/s | **384 GB/s Laptop** (NVIDIA : desktop 12GB=672, laptop 8GB=384) |
| CPU/NPU | Dragon Range / ~16 TOPS | **Strix Point** (4 Zen5+6 Zen5c, 10C/20T) / **50 TOPS marketing** (AMD) |
| NPU_TOPS | utilisé dans planner | `NPU_TOPS_MARKETING=50` ≠ `NPU_TOPS_EFFECTIVE=MEASURED` (6.65-8.69 kernel-only) |

Fichiers corrigés : SPEC, DOC_EXHAUSTIVE, OBJECTIF, BLINDSPOTS_HW, RAPPORT_FLASH_35B_CORRIGE,
RECHERCHE, PLAN_EXECUTION, place_expert.py (BW_RTX 384e9).

## 2. LE FINDING MAJEUR : le backbone dense peut dominer le trafic (vLLM #51197)

Analyse publique sur Qwen3.6-35B-A3B : le **backbone GDN dense BF16 ≈ 2.95 GB / 3.7 GB du
trafic poids par étape decode (~80%)**, contre ~0.75 GB pour les experts NVFP4/FP8.

**Conséquence** : optimiser parfaitement l'expert cache/prefetch peut avoir PEU de gain si le
backbone dense domine déjà. Il faut la métrique :
```
bytes/token = dense_backbone_bytes + routed_expert_bytes + shared_expert_bytes
            + attention_state + GDN_state + other_state
```
→ **deux caches logiques** : EXPERT_RESIDENCY + DENSE_STATE/WEIGHT_REUSE.

## 3. LA CORRECTION ARCHITECTURALE : le niveau ENGINE manque

```
expert → device → engine → kernel → format → path
ex. NVFP4 → RTX → FlashInfer → SM120 kernel A
    Q4    → RTX → llama.cpp  → MMQ
    Q4    → XDNA → dequant → INT8 → IRON/XRT
```
Le recipe vLLM Qwen3.6 : choix kernel XQA decode SM120 selon version (≥0.28.0), modelopt_fp4,
Marlin MoE, block-size 128, TRT-LLM attention. → format seul ne suffit pas.

## 4. LES 10 VRAIS BLOCAGES (des 66 angles morts)

| # | Blocage | Preuve | État |
|---|---|---|---|
| 1 | Profil RTX faux | NVIDIA : 8GB/384 GB/s | ✅ corrigé |
| 2 | Modèle exact non verrouillé | Qwen3.5/3.6/3.8 différents | 🔴 à verrouiller |
| 3 | Dense backbone ignoré | vLLM #51197 ~80% | 🔴 ajouter bytes/token dense |
| 4 | Engine/kernel absent du choix | vLLM/FlashInfer/SM120 | 🔴 kernel_registry + engine |
| 5 | Runtime XDNA dominé dispatch/sync | 3064 dispatch, 81.32 ms | 🔴 résoudre (runlist, persistent ctx) |
| 6 | PCIe réel inconnu | repo UNKNOWN | 🔴 mesurer cible |
| 7 | XDNA driver/runtime versioning | issues amdxdna/XRT | 🔴 hardware_hash |
| 8 | Prefetch sans deadline | lead-time non modélisé | 🟠 deadline scheduler |
| 9 | VRAM cache sans GDN/KV/workspace | rapport Flash | 🟠 cache_budget = VRAM−w−KV−ws |
| 10 | Pas de boucle Runtime→Profiler→D2 | README | 🔴 runtime/ + feedback/ |

## 5. LES TESTS PRIORITAIRES (P0.1-P0.10) — à préparer

| Test | Mesure | Fichier prévu |
|---|---|---|
| P0.1 | GPU SM120 id, VRAM usable, clock/power/temp | collectors/hw_discovery (cible) |
| P0.2 | PCIe gen/width + H2D/D2H 4K→64M | collectors/ssd_ddr_pcie (cible) |
| P0.3 | kernel NVFP4 M/N/K réel + workspace | kernels/kernel_registry (mesure) |
| P0.4 | GDN dense backbone bytes/token + state | static/bytes_per_token (+dense) |
| P0.5 | expert bytes/cache hit/reuse distance | runtime/experiment_cache |
| P0.6 | XDNA DDR→NPU DMA/submit/completion/compute | collectors/amd_npu |
| P0.7 | overlap RTX+XDNA+SSD | runtime/synchronizer |
| P0.8 | prediction top1/2/4/8 + lead time + waste | feedback/predictor |
| P0.9 | Flash SSD cold/warm/RAM/VRAM | experiments/flash_path |
| P0.10 | end-to-end prefill/decode 4K/32K/128K | experiments/full_matrix |

## 6. JARVIX-MEMORY (oublié — à relier)

- `gat45/jarvix-memory` = projet mémoire JARVIX (gestion mémoire conversationnelle).
- Lien : le D2 Runtime doit exposer ses décisions de résidence d'experts à la couche mémoire
  JARVIX (promotion/demotion/eviction en fonction du contexte conversationnel).
- Intégration prévue : `runtime/` → jarvix-memory (persistance + hotness par session).

## 7. SOURCES PUBLIQUES (ajoutées au registre)
- NVIDIA RTX 50 Laptop specs (384 GB/s)
- AMD Ryzen AI 9 365 (Strix Point, 50 TOPS)
- AMD AIE-ML v2 memory (64 KB/tile, 8 banks, DMA)
- vLLM Qwen3.6-35B-A3B recipe (NVFP4, SM120 kernel selection)
- FlashInfer SM120 NVFP4 issues (kernels par version)
- Qwen3.6 GDN bandwidth issue (vLLM #51197 — backbone ~80%)
- AMD XDNA driver telemetry compat issue
- gat45/jarvix-memory (couche mémoire)

## 8. PROCHAINE ACTION
Exécuter P0.1-P0.2 sur la machine cible (hw_discovery + ssd_ddr_pcie) pour remplacer les
UNKNOWN du HardwareProfile, puis P0.4 (bytes/token dense backbone) qui est le finding critique.