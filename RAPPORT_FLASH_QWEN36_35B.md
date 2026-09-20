# RAPPORT FLASH — QWEN3.6-35B-A3B
# Établi 2026-09-20 · Dossier npu-rtx/ · Pipeline identique au Flash Qwen3.8, recalculé pour 35B-A3B
# Séparation STRICTE : faits documentés / dérivations / hypothèses.
# ⚠️ Ne jamais figer "Q4 = X GB" : les tailles GGUF varient selon le quantizer (~20-22 GB Q4_K_M).
# À vérifier contre le checkpoint réel (config.json + convert.log) avant de conclure.

---

## 0. OBJECTIF

Le problème n'est PAS « faire tenir 35B en VRAM » mais :
> Comment exécuter un MoE 35B avec ~6.5 GiB dispo (8 - 1.5 WDDM) via SSD → RAM → PCIe → RTX 5070
> (+ XDNA2), sans charger les 35B en mémoire ?

```
          Qwen3.6-35B-A3B
               │
      ┌────────┴────────┐
      │                 │
   backbone         256 experts
   résident         SSD / cache
      │                 │
      │         ┌───────┴───────┐
      │         │               │
      │       cache           SSD
      │         │               │
      │         └───────┬───────┘
      │                 │
      │                RAM
      │                 │
      │           PCIe / DMA
      │      ┌──────────┴──────────┐
      │      │                     │
      │    RTX 5070              XDNA2
```

## 1. FAITS DOCUMENTÉS (config officielle — à vérifier contre le checkpoint réel)

| Paramètre | Valeur |
|---|---|
| Paramètres totaux | ~35B |
| Actifs/token | ~3B |
| Couches | 40 (30 GDN : 10 Attention = 3:1) |
| Hidden size | 2048 |
| Experts | 256 |
| Routed actifs | 8 |
| Shared | 1 (9 experts actifs/token) |
| Expert intermediate | 512 |
| KV heads | 2 |
| Vocab | 248 320 |
| Contexte natif | 262 144 |
| Attention | Gated Attention (10 couches) |
| Linear attention | Gated DeltaNet (30 couches) |

Source : config.json officiel (dans models/qwen36_35b_a3b/config.json).

## 2. DÉRIVATIONS (calculées — Static Oracle 35B)

### Taille d'un expert (gate/up merged + down)
```
gate/up : 2048 × 1024 (merged 2×512) · down : 512 × 2048
params/expert = 2048×1024 + 512×2048 = 3 145 728 ≈ 3.146M
```
| Format | Poids/expert |
|---|---|
| BF16 | 6.0 MiB |
| INT8 | 3.0 MiB |
| Q8_0 | 3.19 MiB |
| Q6_K | 2.46 MiB |
| **Q4/NVFP4** | **1.69 MiB** |
| Q3 | 1.31 MiB |
| Q2 | 0.99 MiB |

### Nombre d'experts
```
256 × 40 = 10 240 routed + 40 shared = 10 280 blocs expert
params routed = 10 240 × 3.146M ≈ 32.2B
BF16 ≈ 64.4 GB  (cohérent avec modèles BF16 ~65-70 GB ; GGUF Q8 ~35 GB, Q4_K_M ~20-22 GB)
```
⚠️ Ces ~64 GB ne sont PAS un working set obligatoire.

### Trafic froid (sans cache) par token
| Format | MiB/token |
|---|---|
| Q4 | 780 |
| Q6_K | 1028 |
| Q8_0 | 1260 |
| BF16 | 2160 |

### Trafic avec cache (Q4, MiB/token)
| Hit | MiB/token |
|---|---|
| 50% | 390 |
| 75% | 195 |
| **90%** | **78** |
| 95% | 39 |
| 99% | 7.8 |

→ **cache 90% = ×10 de trafic** (780 → 78 MiB/token). Même phénomène que Flash (1.236→0.124 GiB).

### Lower bounds (placeholder BW — à remplacer par mesures profiler-v3)
```
PCIe 20 GB/s : Q4 cold = 40.9 ms/token · Q4 hit90 = 4.09 ms/token
```
⚠️ placeholders, PAS des mesures.

## 3. HYPOTHÈSES (à valider par Dynamic)

1. **VRAM budget = 6.5 GiB** (8 - 1.5 WDDM reserve) — contrainte dure
2. **L1 XDNA2 ≤ 64 KiB** — contrainte de plan/kernel
3. **2048/512 multiples de 8** → pas de pénalité d'alignement mmul 8x8x8 (contrairement à 3584)
4. **GGUF Q4_K_M ~20-22 GB varie selon quantizer** → NE PAS figer, extraire du checkpoint réel
5. **Conversion NVFP4→INT8 = coût réel** (résultat Flash : 138→278 µs à froid) à re-mesurer 35B

## 4. CE QUI DOIT RÉSIDER DANS LES 6.5 GiB

```
backbone + active experts + expert cache + GDN state (30 layers) + attention KV (10 layers)
+ workspace + CUDA/WDDM overhead
```
→ **jamais** `model_memory = weights + KV`. GDN state, attention KV, expert cache, workspace
sont comptés SÉPARÉMENT.

## 5. MÉTRIQUES OBLIGATOIRES (par token)

```
token_id layer expert_ids expert_weights
cache_hit cache_miss
SSD_read_bytes/us RAM_stage_bytes/us PCIe_read_bytes/us DMA_us conversion_us
RTX_compute_us XDNA_compute_us sync_us
GDN_us QSA_us KV_bytes
VRAM_peak RAM_peak prefetch_hit prefetch_waste
total_us tok/s
```
+ 3 compteurs indispensables 35B : **expert_reuse_distance, expert_cooccurrence, prefetch_accuracy**.

## 6. LES 10 EXPÉRIENCES (Phase C)

| # | Expérience | Objectif |
|---|---|---|
| T1 | dense baseline RTX-only | PCIe/launch/memory/compute |
| T2 | small MoE RTX-only | routing/dispatch |
| T3 | 35B cache OFF (Q4, SSD) | SSD/RAM/PCIe/ms par token |
| T4 | 35B cache ON 90% | hit/miss/bytes/token — **expérience clé** |
| T5 | T4 + prefetch | latency/SSD/stall |
| T6 | precision Q4/Q6/Q8/INT8 (même routing) | format réel |
| T7 | PLE placement | cache/prefetch |
| T8 | GDN/QSA séparés | state/KV/workspace |
| T9 | RTX only | référence |
| T10 | RTX + XDNA2 | hybride, mêmes experts |

## 7. MODÈLE DE COÛT (contraintes dures + score)

```
Cost(plan) = α·latency + β·PCIe_bytes + γ·SSD_bytes + δ·RAM_pressure
             + ε·VRAM_pressure + ζ·conversion + η·sync
avec contraintes DURES :
  VRAM > 6.5 GiB → impossible
  RAM > budget   → impossible
  tile > 64 KiB  → impossible
```
(α..η calibrés par profiler-v3, jamais ASSUMED comme MEASURED.)

## 8. BOTTLENECK ATTENDU (35B, à mesurer)

Avec ~3B actifs/token, NE PAS optimiser "35B compute" mais **3B compute + expert movement**.
→ le bottleneck peut être SSD/PCIe/DMA/launch/sync AVANT le compute. profiler-v3 doit le révéler.

## 9. SOURCES
- config officiel : huggingface.co/Qwen/Qwen3.6-35B-A3B (à vérifier)
- Static Oracle 35B : `static/static_oracle_35b.py` + `models/qwen36_35b_a3b/config.json`
- Pipeline : voir README.md + PLAN_CAMPAGNE_PARETO.md + STATIC_ORACLE_QWEN38.md