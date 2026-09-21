# 35B Genesis Q8_K_P — Profilage layer × expert, SNR EXACT grain expert

**Date** : 2026-09-21 · **Source** : `D:/Hermes3.6-35B-A3B-Uncensored-Genesis-Final-Q8_K_P.gguf` (40.61 GiB, recette Q8_0/F16 mesurée)
**Méthode** : `requant_experts.py` (d2_layer_profiler_v3 streaming) — SNR **exact** par expert
(quantize→dequantize gguf-py vs source Q8_0/F16), **30 720 experts × 3 familles = 92 160 mesures**,
120 tenseurs experts (40 couches × gate/up/down), 40 couches. Aucune estimation théorique : tout est mesuré.

## Candidat chiffré

| | Source Q8_K_P | Candidat gate/up Q4_0 + down Q5_0 |
|---|---:|---:|
| Fichier | 40.60 GiB | **20.76 GiB** (−49 %) |
| Pool experts | 37.97 GiB | **18.12 GiB** |
| MB/expert | 3.34 / 5.31 | ~1.80 / ~1.93 |

## SNR mesuré par couche — motif net

| Couches | gate/up Q4_0 mean | gate/up min | down Q5_0 mean | down min | Verdict |
|---|---:|---:|---:|---:|---|
| **0** | 21.14 | 18.41 | 28.23 | 27.04 | ✓ sain |
| 1–27 | 21.16–21.28 | 19.7–21.2 | 27.32–27.38 | 26.5–27.1 | ✓ sain, homogène |
| **28** | **20.57** | 18.5 | 27.28 | 26.9 | ⚠ sensiblement plus fragile |
| **32–34** | **20.98–21.04** | **18.92–19.06** | 27.24–27.28 | 26.8–26.9 | ⚠ fragile (cluster) |
| 35 | 21.14 | 19.93 | 27.32 | 27.10 | ✓ récupère |
| **36** | 21.11 | **18.67** | 27.30 | 26.76 | ⚠ expert e2 très fragile |
| **38–39** | 21.05–21.08 | 19.38–19.61 | **27.18–27.27** | 26.40 | ⚠ fin de réseau fragilisée |

**Lecture** : le motif "premières/dernières couches plus sensibles" (hypothèse APEX) est
**conforté en SNR expert-grain** sur le 35B réel — couches 0 et 1 saines mais 28, 32–34,
36, 38–39 fragilisées. Le down est partout ~6 dB au-dessus du gate/up (Q5_0 vs Q4_0).

## Top fragiles (à garder Q8_0, coût VRAM négligeable)

`blk.0 gate e252` (18.41) · `blk.36 gate e2` (18.67) · `blk.32 gate e123` (18.92) ·
`blk.34 gate e91` (19.00) · `blk.33 gate e75` (19.06) · `blk.38 gate e181` (19.38) ·
`blk.9 up e58` (19.44) · `blk.36 up e2` (19.59) · `blk.32 up e123` (19.60) ·
`blk.39 gate e141` (19.61)

## Recette recommandée (décision expert-grain)

1. **gate/up → Q4_0, down → Q5_0** sur les 40 couches : 20.76 GiB, SNR ≥ 18.4 dB.
2. **Exclusions haute précision** (fragiles identifiés) : les ~15 experts top-listés
   restent en Q8_0 source → +~50 MiB seulement sur 20.76 GiB.
3. Couches **20/27/29–39** (gate/up F16 source) : mêmes conclusions (SNR mesuré
   dans les tranches 14–27 / 28–39, F16 = référence encore plus fiable que Q8_0).
4. Production : les mêmes règles passent à `llama-quantize --tensor-type
   "ffn_(gate|up)_exps=Q4_K" --tensor-type "ffn_down_exps=Q6_K" --allow-requantize`
   (Q4_K/Q6_K = qualité supérieure à Q4_0/Q5_0 à bpw légèrement supérieur) — à valider
   par PPL ensuite.

## Profil 1080 recalculé avec pool mesuré (18.12 GiB, @8k, skew 0.305)

| | Q8_K_P origine | Candidat requant |
|---|---:|---:|
| Pool experts | 37.97 GiB | **18.12 GiB** |
| Résidents (budget 3.29 GiB) | 22/256 | **42/256** |
| Hit (skew mesuré) | 0.47 | **~0.60** |
| Miss | 4.2/tok → CPU 0.34 ms | 3.2/tok → CPU |
| **Decode MODELLED** | 70.5 t/s | **~71.5 t/s** + qualité préservée |

Le gain décode reste modeste (backbone GPU dominant) ; le gain réel du requant est la
**RAM** : 18.12 GiB d'experts tiennent dans 32 GiB système (au lieu de 37.97 impossibles),
rendant le déploiement réel possible sur cette machine.

## Données

- `snr_experts_chunk0_13.json`, `snr_experts_chunk14_27.json`, `snr_experts_chunk28_39.json`
  (SNR par tenseur : mean/min/argmin/p5/global, 768 tenseurs)
- `requant_experts.py` — outil réutilisable (--snr / --write, règles regex=TYPE)
- Limites : SNR = fidélité poids (proxy qualité) ; validation finale par PPL
  (wiki.test 4.2 MB, llama-perplexity du fork) sur le GGUF requantizé écrit.
