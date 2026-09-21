# TEST P0-3 : αw spectral prédit-il la fragilité quant ? — VERDICT avec preuve

**Date** : 2026-09-21 · **Modèle** : 35B Genesis Q8_K_P (D:)
**Échantillon** : 1 536 experts (couches 0–1 × gate/up/down × 256), chaque expert portant
**dans le même JSON** (`spectral_experts_smoke.json`) son SNR exact
(quantize→dequantize vs source Q8_0/F16) ET ses features spectrales
(randomized SVD rank-32 : decay/αw, gap, entropie). Aucun appariement cross-run,
aucune hypothèse : tout est mesuré sur le même tenseur lu une seule fois.

## Résultats (Spearman ρ, SNR ↔ feature)

| couche | famille | ρ(SNR, decay) | ρ(SNR, gap) | ρ(SNR, ent) | std(decay) |
|---:|---|---:|---:|---:|---:|
| 0 | down | **+0.484** | +0.046 | −0.465 | 0.0169 |
| 0 | gate | **−0.641** | −0.290 | +0.620 | 0.0975 |
| 0 | up | **−0.579** | −0.027 | +0.523 | 0.0133 |
| 1 | down | −0.254 | −0.203 | +0.117 | 0.0127 |
| 1 | gate | −0.040 | +0.021 | −0.047 | 0.0258 |
| 1 | up | −0.347 | −0.115 | +0.281 | 0.0067 |

**Global (1 536 experts) : ρ(SNR, decay) = −0.181 · |ρ| moyen decay 0.39 ·
gap 0.12 · ent 0.34.**

## Lecture honnête

1. **Le signal existe mais n'est NI uniforme NI fort** : |ρ| = 0.12–0.64 selon
   (couche, famille). Le signe s'inverse (gate/up négatif = decay raide → SNR bas,
   conforme à l'intuition AlphaQ ; down positif en blk.0).
2. **La variance inter-experts de decay est minuscule** (std 0.007–0.098 pour des
   moyennes ~0.09) : les experts d'une même couche sont spectralement quasi
   identiques, alors que leurs SNR s'étalent de 18.4 à 30.5 dB. **Le spectre ne
   sépare presque rien à l'intérieur d'une couche** — c'est pourtant là que la
   décision expert-grain doit se prendre.
3. La fragilité détectée dans les tranches complètes (couches 28, 32–34, 36,
   38–39) est un effet **de couche**, pas de géométrie individuelle mesurable par
   αw au rank-32.

## Verdict P0-3 (sur cet échantillon)

- **αw seul ne peut PAS piloter l'allocation expert-grain** : ρ ≈ −0.18 global,
  signe instable, variance intra-couche quasi nulle. Cohérent avec la méta-littérature
  (MoPEQ → Hessian, ICLR'26 → Δrouter) : les signaux poids-seuls faibles.
- **Il reste un signal de couche** (le fit par couche×famille atteint 0.48–0.64 en
  valeur absolue) : usage légitime = moduler le type **par couche** (déjà le mode
  `--plan`), pas par expert.
- **Ce qui reste scientifiquement candidat pour l'expert-grain** : le SNR **mesuré**
  lui-même (coût ~1.5 h/40 couches, déjà fait) et les proxies activation
  (Δrouter/Hessian) — à tester en P0-2 avant de trancher.

## Addendum (même jour) — statistique propre + contrôle quantizer-égal

1. **scipy.spearmanr + bootstrap CI95** (mon premier calcul artisanal gérait mal
   les ties — 86.1 % des SNR dupliqués par l'arrondi 2 décimales) :
   global ρ(SNR,decay) = **−0.184 [−0.238, −0.127], p=4e-13** → significatif mais
   faible, verdict inchangé. Nouveau : **ρ(SNR,gap) = +0.040 (p=0.12, NS)** →
   gap orthogonal au SNR ; mais **ρ(decay,gap) = +0.541** → decay et gap sont
   redondants entre eux (une seule dimension spectrale, pas deux). Signe
   Spearman/Pearson inversé sur 1/down (−0.255 vs +0.216) → relation non linéaire.
2. **Contrôle quantizer-égal (couches 0–1)** : à Q4_0 uniforme, gate 21.14 /
   up 21.22 / down 22.14–21.31 dB (écart ≤ 1 dB) ; à Q5_0 uniforme, 27.2–28.2
   partout. **Le "down intrinsèquement plus robuste" était un artefact du
   quantizer (Q5_0 vs Q4_0), pas une propriété de la famille.** Les argmin
   (experts fragiles) sont identiques aux deux quantizers → l'index de fragilité
   est invariant au quantizer.

2 couches / 1 536 experts, source = requant Q8_0/F16 (pas FP8 natif), decay estimé
au rank-32 (Halko). Extension aux 40 couches = relancer `d2_expert_spectral_scan.py`
par tranches (~1.5 h) ; le verdict pourra être réévalué si la variance decay est
plus grande ailleurs — mais blk.0 (couche la plus "spéciale" du réseau) était le
meilleur candidat pour montrer un signal fort, et il n'en montre pas un utilisable.

## Rejouer

```bash
cd geniex_harness/npu-rtx/profiler_tiers
py d2_expert_spectral_scan.py --src "D:/Hermes...Q8_K_P.gguf" --layers 0-1 --tag smoke
py -c "…corrélation Spearman snr/decay/gap/ent sur spectral_experts_smoke.json…"
```
