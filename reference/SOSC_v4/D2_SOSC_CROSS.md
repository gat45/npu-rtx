# SOSC v4 × D2 — Cross-Reference Intégrale

## Les 3 projets en regard

| Dimension | D2-Bayesian-Geometry | D2-Quant-Planner | SOSC v4 |
|-----------|---------------------|-------------------|---------|
| Domaine | SNN + LLM (théorie) | LLM (pratique) | ESN + LIF + LLM |
| Métrique clé | α_w (heavy-tail) | α_w → precision | r_eff + α_w + SC |
| Cible | Criticité biologique | Quantification GPU | Performance ESN |
| Validation | SNN simulé | 10 LLMs, XDNA2 | 22 tests ESN/LIF |
| R² α_w → cible | Non mesuré | Non mesuré | **-0.08** ❌ |
| R² r_eff → cible | Non mesuré | Non mesuré | **0.71** ✅ |

---

## 1. Divergence centrale : α_w vs r_eff

### D2-Bayesian-Geometry
```
α_w = -2 · slope(log σ_i, log i)
SNN: α_w = 2.17 ✅  (validé biologiquement)
LLM: α_w = 3.08     (Coppola 2026)
Postulat: α_w est la métrique spectrale fondamentale
```

### SOSC v4
```
r_eff = (Σσ²)² / Σσ⁴
α_w seul → quant_error: R² = -0.08 ❌
r_eff + α_w + SC → quant_error: R² = 0.71 ✅

r_eff = participation ratio (Rajan & Abbott 2006)
Mesure combien de directions portent l'information
```

### Ce que ça signifie
- **α_w mesure la queue de distribution** (heavy-tail exponent)
- **r_eff mesure la dimension effective** (participation ratio)
- Les deux sont utiles mais mesurent des choses différentes
- α_w seul est insuffisant pour la quantification (R²=-0.08)
- r_eff est plus robuste pour prédire la fragilité

---

## 2. Convergences

| Concept | D2-Bayesian | D2-Quant | SOSC v4 | Statut |
|---------|-------------|----------|---------|--------|
| ρ ≈ 1.04 criticité | ✅ Validé SNN | Utilisé comme cible | ❌ Falsifié ESN | Dépend du domaine |
| α_w attention < MLP | ✅ 10/10 LLMs | Implicite | ✅ P3 10 LLMs | **Validé** |
| Non-normalité H | ✅ Henrici/Kreiss | — | ✅ T4 RhoNorm | **Validé** |
| Quantification adaptative | — | α_w → precision | r_eff → risk | Divergent |
| K → Energy | — | Tile_util → TPS | r=0.86 | **Validé** |
| Dale law | Simulé | — | Testé P6 | **Validé** |

---

## 3. Ce que SOSC apporte à D2

### 3.1 Meilleure prédiction de quantification
```python
# D2 actuel (D2-Bayesian, D2-Quant)
risk = alpha_w * (1 + density)           # R² non mesuré

# SOSC v4
risk = 0.6 * (1 - r_eff/min(N,M)) + 0.4 * min(alpha_w/100, 1)  # R²=0.71
```

### 3.2 TIN pour politique adaptative
```python
# Actuel: même politique pour toutes les tâches
# SOSC: TIN adapte la politique
if TIN.regime == "Amplification":
    proteger_OSI = True   # moins quantifier
elif TIN.regime == "Memory":
    compresser_plus = True
```

### 3.3 K → Energy pour coût matériel
```python
# D2 actuel: Tile_util seul
# SOSC: K predit consommation
energie = 0.86 * K  # r=0.86 validé
```

---

## 4. Ce que D2 apporte à SOSC

### 4.1 α_w corrigé (Hill/CSN)
```python
# SOSC actuel: SVD complet → α_w approximatif
# D2-Bayesian: Hill/CSN sur queue des valeurs propres
def alpha_w_hill(W, x_min_percentile=80):
    s = np.linalg.svd(W, compute_uv=False)
    s2 = s**2
    x_min = np.percentile(s2, x_min_percentile)
    tail = s2[s2 > x_min]
    return 1 + len(tail) / np.sum(np.log(tail / x_min))
```

### 4.2 Henrici / Kreiss (non-normalité)
```python
# SOSC: H simple = ||WWT - WTW|| / ||W||²
# D2-Bayesian: Henrici measure + Kreiss constant + pseudospectre
def henrici(W):
    n = W.shape[0]
    return np.linalg.norm(W @ W.T - W.T @ W, 'fro') / np.linalg.norm(W, 'fro')
```

### 4.3 Branching parameter (LIF)
```python
# D2-Bayesian: m ≈ 0.98 à σ_B=1.04
# SOSC: OSI_spike → m, r=-0.66
# Complémentaire: LIF + spectral
```

---

## 5. Contradictions à résoudre

| Point | D2-Bayesian | SOSC v4 | Résolution |
|-------|-------------|---------|------------|
| α_w SNN | 2.17 ± 0.09 | Non mesuré | À mesurer sur nos ESN |
| α_w GPT-2 | 3.08 (Coppola) | α_HT≈1.13 (SOSC) | Définition différente (α_w=2β vs α_HT) |
| ρ=1.04 universel | Validé SNN | Falsifié ESN | ρ_c = f(domaine) |
| r_eff > α_w | Non testé | R²=0.71 vs -0.08 | À intégrer dans D2 |

### Correction de la divergence α_w
```
D2-Bayesian: α_w = -2 · slope(log σ_i, log i)  → GPT-2: 3.08
SOSC v4:     α_HT = -2 · slope(log σ_i, log i)  → Qwen: 0.98

Cause probable: D2 utilise tout le spectre, SOSC utilise un fit linéaire
complet. Mais nos mesures sur Qwen (p3_cross_model.csv) donnent α_HT≈0.98,
pas 3.08. Différence de méthode ou de modèle ?
```

---

## 6. Intégration recommandée

```
D2-Bayesian-Geometry (théorie)
  ├── α_w Hill/CSN
  ├── Henrici/Kreiss
  ├── Branching LIF
  │
  + SOSC v4 (correction empirique)
  │   ├── r_eff > α_w pour quantification
  │   ├── TIN pour politique adaptative
  │   ├── K → Energy pour coût
  │   └── ρ_c variable (pas 1.04 fixe)
  │
  v
D2-Quant-Planner (pratique)
  ├── risk = f(r_eff, α_w, SC)
  ├── VRAM-constrained optimizer
  ├── Tile_util model (XDNA2)
  └── ILP solver
```

---

## 7. Fichiers

```
SOSC_v4/
  D2_VS_LITTERATURE.md          — Positionnement littérature
  d2_sosc_integration.py        — Risk model (r_eff+α_w+SC)
  d2_sosc_v2.py                 — VRAM optimizer
  P3_LLM_spectral_analysis.ipynb — Preuve R²=0.71

D2-Bayesian-Geometry/
  simulations/measure_alpha_w.py — α_w Hill/CSN
  simulations/build_bio_W.py     — SNN biologique
  docs/D2_paper.md               — Théorie

D2-Quant-Planner/
  d2_rtx_gguf_profiler.py       — Scanner GGUF
  d2_compiler_v2.py              — Compilateur ILP
  alpha_spectral_scanner.py      — Scanner α_w
```
