# SOSC — Cross-Reference Final + Gaps
## What we have, what we miss, what to test next

---

## 1. VALIDATED CORE

```
S(t,θ) = [β, ρ, K, η, Ω, ITI]

β = -dlogP/dlogf              spectral organization
ρ = σ(W) / 1.04               dynamical criticality
K = d / √N                    connectivity level
η = α - 0.5                   scaling regime (d ∝ N^α)
Ω = C_clustering / L_path      topology (small-world ratio)
ITI = I(X_t ; X_{t-τ})        temporal information structure
```

### TIN (Task Information Nature)

```
TIN = softmax(a_OSI, a_MC, a_TTI)    with ΣR_i = 1 guaranteed

Regimes:
  R_OSI > 0.5  → Amplification  (Lorenz)
  R_MC  > 0.5  → Memory         (NARMA10, noisy sine)
  R_TTI > 0.5  → Temporal       (Mackey τ≥50)
```

### Performance

```
ΔLoss = F(S_system, S_task, θ_architecture)
```

---

## 2. WHAT WE CONFIRMED

| Claim | Evidence | Status |
|-------|----------|--------|
| Static metrics (ρ, H, α_w) → ΔMSE ≈ 0 | R²≈0 on all ESN tasks | ✅ CONFIRMED |
| OSI_dynamic → ΔMSE | R²=0.97 on Lorenz | ✅ CONFIRMED |
| MC → ΔMSE (memory tasks) | R²=0.97 on NARMA10 | ✅ CONFIRMED |
| TTI → ΔMSE (temporal tasks) | R²>0.96 on Mackey τ≥50 | ✅ CONFIRMED |
| TIN separates regimes | 6 tasks, 3 regimes, perfect | ✅ CONFIRMED |
| Axis independence | max cross-corr = 0.18 | ✅ CONFIRMED |
| r_eff → quant_error | R²=0.71 across 10 LLMs | ✅ CONFIRMED |
| OSI_spike → m avalanche | r=-0.66 on LIF networks | ✅ CONFIRMED |
| Dynamics > Structure for LIF | ρ, α_w → m: R²≈0 | ✅ CONFIRMED |

---

## 3. WHAT WE FALSIFIED (removed)

| Claim | Replaced by | Evidence |
|-------|-------------|----------|
| ρ = 1.04 universal | ρ_opt = f(task, architecture) | Lorenz: ρ=1.20 > ρ=1.04 |
| K* = 2.17 universal | K = d/√N, class-dependent | K_bio=0.14, K_digital=1.69, K_neuro=6.27 |
| C₀ = 0.0380 universal | Removed | Not reproduced on ESN |
| β_LLM = 0.31-0.47 | β_system = β measured | β≈0.98 on Qwen2-1.5B |
| σ_B LLM ≈ 1.04 | Removed | GPT-2: σ_B≈20.3 |
| skip_every | Removed | Diverges when ρ→1 |
| AC and TTI orthogonal | corr=0.93 → same dimension | AC is linear proxy for TTI |

---

## 4. WHAT WE MISS (gaps identified)

### Gap 1: ITI replaces AC

```
AC(τ) = E[x_t · x_{t-τ}]           linear, collapses at long τ
ITI(τ) = I(X_t ; X_{t-τ})          nonlinear, captures full structure
```

**Test:** Replace AC by ITI in TIN, re-run ablation on Mackey τ=50,100.
**Expected:** R²(OSI+MC+ITI) > R²(OSI+MC+AC) specifically for long τ.
**Not yet done** because our E2 TTI analysis was per-tau only, not per-trial.

### Gap 2: Transfer Entropy (information flow)

```
MC = Σ corr²(ŷ_k, u_k)             passive memory (can I recall?)
TE = I(X_source ; X_target | past)   active transfer (does info flow?)
```

**Hypothesis:** Networks with same MC can have different TE.
**Not measured** in any of our experiments.

### Gap 3: Topology Ω (small-world ratio)

```
Ω = C_clustering / L_path
```

**Problem:** Our ESNs are random sparse graphs (no clustering).
**Missing:** Compare ESNs with same K but different Ω (random vs small-world vs scale-free).
**Hypothesis:** Ω explains residual variance after OSI+MC+TTI.

### Gap 4: Dynamical reserve R_d

```
R_d = 1 - |ρ - 1|
R_d = 1 at critical point, 0 far from it.
```

**Problem:** ρ alone doesn't capture "how much room before explosion/extinction."
**Not tested** as a predictor.

### Gap 5: K-OSI coupling

```
OSI ≈ g(K, ρ) = K^λ · (1 - ρ)^μ
```

**Hypothesis:** OSI is not fundamental — it's a consequence of K and ρ.
**Test:** Regress OSI = f(K, ρ). If R² > 0.8, OSI is derived, not fundamental.
**Not tested** — would change the SOSC hierarchy.

### Gap 6: Cost model

```
C_SOSC = a·K + b·|ρ-ρ*| + c·(MC×ITI) + d·r_eff + e·Ω
```

**Hypothesis:** SOSC predicts computational cost (VRAM, tokens/s, energy).
**Test:** Measure real LLM cost vs SOSC vector.
**Not tested** — requires GPU profiling.

### Gap 7: Efficiency

```
Efficiency_SOSC = (MC + ITI) / (K × C_ρ)
```

**Biological interpretation:** A brain is efficient because it maximizes information per unit connectivity-cost.
**Not tested** — would require comparing biological, digital, neuromorphic systems on same tasks.

---

## 5. THE 5 CRITICAL REMAINING TESTS

| # | Test | What it proves | Effort |
|---|------|----------------|--------|
| T1 | OSI = f(K, ρ) regression | OSI derived or fundamental? | 1h (data exists) |
| T2 | ITI replaces AC in TIN | ITI > AC for long τ | 2h (new dataset) |
| T3 | Same K, different Ω | Topology matters beyond K | 4h (new generators) |
| T4 | Mediation: K → ρ → OSI | Causal chain or independent? | 2h (SEM on existing data) |
| T5 | K_c(η, ρ) transition | Phase transition in K space | 4h (sweep N,d large) |

---

## 6. THE 2 MISSING DIMENSIONS

```
Current SOSC:     S = [β, ρ, K, η, Ω, ITI]
                    structure  dynam.  topol.  scale  net.  info

Missing layer:    Task = [OSI, MC, TTI/TEI]    
                              amplif.  mem.  transfer
```

The bridge between system state and task performance is:

```
Performance = F(S_system, S_task)
```

We have both vectors but we never formally coupled them into a single predictive equation with learned weights.

---

## 7. FINAL FORM THAT SURVIVES

```
SOSC is not a theory of universal constants (ρ=1.04, K*=2.17).
SOSC is a regime classification framework:

  S(t) = [β, ρ, K, η, Ω, ITI]    ← system state
  TIN  = [R_OSI, R_MC, R_TTI]   ← task regime
  ΔLoss = F(S, TIN, θ)           ← performance

Core insight:
  Systems that perform well maintain a dynamical compromise
  between propagation (ρ, OSI), memory (MC), and information (ITI).
```

---

## 8. FILES UPDATED

```
SOSC_v4/
  BILAN_FINAL.md              — Complete bilan
  SOSC_v4_PUBLICATION_READY.md — Publication-ready document
  cross_reference_final.md    — S1-23 × v4 cross-ref
  SOSC_CORE.md                — Core surviving formulas
  SOSC_DIFF.md                — Version diff V1→V4
```

**Missing tests identified:** 7 gaps, 5 critical tests, 2 missing dimensions.
