# Convergence Characterization — Design Matrix

## Overview

Systematic characterization of VB-PCA convergence across data regimes, prior
settings, stopping criteria, and algorithmic variants. Builds on evidence from
existing diagnostic scripts (`_convergence_analysis.py`,
`_convergence_diagnostic.py`, `_rms_oscillation_trace.py`,
`_rms_original_vs_internal.py`).

The closed-form ELBO gives us a theoretically grounded convergence guarantee
that the original implementation never exploited. If convergence is fast
(6-50 iters), full-covariance VB-PCA may be tractable at scales previously
assumed to require the diagonal approximation.

---

## Established findings (from existing scripts)

| Finding | Source | Status |
|---------|--------|--------|
| Angle converges by iter 5-7 for dense data | `_convergence_analysis.py` | Confirmed |
| RMS 2-cycle oscillation on dense data (constant amplitude) | `_rms_oscillation_trace.py` | Confirmed |
| Oscillation enters at the bias update (uses previous iter's err_mx) | `_rms_oscillation_trace.py` | Confirmed |
| Rotation (rotate2pca) does NOT cause oscillation | `_rms_oscillation_trace.py` | Confirmed |
| Pre-centering + bias=0 eliminates oscillation entirely | `_rms_oscillation_trace.py` | Confirmed |
| Pre-centering + bias=1 also eliminates oscillation | `_rms_oscillation_trace.py` | Confirmed |
| 2-pt moving average on RMS eliminates oscillation | `_convergence_analysis.py` | Confirmed |
| `niter_broadprior=100` wastes 92+ iters on dense data | `_convergence_diagnostic.py` | Confirmed |
| Missing data: angle stalls at ~1e-4, RMS converges by iter 25-50 | `_convergence_analysis.py` | Confirmed |
| Noise variance also oscillates with period 2 | `_rms_original_vs_internal.py` | Confirmed |

---

## Design factors

### Factor 1: Data geometry

Extends existing `scenarios` list from `_convergence_analysis.py`.

| ID | n | p | k | Label | Notes |
|----|---|---|---|-------|-------|
| D1 | 20 | 50 | 3 | Small dense | Existing |
| D2 | 50 | 100 | 5 | Medium dense | Existing |
| D3 | 100 | 200 | 10 | Large dense | Existing |
| D4 | 200 | 200 | 5 | Square dense | Existing |
| D5 | 500 | 200 | 10 | Tall dense | New: scalability |
| D6 | 1000 | 100 | 10 | Very tall dense | New: scalability frontier |
| D7 | 200 | 500 | 5 | Wide dense | New: p >> n regime |
| D8 | 5000 | 500 | 20 | HPC scale | New: tractability test |

### Factor 2: Missingness pattern

Extends existing MCAR-only grid with structured patterns from `stability_analysis.py`.

| ID | Pattern | Fraction | Notes |
|----|---------|----------|-------|
| M0 | Complete | 0% | Baseline (oscillation present) |
| M1 | MCAR | 10% | Existing |
| M2 | MCAR | 30% | Existing |
| M3 | MCAR | 50% | Existing |
| M4 | MCAR | 75% | Ilin & Raiko worst case (Fig 10) |
| M5 | MNAR censored | 15% | From stability_analysis.py |
| M6 | Block missing | 15% | From stability_analysis.py |

### Factor 3: Prior hyperparameters (hp_va, hp_vb, hp_v)

Currently hardcoded at 0.001. Requires #50 to sweep.

| ID | hp_va | hp_vb | hp_v | Label |
|----|-------|-------|------|-------|
| P0 | 1e-3 | 1e-3 | 1e-3 | Legacy default |
| P1 | 1e-6 | 1e-6 | 1e-6 | Near-uninformative |
| P2 | 1e-1 | 1e-1 | 1e-1 | Moderate |
| P3 | 1.0 | 1.0 | 1.0 | Strong |
| P4 | 10.0 | 10.0 | 10.0 | Very strong |
| P5 | 1e-6 | 1e-6 | 1.0 | Uninformative A,S / strong noise |
| P6 | 1.0 | 1.0 | 1e-6 | Strong A,S / uninformative noise |

### Factor 4: Convergence criterion

What to check, thresholds, and computational point.

| ID | Criterion | Threshold | Notes |
|----|-----------|-----------|-------|
| C0 | Raw RMS rel_delta | 1e-4 | Current default (broken on dense) |
| C1 | Smoothed RMS (2-pt MA) rel_delta | 1e-4 | Proven to work all regimes |
| C2 | Subspace angle | 1e-8 | Optimal for dense, stalls on missing |
| C3 | ELBO absolute decrease | 1e-6 | Theoretically grounded, currently OFF |
| C4 | ELBO relative decrease | 1e-6 | Scale-invariant version |
| C5 | ELBO curvature (2nd difference) | 1e-4 | When improvement rate itself stabilizes |
| C6 | Composite: angle AND smoothed RMS | 1e-8/1e-4 | Belt and suspenders |
| C7 | Composite: ELBO + angle + smoothed RMS | varies | Regime-adaptive |

### Factor 5: RMS computation timing

Existing scripts show oscillation enters at bias update. Test where in the
coordinate cycle RMS is evaluated.

| ID | Computation point | Notes |
|----|-------------------|-------|
| T0 | After scores+loadings (current) | Asymmetric: S new, A new, mu old |
| T1 | After full cycle (S, A, mu, V) | All parameters at iteration-end |
| T2 | Average of pre- and post-cycle | Period-2 average |
| T3 | After loadings only (before noise) | Skip noise influence |

### Factor 6: Algorithmic variant

| ID | Variant | Notes |
|----|---------|-------|
| A0 | VB (use_prior=True, use_postvar=True) | Full VB-PCA |
| A1 | MAP (use_prior=True, use_postvar=False) | No posterior variance |
| A2 | PPCA (use_prior=False, use_postvar=False) | ML estimation |

### Factor 7: Pre-centering

| ID | Pre-centering | Bias | Notes |
|----|---------------|------|-------|
| R0 | No | bias=1 | Current default (oscillation present) |
| R1 | Row-mean | bias=0 | Eliminates oscillation |
| R2 | Row-mean | bias=1 | Also eliminates oscillation |

### Factor 8: niter_broadprior

| ID | Value | Notes |
|----|-------|-------|
| B0 | 0 | No suppression |
| B1 | 5 | Minimal |
| B2 | 10 | Proposed new default |
| B3 | 50 | Intermediate |
| B4 | 100 | Current default |

### Factor 9: Signal-to-noise ratio (SNR)

Proof 3 predicts ρ depends on SNR, but the current grid controls it only
implicitly via geometry. Making noise scale an explicit factor enables
direct verification of the rate expression.

| ID | σ_noise | Approx SNR label | Notes |
|----|---------|-------------------|-------|
| N0 | 0.1 | High SNR | Strong signal |
| N1 | 0.5 | Moderate SNR | Current default in stability_analysis.py |
| N2 | 1.0 | Low SNR | Comparable to signal |
| N3 | 2.0 | Very low SNR | Noise-dominated |

### Factor 10: Condition number of true loadings A_true

Proof 3's rate depends on κ(A^T A). Default `randn` loadings give κ ≈ O(1).
Collinear components test the ill-conditioned regime.

| ID | κ(A_true) | Construction | Notes |
|----|-----------|--------------|-------|
| K0 | ~1 | Orthogonal A (QR of randn) | Best-case |
| K1 | ~10 | Mild collinearity (geometric singular value decay) | Typical |
| K2 | ~100 | Moderate ill-conditioning | Stresses ARD |
| K3 | ~1000 | Near-rank-deficient | Worst-case for convergence |

### Factor 11: Overspecification ratio (k_fit / k_true)

When k_fit > k_true, ARD must prune surplus components. This is a known
slow regime — the extra components decay slowly under weak priors.

| ID | Ratio | Notes |
|----|-------|-------|
| O0 | 1× | Correctly specified |
| O1 | 2× | Moderate overspecification |
| O2 | 3× | Strong overspecification |

### Factor 12: Stopping threshold ε

The same criterion at different thresholds fires at different iterations —
this is a wall-time knob distinct from which criterion is used.

| ID | ε | Notes |
|----|---|-------|
| E0 | 1e-4 | Coarse (fast) |
| E1 | 1e-6 | Moderate |
| E2 | 1e-8 | Fine (current minangle default) |
| E3 | 1e-10 | Very fine (overkill?) |

### Factor 13: Patience window τ

Consecutive sub-threshold iterations before declaring convergence.
Trades robustness for speed.

| ID | τ | Notes |
|----|---|-------|
| W0 | 1 | Aggressive (stop on first trigger) |
| W1 | 3 | Moderate |
| W2 | 5 | Conservative |
| W3 | 10 | Very conservative |

### Factor 14: Initial broad prior value v_a^(0)

Currently hardcoded at 1000. Determines how broad the broad-prior phase is;
interacts with niter_broadprior.

| ID | v_a^(0) | Notes |
|----|---------|-------|
| I0 | 100 | Less broad — ARD activates sooner |
| I1 | 1000 | Current default |
| I2 | 10000 | Very broad — slower ARD activation |

### Factor 15: Probe fraction

Fraction of observed entries held out for holdout-based metrics.
Currently hardcoded at 0.10 in `_PROBE_FRACTION`.

| ID | Fraction | Notes |
|----|----------|-------|
| F0 | 0.05 | More training data, noisier holdout |
| F1 | 0.10 | Current default |
| F2 | 0.20 | Better holdout estimate, less training |

---

## Response variables (what we measure per run)

### Per-iteration traces
- `rms[t]` — raw RMS at each iteration
- `rms_smooth[t]` — 2-point moving average RMS
- `angle[t]` — subspace angle (requires #52)
- `cost[t]` — ELBO/cost (requires `cfstop` enabled)
- `noise_var[t]` — noise variance trajectory
- `delta_cost[t]` — ELBO decrease per iteration
- `delta2_cost[t]` — curvature: second difference of ELBO

### Scalar summaries — posterior quality ("forecast skill" axis)

Analogous to relWIS in the TICCS poster: these measure how well the
algorithm recovered the *known* generating model, not just how well it
fit the observed data. Scored against $(A_{\text{true}}, \sigma^2_{\text{true}}, k_{\text{true}})$,
not training-set reconstruction.

**Rank selection (ARD correctness):**
- `rank_recovered` — ARD-pruned rank vs true rank
- `rank_exact` — binary: did ARD recover the true rank?
- `rank_over` / `rank_under` — directional error
- `effective_rank` — continuous: $\|A\|_F^2 / \|A\|_2^2$ (stable rank)

**Posterior calibration:**
- `coverage_90` — posterior interval coverage at 90% level on held-out entries
- `mean_interval_width` — sharpness: mean $2 z_{\alpha/2} \sqrt{v_{ij}}$ on holdout
- `CRPS_holdout` — continuous ranked probability score (via `scoringrules.crps_gaussian`)
- `log_pred_density` — mean log predictive density on holdout entries
- `PIT_ks_stat` — Kolmogorov-Smirnov statistic of PIT values vs Uniform (calibration)

**Point-estimate accuracy:**
- `rms_final` — final reconstruction error (training set)
- `RMSE_holdout` — reconstruction error on held-out entries
- `angle_final` — subspace angle between $\hat{A}$ and $A_{\text{true}}$
- `loadings_frobenius` — Frobenius error of loadings after Procrustes alignment
- `noise_var_error` — $|\hat{\sigma}^2 - \sigma^2_{\text{true}}|$

### Scalar summaries — wall time ("surveillance cost" axis)

- `n_iter` — iterations to convergence (by each criterion)
- `converged` — did the criterion fire before maxiters?
- `stop_reason` — which criterion triggered first
- `wall_time` — seconds to convergence
- `n_warmup_wasted` — iterations spent in broad-prior phase after angle has converged

### Scalar summaries — convergence diagnostics

- `oscillation_amplitude` — max(rms[even]) - min(rms[odd]) in last 20 iters
- `convergence_rate` — empirical ρ̂: slope of log(criterion) vs iteration (linear fit)
- `elbo_monotonicity_violations` — count of iterations where ELBO decreased (should be 0)

---

## Experiment tiers

### Tier 1: Oscillation root cause (quick, local)

**Goal**: Determine if RMS computation timing eliminates oscillation.

**Factors**: T0-T3 × R0-R2 × {D2} × {M0, M2}
**Grid size**: 4 × 3 × 1 × 2 = 24 runs
**Extends**: `_rms_oscillation_trace.py::trace_rms_within_iteration()`
**Blocked by**: Nothing (can instrument with monkey-patching)

### Tier 2: ELBO as convergence metric (quick, local)

**Goal**: Enable cost computation, compare ELBO convergence to angle/RMS.

**Factors**: C0-C5 × {D1-D4} × {M0-M3} × {P0}
**Grid size**: 6 × 4 × 4 = 96 runs
**Extends**: `_convergence_analysis.py` (add cfstop=[10, 1e-6, 1e-3])
**Blocked by**: Nothing (cfstop already exists, just disabled)

### Tier 3: Prior sensitivity (requires #50)

**Goal**: Map how hp_va/hp_vb/hp_v affect convergence rate and oscillation.

**Factors**: P0-P6 × {D2, D3} × {M0, M2, M4} × {C3, C2}
**Grid size**: 7 × 2 × 3 × 2 = 84 runs
**Extends**: New
**Blocked by**: #50 (expose priors)

### Tier 4: niter_broadprior profiling (requires #50, #52)

**Goal**: Find minimal safe niter_broadprior per regime.

**Factors**: B0-B4 × {D1-D4} × {M0-M4} × {P0, P1}
**Grid size**: 5 × 4 × 5 × 2 = 200 runs
**Extends**: `_convergence_analysis.py` (currently only B0)
**Blocked by**: #50, #52

### Tier 5: Full convergence characterization (HPC, requires all Phase 1)

**Goal**: The paper's main empirical result.

**Factors**: D1-D8 × M0-M6 × P0-P4 × C3,C5,C7 × A0-A1 × R0,R1 × N0-N3 × K0-K2 × O0-O1
**Grid size**: 8 × 7 × 5 × 3 × 2 × 2 × 4 × 3 × 2 = 80,640 configs
**Reps**: 3 seeds each = 241,920 total (HPC; subsample adaptively if needed)
**Note**: SNR (N0-N3), κ(A) (K0-K2), and overspec (O0-O1) added because Proof 3
predicts ρ depends on these. Without them the regression model for ρ cannot
be verified against theory. Can subsample: Latin hypercube over continuous
factors × full factorial over discrete.
**Blocked by**: #50, #52, Tier 1-2 results (to pick best criterion)

### Tier 6: Scalability frontier (HPC)

**Goal**: At what (n, p) does full-covariance VB become slower than diagonal?

**Factors**: D5-D8 × M0,M2,M4 × {best prior from Tier 3} × {best criterion from Tier 5}
**Grid size**: 4 × 3 = 12 configs, time-profiled
**Blocked by**: Tier 5 results, #16 (diagonal covariance, if we want to compare)

### Tier 7: Bayesian stacking & Pareto front (requires Tier 5)

**Goal**: Ensemble across stopping criteria, weight by downstream performance;
map the Pareto front of composite quality vs. wall time.

**Metrics for stacking weights** ("forecast skill" — all scored against known truth):
- Coverage at 90% nominal level
- Mean interval width (sharpness)
- CRPS on held-out entries (via `scoringrules.crps_gaussian`)
- Posterior predictive log-likelihood (via `scipy.stats.norm.logpdf`)
- PIT uniformity (KS statistic via `scipy.stats.kstest`)
- ARD rank fidelity (|k_recovered − k_true|)
- Subspace angle to ground truth loadings
- Loadings Frobenius error (Procrustes-aligned)

**Stacking method**: Yao et al. (2018) — convex optimization of LOO weights
via `arviz.weight_models()`. Each "model" is the same VB-PCA fit stopped by
a different criterion; stacking picks the optimal posterior-average ensemble.

**Pareto front** (exact analogy to poster.tex surveillance design):
- x-axis: wall time (iterations × per-iter cost) — "surveillance cost"
- y-axis: composite quality score — "forecast skill"
- Composite quality = scalarized (coverage gap, rank error, CRPS, ...) with
  weights traced by λ to sweep the front
- Post-hoc extraction from Tier 5 traces via `pymoo` (no additional fits)
- Composite metric weights learned via `optuna` Bayesian optimization
- 3D Pareto surface in supplementary: wall time × rank error × coverage gap
- Regime-conditional fronts: complete vs. sparse missingness

**Blocked by**: Tier 5

**Python packages consumed at this tier:**
- `scoringrules` — CRPS, log score (proper scoring primitives)
- `arviz` — Yao (2018) stacking via `az.weight_models()`
- `pymoo` — Pareto front extraction from existing grid
- `optuna` — composite metric weight learning (lightweight BO)

**Conditional dependency — pp-eigentest:**
pp-eigentest builds its own composite grid for rank test fidelity,
conditioned on VBPCApy's tuned defaults from this sweep. It is a
separate optimisation in the pp-eigentest package, not a consumer of
these traces.

---

## Execution plan

```
Tier 1 (now)  ─── Tier 2 (now)  ──┐
                                    ├─── Tier 3 (after #50) ──┐
                                    │                          │
                                    │                          ├── Tier 4 ── Tier 5 (HPC) ── Tier 6
                                    │                          │                              Tier 7
                                    └──────────────────────────┘
```

Tiers 1-2 can start immediately with no code changes.
Tier 3 requires #50 (expose priors).
Tier 4 requires #50 + #52 (expose priors + store angle).
Tiers 5-7 require all Phase 1 issues + results from earlier tiers.

---

## Connection to lawful representation ideas

The VB-PCA ELBO has structure that maps onto a constraint hierarchy:

| Tier | Property | VB-PCA instantiation |
|------|----------|---------------------|
| **Embedded (E)** | Holds by construction | ELBO monotonic improvement under coordinate ascent; posterior variance positivity |
| **Penalized (P)** | Enforced by stopping rule | Convergence tolerances (angle < ε, ELBO Δ < ε) |
| **Diagnostic (D)** | Post-hoc measurement | RMS, coverage, ARD sparsity, oscillation index |

Structured observables on the iteration trajectory:
- $\mathcal{O}_{\text{ELBO}}(\theta_t) = \mathcal{L}(\theta_t) - \mathcal{L}(\theta_{t-1})$ — improvement per step
- $\mathcal{O}_{\text{angle}}(\theta_t) = \angle(W_t, W_{t-1})$ — subspace stability
- $\mathcal{O}_{\text{curv}}(\theta_t) = |\Delta\mathcal{L}_t - \Delta\mathcal{L}_{t-1}|$ — improvement rate change

Curvature-based stopping is second-order convergence detection: when
$\mathcal{O}_{\text{curv}} < \epsilon$, the algorithm is in the asymptotic
linear-convergence regime and further iterations yield diminishing returns
at a predictable rate.

The convergence rate itself is characterizable from the ELBO Hessian
(condition number of the variational objective), giving a theoretical
$O(1/t)$ bound for coordinate ascent on the convex bound.

---

## Analogy to poster.tex: two-axis Pareto framework

The experiment design follows the same closed-loop pattern as the TICCS
surveillance optimization poster:

| TICCS (poster) | VBPCApy (convergence paper) |
|---|---|
| Surveillance plan (streams, frequency, spatial) | Algorithmic config (priors, stopping rule, warmup, centering) |
| Known latent hazard λ(t) | Known generative truth (A, S, σ², k) |
| Observation model (ascertainment, survey noise) | Missingness pattern (MCAR/MNAR/block + fraction) |
| Forecast skill (relWIS, peak timing) | Composite posterior quality (coverage, ARD, CRPS, ...) |
| Surveillance cost ($) | Wall time (iters × per-iter cost) |
| "Score against known λ(t), not noisy counts" | "Score against known (A, σ², k), not training reconstruction" |
| Closed-loop: propose → simulate → train → score | Closed-loop: propose config → fit synthetic → evaluate → rank |

The Pareto front maps quality vs. cost in both settings. Practitioners
pick their operating point on the front; the "recommended default" sits
at the knee.

---

## Python ecosystem leverage (no reinvention)

| Capability | Package | Used in |
|---|---|---|
| Proper scoring (CRPS, log score) | `scoringrules` (v0.9, Jan 2026) | Tier 7 stacking + all endpoint metrics |
| Bayesian stacking (Yao 2018) | `arviz.weight_models()` (v1.0) | Tier 7 |
| Pareto extraction from grid | `pymoo` (v0.6) | Tier 7 post-hoc |
| Composite weight learning | `optuna` (v4.8) | Tier 7 BO over (w, ε, τ) |
| PIT / calibration | `scipy.stats.kstest` | All endpoint metrics |
| Rank significance | `pp_eigentest.eigentest()` | Separate sweep in pp-eigentest, conditioned on tuned VBPCApy defaults |
| Procrustes alignment | `scipy.linalg.orthogonal_procrustes` | Loadings error metric |
| MCMC diagnostics (if needed) | `arviz` (R-hat, ESS) | — |
| Calibration framework | `netcal` (v1.3) | Supplementary figures |

VI/EM convergence diagnostics are a genuine ecosystem gap — no Python
package exists for this. The composite Bayesian convergence metric and
trace-replay infrastructure are novel contributions.

---

## Orchestration assessment

Both VBPCApy and pp-eigentest need the same infrastructure:
- Controlled data generation (SNR, κ, missingness)
- Per-run endpoint evaluation (scoringrules, scipy, Procrustes)
- Trace recording + post-hoc replay
- Pareto extraction (pymoo) + weight learning (optuna)
- Bayesian stacking (arviz)

`stability_analysis.py` already has a 500-line sweep harness. The convergence
sweep will be larger. pp-eigentest will need a parallel harness conditioned
on VBPCApy's tuned defaults. Six external packages plus custom trace replay
is enough moving parts that a thin shared Python evaluation package
(`eval-harness` or similar) is justified — not a heavy framework, but shared:

1. **Data generation** (~50 lines): low-rank + noise + missingness + controlled κ/SNR
2. **Endpoint battery** (~150 lines): CRPS, log-pred, PIT, coverage, Procrustes, etc.
3. **Trace format** (~100 lines): save/load npz, post-hoc criterion replay
4. **Pareto + stacking glue** (~100 lines): pymoo extraction, optuna objective, arviz stacking

~400 lines total. Lives under the Prior Lab org alongside VBPCApy and
pp-eigentest. Both packages depend on it. This is the Python-side analog
of ModelCriticism.jl but much thinner — no model worlds, just the
evaluation/scoring/Pareto layer.
