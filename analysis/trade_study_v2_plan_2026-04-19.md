# Trade Study v2 — Multi-Stage Convergence Optimization Plan

> **Date**: April 20, 2026 (rev 2); **reframed** July 24, 2026 (rev 3).
> **Status**: Draft
> **Depends on**: #101 (configurable criterion ordering and per-criterion enable/disable) — merged as #102; predictive-variance fix (`feature/predictive-variance`).

## Update 2026-07-24 — Empirical findings that reframe the study

Three findings from the convergence-trace study (`analysis/trade_study/convergence_trace.py`)
and the coverage assessment substantially simplify the optimization problem.

### Finding 1 — Quality converges by iteration 3–18 in every regime

A re-fit sweep (fixed `k=true_rank`, **all** convergence criteria disabled so each fit runs
exactly `maxiters`, `niter_broadprior` swept) shows holdout RMSE and MAE reach within 5% of
their converged value by **iteration 3–18** (coverage by iteration 2) across all 8 regimes
spanning small-p / transition / large. `maxiters=200` (default) or `500` is **10–150× more
than needed**. → `maxiters` is purely a *safety ceiling*; the real question is which stopping
rule lands closest to the per-regime knee.

### Finding 2 — `niter_broadprior` is pure waste for quality

`bp ∈ {0, 25, 50, 100}` give **identical** final RMSE/MAE/coverage (to 3–4 decimals); bp only
*delays* reaching the knee (e.g. small-p 50×10: knee 12→18 as bp 0→25). → default
`niter_broadprior=0`. bp is retained as a factor **only** to test its effect on rank recovery
/ ARD pruning (its sole remaining justification), not quality.

### Finding 3 — Under-coverage was a reporting bug, now fixed (coverage leaves the objective set)

Coverage sat at 48–65% (nominal 95%) because the reported reconstruction variance
(`variance_`) captured only the epistemic uncertainty of the denoised mean `E[AS+μ]` and
**omitted the observation-noise variance σ²**. This was **not** a convergence, prior, or
broadprior effect: sweeping `hp_va/hp_vb/hp_v/va_init` over orders of magnitude moved coverage
by **0%**. The fix (`predictive_variance_ = variance_ + σ²`, non-breaking, on
`feature/predictive-variance`) restores **94–96%** coverage across all regimes.

### How the plan changes

- **Coverage is removed as an optimization objective.** With `predictive_variance_` it is ~95%
  regardless of config, so it becomes a pass/fail calibration *check*, not a Pareto axis. The
  scorer (`_world.py`) must score coverage from `predictive_variance_`, not `variance_`.
- **Objectives → multi-objective Pareto, no scalarization** (locked with user): `holdout_rmse`,
  `holdout_mae`, `rank_mae` vs `total_iters`. **Quality guards**: RMSE/MAE must not regress >5%
  vs library defaults; `rank_mae` is a tie-breaker. Nothing can be silently traded away (this
  fixes the v3 regression where optimization bought rank recovery by hurting RMSE/MAE).
- **`rank_mae` is the primary remaining quality lever.** Default recovers the true rank only
  ~33% of the time; optimized configs reached 75–81%. Rank recovery — not RMSE/coverage — is
  where tuning pays off.
- **Stage 2 recentred on the knee.** The central question is now *which*
  `criterion_order × active_criteria × threshold` stops closest to the per-regime knee
  **without firing before it**. Because `convergence_check()` evaluates every criterion each
  iteration, this is answered by **offline replay** of the captured learning curves
  (`convergence_replay.py`, to build) — no new fits. `maxiters` small (~50) as safety;
  `niter_broadprior=0`.
- **`total_iters` is now the dominant efficiency prize**: stopping at the knee (~15–25) vs the
  default (~200) is an **8–13× reduction** with no quality loss.

The multi-stage structure below is retained, but read Stage 1 objectives as
(RMSE, MAE, rank_mae) — coverage dropped — and Stage 2 as the knee-targeting replay above.

## Motivation

The v1 trade study (Phase 1–3 in `analysis/trade_study/`) treated all 13 factors jointly. This caused two problems:

1. **Broadprior confound**: `niter_broadprior` masks convergence for the first N iterations (convergence stop is suppressed inside `_log_and_check_convergence` during the broadprior warmup). This dilutes the measured sensitivity of prior hyperparameters (hp_va, hp_vb, hp_v, va_init). Sobol S1 indices for these factors are unreliable when bp is active. Setting bp wrongly can also waste compute — if bp is too long the model burns iterations under the broad prior, but if bp is too short (or off) with poorly-initialised priors, the ARD mechanism may not fire correctly.
2. **total_iters as objective**: Iteration count is tightly coupled with convergence threshold parameters. Jointly optimizing thresholds to minimize iters is trivial — just loosen them. The interesting question is which configs reach good quality fastest.

### Observable correlation analysis (v1 data)

Spearman correlations pooled from 377 Phase 3 trials across 4 regimes:

| Pair | ρ | Implication |
|---|---|---|
| holdout_rmse ↔ holdout_mae | +0.995 | Near-perfect redundancy — keep only holdout_rmse |
| coverage_95 ↔ calibration_gap | −1.000 | Identical information (by construction) — keep only coverage_95 |
| holdout_rmse ↔ interval_score | +0.878 | Strongly redundant — drop interval_score from objectives |
| holdout_rmse ↔ coverage_95 | +0.089 | **Independent** |
| rank_mae ↔ everything | |ρ| < 0.13 | **Independent** |
| wall_seconds ↔ holdout_rmse | +0.361 | Weakly correlated; prefer total_iters as cost axis |

**Three independent quality axes**: holdout_rmse, coverage_95, rank_mae. All others are either redundant or derivable.

## Design principles

1. **Separate "converge to what" from "how fast"** — Stage 1 (quality ceiling, bp OFF) → Stage 2 (convergence efficiency including bp, 2-objective) → Stage 3 (integration & comparison)
2. **Exploit mixed-factor QMC** — `build_grid(method="sobol")` handles discrete+continuous via bucket mapping; **requires no continuity** (uses Sobol QMC with bucket-mapped discrete factors)
3. **`screen(method="sobol")` requires ALL-continuous factors** — Sobol sensitivity analysis needs continuous inputs; discrete/categorical factors must be excluded or held fixed. Use full-factorial or Sobol QMC for discrete sensitivity.
4. **Broadprior OFF in Stage 1** — fixes niter_broadprior=0 so prior sensitivity is not confounded by the warmup suppression. Broadprior is re-introduced in Stage 2 as a discrete factor alongside convergence settings, where its interaction with convergence stop is the point.
5. **2-objective Stage 2** — weighted quality composite (holdout_rmse, coverage_95, rank_mae) vs total_iters, since the three quality metrics are uncorrelated with each other but we want a single quality-vs-efficiency tradeoff.
6. **Three-way final comparison** — library defaults vs scikit-learn PCA (existing `compare.py` comparator) vs v2-optimized.

## Factor partitioning

| Group | Factors | Types | Stage |
|---|---|---|---|
| Prior / ARD | hp_va, hp_vb, hp_v, va_init | 4 continuous | Stage 1 |
| Structural | xprobe_fraction | 1 continuous | Stage 1 |
| Broadprior | niter_broadprior | 1 discrete (5 levels: 0, 25, 50, 100, 200) | **Stage 2** (OFF in Stage 1) |
| Convergence thresholds | minangle, cfstop_rel, rmsstop_atol, rmsstop_rtol | 4 continuous | Stage 2 |
| Convergence discrete | maxiters, patience, rmsstop_window | 3 discrete | Stage 2 |
| Criterion ordering | criterion_order | 1 categorical (6 levels, see below) | Stage 2 |
| Criterion enablement | active_criteria | 1 categorical (5 presets, see below) | Stage 2 |

**Key change from v1 plan**: `niter_broadprior` moves from Stage 1 to Stage 2. In Stage 1 it is fixed at 0 so that Sobol screening of the prior hyperparameters is unconfounded. In Stage 2, bp is varied jointly with convergence factors — this is where the interaction between broadprior warmup and convergence stop is directly studied. The risk of bp=0 producing lower Stage 1 quality ceilings is mitigated by the generous convergence budget (maxiters=1000, patience=1) — with enough iterations the ARD priors can still tighten without the broadprior warmup.

### `criterion_order` levels

These map to the `criterion_order` parameter from #101, which controls the
priority sequence in `convergence_check()`. The first criterion that fires wins.

| Level | Order | Rationale |
|---|---|---|
| `default` | angle → earlystop → rms_plateau → cost → composite → slowing_down | Current behavior |
| `cost_first` | cost → angle → composite → rms_plateau → earlystop → slowing_down | ELBO-driven: theoretically grounded, monotonic |
| `composite_first` | composite → cost → angle → rms_plateau → earlystop → slowing_down | Composite leads: tests the learned metric |
| `rms_first` | rms_plateau → angle → cost → composite → earlystop → slowing_down | RMS-driven: practical, fast |
| `cost_angle` | cost → angle → rms_plateau → composite → earlystop → slowing_down | ELBO then angle: two strongest criteria first |
| `angle_cost` | angle → cost → composite → rms_plateau → earlystop → slowing_down | Angle then ELBO: slight reorder of default |

### `active_criteria` presets

Rather than 6 independent booleans (64 combos, most nonsensical), we define
meaningful presets via the `convergence_criteria` dict from #101. Disabled
criteria are still evaluated for diagnostics but excluded from the stop decision.

| Preset | Enabled criteria | Rationale |
|---|---|---|
| `all` | angle, earlystop, rms_plateau, cost, composite, slowing_down | Kitchen sink — current behavior when all thresholds set |
| `core` | angle, cost, rms_plateau | The three individually strongest criteria |
| `elbo_only` | cost | Pure ELBO stopping — tests theoretical minimum |
| `angle_elbo` | angle, cost | Two theoretically grounded criteria |
| `practical` | angle, rms_plateau, composite | No raw ELBO — criteria that work without cfstop enabled |

## Observables

### Stage 1 objectives (3 independent quality axes)

| Observable | Direction | Role | Rationale |
|---|---|---|---|
| holdout_rmse | MIN | **Pareto objective** | Primary accuracy axis |
| coverage_95 | MAX | **Pareto objective** | Independent calibration axis (ρ=0.089 with RMSE) |
| rank_mae | MIN | **Pareto objective** | Most independent axis (all |ρ| < 0.13) |

### Stage 2 objectives (2: quality composite vs efficiency)

The three quality metrics are uncorrelated (|ρ| < 0.13), so a weighted sum
captures them without information loss. Total_iters is the efficiency axis.

| Observable | Direction | Role | Definition |
|---|---|---|---|
| quality_composite | MIN | **Objective 1** | w₁·norm(holdout_rmse) + w₂·norm(1−coverage_95) + w₃·norm(rank_mae) |
| total_iters | MIN | **Objective 2** | Sum of iterations across all k-trials |

Default weights: w₁=0.5, w₂=0.25, w₃=0.25 (RMSE is the primary metric,
coverage and rank are secondary). Normalisation is min-max within each
phase's sample. Weights are configurable per run.

> **Rationale for 2-objective instead of 4**: With 4 objectives (3 quality + iters),
> NSGA-II performance degrades and the Pareto front becomes hard to interpret.
> Since the 3 quality metrics are empirically uncorrelated, a weighted sum
> preserves all information while keeping the Pareto front 2D (quality vs cost).
> The knee of this front directly answers "what convergence settings give the
> best quality per iteration?"

### Tracked (not optimised)

| Observable | Direction | Why tracked |
|---|---|---|
| holdout_mae | MIN | ρ=0.995 with RMSE — redundant but useful for reporting |
| calibration_gap | MIN | ρ=−1.0 with coverage — identical information |
| interval_score | MIN | ρ=0.878 with RMSE — redundant |
| wall_seconds | MIN | Platform-dependent; prefer total_iters |
| best_k_iters | MIN | Diagnostic |

## Constraints

| Name | Observable | Condition | Scope |
|---|---|---|---|
| min_coverage | coverage_95 | ≥ 0.40 | Family A (well-specified) |
| min_coverage_b | coverage_95 | ≥ 0.30 | Family B (mis-specified) |
| max_rank_mae | rank_mae | ≤ 3.0 | All |

## Data regimes

Carried forward from v1 with additions:

**Screen regimes** (4, one per missingness type):

| n | p | rank | missingness | noise |
|---|---|---|---|---|
| 50 | 30 | 5 | mcar | 0.5 |
| 100 | 100 | 5 | complete | 0.5 |
| 200 | 50 | 2 | mnar_censored | 1.0 |
| 200 | 200 | 10 | block | 1.0 |

**Study regimes** (6, Family A + Family B):

- Family A: (50,30,5,mcar,0.5), (100,100,5,complete,0.5), (200,100,5,complete,1.0)
- Family B: (100,50,2,mnar_censored,0.5), (200,200,10,block,1.0), (100,100,5,mnar_censored,1.0)

**Validation regimes** (4, held-out — never seen during refinement):

| n | p | rank | missingness | noise |
|---|---|---|---|---|
| 70 | 50 | 5 | mcar | 0.3 |
| 100 | 200 | 10 | complete | 0.5 |
| 200 | 50 | 2 | mnar_censored | 1.0 |
| 150 | 150 | 5 | block | 0.5 |

---

## Stage 1: Quality Ceiling (broadprior OFF)

**Goal**: Find prior/structural configs that maximise quality with generous convergence and **niter_broadprior=0**. Convergence params are fixed permissively (maxiters=1000, minangle=1e-10, patience=1) so they don't limit achievable quality. Broadprior is off so that Sobol screening of the 5 continuous prior/structural factors is unconfounded.

**Fixed convergence/bp settings for all of Stage 1**:
- `niter_broadprior = 0`
- `maxiters = 1000`
- `minangle = 1e-10`
- `patience = 1`
- `cfstop_rel = 1e-8` (very tight — won't fire early)
- `rmsstop = [200, 1e-6, 1e-5]` (very tight — won't fire early)

**Pareto objectives**: holdout_rmse, coverage_95, rank_mae (3 objectives — no total_iters).

### Phase 1.0 — Sobol screening (continuous prior/structural factors)

- **Method**: `screen(method="sobol")` — **continuous factors ONLY** (required by Sobol sensitivity analysis which needs continuous inputs for derivative estimation)
- **Factors**: hp_va, hp_vb, hp_v, va_init, xprobe_fraction (5 continuous)
- **Budget**: n_samples=256 → 256 × (2×5+2) = 3,072 evals × 4 screen regimes = **12,288 evals**
- **Tools**: `screen(method="sobol")`, `reduce_factors(threshold=0.02)`
- **Output**: S1/ST indices per factor per regime; factors below threshold dropped. Expect to reduce from 5 to 3–4 active continuous factors.

### Phase 1.1 — Sobol QMC exploration (reduced priors)

- **Method**: `build_grid(method="sobol")` — continuous factors only (survivors from 1.0). **Sobol QMC** (quasi-Monte Carlo) fills the space uniformly; unlike `screen(method="sobol")`, it handles mixed types via bucket mapping, but here all factors are continuous so it's straightforward.
- **Factors**: Surviving continuous from Phase 1.0
- **Budget**: n_samples=256 × 6 study regimes = **~1,536 evals**
- **Aggregation**: Per-config median and p90 across regimes
- **Tools**: `build_grid(method="sobol", n_samples=256)`, `run_grid`, `extract_front`
- **Output**: Quality Pareto front on 3 objectives; top-30 by weighted sum

### Phase 1.2 — Adaptive refinement (Optuna NSGA-II)

- **Method**: `run_adaptive` — zooms in around Phase 1.1 Pareto front
- **Factors**: Same as Phase 1.1; per-regime `VBPCASimulator(regime_defaults=...)`
- **Budget**: n_trials=200 per regime, 6 regimes = **~1,200 evals**
- **Tools**: `run_adaptive(n_trials=200)`, `extract_front`, `hypervolume`
- **Output**: Refined 3D quality Pareto front; top configs to pin for Stage 2

### Phase 1.3 — Cross-validation on held-out regimes

- **Method**: `run_grid` on VALIDATION_REGIMES with top-20 from Phase 1.2
- **Budget**: 20 × 4 = **80 evals**
- **Tools**: `run_grid`, `feasibility_filter(ALL_CONSTRAINTS)`
- **Output**: Confirmed robust quality configs. Failures dropped. **3–5 survivors pinned for Stage 2.**

---

## Stage 2: Convergence Efficiency (broadprior as factor)

**Goal**: Among quality-feasible configs from Stage 1, find convergence settings (including broadprior) that minimise total_iters with minimal quality degradation.

**Prior/structural factors**: Pinned at 3–5 Pareto-optimal configs from Stage 1.3 (treated as environmental strata).

**Objectives**: 2 — quality_composite (MIN) vs total_iters (MIN).

> **Broadprior interaction**: With bp=0, convergence criteria are active from iteration 1 — the model converges as fast as the criteria allow. With bp>0, convergence stop is **suppressed** for the first `niter_broadprior` iterations (see `_log_and_check_convergence`), guaranteeing at least that many iterations. This means bp interacts with every convergence factor: a high bp with loose thresholds wastes the most compute; a moderate bp with tight thresholds pays a fixed warmup cost then stops quickly. This interaction is exactly what Stage 2 is designed to measure.

### Phase 2.0 — Full-factorial pilot on discrete/categorical convergence factors

- **Method**: `build_grid(method="full")` — discrete/categorical factors only
- **Factors** (all discrete/categorical — no continuity requirement):
  - `niter_broadprior` (5 levels: 0, 25, 50, 100, 200)
  - `criterion_order` (6 levels)
  - `active_criteria` (5 presets)
  - `maxiters` (4 levels: 100, 200, 500, 1000)
  - `patience` (4 levels: 1, 2, 3, 5)
  - `rmsstop_window` (3 levels: 50, 100, 200)
- **Grid**: 5 × 6 × 5 × 4 × 4 × 3 = 7,200 configs × 2 pinned priors × 1 regime = **~14,400 evals**
  - Use only 2 pinned priors (best-quality + most-robust from Stage 1) and 1 representative regime to keep budget feasible
  - Many combos are inert (e.g., `rms_first` + `elbo_only` — rms isn't enabled). Post-hoc filtering removes dominated/inert rows.
- **Purpose**: Exhaustively map discrete landscape; identify dead combinations and dominated orderings. Crucially, quantifies the bp × convergence interaction.
- **Tools**: `build_grid(method="full")`, `run_grid`, `pareto_rank` (on 2 objectives: quality_composite, total_iters)
- **Output**: Surviving discrete levels. Expect to reduce criterion_order to ~3, active_criteria to ~2–3, and identify whether bp>0 is ever Pareto-optimal vs bp=0.

### Phase 2.1 — Sobol QMC exploration (mixed convergence + bp)

- **Method**: `build_grid(method="sobol")` — **Sobol QMC** handles mixed continuous+discrete via bucket mapping (no continuity requirement unlike `screen(method="sobol")`)
- **Factors**: minangle, cfstop_rel, rmsstop_atol, rmsstop_rtol (4 continuous) + surviving discrete from Phase 2.0 (including niter_broadprior)
- **Budget**: n_samples=256 × 3 pinned priors × 4 regimes = **~3,072 evals**
- **Objectives**: quality_composite (MIN) vs total_iters (MIN) — 2D Pareto
- **Tools**: `build_grid(method="sobol", n_samples=256)`, `run_grid`, `extract_front`
- **Output**: 2D convergence Pareto front (quality degradation vs iters saved); knee identification

### Phase 2.2 — Adaptive refinement (Optuna NSGA-II)

- **Method**: `run_adaptive` — zooms into knee of efficiency front
- **Factors**: Same as Phase 2.1; per-(regime × pinned-prior) strata
- **Budget**: n_trials=200 per stratum, ~12 strata = **~2,400 evals**
- **Objectives**: quality_composite, total_iters (2 objectives)
- **Tools**: `run_adaptive(n_trials=200)`, `extract_front`, `hypervolume`
- **Output**: Refined 2D Pareto front; candidate recommended convergence configs

### Phase 2.3 — Validation

- **Method**: `run_grid` on VALIDATION_REGIMES × top candidates
- **Budget**: ~10 × 5 × 4 = **~200 evals**
- **Tools**: `run_grid`, `feasibility_filter`, report decomposed quality metrics (not just composite)
- **Output**: Final recommended convergence+bp defaults; quality degradation budget confirmed. Report individual holdout_rmse, coverage_95, rank_mae (not just composite) to verify no single metric degraded unacceptably.

---

## Stage 3: Integration & Deliverables

### Phase 3.0 — Full-combination confirmation

- **Method**: `run_grid` — best prior configs × best convergence configs × all regimes
- **Budget**: ~5 × 5 × 10 = **~250 evals**
- **Purpose**: Verify Stage 1 + Stage 2 winners compose well (no cross-stage interaction effects)
- **Tools**: `run_grid`, `extract_front`, `hypervolume`

### Phase 3.1 — Three-way comparison

Run head-to-head comparison of three conditions:

1. **Library defaults** — current `_build_options()` defaults, no user overrides
2. **scikit-learn PCA** — existing comparator from `compare.py` (impute → PCA → rank selection)
3. **v2-optimized** — best combined prior + convergence config from Phase 3.0

- **Method**: Extend existing `compare.py` framework (which already has `"sklearn_pca"` and `"vbpca_default"` conditions) to add `"vbpca_optimized"` with v2 settings
- **Regimes**: Full grid from `compare.py` (7×7×3 n/p/rank × 4 missingness) + VALIDATION_REGIMES
- **Tools**: `compare.py` infrastructure, `plot_front`, `plot_parallel`, `plot_calibration`
- **Output**: Speedup and quality improvement quantified; paper figures. Key metrics: total_iters reduction, holdout_rmse change, coverage_95 change, rank_mae change — all individually reported, not just composite.

### Phase 3.2 — Stacking (optional)

- **Method**: `stack_scores` — across criterion variants at recommended operating point
- **Tools**: `stack_scores`, `ensemble_predict`
- **Purpose**: If multiple convergence criteria give similar quality, ensemble-average posteriors for better calibration

---

## Method × factor type compatibility

| Method | Continuous | Discrete | Categorical | Continuity required? | Used in |
|---|---|---|---|---|---|
| `screen(method="sobol")` | ✓ | ✗ (filtered) | ✗ (filtered) | **Yes** — Sobol sensitivity needs continuous inputs | Phase 1.0 |
| `build_grid(method="full")` | ✗ (needs levels) | ✓ | ✓ | No | Phase 2.0 |
| `build_grid(method="sobol")` | ✓ (linear map) | ✓ (bucket map) | ✓ (bucket map) | **No** — Sobol QMC sequence with bucket mapping | Phase 1.1, 2.1 |
| `build_grid(method="lhs")` | ✓ | ✓ | ✓ | No | Alternative |
| `build_grid(method="halton")` | ✓ | ✓ | ✓ | No | Alternative |
| `run_adaptive` (NSGA-II) | suggest_float | suggest_categorical | suggest_categorical | No | Phase 1.2, 2.2 |
| `run_grid` | Any (pre-built) | Any | Any | No | All phases |

> **Critical distinction**: `screen(method="sobol")` performs Sobol **sensitivity analysis** (computing S1/ST indices via Saltelli's scheme) and requires continuous factors because it takes derivatives in factor space. `build_grid(method="sobol")` generates a **Sobol QMC** quasi-random sequence for space-filling and maps discrete/categorical factors via bucket thresholds — no continuity required.

## Budget summary

| Phase | Evals | Purpose |
|---|---|---|
| 1.0 | ~12,288 | Sobol screening (prior/structural, bp OFF) |
| 1.1 | ~1,536 | QMC exploration |
| 1.2 | ~1,200 | Adaptive refinement |
| 1.3 | 80 | Validation |
| 2.0 | ~14,400 | Discrete factorial (convergence + bp) |
| 2.1 | ~3,072 | QMC exploration (mixed) |
| 2.2 | ~2,400 | Adaptive refinement |
| 2.3 | ~200 | Validation |
| 3.0–3.2 | ~500 | Confirmation + three-way comparison |
| **Total** | **~35,676** | |

Phase 2.0 dominates at ~14.4K. This can be reduced by:
- Using fewer pinned priors (1 instead of 2) → ~7,200
- Testing on 2 regimes instead of 1 to get more signal (trades budget for robustness)
- Pre-filtering: exclude logically-inert combos before running (e.g., if `active_criteria=elbo_only`, then `criterion_order` variants that only reorder non-ELBO criteria are equivalent — collapse to 1 level)

Phase 1.0 is second at ~12.3K. Can reduce to ~6.1K with n_samples=128 (still acceptable for 5 factors).

## Key differences from v1

| Aspect | v1 | v2 |
|---|---|---|
| Stages | 1 (all factors together) | 3 (quality → efficiency → integration) |
| Broadprior in Stage 1 | Confounded with priors | **OFF** (fixed at 0) — clean screening |
| Broadprior handling | Single factor | Stage 2 factor — studied jointly with convergence where the interaction matters |
| Stage 2 objectives | 4 independent Pareto | **2**: quality_composite vs total_iters |
| Quality composite | N/A | w₁·RMSE + w₂·(1−cov95) + w₃·rank_mae (uncorrelated components) |
| Discrete factors | Invisible to Sobol screening | Full factorial (Phase 2.0) + Sobol QMC bucket mapping |
| criterion_order | Not a factor | Categorical factor, 6 levels (Stage 2) — via #102 `criterion_order` param |
| active_criteria | Not a factor | Categorical factor, 5 presets (Stage 2) — via #102 `convergence_criteria` dict |
| coverage_95 | Tracked only | **Pareto objective** (Stage 1), part of composite (Stage 2) |
| Final comparison | v1 defaults vs optimized | **Library defaults vs scikit-learn vs v2-optimized** |
| Validation | None | Held-out regimes (Phase 1.3, 2.3) |
| Interaction check | None | Phase 3.0 cross-check |

## Open decisions

1. **Phase 1.0 budget** — 12.3K evals for Sobol screening. Acceptable, or reduce n_samples to 128 (~6.1K)?
2. **Pinned prior count for Stage 2** — 2 (Phase 2.0 budget feasible) or 3 (more robust, ~50% more evals)?
3. **Quality composite weights** — w₁=0.5, w₂=0.25, w₃=0.25 proposed. Adjust?
4. **Phase 2.0 pre-filtering** — Should we programmatically collapse logically-equivalent criterion_order × active_criteria combos before running, or let the factorial run and filter post-hoc?

## Prerequisites

- [x] `criterion_order` parameter in `_converge.py` / `_pca_full.py` (#101 → #102)
- [x] `convergence_criteria` dict for per-criterion enable/disable (#101 → #102)
- [ ] Forward `criterion_order` and `convergence_criteria` in `_world.py` `VBPCASimulator`
- [ ] Add `CRITERION_ORDER_LEVELS` and `ACTIVE_CRITERIA_PRESETS` dicts to `_common.py`
- [ ] Implement `quality_composite` scorer (weighted min-max normalised sum)
- [ ] Define `active_criteria` preset → `convergence_criteria` dict mapping in `_common.py`
- [ ] Define `criterion_order` level → list mapping in `_common.py`
- [ ] Update `_common.py` with v2 factor/observable definitions
- [ ] Update `_world.py` to pass `criterion_order` and `convergence_criteria` through to VBPCA
