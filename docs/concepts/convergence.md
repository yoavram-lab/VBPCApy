# Convergence Criteria

Each fit run (including every candidate $k$ tried in `select_n_components`)
iterates the EM loop until one of the following criteria triggers or `maxiters`
is reached.

## Criteria (evaluated in priority order)

### 1. Subspace angle — `minangle`

Stops when the principal angle between successive loading matrices falls below
the threshold.

| Option | Default | Description |
|--------|---------|-------------|
| `minangle` | `1e-8` | Angle threshold in radians |

### 2. Early stopping on probe RMS — `earlystop`

When a probe set is provided (via `xprobe` or `xprobe_fraction`), stops if the
probe RMS starts increasing (overfitting signal). The returned factors are
restored to the lowest-probe-RMS state; the complete trace remains available,
with `best_probe_iteration_`, `best_probe_rms_`, and `returned_iteration_`
identifying the selected state.

| Option | Default | Description |
|--------|---------|-------------|
| `earlystop` | `False` | Enable probe-based early stopping |

### 3. RMS plateau — `rmsstop`

Compares the current RMS to the value `window` iterations ago. Stops if RMS did
not worsen and the absolute change is below `abs_tol` or the relative change is
below `rel_tol`.

| Option | Default | Description |
|--------|---------|-------------|
| `rmsstop` | `[100, 1e-4, 1e-3]` | `[window, abs_tol, rel_tol]` |

### 4. Cost / ELBO plateau — `cfstop`

Same interpretation as `rmsstop` but applied to the variational cost (negative
ELBO).

| Option | Default | Description |
|--------|---------|-------------|
| `cfstop` | `[]` (disabled) | `[window, abs_tol, rel_tol]` |

### 5. Relative ELBO decrease — `cfstop_rel`

Stops when the fractional improvement in variational free energy (negative
ELBO, so lower is better) drops below a threshold. A worsening step does not
count as convergence.

| Option | Default | Description |
|--------|---------|-------------|
| `cfstop_rel` | disabled | Relative improvement threshold |

### 6. ELBO curvature — `cfstop_curv`

Stops only when the current free-energy slope and its second difference are
both smaller than the threshold. Requiring a small slope prevents a constant
but steep descent (zero curvature) from being mistaken for convergence, and a
worsening step does not satisfy the criterion.

| Option | Default | Description |
|--------|---------|-------------|
| `cfstop_curv` | disabled | Joint absolute slope and curvature threshold |

### 7. Composite criteria — `composite_stop`

Require multiple criteria to trigger simultaneously. Pass a dict specifying
which criteria must all be satisfied:

```python
model = VBPCA(
    n_components=5,
    composite_stop={"angle": 1e-4, "rms": 1e-3, "elbo_rel": 1e-4},
)
```

The valid keys are `angle`, `rms`, and `elbo_rel`; each value is a scalar
threshold. The `rms` and `elbo_rel` checks require a non-worsening final step.

## Patience

All criteria support a **criterion-specific patience window**: the same
criterion must be satisfied for $N$ consecutive eligible iterations before
convergence is declared. Alternating hits from different criteria are tracked
separately and cannot jointly satisfy patience.

| Option | Default | Description |
|--------|---------|-------------|
| `patience` | `1` | Consecutive iterations required |

## Other stopping conditions

- **Hard cap:** `maxiters` (default 1000).
- **Broad-prior warmup:** during the first `niter_broadprior` iterations (default
  100), stopping messages are suppressed when `use_prior` is active, allowing the
  model to settle before ARD engages.

A fit that reaches `maxiters` without accepting a criterion reports
`convergence_reason_ == "maxiters"` and `converged_ == False`. Probe-based early
stopping also sets `converged_ == False` because it terminates for validation
rather than numerical convergence. VBPCA warns whenever the resolved options have
`maxiters <= niter_broadprior`: that configuration leaves no eligible
post-warmup iteration in which a convergence criterion can stop the fit.

The legacy `slowing_down` name is unavailable in live fitting: the fitter does
not supply its backtracking counter. Recommendations and resolved fit options
disable it. An explicit `True` request emits a warning and resolves to `False`,
without mutating the caller's dictionary or changing the already inert rule's
fit endpoints. `criterion_available_slowing_down` is zero throughout the returned
learning curve, distinguishing an unavailable event stream from an evaluated
criterion that did not fire. The legacy helper can still interpret an explicitly
supplied diagnostic counter; this does not enable it in the fitter.

An accepted heuristic stopping rule does not certify posterior stability or
predictive calibration. Changing a recommended threshold or policy requires
separate evidence on the intended regimes and independent confirmation.

## Criterion ordering — `criterion_order`

Individual criteria have **OR semantics** by default. Criteria are evaluated in
a fixed priority order (angle first), and the first eligible
criterion whose own patience streak is complete wins. Reordering determines
the winner when multiple criteria are ready on the same iteration; it does not
require an earlier criterion to converge before a later one may stop the fit.
Use `composite_stop` and disable its individual constituents when AND semantics
are required.

| Option | Default | Description |
|--------|---------|-------------|
| `criterion_order` | `None` (use default) | List of criterion names in priority order |

Valid criterion names: `angle`, `earlystop`, `rms_plateau`, `cost`,
`composite`, `slowing_down`.

The learning curve records a numeric `criterion_satisfied_<name>` trace for
every criterion, including disabled or nonwinning criteria. Cost rules also
have separate `criterion_satisfied_cost_plateau`,
`criterion_satisfied_cfstop_rel`, and `criterion_satisfied_cfstop_curv` traces.
These traces allow counterfactual stopping policies to be compared after a
long-running fit.

## Per-criterion enable/disable — `convergence_criteria`

Selectively disable individual criteria without zeroing their thresholds.
Disabled criteria are still evaluated for diagnostics but excluded from the
stop decision.

| Option | Default | Description |
|--------|---------|-------------|
| `convergence_criteria` | `None` (all enabled) | Dict mapping criterion names to booleans |
