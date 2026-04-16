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
probe RMS starts increasing (overfitting signal).

| Option | Default | Description |
|--------|---------|-------------|
| `earlystop` | `False` | Enable probe-based early stopping |

### 3. RMS plateau — `rmsstop`

Compares the current RMS to the value `window` iterations ago. Stops if the
absolute change is below `abs_tol` or the relative change is below `rel_tol`.

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

Stops when the fractional ELBO improvement drops below a threshold.

| Option | Default | Description |
|--------|---------|-------------|
| `cfstop_rel` | disabled | Relative improvement threshold |

### 6. ELBO curvature — `cfstop_curv`

Stops when the second difference of the ELBO stabilises.

| Option | Default | Description |
|--------|---------|-------------|
| `cfstop_curv` | disabled | Curvature threshold |

### 7. Composite criteria — `composite_stop`

Require multiple criteria to trigger simultaneously. Pass a dict specifying
which criteria must all be satisfied:

```python
model = VBPCA(
    n_components=5,
    composite_stop={"rmsstop": [50, 1e-4, 1e-3], "cfstop": [50, 1e-3, 1e-2]},
)
```

## Patience

All criteria support a **patience window**: the criterion must be satisfied for
$N$ consecutive iterations before convergence is declared.

| Option | Default | Description |
|--------|---------|-------------|
| `patience` | `1` | Consecutive iterations required |

## Other stopping conditions

- **Slowing-down guard:** internal backtracking hits 40 steps.
- **Hard cap:** `maxiters` (default 1000).
- **Broad-prior warmup:** during the first `niter_broadprior` iterations (default
  100), stopping messages are suppressed when `use_prior` is active, allowing the
  model to settle before ARD engages.
