# Cross-Validation & Diagnostics

This tutorial demonstrates K-fold cross-validated model selection and
post-fit diagnostic inspection.

## Generate data

```python
import numpy as np

rng = np.random.default_rng(42)

n_features, n_samples, true_rank = 40, 150, 4
noise_std = 0.5

W = rng.standard_normal((n_features, true_rank))
S = rng.standard_normal((true_rank, n_samples))
X = W @ S + noise_std * rng.standard_normal((n_features, n_samples))

# 15% missing
mask = rng.random(X.shape) > 0.15
X_obs = np.where(mask, X, np.nan)
```

## Cross-validate component count

`cross_validate_components` partitions the *observed entries* (not full rows)
into folds. Each fold holds out a subset of entries, fits on the rest, and
evaluates reconstruction on the held-out entries.

```python
from vbpca_py import cross_validate_components, CVConfig

cfg = CVConfig(
    n_splits=5,
    metric="prms",       # probe RMS on held-out entries
    one_se_rule=True,    # prefer simpler model within 1 SE of best
    seed=42,
)

best_k, results = cross_validate_components(
    X_obs,
    mask=mask,
    components=range(1, 10),
    config=cfg,
    maxiters=200,
)

print(f"Selected k = {best_k}  (true rank = {true_rank})")
```

### Inspect CV results

```python
for entry in results:
    k = entry["k"]
    mean_metric = entry["mean"]
    se = entry["se"]
    marker = " <-- selected" if k == best_k else ""
    print(f"  k={k:2d}  prms={mean_metric:.4f} ± {se:.4f}{marker}")
```

### 1-SE rule

When `one_se_rule=True`, the selected $k$ is the smallest value whose mean
metric is within one standard error of the overall best. This guards against
overfitting by favouring parsimony.

## Fit the final model

```python
from vbpca_py import VBPCA, make_xprobe_mask

# Create a probe set for monitoring
X_train, X_probe = make_xprobe_mask(X_obs, fraction=0.10, rng=rng)

model = VBPCA(n_components=best_k, maxiters=300, verbose=1)
model.fit(X_train, mask=~np.isnan(X_train), xprobe=X_probe)
```

## Convergence diagnostics

### Learning curves

```python
print(f"Final RMS:       {model.rms_:.6f}")
print(f"Final probe RMS: {model.prms_:.6f}")
print(f"Final cost:      {model.cost_:.2f}")
```

### Explained variance

```python
evr = model.explained_variance_ratio_
if evr is not None:
    total = 0.0
    for i, v in enumerate(evr):
        total += v
        print(f"  PC{i+1}: {v:.3f}  (cumulative: {total:.3f})")
```

## Plotting

Install the plotting extra if you haven't already:

```bash
pip install vbpca_py[plot]
```

```python
from vbpca_py.plotting import scree_plot, loadings_barplot, variance_explained_plot

# Scree plot — explained variance ratio per component
fig = scree_plot(model, cumulative=True)

# Loadings bar plot — feature importance for the first component
fig = loadings_barplot(model, component=0, top_n=15)

# Absolute variance per component
fig = variance_explained_plot(model)
```

### Scree plot

The scree plot shows the explained variance ratio for each component. A sharp
"elbow" suggests the number of meaningful components. The cumulative line shows
the total variance explained.

### Loadings bar plot

The loadings bar plot for a given component highlights which features
contribute most. Use `top_n` to show only the highest-magnitude features.

## Next steps

- [Convergence Tuning](../howto/convergence.md) — fine-tune stopping criteria for
  tricky datasets.
- [Model Selection concepts](../concepts/model-selection.md) — understand the
  sweep logic and 1-SE rule in depth.
