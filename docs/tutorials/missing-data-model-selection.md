# Missing Data & Model Selection

This tutorial demonstrates VBPCApy's core strength: fitting PCA directly on
incomplete data and automatically selecting the number of components.

## Generate data with missing entries

```python
import numpy as np

rng = np.random.default_rng(42)

n_features, n_samples, true_rank = 60, 200, 5
noise_std = 0.5

W_true = rng.standard_normal((n_features, true_rank))
S_true = rng.standard_normal((true_rank, n_samples))
X_clean = W_true @ S_true + noise_std * rng.standard_normal((n_features, n_samples))

# Mask 20% of entries as missing
missing_rate = 0.20
mask = rng.random((n_features, n_samples)) > missing_rate  # True = observed
X_obs = np.where(mask, X_clean, np.nan)

print(f"Data: {n_features} features × {n_samples} samples")
print(f"Missing: {100 * (1 - mask.mean()):.1f}%")
```

## Create a probe set

A **probe set** holds out a small fraction of the observed entries. The model
never sees these during fitting, so they provide an unbiased estimate of
reconstruction quality.

```python
from vbpca_py import make_xprobe_mask

X_train, X_probe = make_xprobe_mask(X_obs, fraction=0.10, rng=rng)
```

`make_xprobe_mask` returns two matrices:

- `X_train` — the training data with probe entries set to NaN.
- `X_probe` — a sparse matrix containing only the held-out probe entries.

## Select the number of components

Use `select_n_components` to sweep over candidate values of $k$ and pick the
best according to the variational cost:

```python
from vbpca_py import select_n_components, SelectionConfig

cfg = SelectionConfig(
    metric="cost",          # select by variational free energy
    patience=2,             # require 2 consecutive worsening trials to stop
    max_trials=12,          # try at most 12 values of k
    return_best_model=True, # keep the fitted model for the best k
)

best_k, metrics, trace, best_model = select_n_components(
    X_train,
    mask=~np.isnan(X_train),
    components=range(1, 15),
    config=cfg,
    maxiters=200,
    xprobe=X_probe,
)

print(f"Selected k = {best_k}  (true rank = {true_rank})")
print(f"  cost = {metrics['cost']:.2f}")
```

!!! tip "Using probe RMS instead"
    You can also select by held-out reconstruction error:

    ```python
    cfg = SelectionConfig(metric="prms", patience=2)
    ```

    This requires a probe set (via `xprobe` or `xprobe_fraction`).

## Inspect the selection trace

The `trace` list contains per-$k$ endpoint metrics. You can plot the cost
or probe RMS across candidates:

```python
ks = [entry["k"] for entry in trace]
costs = [entry["cost"] for entry in trace]

# Quick text summary
for entry in trace:
    marker = " <-- best" if entry["k"] == best_k else ""
    print(f"  k={entry['k']:2d}  cost={entry['cost']:.2f}{marker}")
```

## Fit the final model

If you used `return_best_model=True`, the fitted model is already available:

```python
model = best_model
```

Otherwise, fit manually at the selected rank:

```python
from vbpca_py import VBPCA

model = VBPCA(n_components=best_k, maxiters=200)
model.fit(X_obs, mask=mask)
```

## Evaluate on held-out entries

```python
X_hat = model.inverse_transform()
held_out = ~mask

rmse_held = float(np.sqrt(np.nanmean((X_hat[held_out] - X_clean[held_out]) ** 2)))
rmse_obs = float(np.sqrt(np.nanmean((X_hat[mask] - X_clean[mask]) ** 2)))

print(f"Reconstruction RMSE — observed: {rmse_obs:.4f}")
print(f"Reconstruction RMSE — held-out: {rmse_held:.4f}")
print(f"Noise σ: {noise_std}")
```

If the model has selected the correct rank, both RMSE values should be close to
the noise standard deviation.

## Explained variance

```python
evr = model.explained_variance_ratio_
if evr is not None:
    for i, v in enumerate(evr[:best_k]):
        print(f"  PC{i+1}: {v:.3f}")
```

## Next steps

- [Preprocessing Mixed Data](preprocessing-mixed-data.md) — handle categorical +
  continuous columns before fitting.
- [Model Selection concepts](../concepts/model-selection.md) — understand the sweep
  logic, metrics, and 1-SE rule.
