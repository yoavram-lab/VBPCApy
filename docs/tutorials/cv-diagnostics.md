# Cross-Validation & Diagnostics

This tutorial chooses the number of components with K-fold cross-validation
over observed entries, then fits the selected model while monitoring a probe
set and inspects its diagnostics.

## Generate data

```python
import numpy as np

rng = np.random.default_rng(42)

n_samples, n_features, true_rank = 150, 40, 4
noise_std = 0.5

signal = rng.standard_normal((n_samples, true_rank)) @ rng.standard_normal(
    (true_rank, n_features)
)
X = signal + noise_std * rng.standard_normal((n_samples, n_features))

observed = rng.random(X.shape) > 0.15  # 15% missing
X_obs = np.where(observed, X, np.nan)
```

## Cross-validate the number of components

`cross_validate_components` splits the *observed entries* (not whole rows)
into folds. For each fold it fits every candidate on the remaining entries and
scores its predictions for the held-out entries. Every training fold keeps at
least one observation in each non-empty row and column. `CVConfig.seed`
controls both the fold assignment and the model initialization.

```python
from vbpca_py import CVConfig, cross_validate_components

config = CVConfig(
    n_splits=5,
    metric="prms",  # held-out RMS error
    one_se_rule=True,  # prefer the smallest count within one standard error
    seed=42,
)

best_k, results = cross_validate_components(
    X_obs,
    components=range(0, 10),  # 0 is the mean-only model
    config=config,
    maxiters=200,
)

print(f"Selected k = {best_k} (generating rank {true_rank})")
```

### Inspect the results

```python
for entry in results:
    marker = "  <- selected" if entry["k"] == best_k else ""
    print(
        f"k={entry['k']:2d}  held-out RMS={entry['mean_prms']:.4f}"
        f" +/- {entry['se_prms']:.4f}{marker}"
    )
```

Each entry reports the mean and standard error of the held-out error over the
folds, and per-fold values such as `prms_fold_1`.

### The one-standard-error rule

With `one_se_rule=True`, the selected count is the smallest candidate whose
mean held-out error is within one standard error of the lowest mean. This
prefers a simpler model when larger ones do not predict clearly better. Rank
zero denotes a mean-only model and is a valid result when no latent component
improves prediction. The rule is a model-selection heuristic, not a hypothesis
test.

## Fit the selected model

This simulated example has a positive rank. If `best_k == 0` on other data,
report the mean-only result; there is no positive-rank model to fit.

A probe set lets the fit report its held-out error at every iteration:

```python
from vbpca_py import VBPCA, make_xprobe_mask

X_train, X_probe = make_xprobe_mask(X_obs, fraction=0.10, rng=rng)

model = VBPCA(n_components=best_k, maxiters=300, record_cost=True, random_state=0)
model.fit(X_train, xprobe=X_probe)
```

`record_cost=True` keeps the variational cost at every iteration; it is off by
default because computing it adds time.

## Convergence diagnostics

```python
print(f"Iterations: {model.n_iter_} ({model.convergence_reason_})")
print(f"Training RMS: {model.rms_:.4f}")
print(f"Probe RMS:    {model.prms_:.4f}")
print(f"Cost:         {model.cost_:.1f}")
```

`learning_curve_` holds the per-iteration history:

```python
curve = model.learning_curve_
print(sorted(key for key in ("rms", "prms", "cost") if key in curve))
print("Iterations recorded:", len(curve["rms"]))
```

## Explained variance

```python
cumulative = 0.0
for i, ratio in enumerate(model.explained_variance_ratio_, start=1):
    cumulative += ratio
    print(f"PC{i}: {ratio:.3f} (cumulative {cumulative:.3f})")
```

## Plotting

The plotting helpers need the `plot` extra (`pip install "vbpca-py[plot]"`).

```python
from vbpca_py.plotting import loadings_barplot, scree_plot, variance_explained_plot

scree = scree_plot(model, cumulative=True)
loadings = loadings_barplot(model, component=0, top_n=15)
variance = variance_explained_plot(model)
```

- `scree_plot` shows the explained variance ratio of each component, with the
  cumulative ratio when `cumulative=True`.
- `loadings_barplot` shows the features with the largest absolute loadings on
  one component; `top_n` limits how many are shown.
- `variance_explained_plot` shows the variance of each component in data units.

Each function returns a Matplotlib figure. The
[plotting how-to](../howto/plotting.md) shows the figures and their options.

## Next steps

- [Convergence Tuning](../howto/convergence.md): stopping criteria and how to
  check a fit.
- [Model Selection concepts](../concepts/model-selection.md): selection rules
  in more detail.
