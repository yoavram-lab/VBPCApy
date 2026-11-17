# Missing Data & Model Selection

This tutorial fits PCA directly to incomplete data and chooses the number of
components by comparing candidate models on held-out entries.

## Generate data with missing entries

The matrix has 200 samples (rows) and 60 features (columns) with a rank-five
signal. Twenty percent of entries are hidden at random and replaced by `NaN`.

```python
import numpy as np

rng = np.random.default_rng(42)

n_samples, n_features, true_rank = 200, 60, 5
noise_std = 0.5

signal = rng.standard_normal((n_samples, true_rank)) @ rng.standard_normal(
    (true_rank, n_features)
)
X_complete = signal + noise_std * rng.standard_normal((n_samples, n_features))

observed = rng.random((n_samples, n_features)) > 0.20  # True = observed
X_obs = np.where(observed, X_complete, np.nan)

print(f"Data: {n_samples} samples x {n_features} features")
print(f"Missing: {100 * (1 - observed.mean()):.1f}%")
```

VBPCApy treats `NaN` entries as unobserved. Only observed entries enter the
likelihood, so no values are imputed before fitting. This is appropriate when
the values are missing at random given the model.

## Hold out probe entries

A **probe set** is a random subset of the observed entries that is removed
before fitting. Because the model never sees these values, its error on them
estimates how well it predicts entries it has not observed.

```python
from vbpca_py import make_xprobe_mask

X_train, X_probe = make_xprobe_mask(X_obs, fraction=0.10, rng=rng)
print("Probe entries:", int(np.isfinite(X_probe).sum()))
```

For dense input, `X_train` is a copy of `X_obs` with the probe entries set to
`NaN`, and `X_probe` has the same shape with only the probe values filled in.

## Select the number of components

`select_n_components` fits one model per candidate number of components and
compares them with a chosen metric.

```python
from vbpca_py import SelectionConfig, select_n_components

config = SelectionConfig(
    metric="prms",  # held-out RMS error on the probe entries
    patience=2,  # stop after two candidates without improvement
    max_trials=12,  # fit at most 12 candidates
    return_best_model=True,  # keep the fitted model for the selected count
)

best_k, metrics, trace, best_model = select_n_components(
    X_train,
    components=range(1, 15),
    config=config,
    maxiters=200,
    xprobe=X_probe,
    random_state=0,
)

print(f"Selected k = {best_k} (generating rank {true_rank})")
print(f"Held-out RMS error at k = {best_k}: {metrics['prms']:.3f}")
```

`prms` is the root mean squared error of the model's predictions for the probe
entries. Selecting by `metric="cost"` instead uses the variational cost (the
negative evidence lower bound) on the training entries and does not need a
probe set. The [model-selection concepts page](../concepts/model-selection.md)
compares the two.

## Inspect the selection trace

`trace` has one entry per fitted candidate:

```python
for entry in trace:
    marker = "  <- selected" if entry["k"] == best_k else ""
    print(
        f"k={entry['k']:2d}  probe RMS={entry['prms']:.3f}  "
        f"cost={entry['cost']:.1f}{marker}"
    )
```

The probe error falls until the generating rank is reached and then stays
flat: additional components neither help nor hurt prediction here, because
automatic relevance determination shrinks unneeded components towards zero.

## The selected model

With `return_best_model=True`, the model fitted at the selected count is
returned. To refit on all observed entries, including the probes, construct
the estimator directly:

```python
from vbpca_py import VBPCA

model = VBPCA(n_components=best_k, maxiters=200, random_state=0)
model.fit(X_obs)
```

## Compare predictions with the hidden values

The data are simulated, so the values hidden by the missingness pattern are
known. The fitted reconstruction predicts every entry, observed or not:

```python
X_hat = model.reconstruction_
missing = ~observed

rmse_missing = float(np.sqrt(np.mean((X_hat[missing] - X_complete[missing]) ** 2)))
rmse_observed = float(np.sqrt(np.mean((X_hat[observed] - X_complete[observed]) ** 2)))

print(f"RMSE on observed entries: {rmse_observed:.3f}")
print(f"RMSE on missing entries:  {rmse_missing:.3f}")
print(f"Noise standard deviation: {noise_std}")
```

Both errors are close to the noise standard deviation. The error on missing
entries is slightly larger, because those values did not inform the fit.

## Explained variance

```python
for i, ratio in enumerate(model.explained_variance_ratio_, start=1):
    print(f"PC{i}: {ratio:.3f}")
```

## Next steps

- [Preprocessing Mixed Data](preprocessing-mixed-data.md): categorical and
  continuous columns.
- [Cross-Validation & Diagnostics](cv-diagnostics.md): choose the number of
  components with K-fold cross-validation.
- [Model Selection concepts](../concepts/model-selection.md): metrics and
  selection rules.
