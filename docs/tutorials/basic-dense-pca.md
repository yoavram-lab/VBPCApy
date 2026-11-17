# Basic Dense PCA

This tutorial fits a variational Bayesian PCA model to a complete dense
matrix with known low-rank structure, then inspects the fitted components,
reconstruction and convergence diagnostics.

## Generate synthetic data

Rows are samples and columns are features, as in scikit-learn. The matrix is
the product of five latent scores per sample and five loadings per feature,
plus Gaussian noise, so the fitted model can be compared with known structure.

```python
import numpy as np

rng = np.random.default_rng(42)

n_samples, n_features, true_rank = 200, 50, 5
noise_std = 0.3

scores_true = rng.standard_normal((n_samples, true_rank))
loadings_true = rng.standard_normal((true_rank, n_features))
noise = noise_std * rng.standard_normal((n_samples, n_features))
X = scores_true @ loadings_true + noise
```

## Fit the model

```python
from vbpca_py import VBPCA

model = VBPCA(n_components=5, maxiters=200, random_state=0)
model.fit(X)
```

`random_state` fixes the random initialization, so repeated fits give the same
result. `maxiters` caps the number of update iterations; the fit stops earlier
when a convergence criterion is met.

## Inspect the results

### Fitted attributes

```python
print("Components:", model.components_.shape)  # (n_components, n_features)
print("Scores:", model.scores_.shape)  # (n_samples, n_components)
print("Mean:", model.mean_.shape)  # (n_features,)
print("Noise variance:", round(float(model.noise_variance_), 4))
```

`scores_` holds the posterior means of the latent scores estimated jointly with
the loadings during fitting. `noise_variance_` is the fitted variance of the
Gaussian observation noise; here it should be close to
`noise_std ** 2 = 0.09`.

!!! note "`fit_transform` and `scores_`"
    Following scikit-learn, `model.fit_transform(X)` is `model.fit(X).transform(X)`.
    `transform` projects rows onto the fitted components with the loadings held
    fixed, so its result can differ slightly from `scores_`, which was estimated
    together with the loadings. [Known limitations](../limitations.md)
    describes the difference.

### Explained variance

```python
evr = model.explained_variance_ratio_
for i, ratio in enumerate(evr, start=1):
    print(f"PC{i}: {ratio:.3f}")
```

Each ratio is the fraction of the variance of the fitted reconstruction that
the component accounts for, so the ratios sum to one over the fitted
components.

### Reconstruction quality

```python
reconstruction = model.reconstruction_
rmse = float(np.sqrt(np.mean((X - reconstruction) ** 2)))
print(f"Reconstruction RMSE: {rmse:.3f} (noise standard deviation {noise_std})")
```

`reconstruction_` equals `model.inverse_transform(model.scores_)`. Its error
against the data is close to the noise standard deviation, because the five
components capture the low-rank signal and leave the noise.

### Convergence diagnostics

```python
print("Iterations:", model.n_iter_)
print("Converged:", model.converged_)
print("Stopping criterion:", model.convergence_reason_)
print(f"Training RMS: {model.rms_:.4f}")
```

`convergence_reason_` names the criterion that stopped the fit, or
`"maxiters"` when the iteration cap was reached first. A criterion stopping
the fit is a heuristic signal; the [convergence guide](../howto/convergence.md)
describes how to check that a fit has stabilized.

## Resolved options

`get_options()` returns every solver setting, with defaults filled in:

```python
options = model.get_options()
for key in ("maxiters", "bias", "rotate2pca", "compat_mode"):
    print(f"{key}: {options[key]}")
```

## Next steps

- [Missing Data & Model Selection](missing-data-model-selection.md): fit
  incomplete data and choose the number of components.
- [VBPCA API reference](../api/vbpca.md): constructor parameters and fitted
  attributes.
