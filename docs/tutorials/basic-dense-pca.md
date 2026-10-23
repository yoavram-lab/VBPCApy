# Basic Dense PCA

These feature-first examples use the explicit pre-0.5 compatibility adapter.
For the native samples-first estimator, see the [0.5 migration guide](../api/migration-05.md).

This tutorial walks through the simplest VBPCApy workflow: fitting a
Variational Bayesian PCA model to fully-observed dense data.

## Generate synthetic data

We create a low-rank matrix with known structure so we can verify the model
recovers it.

```python
import numpy as np

rng = np.random.default_rng(42)

n_features, n_samples, true_rank = 50, 200, 5
noise_std = 0.3

# True latent factors
W_true = rng.standard_normal((n_features, true_rank))
S_true = rng.standard_normal((true_rank, n_samples))
X = W_true @ S_true + noise_std * rng.standard_normal((n_features, n_samples))
```

!!! note "Legacy-adapter features × samples"
    The explicit compatibility adapter used here expects each **column** to be
    one observation. The native 0.5 estimator follows scikit-learn's
    row-per-sample convention. See
    [Data Convention](../concepts/data-convention.md) for details.

## Fit the model

```python
from vbpca_py.legacy import VBPCA

model = VBPCA(n_components=5, maxiters=200, verbose=1)
scores = model.fit_transform(X)
```

`fit_transform` runs the EM algorithm and returns the score matrix
$S \in \mathbb{R}^{K \times N}$.

## Inspect the results

### Learned attributes

```python
print("Components shape:", model.components_.shape)   # (50, 5)
print("Scores shape:", model.scores_.shape)            # (5, 200)
print("Mean shape:", model.mean_.shape)                # (50,)
print("Noise variance:", model.noise_variance_)
```

### Explained variance

```python
evr = model.explained_variance_ratio_
for i, v in enumerate(evr):
    print(f"  PC{i+1}: {v:.3f}")
print(f"  Total: {sum(evr):.3f}")
```

### Reconstruction quality

```python
recon = model.inverse_transform()
rmse = float(np.sqrt(np.mean((X - recon) ** 2)))
print(f"Reconstruction RMSE: {rmse:.4f}  (noise σ = {noise_std})")
```

The reconstruction RMSE should be close to the noise standard deviation,
confirming the model has recovered the underlying signal.

### Convergence diagnostics

```python
print(f"Final RMS:  {model.rms_:.6f}")
print(f"Final cost: {model.cost_:.2f}")
```

## Resolved options

Call `get_options()` to see all defaults merged with your overrides:

```python
opts = model.get_options()
for k, v in sorted(opts.items()):
    print(f"  {k}: {v}")
```

## Next steps

- [Missing Data & Model Selection](missing-data-model-selection.md) — handle
  incomplete observations and choose $k$ automatically.
- [VBPCA API reference](../api/vbpca.md) — full list of constructor parameters and
  learned attributes.
