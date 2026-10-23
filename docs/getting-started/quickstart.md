# Quick Start

The native 0.5 estimator consumes **samples x features** and defaults
to modern observation semantics. Dense NaNs are missing; numeric zeros are
observed. For the pre-0.5 feature-first API, use the explicit
[legacy adapter](../api/migration-05.md#explicit-legacy-api-and-serialized-models).

## Fit, project and reconstruct

```python
import numpy as np
from vbpca_py import VBPCA

X = np.random.default_rng(23).normal(size=(12, 4))
X[0, 0] = np.nan
model = VBPCA(2, maxiters=4, niter_broadprior=0, verbose=0,
              random_state=23, num_cpu=1, runtime_tuning="off")
Z = model.fit_transform(X)
reconstruction = model.inverse_transform(Z)
assert Z.shape == (12, 2)
assert reconstruction.shape == X.shape
assert model.components_.shape == (2, 4)
```

These short fixed-iteration examples demonstrate the API. They do not establish
posterior convergence. `scores_` retains joint variational training means;
`fit_transform(X)` returns the same conditional projections as `fit(X).transform(X)`.

New samples use the fitted state without refitting:

```python
new_scores = model.transform(X[:3])
posterior = model.transform_posterior(X[:3])
assert new_scores.shape == (3, 2)
assert posterior.covariance.shape == (3, 2, 2)
```

## Explicit observations

Direct masks and probes follow the sample-first layout:

```python
mask = np.ones_like(X, dtype=bool)
mask[1, 2] = False
model.fit(X, mask=mask)
masked_scores = model.transform(X, mask=mask)
assert masked_scores.shape == (12, 2)
```

Sparse stored entries, including stored zeros, are observations; absent entries
are missing. Construct coordinates explicitly when zeros are observed:

```python
from scipy.sparse import csr_matrix

values = np.array([[0.0, 1.0, 2.0], [2.0, 0.0, 3.0], [1.0, 2.0, 0.0]])
rows, columns = np.indices(values.shape)
sparse = csr_matrix((values.ravel(), (rows.ravel(), columns.ravel())),
                    shape=values.shape)
sparse_model = VBPCA(1, maxiters=4, niter_broadprior=0, verbose=0,
                     random_state=23, num_cpu=1, runtime_tuning="off").fit(sparse)
assert sparse_model.transform(sparse).shape == (3, 1)
```

## Encode mixed original variables

Consume a raw-variable mask before expanding categories. The encoder and
native model both use samples × features; the inverse decoder accepts the
model's reconstructed encoded features directly.

```python
from vbpca_py import AutoEncoder, ObservationMasker

raw = np.array([["west", 10.], ["east", np.nan], ["north", 30.],
                ["west", 40.], ["east", 50.], ["north", 60.]], dtype=object)
raw_mask = np.ones(raw.shape, dtype=bool)
raw_mask[0, 0] = False
observed_raw = ObservationMasker().fit_transform(raw, mask=raw_mask)
encoder = AutoEncoder(column_types=["categorical", "continuous"])
encoded = encoder.fit_transform(observed_raw)
mixed_model = VBPCA(1, maxiters=4, niter_broadprior=0, verbose=0,
                    random_state=23, num_cpu=1, runtime_tuning="off").fit(encoded)
decoded = encoder.inverse_transform(
    mixed_model.inverse_transform(mixed_model.transform(encoded)))
assert encoded.shape == (6, 4)
assert np.isnan(encoded[0, :3]).all()
assert decoded.shape == raw.shape
```

The encoder inverse restores original variables and continuous input units.
Reconstruction predicts values; it does not restore missing placeholders.
For validation, fit preprocessing within each training split as in the
[executed gallery](../gallery/index.md).

A sparse constructor applied directly to a dense table can discard observed
zeros. See [data conventions](../concepts/data-convention.md) and the
[migration guide](../api/migration-05.md) for sparse masks, output budgets,
rank zero and serialized artifacts. Public component-selection helpers also
consume samples x features. The
[complete mixed-data Pipeline example](../tutorials/missing-aware-pipeline.md)
demonstrates raw-mask routing, row validation, GridSearchCV, named output,
new-row projection and inverse reconstruction. A width-changing encoder must
not receive a stale raw mask downstream.
