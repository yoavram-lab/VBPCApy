# Data orientation and observations

Native `vbpca_py.VBPCA` consumes **samples x features**. The explicit
`vbpca_py.legacy.VBPCA` adapter and low-level `pca_full` retain **features x
samples**. Axes are selected by API, never inferred from matrix dimensions.
See the [0.5 migration guide](../api/migration-05.md) for the transition.

| Boundary | Input | Latent output | Components | Mean |
| --- | --- | --- | --- | --- |
| Native estimator | `(n_samples, n_features)` | `(n_samples, q)` | `(q, n_features)` | `(n_features,)` |
| Legacy adapter | `(n_features, n_samples)` | `(q, n_samples)` | `(n_features, q)` | historical column layout |
| Low-level solver | `(n_features, n_samples)` | result `S: (q,n_samples)` | result `A: (n_features,q)` | historical result keys |

The native estimator defaults to `compat_mode="modern"`. The legacy adapter
and low-level solver retain `strict_legacy` as their default. Selecting an
explicit compatibility mode also selects its documented numerical and
observation semantics. Frozen environments retain their registered versions.

## Dense observations

Under modern semantics, a finite numeric zero is observed and NaN is missing.
An entry-wise mask passed as `mask=...` marks observations in the same layout
as the input. A masked placeholder is excluded before projection or fitting.
Infinity at an observed position is rejected. A native fit rejects entirely
unobserved features; individual missing sample rows are supported.

Preprocessors consume samples x features. They preserve missingness through
NaNs, so native estimator composition needs no transpose:

```python
import numpy as np
from sklearn.pipeline import Pipeline
from vbpca_py import MissingAwareStandardScaler, VBPCA

X = np.random.default_rng(23).normal(size=(12, 4))
X[0, 0] = np.nan
pipeline = Pipeline([
    ("scale", MissingAwareStandardScaler()),
    ("pca", VBPCA(2, maxiters=4, niter_broadprior=0, verbose=0,
                  random_state=23, num_cpu=1, runtime_tuning="off")),
])
Z = pipeline.fit_transform(X)
assert Z.shape == (12, 2)
reconstruction = pipeline.inverse_transform(Z)
assert reconstruction.shape == X.shape
```

A raw mask cannot be blindly forwarded across an encoder that changes feature
width. The mask-routing and complete preprocessor conformance work is tracked
by #178/#36. Direct preprocessing fits accept `y=None` and keyword-only `mask`.

## Sparse observations

Sparse **stored entries are observed**, including stored zeros; absent entries
are missing. Construct sparse coordinates explicitly when zero observations
must be retained. Constructing CSR directly from a dense table generally drops
numeric zeros and changes the observation set. Converting/slicing a sparse
container must preserve stored zeros to retain this contract.

Native direct calls also accept explicit sparse masks, or a dense mask within
the configured budget. A true mask coordinate absent from input storage means
an observed zero value; it need not equal `spones(X)`. Duplicate mask entries
combine as Boolean observations. NaN values remain missing, and entries outside
the mask are excluded. CSR matrices and supported SciPy sparse arrays normalize
to the same checked CSR boundary.

Automatic native probes partition the observation coordinates without removing
unselected stored zeros. Sparse explicit probes retain their stored zero values.
A fully missing new row projects to the latent prior; an entirely unobserved
scoring table is rejected.

## Training scores, projection and selection

Native `scores_` stores joint variational training means. `transform(X_new)`
conditions new rows under the fitted state. `fit_transform(X)` is exactly
`fit(X).transform(X)`; it need not equal `scores_`. `transform_posterior` exposes
conditional score covariances, and `inverse_transform(Z)` returns samples x
features, including rank-zero mean-only models.

The existing cellwise selection helpers currently retain their feature-first
boundary and compatibility-model return values while #180 integration proceeds.
Transpose explicitly at that documented boundary, rather than changing the
layout used by native estimators or preprocessors. Their PRMS/Brier selection
target differs from native row-wise marginal Gaussian scoring.
