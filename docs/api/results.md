# Fit results and learning curves

The low-level `pca_full()` function returns a regular dictionary with the
historical keys. Its return annotation is `PCAResult`, available from
`vbpca_py.results`. The type describes the current runtime contract; it does
not wrap or copy the fitted arrays.

```python
import numpy as np

from vbpca_py._pca_full import pca_full
from vbpca_py import PCAResult

data = np.array([[1., 2., 3.], [3., 1., 2.]])
result: PCAResult = pca_full(
    data, n_components=1, maxiters=2, niter_broadprior=0, random_state=4
)
loadings = result["A"]  # features by components
scores = result["S"]    # components by samples
```

This two-iteration example demonstrates the return structure; a scientific
fit needs separate convergence checks.

`Xrec`, `Vr`, `Vpred`, `ExplainedVar` and `ExplainedVarRatio` are `None` when
diagnostics are disabled. Covariances can be a sequence of blocks or a stacked
array; `Isv`, when present, maps samples to shared score covariance blocks.
Float-array aliases specify float64 dtype. Pattern indices use NumPy's default
integer dtype, which can depend on platform and NumPy version. Shape and ownership requirements
remain documented runtime contracts, rather than checks performed by typing.

`LearningCurve` describes both metric histories and fit metadata. For example,
`rms` is a list of floats, `returned_iteration` is an integer,
`convergence_reason` is a string and optional `prior_trace` is a list of
vectors. The estimator exposes this mapping through `learning_curve_` after
fitting. Optional fields can be absent; use `get()` when reading them.

```python
curve = result["lc"]
returned_iteration = curve.get("returned_iteration")
best_probe_rms = curve.get("best_probe_rms")
```

Strict type checking checks example consumers as well as the library. The test
suite runs the result and `pca_full` docstring examples and checks that unknown
result keys and incorrectly typed metadata produce type-checking errors.

::: vbpca_py.results
    options:
      docstring_style: google
      members:
        - PCAResult
        - LearningCurve
        - CovarianceResult
        - HyperparameterResult
