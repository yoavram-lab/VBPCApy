# vbpca_py

Variational Bayesian PCA (Illin and Raiko, 2010) with support for missing data, sparse masks, optional bias terms, and an orthogonal post-rotation to a PCA basis. The implementation follows the original MATLAB reference while adding Python-native APIs and fast C++ extensions for heavy routines.

## Features
- Variational Bayesian PCA on dense or sparse data with explicit missing-entry masks.
- Optional bias (per-feature mean) estimation and rotation to a PCA-aligned solution.
- Support for shared observation patterns to reuse factorizations and speed inference.
- Posterior covariances for scores and loadings; probe-set RMS for held-out validation.
- Direct access to reconstructions and per-entry marginal variances from `pca_full` or `VBPCA`.
- C++ extensions via pybind11 for performance-critical routines.
- Missing-aware preprocessing utilities (one-hot encode, standardize, min-max, auto-routing) that preserve NaNs/masks for generative reconstruction.
- `VBPCA` sklearn-like wrapper for `pca_full` (fit/transform/inverse_transform) with mask support.

## Installation
Requirements: Python >= 3.11, a C++14 compiler, and Eigen headers. Eigen is located automatically via `EIGEN_INCLUDE_DIR`, `$CONDA_PREFIX/include/eigen3`, `/opt/homebrew/include/eigen3`, or `/usr/local/include/eigen3`.

```bash
pip install .
# Development tools
pip install .[dev]
# Optional data utilities
pip install .[data]
# Optional plotting utilities
pip install .[plot]
# Optional Octave bridge (only needed to run MATLAB/Octave helpers/tests)
pip install .[octave]
```

## Quick start
```python
import numpy as np
from vbpca_py.estimators import VBPCA
from vbpca_py.preprocessing import AutoEncoder

# 50 features, 200 samples
x = np.random.randn(50, 200)

# Optional mask (1 = observed, 0 = missing); omit for fully observed data
mask = np.ones_like(x)

model = VBPCA(n_components=5, maxiters=100)
scores = model.fit_transform(x, mask=mask)
recon = model.inverse_transform()

# Access reconstruction + marginal variance directly from the estimator
recon = model.reconstruction_
var = model.variance_

# Learning-curve summaries
print("RMS", model.rms_)
print("Probe RMS", model.prms_)
print("Final cost", model.cost_)

# Missing-aware preprocessing pipeline (categorical + continuous)
auto = AutoEncoder(cardinality_threshold=10, continuous_scaler="standard")
z = auto.fit_transform(x, mask=mask)
model = VBPCA(n_components=5, maxiters=100)
scores = model.fit_transform(z, mask=np.ones_like(z, dtype=bool))
z_recon = model.inverse_transform()
x_recon = auto.inverse_transform(z_recon)
```

### Options highlights
- `mask` / `pattern_index`: handle missing entries and reuse observation patterns.
- `bias`: toggle mean estimation; `init`: control initial factors.
- `probe`: pass probe data/masks to monitor held-out RMS during fitting.
- `maxiters`, `tol`, `verbose`: convergence control and logging.
- `rotation`: final orthogonal rotation to a PCA-aligned solution.

See [src/vbpca_py/_pca_full.py](src/vbpca_py/_pca_full.py) for the full set of options.
See `vbpca_py.estimators.VBPCA` for the stable public API.

### Direct access via `pca_full`

```python
from vbpca_py import pca_full

result = pca_full(x, 5, maxiters=200)

# Reconstruction and marginal variance
xrec = result["Xrec"]
vr = result["Vr"]

# Monitoring metrics
rms = result["RMS"]             # last lc["rms"]
probe_rms = result["PRMS"]      # last lc["prms"]
cost = result["Cost"]           # last lc["cost"] (NaN if not computed)
lc = result["lc"]               # full learning curves
```

## Testing and development
Run the test suite:
```bash
pytest -q
```

Legacy MATLAB/Octave helpers (in `tools/`) are optional; they require Octave installed plus the `octave` extra if you want to call them from Python or run any Octave-dependent tests.

## Citing
If you use this package, please cite Illin and Raiko (2010) and the forthcoming JOSS article for this implementation once available.
