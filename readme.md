# vbpca_py

Variational Bayesian PCA (Illin and Raiko, 2010) with support for missing data, sparse masks, optional bias terms, and an orthogonal post-rotation to a PCA basis. The implementation follows the original MATLAB reference while adding Python-native APIs and fast C++ extensions for heavy routines.

## Features
- Variational Bayesian PCA on dense or sparse data with explicit missing-entry masks.
- Optional bias (per-feature mean) estimation and rotation to a PCA-aligned solution.
- Support for shared observation patterns to reuse factorizations and speed inference.
- Posterior covariances for scores and loadings; probe-set RMS for held-out validation.
- Direct access to reconstructions and per-entry marginal variances from the sklearn-like `VBPCA` estimator.
- C++ extensions via pybind11 for performance-critical routines.
- Missing-aware preprocessing utilities (one-hot encode, standardize, min-max, auto-routing) that preserve NaNs/masks for generative reconstruction.
- `VBPCA` sklearn-like wrapper (fit/transform/inverse_transform) with mask support.
- Empirical risk minimzation based model selector for number of PCs which best reconstruct the empirical data.

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

# Inspect the resolved options (defaults + your overrides)
print(model.get_options())

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

### Convergence and stopping
Each fit (including every k tried in `select_n_components`) runs the PCA_FULL EM loop until one of these criteria triggers or `maxiters` is reached:
- Subspace angle below `minangle` (default `1e-8`).
- Probe RMS increase when `earlystop` is truthy.
- RMS plateau via `rmsstop = [window, abs_tol, rel_tol]` (default `[100, 1e-4, 1e-3]`, enabled). Meaning: compare the latest RMS to the value `window` iterations ago; stop if the absolute change is < `abs_tol` or, when finite, the relative change is < `rel_tol`.
- Cost plateau via `cfstop = [window, abs_tol, rel_tol]` (default `[]`, disabled). Same interpretation as `rmsstop` but on cost.
- Slowing-down guard when internal backtracking hits 40 steps.
- Hard cap `maxiters` (default 1000).

Notes:
- `niter_broadprior` (default 100) suppresses stopping messages while the broad-prior warmup runs when `use_prior` is on.
- All options are case-insensitive and passed through the `VBPCA` constructor (or forwarded by `select_n_components`).
- Reference implementation lives in [src/vbpca_py/_pca_full.py](src/vbpca_py/_pca_full.py) and [src/vbpca_py/_converge.py](src/vbpca_py/_converge.py).

### Public API surface
- Core estimator: `VBPCA(n_components, bias=True, maxiters=None, tol=None, verbose=0, **opts)` with `fit`, `transform`, `fit_transform`, `inverse_transform`, learned attributes (`components_`, `scores_`, `mean_`, `rms_`, `prms_`, `cost_`, `variance_`, `reconstruction_`, `explained_variance_`, `explained_variance_ratio_`), and `get_options()` to inspect merged defaults.
- Model selection: `select_n_components(x, *, mask=None, components=None, config=None, **opts)` sweeps ks and returns `(best_k, best_metrics, trace, best_model)`. `components` defaults to `1..min(n_features, n_samples)`. `SelectionConfig(metric="prms"|"rms"|"cost", patience=None, max_trials=None, compute_explained_variance=True, return_best_model=False)` controls sweep stopping and retention. `**opts` flow through to `VBPCA`/`pca_full` (e.g., `maxiters`, `minangle`, `rmsstop`, `cfstop`, `earlystop`, `rotate2pca`). `trace` holds per-k endpoint metrics; `best_metrics` is the winning entry.
- Preprocessing: missing-aware encoders and scalers
  - `MissingAwareOneHotEncoder`: categorical OHE respecting masks/NaNs; `handle_unknown="ignore"|"raise"`, optional mean-centering.
  - `MissingAwareStandardScaler` and `MissingAwareMinMaxScaler`: continuous scaling while ignoring masked entries.
  - `AutoEncoder`: column-wise router that applies the above per column with `cardinality_threshold`, `continuous_scaler` (`"standard"` or `"minmax"`), optional `column_types` override, and `fit/transform/inverse_transform` for round-tripping mixed data with masks.

All options are consumed via the `VBPCA` estimator. Call `model.get_options()` after construction to view the merged defaults and your overrides. The canonical reference list lives in [src/vbpca_py/_pca_full.py](src/vbpca_py/_pca_full.py). See [src/vbpca_py/estimators.py](src/vbpca_py/estimators.py) and [src/vbpca_py/preprocessing.py](src/vbpca_py/preprocessing.py) for the stable public APIs.

### Autoencoding workflow 

The package includes missing-aware preprocessing and an autoencoder-style inverse transform to map back to the original feature space:

```python
from vbpca_py.preprocessing import AutoEncoder

auto = AutoEncoder(cardinality_threshold=10, continuous_scaler="standard")
z = auto.fit_transform(x, mask=mask)          # encodes continuous + categorical
model = VBPCA(n_components=5, maxiters=100)
scores = model.fit_transform(z, mask=np.ones_like(z, dtype=bool))
z_recon = model.inverse_transform()
x_recon = auto.inverse_transform(z_recon)     # decode back to original space
```

## Testing and development
Install in developer mode:
```bash
pip install -e .
```
Run lint check:
```bash
just lint
```
Run type checking:
```bash
just typecheck
```
Run the test suite:
```bash
just test
```
Run the entire validation pipeline (lint -> typecheck -> test):
```bash
just ci
```
Legacy MATLAB/Octave helpers (in `tools/`) are optional; they require Octave installed plus the `octave` extra if you want to call them from Python or run any Octave-dependent tests.


## Citing
If you use this package, please cite Illin and Raiko (2010) and the forthcoming JOSS article for this implementation once available.
