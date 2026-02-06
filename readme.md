# vbpca_py

Variational Bayesian PCA (Illin and Raiko, 2010) with support for missing data, sparse masks, optional bias terms, and an orthogonal post-rotation to a PCA basis. The implementation follows the original MATLAB reference while adding Python-native APIs and fast C++ extensions for heavy routines.

## Features
- Variational Bayesian PCA on dense or sparse data with explicit missing-entry masks.
- Optional bias (per-feature mean) estimation and rotation to a PCA-aligned solution.
- Support for shared observation patterns to reuse factorizations and speed inference.
- Posterior covariances for scores and loadings; probe-set RMS for held-out validation.
- C++ extensions via pybind11 for performance-critical routines.

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
from vbpca_py.pca_full import pca_full

# 50 features, 200 samples
x = np.random.randn(50, 200)

# Optional mask (1 = observed, 0 = missing); omit for fully observed data
mask = np.ones_like(x)

result = pca_full(x, n_components=5, mask=mask, bias=True, maxiters=100)

loadings = result["a"]       # shape (features, components)
scores = result["s"]         # shape (components, samples)
mu = result["mu"]            # feature means (bias)
rms = result["rms"]          # reconstruction RMS on the data
```

### Options highlights
- `mask` / `pattern_index`: handle missing entries and reuse observation patterns.
- `bias`: toggle mean estimation; `init`: control initial factors.
- `probe`: pass probe data/masks to monitor held-out RMS during fitting.
- `maxiters`, `tol`, `verbose`: convergence control and logging.
- `rotation`: final orthogonal rotation to a PCA-aligned solution.

See `src/vbpca_py/pca_full.py` for the full set of options.

## Testing and development
Run the test suite:
```bash
pytest -q
```

Legacy MATLAB/Octave helpers (in `tools/`) are optional; they require Octave installed plus the `octave` extra if you want to call them from Python or run any Octave-dependent tests.

## Citing
If you use this package, please cite Illin and Raiko (2010) and the forthcoming JOSS article for this implementation once available.
