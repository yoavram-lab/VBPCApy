# VBPCApy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Variational Bayesian PCA (Illin and Raiko, 2010) with support for missing data, sparse masks, optional bias terms, and an orthogonal post-rotation to a PCA basis. The implementation follows the original MATLAB reference while adding Python-native APIs and fast C++ extensions for heavy routines.

## Statement of need

Missing values are common in scientific and industrial tabular datasets, but many PCA workflows either impute first (which can bias uncertainty estimates) or drop incomplete samples/features. VBPCApy provides a practical Variational Bayesian PCA implementation that models missingness directly, exposes uncertainty-aware outputs, and preserves compatibility with the legacy MATLAB full-rank PCA behavior.

This package targets researchers and practitioners who need:
- robust latent-factor modeling with incomplete observations,
- explicit uncertainty terms (posterior covariances, marginal variances), and
- a Python API suitable for reproducible pipelines while retaining a parity path to legacy workflows.

## Scope and reference behavior

VBPCApy implements the Illin & Raiko (2010) VB-PCA formulation with modern Python ergonomics. The default `compat_mode="strict_legacy"` preserves historical behavior; `compat_mode="modern"` is available for updated semantics in selected preprocessing/masking cases.

## Runtime backend selection

The package now uses optimized kernels by default when extension modules are available in your Python environment. You do not need to enable a special "fast mode".

- **Default behavior**: dense/sparse/noise update kernels are selected automatically from data and mask structure.
- **Fallback behavior**: if an extension cannot be imported (for example, partial build or unavailable binary), the equivalent NumPy/SciPy implementation is used automatically.
- **Behavioral compatibility**: `compat_mode` controls numerical compatibility semantics (`strict_legacy` vs `modern`) and is independent of whether C++ or Python kernels execute.
- **Thread control**: thread counts can be constrained with environment variables (for example `VBPCA_NUM_THREADS`, and operation-specific overrides used by some kernels).

In short: backend selection affects runtime, not model semantics.

## Features
- Variational Bayesian PCA on dense or sparse data with explicit missing-entry masks.
- Optional bias (per-feature mean) estimation and rotation to a PCA-aligned solution.
- Support for shared observation patterns to reuse factorizations and speed inference.
- Posterior covariances for scores and loadings; probe-set RMS for held-out validation.
- Direct access to reconstructions and per-entry marginal variances from the sklearn-like `VBPCA` estimator.
- C++ extensions via pybind11 for performance-critical routines.
- Missing-aware preprocessing utilities (one-hot encode, standardize, min-max, auto-routing) that preserve NaNs/masks for generative reconstruction.
- `VBPCA` sklearn-like wrapper (fit/transform/inverse_transform) with mask support.
- Empirical risk minimization based model selector for number of PCs which best reconstruct the empirical data.

## Installation

### Requirements
- **Python**: >= 3.11
- **C++ Compiler**: C++14 compatible compiler (gcc, clang, MSVC)
- **Eigen**: Linear algebra library (version 3.x)

### Install Eigen (required for building)

**Ubuntu/Debian:**
```bash
sudo apt-get install libeigen3-dev
```

**macOS (Homebrew):**
```bash
brew install eigen
```

**Conda/Mamba:**
```bash
conda install -c conda-forge eigen
```

**Manual Installation:**
Download from [eigen.tuxfamily.org](https://eigen.tuxfamily.org/) and set the `EIGEN_INCLUDE_DIR` environment variable:
```bash
export EIGEN_INCLUDE_DIR=/path/to/eigen3
```

Eigen is located automatically via `EIGEN_INCLUDE_DIR`, `$CONDA_PREFIX/include/eigen3`, `/opt/homebrew/include/eigen3`, `/usr/include/eigen3`, or `/usr/local/include/eigen3`.

### Install VBPCApy

**From source:**
```bash
git clone https://github.com/yoavram-lab/VBPCApy.git
cd VBPCApy
pip install .
```

**With optional dependencies:**
```bash
# Development tools (pytest, ruff, mypy, just)
pip install .[dev]
# Optional data utilities (pandas)
pip install .[data]
# Optional plotting utilities (matplotlib)
pip install .[plot]
# Optional Octave bridge (only needed to run MATLAB/Octave helpers/tests)
pip install .[octave]
# Install everything
pip install .[dev,data,plot,octave]
```

**Using uv (recommended for Python env management):**
```bash
# Sync core developer environment
uv sync --extra dev --extra data --extra plot

# Include optional Octave Python bridge packages
uv sync --extra dev --extra data --extra plot --extra octave
```

## Quick start
```python
import numpy as np
from vbpca_py import VBPCA, AutoEncoder

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
- `compat_mode`: compatibility policy for sparse empty-row/column handling (`strict_legacy` default, `modern` available).

Legacy option note:
- The project still accepts several MATLAB-compatibility-era option names.
- `autosave` and `filename` are currently accepted for compatibility but have no effect in the Python package.
- Pre-release cleanup will prefer removing no-op compatibility options while retaining options that still control behavior (`algorithm`, `rotate2pca`, `num_cpu`, probe settings, and convergence settings).

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

### API
All public APIs can be imported directly from `vbpca_py`:
```python
from vbpca_py import (
    VBPCA,
    select_n_components,
    SelectionConfig,
    AutoEncoder,
    MissingAwareOneHotEncoder,
    MissingAwareStandardScaler,
    MissingAwareMinMaxScaler,
)
```

**Core estimator:** `VBPCA(n_components, bias=True, maxiters=None, tol=None, verbose=0, **opts)` with `fit`, `transform`, `fit_transform`, `inverse_transform`, learned attributes (`components_`, `scores_`, `mean_`, `rms_`, `prms_`, `cost_`, `variance_`, `reconstruction_`, `explained_variance_`, `explained_variance_ratio_`), and `get_options()` to inspect merged defaults.

**Model selection:** `select_n_components(x, *, mask=None, components=None, config=None, **opts)` sweeps ks and returns `(best_k, best_metrics, trace, best_model)`. `components` defaults to `1..min(n_features, n_samples)`. `SelectionConfig(metric="prms"|"rms"|"cost", patience=None, max_trials=None, compute_explained_variance=True, return_best_model=False)` controls sweep stopping and retention. `**opts` flow through to `VBPCA`/`pca_full` (e.g., `maxiters`, `minangle`, `rmsstop`, `cfstop`, `earlystop`, `rotate2pca`). `trace` holds per-k endpoint metrics; `best_metrics` is the winning entry.

**Preprocessing:** missing-aware encoders and scalers
  - `MissingAwareOneHotEncoder`: categorical OHE respecting masks/NaNs; `handle_unknown="ignore"|"raise"`, optional mean-centering.
  - `MissingAwareStandardScaler` and `MissingAwareMinMaxScaler`: continuous scaling while ignoring masked entries.
  - `AutoEncoder`: column-wise router that applies the above per column with `cardinality_threshold` (integer columns with uniques <= threshold are treated as categorical), `continuous_scaler` (`"standard"` or `"minmax"`), `handle_unknown` (ignore vs raise unseen categories), `mean_center_ohe` (center one-hot columns), optional `column_types` override (force categorical/continuous per column), and `fit/transform/inverse_transform` for round-tripping mixed data with masks.

All options are consumed via the `VBPCA` estimator. Call `model.get_options()` after construction to view the merged defaults and your overrides. The canonical reference list lives in [src/vbpca_py/_pca_full.py](src/vbpca_py/_pca_full.py). See [src/vbpca_py/estimators.py](src/vbpca_py/estimators.py), [src/vbpca_py/model_selection.py](src/vbpca_py/model_selection.py), and [src/vbpca_py/preprocessing.py](src/vbpca_py/preprocessing.py) for the stable public APIs.

### Autoencoding workflow 

The package includes missing-aware preprocessing and an autoencoder-style inverse transform to map back to the original feature space:

```python
from vbpca_py import AutoEncoder, VBPCA

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
pip install -e . pytest-cov
```

The project uses [just](https://just.systems/) as a command runner. Install it separately:
```bash
# macOS
brew install just

# Linux
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin

# Or via cargo
cargo install just
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
Run tests with coverage:
```bash
just test-cov
```

Run performance benchmarks (excluded from default CI):
```bash
# full perf suite
just bench

# scaling-only suite
just bench-scale

# Python vs Octave compare suite
just bench-octave
```

Run the script-based comparison study (mean+PCA vs MICE+PCA vs VBPCA):
```bash
# quick pilot end-to-end (replicates + summaries + paper artifacts)
just bench-study-pipeline

# full sweep + aggregation + paper outputs
just bench-study-full
just bench-study-summary
just bench-study-paper

# deterministic reproducibility check for a fixed-seed pilot setting
just bench-study-repro
```

The study writes `results/replicates*.csv`, summary tables, and paper-ready artifacts under `results/paper/`. See `scripts/benchmark_study.md` for details.


### Full legacy parity test requirements
`just test` runs the standard suite and may skip optional Octave-backed parity tests if Octave tooling is unavailable.

To run **all** parity tests (including sparse MEX-backed regression paths), you must install:
- `octave`
- `octave-dev` (provides `mkoctfile` on Ubuntu/Debian)

Example (Ubuntu/Debian):
```bash
sudo apt-get install octave octave-dev
```

Then run:
```bash
just test-all
```

This command rebuilds host-specific MEX helpers in `tools/` before testing.

If you previously ran tests on another OS/architecture, clear stale binaries before rerunning:
```bash
just mex-clean
```
Run the entire validation pipeline (lint -> typecheck -> test):
```bash
just ci
```
Legacy MATLAB/Octave helpers (in `tools/`) are optional; they require Octave installed plus the `octave` extra if you want to call them from Python or run any Octave-dependent tests.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with clear commit messages
4. Run the test suite (`just ci`) to ensure all tests pass
5. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

## Community Guidelines

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and constructive in all interactions
- Follow the project's coding style (enforced by `ruff`)
- Write tests for new functionality
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.edge 

## Citation

If you use this package in your research, please cite:

**For the implementation:**
```bibtex
@software{vbpca_py2026,
  author = {Naim, Shany and Macdonald, Joshua and Ram, Yoav},
  title = {{VBPCApy}: Variational Bayesian PCA with Missing Data Support},
  year = {2026},
  url = {https://github.com/yoavram-lab/VBPCApy},
  version = {0.1.0},
}
```

**For the algorithm:**
```bibtex
@article{ilin2010practical,
  title={Practical Approaches to Principal Component Analysis in the Presence of Missing Values},
  author={Ilin, Alexander and Raiko, Tapani},
  journal={Journal of Machine Learning Research},
  volume={11},
  pages={1957--2000},
  year={2010}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## Acknowledgments

This implementation is based on the Variational Bayesian PCA algorithm described by Ilin and Raiko (2010), with inspiration from the original MATLAB reference implementation.
