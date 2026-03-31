# VBPCApy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Variational Bayesian PCA (Ilin and Raiko, 2010) with support for missing data, sparse masks, optional bias terms, and an orthogonal post-rotation to a PCA basis. The implementation follows the original MATLAB reference while adding Python-native APIs, fast C++ extensions for heavy routines, and runtime autotuning for thread/accessor/covariance writeback modes.

## Statement of need

Missing values are common in scientific and industrial tabular datasets, but many analysis pipelines either impute first (which can mask uncertainty and affect downstream inference) or drop incomplete samples/features. This pattern is widespread across PCA, matrix factorization, and related dimensionality-reduction workflows. VBPCApy provides a practical Variational Bayesian PCA implementation that models missingness directly and exposes posterior uncertainty outputs (for example, marginal variances and covariance-derived diagnostics) alongside reconstructions. Relative to common impute-then-analyze workflows, this enables uncertainty-aware latent-factor analysis in a single reproducible Python API.

This package targets researchers and practitioners who need:
- robust latent-factor modeling with incomplete observations,
- explicit uncertainty terms (posterior covariances, marginal variances), and
- a Python API suitable for reproducible pipelines while retaining a parity path to legacy workflows.

## Scope and reference behavior

VBPCApy implements the Ilin & Raiko (2010) VB-PCA formulation with modern Python ergonomics. The default `compat_mode="strict_legacy"` preserves historical behavior; `compat_mode="modern"` is available for updated semantics in selected preprocessing/masking cases.

## Runtime backend selection

The package uses optimized c++ kernels.

- **Default behavior**: dense/sparse/noise/rotate kernels are selected automatically from data and mask structure.
- **Build requirement**: extension modules must be available in the installed package (source build or wheel).
- **Behavioral compatibility**: `compat_mode` controls numerical compatibility semantics (`strict_legacy` vs `modern`) and is independent of kernel dispatch.
- **Thread control**: thread counts can be constrained with environment variables (for example `VBPCA_NUM_THREADS`, and operation-specific overrides used by some kernels).

In short: backend selection affects runtime, not model semantics.

## Features
- Variational Bayesian PCA on dense or sparse data with explicit missing-entry masks.
- Optional bias (per-feature mean) estimation and rotation to a PCA-aligned solution.
- Support for shared observation patterns to reuse factorizations and speed inference.
- Posterior covariances for scores and loadings; probe-set RMS for held-out validation.
- Direct access to reconstructions and per-entry marginal variances from the sklearn-like `VBPCA` estimator.
- C++ extensions via pybind11 for performance-critical routines; runtime autotune selects thread counts, buffered accessors, and covariance writeback modes.
- Missing-aware preprocessing utilities (one-hot encode, standardize, min-max, auto-routing) that preserve NaNs/masks for generative reconstruction.
- scikit-learn-compatible `VBPCA` estimator (`fit`/`transform`/`inverse_transform`) with mask support.
- Empirical model selection for the number of latent components via `select_n_components`.

## Installation

### Requirements
- **Python**: >= 3.11
- **C++ Compiler**: C++14 compatible compiler (gcc, clang, MSVC)
- **Eigen**: Linear algebra library (version 3.x)
- **Matplotlib** *(optional)*: install via `pip install vbpca_py[plot]` for monitoring displays and plotting utilities

### Install Eigen (only needed for building from source)

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

**From PyPI** (recommended — pre-built wheels for Python 3.11–3.14, Linux/macOS/Windows):
```bash
pip install vbpca-py
```

**With plotting support:**
```bash
pip install vbpca-py[plot]
```

**From source** (requires Eigen and a C++14 compiler):
```bash
git clone https://github.com/yoavram-lab/VBPCApy.git
cd VBPCApy
pip install .
```

**With optional dependencies (from source):**
```bash
# Development tools (pytest, ruff, mypy, just)
pip install .[dev]
# Plotting utilities (matplotlib)
pip install .[plot]
# Optional data utilities (pandas)
pip install .[data]
# Benchmark + plotting stack (joblib, pandas, scikit-learn, seaborn)
pip install .[benchmark]
# Optional Octave bridge (only needed to run MATLAB/Octave helpers/tests)
pip install .[octave]
# Install everything
pip install .[dev,plot,data,benchmark,octave]
```

**Using uv (recommended for Python env management):**
```bash
# Sync core developer environment
uv sync --extra dev --extra data

# Include benchmark + plotting dependencies
uv sync --extra dev --extra data --extra benchmark

# Include optional Octave Python bridge packages
uv sync --extra dev --extra data --extra benchmark --extra octave
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

### Sparse data quick start
```python
import scipy.sparse as sp
from vbpca_py import VBPCA

# CSR input with explicit stored zeros where observed
x_sparse = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])

# Mask for sparse must match the sparsity pattern (spones(X)); omit to infer from X
mask = x_sparse.copy()
mask.data[:] = 1.0

model = VBPCA(n_components=2, maxiters=100)
scores = model.fit_transform(x_sparse, mask=mask)
```

- Sparse inputs must be CSR/CSC; they remain sparse throughout computation.
- For sparse data, the observation set is the stored entries of `X` (including stored zeros); if you pass a mask it must match `spones(X)` exactly.
- For dense data, pass a dense mask of 0/1 with the same shape; a mask is required for dense inputs when any missingness exists.
- `transform()` only returns training scores; it does not accept new data. Use `fit_transform` on the training set.
- To encode wide numeric categorical columns sparsely, use `MissingAwareSparseOneHotEncoder` (one column at a time, CSR input) and keep the mask `None`.


### Options highlights
- `mask` / `pattern_index`: handle missing entries and reuse observation patterns.
- `bias`: toggle mean estimation; `init`: control initial factors.
- `probe`: pass probe data/masks to monitor held-out RMS during fitting.
- `maxiters`, `tol`, `verbose`: convergence control and logging.
- `rotation`: final orthogonal rotation to a PCA-aligned solution.
- `compat_mode`: compatibility policy for sparse empty-row/column handling (`strict_legacy` default, `modern` available).
- `runtime_tuning` / `num_cpu`: runtime policy and threading controls. `runtime_tuning="safe"` (default) runs a short probe to pick threads, accessor mode (`legacy` vs `buffered`), and covariance writeback (`python`, `bulk`, `kernel`) based on measured speed. `runtime_tuning="aggressive"` tries a wider search. Per-kernel env vars include `VBPCA_NUM_THREADS`, `VBPCA_SCORE_THREADS`, `VBPCA_LOADINGS_THREADS`, `VBPCA_NOISE_THREADS`, `VBPCA_RMS_THREADS`.
- `auto_pattern_masked`: when true, reuse dense mask patterns even with `uniquesv=0` to reduce repeated per-column score covariance work (default off for parity).

### Runtime tuning and fast-mode sweep suggestions
- Default runtime behavior uses `runtime_tuning="safe"` to measure and choose `num_cpu`, accessor mode, and covariance writeback mode; you can still pin `num_cpu` explicitly or override with env vars.
- `runtime_tuning="aggressive"` expands the search if you want maximum throughput and can tolerate a slightly longer probe.
- Fast sweep preset: use `runtime_tuning="safe"`, `SelectionConfig(compute_explained_variance=False, patience=2, max_trials=5)`, and cap the k sweep to a modest window (e.g., 25–45 for tall/wide matrices).

### Public API policy
- Stable public imports are those re-exported from `vbpca_py` in [src/vbpca_py/__init__.py](src/vbpca_py/__init__.py).
- Modules and symbols prefixed with `_` are internal implementation details and may change without deprecation.
- For forward compatibility, prefer `from vbpca_py import ...` over importing from internal modules.

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
    MissingAwareSparseOneHotEncoder,
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

## Choosing sparse vs dense input

| Scenario | Format | Why |
|---|---|---|
| High-dimensional data with structural zeros (genomics count matrices, one-hot-encoded surveys) | **Sparse CSR/CSC** | The CSR sparsity pattern acts as an implicit observation mask; sparse kernels avoid materialising the full matrix. |
| Moderate dimensions with random missingness (NaN-masked tabular data) | **Dense + explicit mask** | Pass a boolean mask of observed entries; dense kernels benefit from BLAS. |

Key API difference:
- **Sparse:** observation mask is inferred from stored entries — `mask = spones(X)`.
- **Dense:** observation mask must be provided explicitly — `mask = ~np.isnan(X)`.

## Plotting utilities

Install the optional plotting extra:
```bash
pip install vbpca_py[plot]
# or
uv sync --extra plot
```

Three convenience functions are provided:
```python
from vbpca_py.plotting import scree_plot, loadings_barplot, variance_explained_plot

model = VBPCA(n_components=5, maxiters=100)
model.fit(x, mask=mask)

scree_plot(model, cumulative=True)           # explained variance per component
loadings_barplot(model, component=0, top_n=10)  # feature importance for one PC
variance_explained_plot(model)               # absolute variance per component
```

## Benchmarking

The project includes several benchmarking recipes via `just`:

| Recipe | Description |
|---|---|
| `just bench` | Kernel-level timing via pytest-benchmark |
| `just bench-scale` | Timing across increasing matrix sizes |
| `just bench-octave` | Python vs Octave comparison (requires `octave` and `uv sync --extra octave`) |
| `just bench-save` / `just bench-compare` | Save and compare baselines |
| `just bench-study-repro` | Validate deterministic reproducibility |

For Octave benchmarks, install Octave first:
```bash
# Ubuntu/Debian
sudo apt-get install octave octave-dev
# macOS
brew install octave
```

Then sync the Octave extra: `uv sync --extra octave`.

## Known limitations

- **`transform(new_data)` is not implemented.** Only training scores are returned. To project new data, refit on the combined dataset.
- **`inverse_transform()` always returns dense output**, even when the input was sparse CSR/CSC.
- **`MissingAwareSparseOneHotEncoder` requires numeric categories.** String categories cannot survive the CSR round-trip.
- **Data convention:** `AutoEncoder` expects samples × features; `VBPCA` expects features × samples. Transpose as needed.

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

List all available recipes:
```bash
just help
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

Validate deterministic reproducibility for a fixed-seed pilot setting:
```bash
just bench-study-repro
```

`pca_full(..., runtime_report=1)` can be used to include a `RuntimeReport`
diagnostic block showing resolved per-kernel thread values and their sources.


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

## Citation

If you use this package in your research, please cite:

**For the implementation:**
```bibtex
@software{vbpca_py2026,
  author = {Macdonald, Joshua and Naim, Shany and Ram, Yoav},
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation is based on the Variational Bayesian PCA algorithm described by Ilin and Raiko (2010), with inspiration from the original MATLAB reference implementation.
