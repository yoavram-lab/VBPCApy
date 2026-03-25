# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-08

### Added
- Core `VBPCA` estimator with sklearn-like `fit`/`transform`/`inverse_transform` API.
- Support for dense and sparse (CSR/CSC) data with explicit missing-entry masks.
- Optional bias estimation and orthogonal post-rotation to PCA basis.
- Posterior covariances for scores and loadings; probe-set RMS for held-out validation.
- C++ extensions via pybind11/Eigen for performance-critical dense, sparse, noise, and rotate kernels.
- Runtime autotuning: thread counts, buffered accessors, and covariance writeback mode selection.
- `select_n_components` model selection with configurable metric, patience, and early stopping.
- `SelectionConfig` dataclass for sweep control.
- Missing-aware preprocessing: `AutoEncoder`, `MissingAwareOneHotEncoder`, `MissingAwareStandardScaler`, `MissingAwareMinMaxScaler`.
- `MissingAwareSparseOneHotEncoder` for sparse categorical encoding preserving CSR structure.
- Optional plotting utilities (`vbpca_py.plotting`): `scree_plot`, `loadings_barplot`, `variance_explained_plot`.
- Property-based tests (hypothesis), integration round-trip tests, and missing-data edge-case tests.
- GitHub Actions CI: lint, format check, mypy --strict, pytest with coverage across Python 3.11–3.13.
- `justfile` command runner with recipes for dev, test, benchmark, and CI workflows.
- CITATION.cff for machine-readable citation metadata.
- CONTRIBUTING.md with developer guidelines.
