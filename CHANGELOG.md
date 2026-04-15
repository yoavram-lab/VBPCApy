# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-15

### Added
- Convergence overhaul: relative ELBO stopping, curvature stopping, composite convergence criteria, and configurable patience windows (#95).
- K-fold cross-validated model selection via `cross_validate_components` and `CVConfig` (#79).
- Preprocessing transforms: `MissingAwareLogTransformer`, `MissingAwarePowerTransformer`, `MissingAwareWinsorizer` (#82).
- Preflight data diagnostics via `DataReport` and `check_data()` (#82).
- Expose prior hyperparameters `hp_va`, `hp_vb`, `hp_v` as VBPCA constructor parameters (#87).
- Expose `niter_broadprior` on VBPCA constructor (#96).
- Expose `va_init` (initial broad prior value) on VBPCA constructor (#97).
- Expose `xprobe_fraction` for auto-generated holdout probe masks, and `make_xprobe_mask` utility (#98).
- Expose `xprobe` parameter in `VBPCA.fit()` for explicit probe data (#86).
- Store subspace angle in learning curves (`lc["angle"]`) (#90).
- GitHub issue templates for bugs, features, and documentation (#83).

### Fixed
- ARD stability with missing data: clamp per-iteration Va shrinkage rate and scale ARD denominator by observed-entry fraction (#86).
- `_marginal_variance` crash when `rmempty` drops columns (#74).
- `variance_` attribute now available on best model returned by `select_n_components` (#85).
- `'rms'` added to `_Metric` type hint in model selection (#57).

### Changed
- Skip octave-parity CI job when only irrelevant files changed (#88).
- Document RMS oscillation workaround (center data before fitting) in Known Limitations.

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
