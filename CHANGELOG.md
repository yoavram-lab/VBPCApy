# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `random_state` constructor kwarg on `VBPCA`: seeds parameter initialization and any auto-generated xprobe mask (`int`, `np.random.Generator`, or `None`, following the sklearn convention). Surfaced via `get_params()`/`set_params()`/`get_options()` (#109).

### Changed
- **Behavior change:** the default (`random_state=None`) now draws fresh entropy on every `fit()` call. Previously, default initialization was silently seeded with a fixed value regardless of configuration, so repeated fits produced identical results without any way to request a different draw. Pass `random_state=<int>` for reproducible runs (#109).
- `recommend_config()`'s `missingness` parameter now warns (`UserWarning`) when passed anything other than the default `"auto"`, instead of silently ignoring it. Recommendations are still bucketed by `p` only — the Option A trade study's example recommendations are too sparse per (p-bucket, missingness) cell (23 points across 3 p-buckets x 4 missingness categories) to bucket on responsibly without shipping unreplicated values (#110, see also #111).

## [0.3.0] - 2026-08-17

### Added
- `recommend_config(n, p, missingness="auto", priority="balanced")`: regime-aware recommended `VBPCA` hyperparameters distilled from the Option A regime-surrogate trade study. The dominant lever is a strong ARD loadings prior (`hp_va` ~0.65-0.75 vs. the library default 0.001), which drives correct rank recovery — the default prior recovers the true rank only ~33% of the time. Exposed as `vbpca_py.recommend_config`.
- `predictive_variance_` fitted attribute: reconstruction variance including observation noise (`variance_ + noise_variance_`). Prediction intervals built from `variance_` alone under-covered noisy held-out entries (~48-65% at nominal 95%); `predictive_variance_` restores coverage to ~94-96% (#104).
- scikit-learn estimator compatibility for `VBPCA` (`get_params`/`set_params`, cloning) (#103, closes #34).
- Configurable convergence-criterion ordering and per-criterion enable/disable via `criterion_order` and `convergence_criteria` constructor kwargs (#102, closes #101).
- Convergence diagnostics exposed as fitted attributes: `n_iter_`, `convergence_reason_`, `learning_curve_` on `VBPCA`, plus per-trial `n_iter`/`convergence_reason` in `select_n_components` traces (#100, closes #99).
- Public documentation site (MkDocs + GitHub Pages).

### Changed
- scikit-learn is now an optional dependency rather than a hard import (#105).

### Fixed
- mypy 2.3.0 strict-mode compatibility.

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
