# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `VBPCA` exposes its automatic relevance determination state:
  `prior_variances_`, `bias_prior_variance_`, `component_relevance_`,
  `effective_rank(threshold=0.01)`, and a per-iteration `prior_trace_` when
  fitted with `record_prior_trace=True` (#251).

### Fixed
- The ARD documentation described $V_a$ as precisions; it is a prior variance,
  and small values prune. The `hp_va`, `hp_vb`, `hp_v`, `niter_broadprior` and
  `va_init` docstrings now state how each enters the updates (#251).

## [0.4.3] - 2026-09-28

### Added
- Dense masked score and loading updates can use exact
  complete-minus-missing sufficient statistics. A conservative density- and
  rank-aware policy selects the formulation automatically and records the
  resolved modes in the runtime report (#210).
- `bias_update_order` selects the MATLAB-compatible mean-first iteration or a
  post-factor mean update that evaluates mean, RMS, and noise from one current
  factor state. Modern mode selects the post-factor order automatically (#222).
- Fitted estimators expose the resolved runtime policy through
  `runtime_policy_report_`, including worker limits and dense-kernel choices
  used by the fit (#211).
- `check_data()` accepts an explicit observation mask, so preprocessing
  diagnostics use the same missing-data contract as model fitting (#213).
- New genomics-dosage and sparse-CSR tutorials document missing-value semantics,
  filtering, scaling, and memory-aware workflows (#181, #213).

### Changed
- `recommend_config()` now selects post-factor mean/RMS ordering after the
  registered modern-defaults study found equal or lower rank error with
  effectively unchanged prediction and calibration. The estimator's
  strict-legacy default remains unchanged (#214, #226, #237).
- Explained-variance diagnostics use fitted low-rank factors instead of forming
  a dense reconstruction, reducing peak memory and work while preserving the
  reported quantity (#212).
- Runtime tuning respects scheduler CPU allocations and reuses one resolved
  policy across compatible component candidates, avoiding oversubscription and
  repeated calibration during rank sweeps (#211).
- Dense masked updates reuse compact masks and workspaces, write covariance
  results directly into NumPy buffers, and select exact
  complete-minus-missing sufficient statistics when density and rank make that
  formulation favorable (#208, #209, #210).

### Fixed
- Bias and centered-data state now persist across iterations, absent means are
  initialized from observed row means, and explicit dense masks exclude masked
  values from that initialization (#222).
- Native extension worker limits now follow the resolved scheduler allocation
  consistently across dense, sparse, rotation, noise, and subtraction kernels
  (#211).
- Registered study manifests retain LF line endings on every platform, keeping
  their byte-level provenance hashes stable in Windows wheel builds (#241).
- CPU-allocation tests isolate every host CPU-count source, including
  `os.process_cpu_count()`, so wheel validation is portable across runners
  (#243).
- RMS autotune threshold tests now isolate the effective CPU-count helper so
  host process limits cannot alter their synthetic cap (#245).

## [0.4.2] - 2026-09-27

### Fixed
- Dense masked runtime autotuning now falls back to the already resolved thread
  policy when an optional benchmark probe encounters a numerical factorization
  failure. The fit continues, the runtime report records the fallback, and no
  failed profile rule is saved (#203).

## [0.4.1] - 2026-09-27

### Added
- The package exports `RANK_ZERO_SELECTION_SUPPORTED` so downstream tools can
  require the explicit mean-only model-selection contract without inspecting
  private APIs (#201).
- Component sweeps and entry-wise cross-validation accept an explicit rank-zero
  candidate, evaluated as a dense mean-only model with held-out predictive RMS.
  Existing defaults remain positive-rank only; variational-cost selection rejects
  rank zero because the objectives are not comparable (#199).
- Convergence-validation manifests now include the exact policy resolved by
  `recommend_config()` and leave-one-criterion-out ablations, while shards
  retain compact versioned learning curves for offline policy replay. Legacy
  version 1 manifests remain readable (#192).

### Fixed
- Entry-wise component cross-validation now reserves a minimum bipartite edge
  cover as permanent training support, then partitions the remaining entries
  across folds. This guarantees that every non-empty row and column remains
  estimable even for large masks with many degree-two samples, where random
  retry-based fold assignment could fail deterministically. Entries in the
  support backbone are not used as validation targets (#187).


## [0.4.0] - 2026-09-25

### Added
- `random_state` constructor kwarg on `VBPCA`: seeds parameter initialization and any auto-generated xprobe mask (`int`, `np.random.Generator`, or `None`, following the sklearn convention). Surfaced via `get_params()`/`set_params()`/`get_options()` (#109).
- `cross_validate_components()` results now retain per-fold iteration counts,
  convergence flags/reasons, and candidate-level convergence summaries so
  callers can audit CV fit quality and iteration-budget saturation (#159).

### Fixed
- Convergence-margin confidence intervals now use a paired, regime-stratified
  bootstrap that preserves the fixed validation design instead of allowing
  pooled bootstrap draws to change the regime composition (#172).
- Dense probe generation now copies an explicit observation mask before
  intersecting it with finite data, so caller-owned and read-only masks are
  preserved (#157).
- Recommendation routing now evaluates both aspect-ratio directions before the
  absolute feature-count guard, so large-but-tall matrices no longer receive a
  wide-data configuration. Large balanced matrices have a distinct
  `large_scale` routing label tied to the single-cell validation regime (#142).
- Warmup/cap warnings now use resolved options, so an explicit `maxiters` that
  is no larger than the default `niter_broadprior` can no longer fail silently
  (#133).
- Convergence plateaus and relative-change rules now reject worsening RMS or
  variational-free-energy trajectories. Curvature stopping additionally
  requires a small first-order slope, so constant steep improvement cannot be
  mistaken for convergence (#141).
- Probe early stopping now restores the model state with the lowest observed
  probe RMS and exposes its iteration and metric. Cost plateau, relative, and
  curvature rules also record separate diagnostic traces (#141).
- Cross-validation now seeds both fold assignment and candidate fits, rejects
  folds that would empty a training row or column, validates small observation
  counts, and refuses to densify sparse inputs silently (#144).
- Component selection now evaluates the requested metric exactly, records cost
  without silently enabling cost-based convergence, and stops after exactly
  the configured number of consecutive non-improving candidates (#140).
- Probe holdouts now remain excluded from training when callers provide an
  explicit observation mask. Sparse probe matrices are accepted without
  densification, and mask-aware probe generation preserves explicitly observed
  zero values (#145).
- `select_n_components()` now respects a caller-supplied `xprobe_fraction` when auto-generating a held-out probe set for the `"prms"`/`"cost"` selection metrics. Previously `_ensure_metric_opts` ignored it entirely and always used a hardcoded 10% probe fraction, regardless of what `xprobe_fraction` (e.g. from `recommend_config()`) was passed in — silently training every model-selection candidate on less data than the caller configured (#122).
- `select_n_components()` no longer invents a 10% probe holdout for `"cost"` or `"rms"` selection when the caller supplied neither `xprobe` nor a positive `xprobe_fraction`. Explicit probe settings are still honored for convergence diagnostics, while `"prms"` retains its historical 10% fallback (#125).
- Convergence-angle logging now handles complete ARD pruning without reducing an empty principal-angle array. Stable empty subspaces have angle zero; a transition to or from an empty subspace uses the maximal angle so it cannot trigger premature angle convergence (#128).
- `select_n_components()` now passes its prepared probe mask explicitly into each candidate fit and seeds that mask from the caller's `random_state`. This prevents a second, differently seeded holdout inside `VBPCA.fit()` and makes a one-candidate sweep match the corresponding direct fit (#130).

### Changed
- Four coarse `recommend_config()` buckets now use validated convergence
  margins instead of suppressing every convergence stop through their full
  iteration budget: `wide_moderate` uses zero warmup/1600 iterations,
  `tall_moderate` zero/400, `tall_extreme` zero/800, and `large_scale`
  zero/400. Across 120 preregistered paired fits, exact rank recovery improved
  from 82.5% to 90.0%, rank MAE from 1.058 to 0.183, and selected-fit budget
  hits fell from 100% to 5%; difficult wide and tall-extreme MNAR fits can
  still reach the cap (#133, #166, #168).
- `cross_validate_components()` now accepts held-out probe RMS (`"prms"`) as
  its sole selection objective. Variational cost remains available in returned
  diagnostics but is no longer mislabeled as a held-out CV metric (#144).
- The inert `VBPCA(tol=...)` compatibility parameter is deprecated and warns
  on `fit()`. Configure `rmsstop`, `cfstop_rel`, or `minangle` explicitly
  instead (#140).
- **Behavior change:** the default (`random_state=None`) now draws fresh entropy on every `fit()` call. Previously, default initialization was silently seeded with a fixed value regardless of configuration, so repeated fits produced identical results without any way to request a different draw. Pass `random_state=<int>` for reproducible runs (#109).
- `recommend_config()`'s `missingness` parameter now warns (`UserWarning`)
  when passed anything other than the default `"auto"`, instead of silently
  ignoring it. Recommendations currently use matrix shape, not missingness;
  the trade-study design is too sparse within missingness-by-shape cells to
  branch on responsibly (#110, see also #111).
- `defaults.py`'s docstring corrected the unsupported "`hp_va` is the dominant lever" claim (the trade study's own marginal sensitivity data doesn't support it) and now documents a real validation: replicated (n_reps=8, seeded) rank_mae for the shipped bucket configs is 28-58% lower than the library default across all three p-buckets, at a 0.4-3.8% cost in holdout RMSE (#111).
- `recommend_config()` now warns when a requested matrix shape falls outside
  the dense Option A validation grid instead of silently extrapolating a
  feature-count bucket (#116).
- `recommend_config()` now routes out-of-grid shapes to five coarse buckets:
  `wide_moderate`, `wide_extreme`, `tall_moderate`, `tall_extreme`, and
  `large_scale`. Each is derived from adaptive search at one representative
  regime, so the function warns that these are coarser than the replicated
  `smallp`/`trans`/`large` recommendations (#120, #142).

## [0.3.0] - 2026-08-17

### Added
- `recommend_config(n, p, missingness="auto", priority="balanced")`:
  regime-aware `VBPCA` configurations distilled from the Option A
  regime-surrogate trade study. Exposed as `vbpca_py.recommend_config`.
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

## [0.1.1] - 2026-04-02

### Added
- Python 3.14 wheels and CI coverage.

### Fixed
- Package `__version__` is now read from distribution metadata instead of a
  separately maintained constant.

## [0.1.0] - 2026-03-31

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
