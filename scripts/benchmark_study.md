# Benchmark Study Workflow

This document describes the script-based benchmark workflow for comparing:

- `mean_pca`: `SimpleImputer(mean)` + `PCA`
- `mice_pca`: `IterativeImputer` + `PCA`
- `knn_pca`: `KNNImputer` + `PCA`
- `vbpca_vb_modern`: VBPCApy (`algorithm="vb"`, `compat_mode="modern"`)

## Fixed modeling settings

- Core study dataset set is explicitly fixed to 4 datasets:
	- `synthetic`
	- `diabetes`
	- `wine`
	- `genomics_like`
- VBPCA: runs model selection over all admissible component counts and uses the
	selected `k` for final reconstruction (`vbpca_model_selection=True`).
	Selection stops when the chosen endpoint metric first worsens from `k-1` to
	`k` (e.g., RMS increases), selecting the previous `k`.
- Selection workflow is two-stage to reduce repeated compute while keeping
	empirical selection:
	1. For each setting, run one anchor model-selection pass to estimate `q*`.
	2. For each replicate, run local model selection over `q* ± window`
	   (default `window=2`) instead of sweeping from 1 upward every time.
- Runtime control for model selection (default in study commands):
	- `vbpca_selection_patience=1`
	- `vbpca_selection_max_trials=8`
	- `vbpca_local_window=2` (robustness check around anchor `q*`)
	- genomics-like profile uses reduced `vbpca_maxiters` (`--vbpca-maxiters-genomics`).
- By default, the selected VBPCA `k` is also used by `mean_pca`, `mice_pca`,
	and `knn_pca`
	in the same replicate (`use_selected_k_for_all_methods=True`) so all methods
	are compared at the empirically selected component count.
- MICE (`IterativeImputer`):
	- `sample_posterior=False`
	- `initial_strategy="mean"`
	- `imputation_order="ascending"`
	- `skip_complete=True`
	- `keep_empty_features=True`
	- `max_iter` from CLI and `tol=1e-2` (default study setting)
	- if convergence warning occurs, automatically retries with higher iteration cap.
- KNN (`KNNImputer`):
	- `n_neighbors` from CLI (`--knn-neighbors`, default `5`)
	- enabled by default (`--include-knn` / `--no-include-knn`).

## Outputs

- `results/replicates.csv`: per-replicate metrics
- `results/summary.csv`: per-method aggregate statistics
- `results/pairwise_summary.csv`: paired delta statistics vs VBPCA
- `results/paper/`: table and figure artifacts for manuscript use

## Core Metrics

Per method and setting:

- `rmse` (held-out masked RMSE)
- `mae` (held-out masked MAE)
- `wall_time_sec`

Statistics reported:

- mean, median, standard deviation on all replications
- empirical percentile CI using 2.5% and 97.5% quantiles

Pairwise outputs (`comparator - vbpca_vb_modern`) include:

- delta mean/median/std
- delta 95% CI
- comparator and VBPCA win rates

## Data types and scaling

- Current benchmark datasets are continuous-valued tabular data:
	- synthetic low-rank Gaussian data
	- synthetic genomics-like data (`genomics_like`): high-dimensional, sparse-loadings latent structure
	- scikit-learn `diabetes` (continuous)
	- scikit-learn `wine` (continuous)
	- optional `breast_cancer` (continuous; out-of-core-set)
- No categorical/ordinal/mixed-feature datasets are included in this benchmark.
- Scaling is applied per replicate using package-native preprocessing:
	- `MissingAwareStandardScaler` fit on observed values only
	- same transform applied to truth and observed matrices
	- avoids leakage from held-out entries while keeping methods comparable.

## Run Commands

Pilot (quick validation):

```bash
just bench-study-pipeline
```

Full sweep:

```bash
just bench-study-full
just bench-study-summary
just bench-study-paper
```

Genetics-like large-loci wall-time pilot (synthetic, sparse missingness behavior, MICE disabled):

```bash
just bench-study-genetics-pilot
```

Core hotspot profiling for genetics-like sparse scale:

```bash
just core-perf-genetics
```

## Reproducibility Check

Run deterministic consistency validation (fixed seed schedule):

```bash
python scripts/validate_benchmark_reproducibility.py --n-jobs 1
```

Using `--n-jobs 1` is recommended for strict reproducibility checks.
