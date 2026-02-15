# Benchmark Study Workflow

This document describes the script-based benchmark workflow for comparing:

- `mean_pca`: `SimpleImputer(mean)` + `PCA`
- `mice_pca`: `IterativeImputer` + `PCA`
- `vbpca_vb_modern`: VBPCApy (`algorithm="vb"`, `compat_mode="modern"`)

## Fixed modeling settings

- VBPCA: runs model selection over all admissible component counts and uses the
	selected `k` for final reconstruction (`vbpca_model_selection=True`).
	Selection stops when the chosen endpoint metric first worsens from `k-1` to
	`k` (e.g., RMS increases), selecting the previous `k`.
- By default, the selected VBPCA `k` is also used by `mean_pca` and `mice_pca`
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
	- scikit-learn `diabetes` (continuous)
	- scikit-learn `wine` (continuous)
	- optional `breast_cancer` (continuous)
- No categorical/ordinal/mixed-feature datasets are included in this benchmark.
- Scaling is applied per replicate using observed entries only:
	- column-wise z-score (`mean`/`std`) fit on observed values
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
