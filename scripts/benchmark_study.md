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
	No floor is applied from requested `n_components`; selection is fully empirical.
- Selection workflow is two-stage to reduce repeated compute while keeping
	empirical selection:
	1. For each setting, run one anchor model-selection pass to estimate `q*`.
	2. For each replicate, run local model selection over `q* ± window`
	   (default `window=3`) instead of sweeping from 1 upward every time.
- Runtime control for model selection (default in study commands):
	- `vbpca_selection_patience=1`
	- `vbpca_selection_max_trials=0` in full run (disables hard trial cap)
	- `vbpca_local_window=3` (robustness check around anchor `q*`)
	- `vbpca_maxiters=80`
	- `vbpca_maxiters_genomics=80`
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

These outputs are generated artifacts and should not be committed to git; they
are reproducible from the `just` command recipes and deterministic seed policy.

Notable uncertainty fields:

- `vbpca_mean_variance`: per-replicate mean of marginal posterior variances.
- `vbpca_median_variance`: per-replicate median of marginal posterior variances.
- `vbpca_median_variance_holdout`: per-replicate median marginal variance on held-out entries.
- `vbpca_median_variance_observed`: per-replicate median marginal variance on observed entries.

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

Uncertainty artifacts in `results/paper/` are generated from
paired per-replicate median marginal variance values on observed vs held-out
entries:

- Figure 2 (`figure2_vbpca_uncertainty.png`): one panel per dataset with paired box plots by dataset-local setting ID.
- Table S0 (`tableS0_setting_key.csv`): setting ID lookup (`setting_id_dataset` = A, B, C, ... within each dataset; includes `missing_setting`).
- Table S5/S6: robust summaries (median, q25, q75, p95, max) split by entry subset (`Observed`, `Held-out`).
- Table S7 (`tableS7_vbpca_uncertainty_replicate_long.csv`): long-form replicate uncertainty values used by Figure 2.

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

The `bench-study-full` recipe is the locked publication/JOSS run profile
(`n_reps=400`, no max-trials cap, 80 VBPCA max iterations).

Genetics-like large-loci wall-time pilot (synthetic, sparse missingness behavior, MICE disabled):

```bash
just bench-study-genetics-pilot
```

Core hotspot profiling for genetics-like sparse scale:

```bash
just core-perf-genetics
```

Default `just` perf recipes now use staged presets for tuning (`quick`/`confirm`)
rather than always forcing 80 iterations. Use `--tuning-stage final` (or
explicit `--maxiters 80`) only for final acceptance checks.

For runtime-policy tuning sweeps, use staged iteration presets via
`scripts/profile_core_vbpca.py`:

- `--tuning-stage quick` → `maxiters=20` (default; fast screening)
- `--tuning-stage confirm` → `maxiters=40` (candidate validation)
- `--tuning-stage final` → `maxiters=80` (final acceptance)

`--maxiters` can still explicitly override the stage mapping.

To include low-overhead per-iteration phase timing summaries in the CSV output,
add `--collect-phase-timings`.

When a very-large sparse case (e.g., 5k+ features, ~98.5% missing) is neutral
or slightly slower under `runtime_tuning=safe`, prefer a runtime profile rule
that pins selected kernel thread settings for that workload bucket rather than
changing global defaults. Example profile fragment:

```json
{
	"schema_version": 1,
	"default_threads": {
		"num_cpu_rms": 2
	},
	"workload_rules": [
		{
			"match": {
				"is_sparse": true,
				"min_features": 4000,
				"min_observed": 2000000
			},
			"threads": {
				"num_cpu_loadings_update": 0,
				"num_cpu_noise_update": 0,
				"num_cpu_rms": 1
			}
		}
	]
}
```

To generate/update a profile JSON from the command line, use:

```bash
python scripts/build_runtime_profile.py \
	--out results/perf_baseline/runtime_profile.json \
	--default-num-cpu-rms 2
```

To append a workload rule seeded from a profiler CSV case row:

```bash
python scripts/build_runtime_profile.py \
	--base-profile results/perf_baseline/runtime_profile.json \
	--out results/perf_baseline/runtime_profile.json \
	--case-csv results/perf_baseline/core_vbpca_genetics_baseline.csv \
	--case-name sparse_genetics_5k \
	--rule-num-cpu-loadings-update 0 \
	--rule-num-cpu-noise-update 0 \
	--rule-num-cpu-rms 1
```

## Reproducibility Check

Run deterministic consistency validation (fixed seed schedule):

```bash
just bench-study-repro
```

Using `--n-jobs 1` is recommended for strict reproducibility checks.
