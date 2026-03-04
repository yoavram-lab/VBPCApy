# Analysis Plan (runbook)

This outlines the end-to-end analyses and the Just targets to execute them.

## Targets
- `just analysis-impute`: Run missing-data benchmarks (core comparators: VBPCA, mean+PCA, MICE+PCA, KNN+PCA; missingness capped at 0.5 by default). Outputs: `results/replicates.csv`, `results/vbpca_selection_trace.csv`.
- `just analysis-dense-runtime`: Dense runtime scaling sweep. Outputs: `results/perf_baseline/dense_runtime.csv`.
- `just analysis-sparse-runtime`: Sparse explicit-mask runtime sweep. Outputs: `results/perf_baseline/sparse_mask_explicit.csv`.
- `just analysis-summary`: Summarize imputation benchmarks into aggregate stats and pairwise deltas. Outputs: `results/summary.csv`, `results/pairwise_summary.csv`.
- `just figures-tables`: Aggregate figures and tables from the above CSVs. Outputs under `results/figures_tables/figures` and `results/figures_tables/tables`.
- `just analysis-all`: Runs all of the above in sequence.

## Notes
- Missingness cap: `--max-missing-rate` defaults to 0.5 in `benchmark_missing_pca.py`; override if needed.
- Comparator presets: `--comparator-preset` in `benchmark_missing_pca.py` (`core`, `light`, `heavy`); defaults to `core`. Use `--include-*` flags to override.
- Seeds: Each script sets its own deterministic seeds; outputs record seeds and comparator flags in the CSVs. No separate manifest is written.
