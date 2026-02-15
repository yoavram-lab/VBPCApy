set shell := ["bash", "-lc"]

# Show available recipes.
help:
	just --list

# Install package + core dev tooling with pip.
dev-install:
	pip install -e . pytest-cov

# Sync a uv-managed environment (Python deps only).
uv-sync:
	uv sync --extra dev --extra data --extra plot

# Sync uv env including optional Octave Python bridge.
uv-sync-octave:
	uv sync --extra dev --extra data --extra plot --extra octave

# Lint Python sources (excluding tests) using Ruff preview rules.
lint:
	ruff check --preview --fix src

# Type-check library code.
typecheck:
	mypy src

# Run the test suite quietly.
test:
	pytest -q

# Run tests with coverage report.
test-cov:
	pytest -q --cov=src/vbpca_py --cov-report=term-missing

# Run pytest perf benchmarks (excluded from default test/ci runs).
bench:
	pytest -q -m perf --benchmark-only --benchmark-sort=mean

# Run only scaling benchmarks across increasing matrix sizes.
bench-scale:
	pytest -q -m perf --benchmark-only -k scaling --benchmark-sort=mean

# Run Python vs Octave comparison benchmarks (Octave test auto-skips if unavailable).
bench-octave:
	pytest -q -m perf --benchmark-only -k compare_dense --benchmark-sort=mean

# Save benchmark baselines per compat mode for later comparisons.
bench-save:
	pytest -q -m perf --benchmark-only --benchmark-save=compat

# Compare current benchmarks with the latest saved baseline.
bench-compare:
	pytest -q -m perf --benchmark-only --benchmark-compare --benchmark-sort=mean

# Run reproducible core VBPCA runtime baseline (dense + sparse cases).
core-perf-baseline:
	python scripts/profile_core_vbpca.py --reps 3 --maxiters 40 --out results/perf_baseline/core_vbpca_baseline.csv

# Run genetics-like large sparse hotspot profiling baseline.
core-perf-genetics:
	python scripts/profile_core_vbpca.py --case-set genetics --reps 2 --maxiters 30 --out results/perf_baseline/core_vbpca_genetics_baseline.csv

# Run script-based benchmark pilot (reduced reps for quick validation).
bench-study-pilot:
	python scripts/benchmark_missing_pca.py --datasets synthetic,diabetes,wine,genomics_like --n-reps 20 --mice-tol 1e-2 --vbpca-selection-patience 1 --vbpca-selection-max-trials 8 --vbpca-local-window 2 --vbpca-maxiters 60 --vbpca-maxiters-genomics 45 --n-jobs -2 --output results/replicates_pilot.csv --selection-trace-output results/vbpca_selection_trace_pilot.csv

# Run full script-based benchmark sweep (publication configuration).
bench-study-full:
	python scripts/benchmark_missing_pca.py --datasets synthetic,diabetes,wine,genomics_like --n-reps 200 --mice-tol 1e-2 --vbpca-selection-patience 1 --vbpca-selection-max-trials 8 --vbpca-local-window 2 --vbpca-maxiters 60 --vbpca-maxiters-genomics 45 --n-jobs -2 --output results/replicates.csv --selection-trace-output results/vbpca_selection_trace.csv

# Run genetics-like large-loci synthetic wall-time comparator sweep (MICE disabled).
bench-study-genetics-pilot:
	python scripts/benchmark_missing_pca.py --datasets genomics_like --mechanisms MCAR --patterns random --missing-rates 0.3 --n-reps 6 --n-components 8 --synthetic-shape 1200x5000 --vbpca-maxiters 60 --vbpca-maxiters-genomics 40 --vbpca-selection-patience 1 --vbpca-selection-max-trials 6 --vbpca-local-window 2 --no-include-mice --n-jobs -2 --output results/replicates_genetics_pilot.csv --selection-trace-output results/vbpca_selection_trace_genetics_pilot.csv

# Aggregate benchmark summaries and paired deltas.
bench-study-summary:
	python scripts/summarize_missing_pca.py --input results/replicates.csv --summary-output results/summary.csv --pairwise-output results/pairwise_summary.csv

# Build paper-facing tables and figures from benchmark outputs.
bench-study-paper:
	python scripts/make_paper_outputs.py --replicates results/replicates.csv --summary results/summary.csv --pairwise results/pairwise_summary.csv --out-dir results/paper
	python scripts/make_selection_supplement.py --replicates results/replicates.csv --selection-trace results/vbpca_selection_trace.csv --out-dir results/paper

# Run pilot + summary + paper output generation end-to-end.
bench-study-pipeline: bench-study-pilot
	python scripts/summarize_missing_pca.py --input results/replicates_pilot.csv --summary-output results/summary_pilot.csv --pairwise-output results/pairwise_summary_pilot.csv
	python scripts/make_paper_outputs.py --replicates results/replicates_pilot.csv --summary results/summary_pilot.csv --pairwise results/pairwise_summary_pilot.csv --out-dir results/paper/pilot
	python scripts/make_selection_supplement.py --replicates results/replicates_pilot.csv --selection-trace results/vbpca_selection_trace_pilot.csv --out-dir results/paper/pilot

# Validate deterministic reproducibility for a fixed-seed pilot setting.
bench-study-repro:
	python scripts/validate_benchmark_reproducibility.py --work-dir results/repro --n-jobs 1

# Ensure Octave + mkoctfile are installed for full legacy parity tests.
check-octave:
	command -v octave >/dev/null || (echo "Missing octave. Install Octave first." && exit 1)
	command -v mkoctfile >/dev/null || (echo "Missing mkoctfile. Install octave-dev (Linux) or Octave development tools." && exit 1)

# Remove host-specific compiled Octave artifacts.
mex-clean:
	rm -f tools/errpca_pt.mex* tools/subtract_mu.mex* tools/*.oct

# Build Octave MEX helpers used by full regression parity tests.
mex-build: check-octave mex-clean
	if cd tools && mkoctfile --mex -O -DNOTHREADS -o errpca_pt errpca_pt.cpp; then echo "Built tools/errpca_pt"; else echo "Warning: could not build tools/errpca_pt (optional true sparse parity may skip)."; fi
	if cd tools && mkoctfile --mex -O -o subtract_mu subtract_mu.cpp; then echo "Built tools/subtract_mu"; else echo "Warning: could not build tools/subtract_mu (sparse SubtractMu parity may fail/skip)."; fi

# Run all tests including Octave/MEX-backed regressions.
test-all: check-octave mex-build
	pytest -q tests

# Run lint, typecheck, and tests in sequence.
ci: lint typecheck test

# Run full CI including Octave parity coverage.
ci-all: lint typecheck test-all
