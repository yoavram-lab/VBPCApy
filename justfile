set shell := ["bash", "-lc"]

# Show available recipes.
help:
	just --list

# Sync a uv-managed environment (Python deps only).
uv-sync:
	uv sync --extra dev --extra data --extra plot

# Sync uv env including optional Octave Python bridge.
uv-sync-octave:
	uv sync --extra dev --extra data --extra plot --extra octave

# Lint Python sources using Ruff preview rules.
lint:
	uv run ruff check --preview --fix src tests scripts analysis

# Check formatting without modifying files.
format-check:
	uv run ruff format --preview --check src tests scripts analysis

# Auto-format all Python sources.
format:
	uv run ruff format --preview src tests scripts analysis

# Type-check library code (strict mode).
typecheck:
	uv run --extra dev --extra plot mypy --strict src

# Run the test suite quietly.
test:
	uv run pytest -q

# Run tests with coverage report (fail under 89 %; Octave parity tests excluded).
test-cov:
	uv run --extra dev python -m pytest -q --cov=src/vbpca_py --cov-report=term-missing --cov-fail-under=89

# Run pytest perf benchmarks (excluded from default test/ci runs).
bench:
	uv run pytest -q -m perf --benchmark-only --benchmark-sort=mean

# Run a lightweight smoke subset (parity + sparse preprocessing).
test-smoke:
	uv run pytest -q tests/test_octave_parity_smoke.py tests/test_preprocessing_sparse.py

# Run only scaling benchmarks across increasing matrix sizes.
bench-scale:
	uv run pytest -q -m perf --benchmark-only -k scaling --benchmark-sort=mean

# Run Python vs Octave comparison benchmarks (Octave test auto-skips if unavailable).
bench-octave:
	uv run pytest -q -m perf --benchmark-only -k compare_dense --benchmark-sort=mean

# Save benchmark baselines per compat mode for later comparisons.
bench-save:
	uv run pytest -q -m perf --benchmark-only --benchmark-save=compat

# Compare current benchmarks with the latest saved baseline.
bench-compare:
	uv run pytest -q -m perf --benchmark-only --benchmark-compare --benchmark-sort=mean

# Run the quick-start missing-data example.
example:
	uv run python scripts/example_missing_pca.py

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
	uv run pytest -q tests

# Validate deterministic reproducibility for a fixed-seed pilot setting.
bench-study-repro:
	uv run --extra data python scripts/validate_benchmark_reproducibility.py

# Run format check, lint, strict typecheck, and tests in sequence.
ci: format-check lint typecheck test-cov

# Run lint, typecheck, and smoke tests only.
ci-smoke: lint typecheck test-smoke

# Run full CI including Octave parity coverage.
ci-all: format-check lint typecheck test-all

# ── Build & publish helpers ──────────────────────────────────────────

# Build sdist + wheel locally and check with twine.
build-check:
	rm -rf dist
	uv build
	uvx twine check --strict dist/*

# Clean-room install test: fresh venv, install wheel + test deps, run test suite.
build-test:
	#!/usr/bin/env bash
	set -euo pipefail
	CLEANROOM=$(mktemp -d)
	trap 'rm -rf "$CLEANROOM"' EXIT
	uv venv "$CLEANROOM/venv"
	source "$CLEANROOM/venv/bin/activate"
	uv build --wheel --out-dir "$CLEANROOM/dist"
	uv pip install --no-cache "$CLEANROOM"/dist/*.whl
	uv pip install pytest pytest-benchmark hypothesis numpy scipy matplotlib
	python -m pytest tests -q -x -m 'not perf and not octave' \
		--ignore=tests/test_octave_parity_smoke.py \
		--ignore=tests/test_rms_regression.py \
		--ignore=tests/test_cost.py \
		--ignore=tests/test_mean.py \
		--ignore=tests/test_rotate.py
	echo "Clean-room test passed."

# Build, test, and check — full pre-publish dry run.
build-all: build-check build-test

# Generate the JOSS paper stability figure (full grid, ~10-20 min).
paper-figure:
	uv run --extra analysis python analysis/stability_analysis.py --fmt png

# Rerun only the coverage sweep (coverage + RMSE figures); reuse existing stability JSON.
paper-coverage:
	uv run --extra analysis python analysis/stability_analysis.py --coverage-only --fmt png

# Regenerate all figures from existing JSON results (no simulation).
paper-plot:
	uv run --extra analysis python analysis/stability_analysis.py --plot-only --fmt png

# Quick smoke run of the stability analysis (~1-2 min).
paper-figure-smoke:
	uv run --extra analysis python analysis/stability_analysis.py --smoke --fmt png

# ── Trade study (hyperparameter optimisation) ────────────────────
# Requires: pip install -e "/path/to/trade-study[all]" in the venv.

# Install local trade-study with adaptive extra (Optuna) into this venv.
trade-a-install-adaptive:
	uv pip install -p .venv/bin/python -e /home/jcmacdo/Documents/GitHub/jcm-sci/trade-study[adaptive]

# ── Trade study: Option A (surrogate-first, rank_mae primary) ────

# Collect surrogate-training data (Sobol design x regime grid, parallel).
trade-a-collect n_samples="96":
	.venv/bin/python -m analysis.trade_study.option_a_pipeline collect --n-samples {{n_samples}} --n-jobs -1

# Fit the regime surrogate and recommend a config per regime.
trade-a-recommend:
	.venv/bin/python -m analysis.trade_study.option_a_pipeline recommend

# Collect then recommend end-to-end.
trade-a-all n_samples="96":
	.venv/bin/python -m analysis.trade_study.option_a_pipeline all --n-samples {{n_samples}} --n-jobs -1

# Generate the Option A figure suite (F1-F7).
trade-a-figures fmt="png":
	.venv/bin/python -m analysis.trade_study.option_a_figures --fmt {{fmt}}

# Hyperparameter sensitivity report from the Option A surrogate-training table.
trade-a-sensitivity fmt="png":
	.venv/bin/python -m analysis.trade_study.sensitivity_report --fmt {{fmt}}

# Sequential retuning: successive-halving + hyperband -> adaptive refinement (v4).
trade-a-retune-seq n_samples="128" n_adaptive="200":
	.venv/bin/python -m analysis.trade_study.v4_phase2_sequential --n-samples {{n_samples}} --n-adaptive {{n_adaptive}}

# Fast smoke run of the v4 sequential retuning workflow.
trade-a-retune-seq-smoke:
	.venv/bin/python -m analysis.trade_study.v4_phase2_sequential --n-samples 16 --n-adaptive 24 --top-k 8 --rungs 10,20,40,80 --max-budget 80

# Full v4 sequential retuning run tuned for the narrowed search workflow.
trade-a-retune-seq-full n_samples="256" n_adaptive="400" top_k="48" max_budget="300" eta="3.0" rungs="25,50,100,200,300":
	.venv/bin/python -m analysis.trade_study.v4_phase2_sequential --n-samples {{n_samples}} --n-adaptive {{n_adaptive}} --top-k {{top_k}} --max-budget {{max_budget}} --eta {{eta}} --rungs {{rungs}}

# 3-way comparison: default vs rank_mae-recommended (surrogate) vs sklearn.
# reps=3 is a fast validation; reps=10 (omit arg) is the full multi-hour grid.
trade-a-compare reps="3" fmt="png":
	.venv/bin/python -m analysis.trade_study.v3_compare --fmt {{fmt}} --reps {{reps}} --surrogate-config analysis/results/optionA/surrogate_train.json

# Full 3-way comparison (reps=10, several hours).
trade-a-compare-full fmt="png":
	.venv/bin/python -m analysis.trade_study.v3_compare --fmt {{fmt}} --surrogate-config analysis/results/optionA/surrogate_train.json

# Build documentation site.
docs:
	uv run --extra docs mkdocs build --strict

# Serve documentation with live reload.
docs-serve:
	uv run --extra docs mkdocs serve
