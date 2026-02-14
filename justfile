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

# Run performance benchmarks (excluded from default test/ci runs).
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
