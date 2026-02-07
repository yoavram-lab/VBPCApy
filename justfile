set shell := ["bash", "-lc"]

# Lint Python sources (excluding tests) using Ruff preview rules.
lint:
	ruff check --preview src

# Type-check library code.
typecheck:
	mypy src

# Run the test suite quietly.
test:
	pytest -q

# Run lint, typecheck, and tests in sequence.
ci: lint typecheck test
