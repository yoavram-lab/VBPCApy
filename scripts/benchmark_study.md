# Encoding-aware selection validation

The #266 change affects categorical scoring and the selected capacity for
mixed encodings. It adds a fitted schema for dense encoders, preserves fitted
centering on later transforms, and decodes nominal categorical predictions
before scoring. Probe RMS remains in encoded units. Fitting a different
encoding can still change the Gaussian model; scoring equivalence refers to
the same decoded predictions, not identical fitted posteriors.

Reproduce the validation from a clean checkout:

```bash
uv sync --extra dev --extra data --extra plot
just ci
uv run pytest -q tests/test_encoding_aware_scores.py
uv run pytest -q tests/test_raw_model_selection.py
just example-raw-cv
just build-check
```

Publication studies already pinned to 0.4.5 retain their registered fitting
policy and full-one-hot scoring contract. Adopting these changes requires a
new software pin and an explicit encoding schema; no production results or
locked benchmark artifacts are regenerated in this PR. The tests compare
decoded scores, original-variable masks and rank-zero CV across encoding
modes, and exercise positive-rank candidates separately.

The #267 raw-table path constructs original-variable cell holdouts before
learning means, scales, category frequencies or block weights. Each fold fits
one encoder and reuses its schema for all capacities. External codebooks can
declare categorical/ordinal levels; an undeclared held-only level raises an
error. The diagnostic tests change held-out values with the training mask fixed
and compare fitted statistics, then check decoded scores and original-unit
numeric RMSE. `scripts/example_raw_cell_cv.py` runs against an installed package
and prints the fold-specific numeric training statistics. The new path leaves
the encoded-array API and locked 0.4.5 publication settings in place; adopting
fold-local preprocessing is a separately registered methodology change.

The `just bench-study-full`, `just bench-study-summary`, and
`just bench-study-paper` sequence referenced in CONTRIBUTING.md has no recipes
in the current justfile. Current benchmark recipes are listed by `just --list`;
use the commands above for this scoring change.

For the 0.4.6 patch, [release validation](../docs/releases/0.4.6.md) adds the
paired fixed-policy fitting check against 0.4.5. Both snapshots must use the
same registered policy module, numerical environment and controlled native
build. This checks the numerical implementation on fixed matrices; adopting
raw-table CV remains a separately registered preprocessing-method change.

## Native failure checks

Runtime failure handling rejects invalid noise variances and sparse pointers,
and propagates native worker errors as Python exceptions. It does not change
update equations, numerical defaults or locked publication settings. Do not
regenerate study artifacts for this repair. Reproduce the process-survival and
numerical checks with:

```bash
uv sync --extra dev --extra plot
uv run pytest -q tests/test_native_failures.py
just numerical-reference
just ci
just build-check
```
