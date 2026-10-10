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
just build-check
```

Publication studies already pinned to 0.4.5 retain their registered fitting
policy and full-one-hot scoring contract. Adopting these changes requires a
new software pin and an explicit encoding schema; no production results or
locked benchmark artifacts are regenerated in this PR. The tests compare
decoded scores, original-variable masks and rank-zero CV across encoding
modes, and exercise positive-rank candidates separately. Fold-local learned
preprocessing is tracked by #267.

The `just bench-study-full`, `just bench-study-summary`, and
`just bench-study-paper` sequence referenced in CONTRIBUTING.md has no recipes
in the current justfile. Current benchmark recipes are listed by `just --list`;
use the commands above for this scoring change.
