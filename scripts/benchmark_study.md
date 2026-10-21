# Encoding-aware selection validation

Result-contract checks run with the regular test suite and strict consumer
typing:

```bash
just typecheck
uv run pytest -q tests/test_result_contracts.py tests/test_sparse_pattern_groups.py
```

These checks cover the existing dictionary keys and heterogeneous learning
curves, plus sparse covariance groups that partition sample columns. The
sparse repair preserves observed zeros and compares shared/unshared fits,
cost histories and uncertainty after consistent component-sign alignment.
It changes a previously failing sparse-sharing path; adopting it in a study
requires recording the new software pin. No frozen benchmark is regenerated.

Encoding-aware selection affects categorical scoring and the selected capacity for
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
locked benchmark artifacts are regenerated for this change. The tests compare
decoded scores, original-variable masks and rank-zero CV across encoding
modes, and exercise positive-rank candidates separately.

The raw-table path constructs original-variable cell holdouts before
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

## Unavailable stopping option and flag validation

Live fitting reports the legacy `slowing_down` event stream as unavailable and
resolves its flag to disabled. Explicit requests warn; removing the inert flag
from recommendations does not change their computed endpoints. Convergence
flags now require booleans, with NumPy boolean scalars normalized into a copied
mapping. Invalid strings/numbers cannot silently enable a stop.

These changes do not restore a new stopping event, retune supported thresholds,
change optimization updates, or alter locked publication software pins. The
existing posterior-stability screen did not qualify a replacement default;
the held-out no-RMS candidate also failed its safety gate. Further policy
confirmation remains separate. Reproduce this option check:

```bash
uv sync --extra dev --extra data --extra plot
uv run pytest -q tests/test_unavailable_slowing_down.py tests/test_converge.py tests/test_defaults.py
just ci
```


## Retained-curve policy replay

The [convergence-detector analysis](../analysis/trade_study/CONVERGENCE_DETECTOR.md)
now provides a policy-only command for the retained v2 screen. It reproduces
selected baseline stops before evaluating exploratory policies, requires a
complete consistent shard set, reports unavailable policies and unstable
endpoints separately, and writes a new provenance-bearing report. No fits,
frozen policies or publication pins change. The completed held-out confirmation
remains negative; replay results alone cannot qualify a replacement default.

## Backend review probes

The [backend review](../analysis/BACKEND_REVIEW_2026_10_20.md) separates
measured implementation costs from proposed statistical changes. Its
prototypes retain full covariance terms and are not installed solver paths.
Run from a clean source checkout with the package installed:

```bash
uv sync --extra dev --extra plot
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VBPCA_NUM_THREADS=1 \
  uv run python analysis/review_backend_costs.py --output /tmp/vbpca-backend-review.json
uv run pytest -q tests/test_backend_rearrangements.py
just numerical-reference
```

The probe writes source/native hashes, numerical versions, platform, process
affinity, thread environment and NumPy build details alongside timings and
allocation/conversion counters. It uses 512 features, 256 samples, ranks 8 and
32, four fixed iterations and 1% missingness. It alternates reference/prototype
timing order after warmup and includes covariance packing. Keep generated JSON
and benchmark outputs outside tracked source. Record actual BLAS pools and
scheduler/node information separately for cluster qualification.

The [algebra-check instructions](../analysis/proof_checks/README.md) pin Lean,
mathlib and SymPy and state the proof boundaries. Follow numerical comparisons
with representative fixed-policy full fits before adopting a rewrite; the
local phase ratios do not establish whole-fit or cross-machine speedups.
