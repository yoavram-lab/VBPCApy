# Algebra checks for backend rearrangements

These checks support two analysis-only prototypes in
[`review_backend_costs.py`](../review_backend_costs.py). They do not change
the installed solver or qualify new statistical defaults.

## Scope and assumptions

The Lean statements quantify over arbitrary finite feature, sample and
component index types, and real-valued arrays. Empty index types are allowed.
They prove finite-sum factorizations, permutation of sums, an explicit
observed/missing partition, and rearrangement of all three covariance terms.
They need no symmetry, positive definiteness or nonzero dimension assumptions
for these algebraic identities. Those properties matter separately when
interpreting the arrays as posterior covariances or solving the updates.

The SymPy check additionally expands the marginal variance of a two-component
bilinear reconstruction. It assumes independent, centered loading, score and
bias errors with finite second moments. Within-vector covariances are retained;
neither diagonality nor Gaussianity is needed for this moment identity. The
fixed dimensions in SymPy are illustrative; Lean handles the finite-sum
rearrangements for arbitrary finite dimensions. Lean does not formalize the
probability argument.

Neither tool proves floating-point stability, solver convergence, posterior
validity, missing-data assumptions, stopping behavior or rank calibration.
In particular, complete-minus-missing subtraction may lose accuracy when
few cells are observed. Numerical eligibility checks and an observed-cell
fallback are still required before adopting it.

## Reproduce

With [elan](https://github.com/leanprover/elan) installed, the toolchain and
dependency graph are pinned in `lean/`:

```bash
cd analysis/proof_checks/lean
lake exe cache get
lake build
```

Run SymPy independently from the repository root. It is a review dependency,
not a VBPCApy runtime dependency:

```bash
uv run --with sympy==1.14.0 python analysis/proof_checks/backend_symbolic.py
uv run pytest -q tests/test_backend_rearrangements.py
```

## Local verification on 20 October 2026

All six Lean theorems compiled with Lean 4.34.1 and mathlib commit
`d13f23b723b8a846827a245b89c10fc7d3f11612`. The local check invoked `lean`
directly with `LEAN_PATH` pointing to an existing read-only cache of the public
dependencies. Each dependency checkout matched the revision in
`lean/lake-manifest.json`. A fresh `lake build` was not run locally.

Each printed theorem uses only `propext`, `Classical.choice` and `Quot.sound`;
none uses `sorryAx`. SymPy 1.14.0 passed both identity checks. Numerical
tests use independent observed-cell sums with positive-definite, full
covariances and complete, random, structured and single-cell masks. Two
six-iteration fits also compare the unadopted noise prototype with the current
solver, retaining its hyperprior, floor and cost calculation. These bounded
checks do not replace release or study qualification.
