# Choose a numerical compatibility mode

The unreleased 0.5 native estimator and public component-selection helpers
default to `compat_mode="modern"`. Use that policy for new native workflows,
including observed zeros and small finite measurements. The explicit
`vbpca_py.legacy` helpers, low-level solver and a directly constructed
`PCAOptions()` retain `strict_legacy` as their default.

Mode and layout are separate choices. Native estimators always accept
samples × features; the legacy adapter and low-level solver accept
features × samples. Setting `compat_mode="strict_legacy"` on a native estimator
does not restore the historical interface or serialized-model layout. The
[0.5 migration guide](../api/migration-05.md) covers those changes.

## What changes between modes

| Solver behavior | `strict_legacy` | `modern` |
| --- | --- | --- |
| Sparse empty-axis detection without an explicit solver mask | Signed row/column sums; observed values can cancel to zero and remove an axis. | Stored non-NaN observations determine occupied axes, including stored zeros. |
| Zero and near-zero normalization | Historical `numpy.isclose(value, 0)` matches are replaced by float64 machine epsilon. | Finite values retain their original magnitude, including exact zero; no epsilon substitution is needed. |
| Observation coordinates in update paths | Retains historical nonzero-based operations, so zeros may be treated differently across preparation/probe paths. | Uses the observation mask through updates, probe exclusion and NaN cleanup. |
| `bias_update_order="auto"` | Resolves to `legacy`. | Resolves to `post_factor`, updating the mean/RMS after the factor sweep. |
| `variance_update_order="auto"` | Resolves to `legacy`. | Resolves to `post_rotation`, updating prior variances after component rotation. |

An explicit `bias_update_order` or `variance_update_order` overrides that
mode's ordering default. The latter controls prior variances, not observation
noise variance. The [algorithm description](../concepts/algorithm.md#update-order)
explains the relationship to returned components.

Machine epsilon here is a historical numerical replacement, not a missing-data
code or a measurement threshold for new data. In modern mode, retain observed
sparse zeros explicitly in storage. A sparse constructor applied to a dense
array can drop those zeros; `eliminate_zeros()` would also remove their
observation status. Use coordinates as in the executed example below.
Without an explicit mask, absent sparse entries are missing in both modes;
neither mode can infer that a discarded entry was an observed zero. A modern
explicit mask can request an observed zero at an absent coordinate.

Dense NaNs are missing at the native estimator boundary. Native masks are
consumed before the solver, and native NaNs remain missing even at positions
marked true. The historical explicit dense-mask interface in the legacy
adapter/low-level solver is authoritative: combine a supplied mask with
`~numpy.isnan(X)` to exclude existing NaNs. Changing numerical mode does not
replace that interface contract. In either mode, observed probes discarded by
empty training-axis removal are refused rather than silently dropped.

## Migrate an existing workflow

For a frozen reproduction, retain the package version, explicit legacy
entry point, mode, initialization/seed and all fitting options. Switching
numerical mode can change both the observation set and update order; it is
not a promise of identical fitted factors or convergence decisions.

For a new native workflow, arrange the observation table as samples × features
and use `VBPCA(...)` or explicitly set `compat_mode="modern"`. Pass aligned
masks/probes in that layout. Keep the [masking Pipeline](../tutorials/missing-aware-pipeline.md)
responsible for missingness before feature expansion. Compare original-unit
predictions and fitting diagnostics when migrating an established application.
The [stopping guide](convergence.md) is a separate choice from compatibility
mode; four-iteration examples below do not qualify convergence or calibration.

The following standalone source uses both entry points explicitly, checks that
modern masked/unmasked tiny measurements agree, preserves a stored sparse
zero and prints the resolved update-order policies. Its modern examples use
the same numerical policy; the historical strict fit is shown without an
equivalence assertion.

```bash
python scripts/example_compatibility_modes.py
```

```python
--8<-- "scripts/example_compatibility_modes.py"
```
