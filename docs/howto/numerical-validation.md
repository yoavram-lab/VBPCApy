# Numerical reference checks

Run the bounded numerical checks before changing the fit loop, update equations
or native kernels:

```bash
uv sync --extra dev --extra plot
just numerical-reference
```

`just ci` includes the same checks. The `numerical-reference` CI job runs on
every pull request and main-branch push, without a path filter or an allowed
failure. Package/native-helper build errors fail the job. Missing imports,
collection errors, skipped checks, expected failures and an empty selection
cannot produce a passing reference run.

## What the checks establish

- Dense score and loading means/covariances are compared with direct NumPy
  solutions of the Gaussian conditional equations. Both observed-cell and
  complement sufficient statistics are exercised, using C/F layouts and one,
  two and four workers.
- Sparse CSC/CSR updates use the same independent equations, including an
  explicitly stored observed zero. Noise corrections are checked against
  independent second-moment calculations, with and without loading covariance.
- Modern-mode fixed-iteration trajectories exercise the actual requested native
  path and compare layouts, worker counts and observed/complement updates.
- Fit-loop RMS and noise updates have independent residual/formula checks.
  Centering, Gaussian data cost and rotation also have direct algebraic checks.
  A seeded strict-legacy end-to-end RMS regression retains its pre-existing
  reference value from revision `5abfeb6761b00c3af00304a375a050ee5bcde4f2`.

The equation checks use `rtol=1e-12`, `atol=1e-13`. Modern trajectory comparisons
use their documented tolerances in `tests/test_update_trajectory.py` (up to
`rtol=5e-9`, `atol=5e-11`); the frozen legacy RMS uses `1e-12` relative and
absolute tolerance. These are small float64 fixtures, not tolerances for all
data, ill-conditioned inputs or arbitrary floating-point backends.

The legacy scalar reference is a compatibility regression, not an independently
established MATLAB oracle. Modern-mode differences in masking and update order
are checked under their own contracts, rather than forcing modern fits to match
legacy endpoints. These checks do not establish posterior calibration,
convergence-policy safety or performance on a scientific study.

## Extended Octave comparisons

`just ci-all` also runs the Octave tests. They require local Octave/MEX tooling
and remain separate from the required bounded reference gate. Optional Octave
availability or helper skips cannot replace or disable the required checks.
The extended CI job is diagnostic; inspect its output when changing the legacy
reference implementations.

## Dtypes, records and ownership in 0.5

Array annotations identify dtype; shape and ownership are checked separately.
The numerical helpers distinguish these representations:

| Representation | Contract |
| --- | --- |
| Incoming dense data/category tables | `InputArray` at conversion boundaries; integer and float32 fit inputs remain accepted |
| Dense solver parameters/covariances | float64, including empty arrays for disabled covariance terms |
| Prepared sparse data | Preserve incoming stored-value dtype until the existing centering/native conversion; avoid an additional eager conversion |
| Dense observation masks | Boolean after normalization; legacy numeric mask consumers retain their accepted weight/storage semantics |
| Native dense masks | Boolean, float64 or uint8 as declared by the binding |
| Pattern/selection indices | Signed platform integer; sparse storage counts can retain int32 or int64 |
| Native sparse pointers/coordinates | Signed int32 at the checked native boundary |
| Row counts | Integer or floating storage inherited from mask summation, including sparse float32/longdouble and unsigned integer masks; counts are distinct from signed indices |
| Configurable one-hot encoder output | Its requested dtype remains dynamic, including working complex/object cases; conversion into the solver is separate |

The sparse residual binding has a heterogeneous typed result: float64 `data`,
int32 `indices`/`indptr`, and an integer shape tuple. Covariance helpers accept
float64 block lists or three-dimensional stacked banks. Their list views borrow
stacked storage; packing a list creates a separate float64 stack, reused within
its existing cache scope. Passing an array to the cached stack helper still
checks the required three-dimensional layout. Read-only and strided input
views stay accepted; an annotation does not promise contiguous or writable
storage.

Live model, bias, update and training records are mutable and slotted. Scalar
shape/RMS configuration and immutable routing/runtime descriptors are frozen
and slotted. A frozen descriptor can **borrow mutable array or encoder
payloads**; it freezes field rebinding, not those payloads. Final-state packaging
borrows the completed training arrays to avoid duplicating them, and fitted
result arrays keep their existing mutability. These internal records no longer
have an instance `__dict__`; use `dataclasses.fields` or `dataclasses.replace`
when inspecting or copying a record.

Noise-complement plans own read-only int32 coordinate buffers. Direct
construction snapshots and validates supplied coordinates; the observation-grid
factory transfers its freshly allocated, already screened buffers without
repeating the partition check. Pickle and deepcopy reconstruct owned read-only
buffers. Encoding descriptors snapshot sequence membership into tuples while
borrowing category-label objects; a schema does not deep-copy arbitrary label
objects. `DataReport` remains a mutable report assembled in place with separate
per-call containers.

For example, immutable encoding metadata can be serialized independently of
the mutable fitted arrays:

```python
import pickle

from vbpca_py import EncodedVariable, EncodingSchema

variable = EncodedVariable("categorical", (0, 1), ("a", "b"), offset=(0.0, 0.0))
schema = EncodingSchema((variable,))
schema.validate(2)
assert schema.feature_groups == (0, 0)
assert pickle.loads(pickle.dumps(schema)) == schema
```

Initialization's omitted-mean calculation is a separate stage before initial
centering. This refactoring retains the existing observed-row sums, division,
zero-observation fallback and operation order. Priors, warm-up, stopping rules,
mask semantics and native numerical equations are unchanged. This helper work
does not complete the estimator orientation and Pipeline migration or announce
a release.
