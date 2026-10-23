# Solver options

The 0.5 development line provides named categorical values in
`vbpca_py.options`. Ordinary string calls to the numerical solver remain
supported. Parsing is explicit, and occurs when resolving a configuration;
estimator constructors retain the values supplied by the caller.

```python
from vbpca_py.options import Algorithm, CompatMode

algorithm = Algorithm.parse("vb")
compatibility = CompatMode.MODERN
```

These values describe requested settings. Runtime eligibility checks still
control execution paths. For example, requesting complement statistics does
not bypass the numerical screening rules. Naming a stopping criterion does
not enable or qualify it.

`parse` normalizes case and surrounding whitespace, and raises for unknown
values. Historical low-level normalization rules remain separate: the
algorithm keyword lowercases without stripping whitespace, and invalid legacy
runtime-mode strings retain their documented fallback behavior. This explicit
parser does not inherit those fallbacks.

Initialization is not a random/custom choice alone. `init` also accepts a
warm-start mapping or a MAT filename; those inputs remain supported. Enum
values serialize as strings in JSON and retain their identity through pickle.
They do not make mappings or arrays immutable.

## Typed configuration and estimator parameters

`PCAOptions` describes requested solver settings. Use the typed argument or
legacy keywords on each low-level call:

```python
import numpy as np
from vbpca_py import PCAOptions, VBPCA
from vbpca_py._pca_full import pca_full
from vbpca_py.options import Algorithm, RuntimeTuning

x = np.random.default_rng(7).normal(size=(5, 12))  # features x samples
config = PCAOptions(
    algorithm=Algorithm.VB,
    maxiters=4,
    niter_broadprior=0,
    verbose=0,
    random_state=7,
    num_cpu=1,
    runtime_tuning=RuntimeTuning.OFF,
)
result = pca_full(x, 2, options=config)
legacy_result = pca_full(x, 2, **config.to_kwargs())

model = VBPCA(2, maxiters=4, niter_broadprior=0, verbose=0, random_state=7)
model.set_params(algorithm=Algorithm.MAP, num_cpu=1)
resolved_request = model.get_config()
keyword_request = model.get_options()
model.fit(x.T)  # native estimator: samples x features
```

For a fully typed direct constructor, use the categorical enum values shown
below. `PCAOptions.from_kwargs(**existing_settings)` is the compatibility
conversion for existing keyword dictionaries: it applies the established
case and numeric normalization rules. The final fitting boundary also checks
cross-option constraints and initialization shapes. Passing `options=` and
solver keyword overrides together raises `TypeError`; update a configuration
with `dataclasses.replace(config, maxiters=20)` before fitting instead.

The estimator exposes all solver controls as explicit constructor parameters,
plus `n_components`, the deprecated inert `tol`, and `xprobe_fraction`.
Pass held-out probe data using `model.fit(x, xprobe=probe)`; it is not a
constructor hyperparameter. Constructor values are stored unchanged. Native sklearn `get_params`,
`set_params`, `clone`, and grid-search parameter selection can inspect every
parameter. Configuration validation happens in `get_config()` or fitting.
This parameter contract does not change the current features × samples
orientation or add projection of new data; those remain separate estimator
migration work. Use the package's model-selection helpers for its existing
missing-entry validation workflow.

### Migrating keyword dictionaries and `.opts`

| Previous use | 0.5 development interface |
| --- | --- |
| `VBPCA(2, **settings)` with recognized settings | Continue using it; every setting is now a named constructor parameter |
| `VBPCA(2, xprobe=probe)` | `VBPCA(2).fit(x, xprobe=probe)` |
| `model.opts["maxiters"] = 20` | `model.set_params(maxiters=20)` |
| Inspect constructor values | `model.get_params(deep=False)` |
| Inspect normalized public requests | `model.get_config()` or `model.get_options()` |
| `pca_full(x, 2, **settings)` | Still supported for recognized keyword names |
| Convert a low-level settings dictionary | `PCAOptions.from_kwargs(**settings)` |
| Misspelled or previously ignored names | Correct/remove them; unknown names now fail clearly |

Estimator names are canonical and case-sensitive, as expected by sklearn.
Recognized low-level keyword names retain their historical case-insensitive
spelling. The generic internal keyword utility can still report unknown names,
but the public solver rejects them. In particular, MATLAB's `autosave` never
saved Python solver state and is now rejected; remove it from Python calls.
Private runtime metadata is not a constructor parameter or public keyword
export. Fitted caches, preflight records, complement plans, and autotuned
execution choices stay in per-fit state. Mutating the dictionary returned by
`get_options()` changes neither the estimator nor another fit.

### Ownership and thread inheritance

The configuration is frozen and slotted. Small stopping arrays, criterion
orders and policy mappings are copied into immutable tuples/mappings. Each
keyword export creates fresh mutable stopping containers. Initialization and
probe payloads are borrowed until fitting; the existing fit preparation owns
its numerical working arrays. Exporting configuration does not copy large
payloads. Mutating a borrowed payload before fitting changes what that fit
receives. A supplied NumPy `Generator` keeps its usual stateful semantics;
use an integer seed for reproducible repeated fits. Pickle/deepcopy retain
configuration, and sklearn clones independently copy mutable constructor
arguments.

`num_cpu=None` means inherit execution policy, including for legacy callers
that explicitly pass `None`. Configuration creation, replacement, pickle and
`get_options()` preserve that request. Environment/profile defaults are resolved
at fit time, so adding a probe does not change precedence and constructing a
configuration does not freeze an environment-derived thread count. A positive
`num_cpu` is an explicit global hint; per-kernel hints keep their existing
precedence. Use `runtime_report=True` and inspect `model.runtime_report_`
(or the low-level result's `RuntimeReport`) for resolved settings and effective
native worker pools. Requested configuration alone is not a report of which
kernels ran or how many workers each used.

::: vbpca_py.configuration.PCAOptions

## Warm-start layouts

The low-level `pca_full` boundary uses features × samples. Let `p` be the
original feature count, `n` the original sample count and `q` the component
count. Initialization axes are explicit even when those dimensions coincide.

| Field | Accepted layout |
| --- | --- |
| `A` | `(p, q)` loadings |
| `S` | `(q, n)` scores |
| `Mu` | `(p,)`, `(p, 1)` or `(1, p)` feature means; empty requests default initialization |
| `Muv` | `(p,)` or `(p, 1)` feature mean variances; empty requests the existing zero fallback |
| `Av` | `(p, q)` diagonal variances, `(p, q, q)` full blocks, or a list of `p` full blocks |
| `Sv` | `(q, k)` diagonal variances, `(k, q, q)` full blocks, or a list of `k` full blocks |
| `Isv` | Optional length-`n` sample-to-source-bank indices, zero-based; MAT row/column vectors and integral floating values are accepted |
| `V` | One noise-variance value, including a one-element MAT array |

Without `Isv`, `k=n` means per-sample score covariances. When target pattern
sharing is active, an explicitly target-sized bank can instead use those target
labels. Supplying `Isv` makes its source-bank mapping authoritative, including
when the bank count happens to equal the sample or component count. Nonintegral,
negative, out-of-range and incompatible indices fail clearly. Full covariance
blocks retain cross terms; a square two-dimensional table remains a diagonal
representation rather than being guessed to be a shared full covariance.

This zero-based contract also applies to MAT files and struct objects. Explicitly
subtract one from native MATLAB `Isv` values at the import boundary before using
them as a Python initialization. The loader does not infer index origin from
the labels or dimensions: an all-positive vector may be valid zero-based data.

Before fitting, mappings, supported struct objects and flat or single-struct
MAT files are normalized into an owned payload. Removing empty features slices
`A`, `Av`, `Mu` and `Muv` together. Removing samples slices `S` and per-sample
`Sv`, or slices `Isv` while retaining the source covariance bank. The caller's
arrays and mapping remain unchanged.

The solver's prepared missingness-pattern labels are separate from source
`Isv`. It first resolves source covariances per retained sample, then chooses
the first retained sample for each destination pattern. A round trip from fit
results must therefore retain `Sv` **and** `Isv`. Matching dimensions or loading
a payload does not establish that it is a valid posterior warm start for a
different dataset or model.

::: vbpca_py.options
