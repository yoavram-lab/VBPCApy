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
