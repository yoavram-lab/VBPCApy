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

::: vbpca_py.options
