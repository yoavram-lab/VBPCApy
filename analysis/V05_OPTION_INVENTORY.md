# Solver option inventory before 0.5 implementation

Source baseline: `16df5d7abc492f0bc261b7ee40bf5507168315d9`. This records requested defaults, not resolved runtime state.

| Low-level keyword | Requested default | Boundary |
|---|---|---|
| `init` | `"random"` | initialization |
| `maxiters` | `1000` | iteration and stopping |
| `bias` | `True` | scientific model |
| `max_dense_bytes` | `2_000_000_000` | output and monitoring |
| `uniquesv` | `False` | execution policy |
| `auto_pattern_masked` | `False` | execution policy |
| `angle_every` | `1` | iteration and stopping |
| `minangle` | `1e-8` | iteration and stopping |
| `algorithm` | `"vb"` | scientific model |
| `niter_broadprior` | `100` | scientific model |
| `hp_va` | `0.001` | scientific model |
| `hp_vb` | `0.001` | scientific model |
| `hp_v` | `0.001` | scientific model |
| `va_init` | `1000.0` | scientific model |
| `random_state` | `None` | initialization |
| `earlystop` | `False` | iteration and stopping |
| `rmsstop` | `np.array([100, 1e-4, 1e-3])` | iteration and stopping |
| `cfstop` | `np.array([])` | iteration and stopping |
| `cfstop_rel` | `None` | iteration and stopping |
| `cfstop_curv` | `None` | iteration and stopping |
| `record_cost` | `False` | output and monitoring |
| `cost_diagnostics` | `"trace"` | output and monitoring |
| `record_prior_trace` | `False` | output and monitoring |
| `composite_stop` | `None` | iteration and stopping |
| `patience` | `1` | iteration and stopping |
| `criterion_order` | `None` | iteration and stopping |
| `convergence_criteria` | `None` | iteration and stopping |
| `verbose` | `1` | output and monitoring |
| `num_cpu` | `None` | execution policy |
| `num_cpu_score_update` | `None` | execution policy |
| `num_cpu_loadings_update` | `None` | execution policy |
| `num_cpu_noise_update` | `None` | execution policy |
| `num_cpu_rms` | `None` | execution policy |
| `runtime_tuning` | `"safe"` | execution policy |
| `runtime_profile` | `None` | execution policy |
| `dense_sufficient_statistics` | `"auto"` | execution policy |
| `noise_sufficient_statistics` | `"auto"` | execution policy |
| `masked_batch_size` | `0` | execution policy |
| `cov_writeback_mode` | `None` | execution policy |
| `log_progress_stride` | `None` | output and monitoring |
| `accessor_mode` | `None` | execution policy |
| `xprobe` | `None` | fit data |
| `rotate2pca` | `True` | scientific model |
| `display` | `False` | output and monitoring |
| `bias_update_order` | `"auto"` | scientific model |
| `variance_update_order` | `"auto"` | scientific model |
| `compat_mode` | `"strict_legacy"` | scientific model |
| `return_diagnostics` | `True` | output and monitoring |
| `runtime_report` | `False` | output and monitoring |
| `explained_var_solver` | `"auto"` | output and monitoring |
| `explained_var_gram_ratio` | `4.0` | output and monitoring |

There are no declared keyword spelling aliases. The legacy keyword merger lowercases names and lets the last case variant win. It retains unknown keys with a warning, except private `_debug_` instrumentation. The modern estimator uses explicit keyword parameters and rejects unknown names through native parameter discovery.

`xprobe` is fit data, not a constructor hyperparameter. Raw masks and probe tables follow their API boundary's orientation; they cannot be guessed from shape. The legacy `tol` parameter is deprecated and inert; it is not a solver keyword and must remain explicit in migration guidance.

Initialization accepts random initialization, `None`, a MAT filename, a mapping, or supported MAT structs. A random/custom enum cannot describe this contract. Preserve this representation until a separately documented ownership-aware initialization record is in place.

Runtime code currently writes resolved CPU counts, statistic choices, covariance/accessor modes, logging stride, derived observed-coordinate metadata and private diagnostic values into the options mapping. The typed migration must separate requested immutable settings from resolved execution choices and mutable iteration state. This inventory does not authorize a new default, removal of legacy fallback behavior, or changing stopping or update order.

## Categorical representations

| Option | Typed values | Compatibility detail |
|---|---|---|
| `algorithm` | `Algorithm`: ppca, map, vb | Legacy lowercases without stripping whitespace |
| `compat_mode` | `CompatMode`: strict_legacy, modern | Normalizes case/whitespace; retains numerical policy meaning |
| `bias_update_order` | `BiasUpdateOrder`: auto, legacy, post_factor | Resolve auto using compatibility mode after validating requested values |
| `variance_update_order` | `VarianceUpdateOrder`: auto, legacy, post_rotation | Separate requested auto from resolved update order |
| `cost_diagnostics` | `CostDiagnostics`: trace, endpoint | Endpoint remains incompatible with cost-consuming stopping/recording |
| dense/noise sufficient statistics | `StatisticsMode`: auto, observed, complement | Eligibility/floating-point screening still controls execution |
| `explained_var_solver` | `ExplainedVarianceSolver`: auto, svd, gram | Preserve ratio fallback in legacy numerical calls |
| `runtime_tuning` | `RuntimeTuning`: off, safe, aggressive | Legacy invalid values fall back to off; modern validated config rejects them |
| `cov_writeback_mode` | `CovarianceWriteback`: auto, python, bulk, kernel | Legacy invalid/None falls back to auto; bulk/kernel share execution today |
| `accessor_mode` | `AccessorMode`: auto, legacy, buffered | Legacy invalid/None falls back to auto |
| convergence names | `Criterion`: angle, earlystop, rms_plateau, cost, composite, slowing_down | Slowing-down remains unavailable and disabled; names do not promise implementation |

## Shape, ownership and validation boundaries

`PCAOptions` will use concrete fields, including optional integer thread
requests, Boolean policies, float thresholds and enums. Stop windows become
immutable tuples in resolved requested configuration; initialization arrays
require explicit copies and validation at fit, not constructor mutation.
Validation distinguishes Boolean values from integer counts. Finite/positive
constraints must follow each option's actual meaning: infinity can disable
legacy rules and must not be rejected globally.

All estimator constructor values remain inspectable and unchanged until fit.
A NumPy `Generator` is mutable; copying/seed behavior needs clone, repeated-fit
and serialization tests rather than an assumption that a frozen config owns
it. Fit data and observation masks are separate from tunable configuration.
Low-level keyword compatibility and strict modern configuration are distinct
entry points so tighter validation does not silently rewrite archived calls.
