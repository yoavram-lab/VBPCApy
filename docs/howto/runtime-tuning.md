# Runtime & Threading

VBPCApy includes C++ kernels for the most expensive operations (score updates,
loading updates, noise updates, RMS computation). Thread counts and memory
policies can be tuned for your hardware.

## Runtime tuning modes

Control the autotuning policy via the `runtime_tuning` parameter:

```python
from vbpca_py import VBPCA

# Default: short probe to pick threads and accessor mode
model = VBPCA(n_components=5, runtime_tuning="safe")

# Wider search, slightly longer startup
model = VBPCA(n_components=5, runtime_tuning="aggressive")

# No autotuning; request num_cpu for native kernels
model = VBPCA(n_components=5, runtime_tuning="off", num_cpu=4)
```

## Pin thread counts

Set a global thread count:

```python
model = VBPCA(n_components=5, num_cpu=8)
```

Automatic tuning is capped by the CPUs available to the process. On Linux,
VBPCApy respects process affinity; it also recognizes `SLURM_CPUS_PER_TASK`,
`PBS_NP`, and `NSLOTS`. Set `VBPCA_NUM_THREADS=$SLURM_CPUS_PER_TASK` in Slurm
launchers as an explicit native-kernel safeguard.

Or use environment variables for per-kernel control:

```bash
export VBPCA_NUM_THREADS=8          # global override
export VBPCA_SCORE_THREADS=4        # score update kernel
export VBPCA_LOADINGS_THREADS=4     # loadings update kernel
export VBPCA_NOISE_THREADS=2        # noise update kernel
export VBPCA_RMS_THREADS=4          # RMS computation kernel
```

Kernel-specific options and an explicitly supplied `num_cpu` take precedence.
Environment overrides apply when the corresponding Python option has not been
set.

`num_cpu` controls VBPCApy's native kernels. NumPy/SciPy linear algebra can use
a separate BLAS thread pool. Set its limits before starting Python, then
measure the combination on the allocated compute node:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  VBPCA_NUM_THREADS=4 python fit.py
```

This is a starting configuration, not a recommendation for every matrix.
Compare native and BLAS thread counts separately. When running several fits
concurrently, measure completed fits per allocation as well as single-fit
latency, and include their combined memory use. Process affinity and scheduler
variables do not necessarily express container CPU quotas; set explicit limits
when the allocation is otherwise ambiguous.

## Noise sufficient statistics

`pca_opts={"noise_sufficient_statistics": "auto"}` is the estimator default.
For dense inputs with at least 4,096 cells, rank at least 8, and at most 10%
missing entries, it computes complete-data covariance contractions and subtracts
the missing-cell terms. All covariance cross terms are retained. Observed zeros
belong to the observed set; data values never determine membership. Shared
covariance patterns are expanded consistently with sample columns.

Choose `"observed"` to retain observed-cell accumulation, or `"complement"` to
request the rearrangement for eligible dense or sparse observations. Fully
observed inputs keep their existing complete-data path. Ineligible coordinates,
nonfinite intermediates, or poorly scaled subtraction use observed accumulation.
The floating-point screening rule is conservative but is not a certified error
bound. This changes summation order, so bitwise equality is not promised.

The runtime report records the requested mode, eligibility, missing-coordinate
storage, temporary preparation grid, covariance packing estimate, and actual
complement/fallback call counts. Preparation uses one temporary boolean grid;
the retained plan contains only missing int32 coordinates. Covariance packing
uses storage proportional to `(features + samples) * rank**2`, outside the
diagnostic output budget below. Sparse automatic mode retains observed sums.

## Memory budget

`max_dense_bytes` bounds the combined persistent storage of the three full-shape
fit diagnostics: reconstruction, latent variance and predictive variance. These
are float64 arrays with the original feature-by-sample shape, even when input
is sparse or empty rows/columns are removed during fitting. The default limit
is 2,000,000,000 bytes. A request above the limit raises `ValueError` before
fitting; exactly the limit is allowed.

```python
from vbpca_py import VBPCA

# Limit the three fit outputs together to 512 MiB.
model = VBPCA(n_components=5, max_dense_bytes=512 * 1024**2)

# Retain fitted factors without materializing those three outputs.
model_without_outputs = VBPCA(n_components=5, return_diagnostics=False)

# Explicitly disable the output limit when sufficient memory is available.
model_without_limit = VBPCA(n_components=5, max_dense_bytes=None)
```

For 5,846 features and 2,504 samples, the three outputs alone need about
335 MiB. Factors, covariance stores, input copies, temporary arrays and allocator
behavior are outside this limit, so it is not a bound on peak process memory.
`return_diagnostics=False` omits those fit outputs; it does not bound temporary
work or make fitting out of core.

The same option separately bounds dense observation-mask handling for sparse
input and the single output of `inverse_transform()`. The latter checks the
actual number of supplied score columns before reconstruction. These operations
do not share a cumulative process-wide allowance. Invalid masks and incompatible
score shapes still raise their own input errors.

Component selection defaults to `SelectionConfig(compute_explained_variance=True)`,
which requests the selected positive-rank model's reconstruction and variance
outputs after its sweep. That operation receives the same combined three-output
check, even when individual candidate fits omit diagnostics. For scalar-only
selection, set `compute_explained_variance=False` and leave candidate
`return_diagnostics=False`. The rank-zero selection baseline returns scalar
metrics and no fitted estimator, so it has no persistent diagnostic outputs
to charge. Its temporary scoring arrays remain outside this output limit.

## Runtime report

Add `runtime_report=1` to the low-level `pca_full()` call to see which thread
counts and accessor modes were selected:

```python
from vbpca_py._pca_full import pca_full

result = pca_full(X, n_components=5, runtime_report=1)
```

The estimator exposes the same dictionary as `model.runtime_report_` when
constructed with `runtime_report=True`.

Component sweeps reuse measured execution settings within compatible
power-of-two rank ranges by default. This reduces repeated tuning during
`select_n_components`; set `SelectionConfig(reuse_runtime_policy=False)` for an
independently tuned comparison. Each trace entry records whether its settings
were measured or reused and the time spent tuning.

Treat the report as evidence about the probe, not a guarantee of optimal
execution. The current dense thread probe uses C-order arrays without
posterior covariance output and does not benchmark the complement path.
Compare complete fixed-iteration fits when the actual workload differs.
Native kernels can clamp requested counts to the available allocation even
when the report retains the requested count. The tuning time budget is checked
between calls and cannot interrupt a slow individual kernel call.

Persistent profiles are opt-in through `runtime_profile`. Until profiles
include hardware and build fingerprints, use separate paths for different
machines, allocations and software builds. Avoid concurrent writers to one
profile file. Rebenchmark after a BLAS, native build or workload change.

## Covariance writeback modes

The `cov_writeback_mode` option controls how posterior covariances are written back
after each update:

| Mode | Description |
|------|-------------|
| `"python"` | Python handling of individual covariance outputs |
| `"bulk"` | Stacked covariance output from native updates |
| `"kernel"` | Currently follows the same stacked-output branch as `"bulk"` |

When `runtime_tuning` is `"safe"` or `"aggressive"`, the writeback mode is
benchmarked and selected automatically. The current aliases do not imply
different implementations or a universal speed ordering.

## Dense sufficient statistics

Masked dense score and loading updates support two exact accumulation forms:

- `"observed"` iterates over observed cells.
- `"complement"` computes complete-data sufficient statistics and subtracts
  missing-cell contributions.

Set `dense_sufficient_statistics="auto"` (the default) to use the
current density and rank policy. The policy is conservative and keeps the
observed-cell implementation for the genomics study workload:

```python
model = VBPCA(n_components=10, dense_sufficient_statistics="auto")
```

The runtime report records the requested mode, resolved score/loading modes,
observed fraction, and thresholds. Explicit modes are useful for benchmarking.

## Posterior variance diagnostics

Full diagnostic output retains the reconstruction mean, posterior variance and
predictive variance at every feature/sample cell. Predictive variance adds the
observation noise to posterior variance. The contraction retains all loading
and score covariance entries, shared pattern covariances and bias uncertainty;
PPCA/MAP omit their absent covariance terms.

Uncertainty is evaluated in rectangular blocks of at most 128 features and
128 samples using matrix products. Covariance packing and outer products use
only those blocks; NumPy scratch storage scales as
`O(128 * rank**2 + 128**2)` rather than additional full feature-by-sample
matrices. This excludes BLAS workspace and the required output itself.
`max_dense_bytes` still checks combined diagnostic output storage, rather than
peak process memory. Use `return_diagnostics=False` to omit full outputs.

The [benchmark procedure](https://github.com/yoavram-lab/VBPCApy/blob/main/scripts/benchmark_study.md#posterior-variance-contractions)
includes covariance packing, numerical error, NumPy allocation peaks and
optional whole-fit measurements. Selected-cell reconstruction remains a separate
API proposal (#20); full output never fills unevaluated cells with zeros.

## Sparse representations and allocation checks

Sparse loading updates consume CSR rows directly. Score updates build a CSC
view once per distinct centered matrix, and reuse it across iterations when
values stay unchanged. Bias updates and bias-enabled rotations invalidate the
cached values, including updates that reuse the original CSR object. Conversion
time is included in score/loadings phase timings and total fit time.

Native score and loading workers reuse their scratch vector across observed
cells. The Linux/glibc CI allocation probe counts malloc/calloc/realloc calls
around warmed kernels, including fixed input/output and Python argument costs.
It compares fixed shape/rank workloads with widely differing observation counts
and rejects growth with `nnz`. This is a deterministic allocation contract,
rather than a portable timing threshold or a claim of zero allocations per fit.
See [reproduction commands](https://github.com/yoavram-lab/VBPCApy/blob/main/scripts/benchmark_study.md#sparse-allocation-and-view-lifetimes).

## Native failures

Dense and sparse update kernels require finite, positive `noise_var`. Invalid
shapes, indices and compressed sparse pointers raise `ValueError` before
workers start. A Cholesky factorization failure raises `RuntimeError`, including
when it occurs in a worker. Native pools join their workers before propagating
an exception to Python; this also applies if launching a later worker fails.

The update kernels allocate their outputs and do not modify supplied data or
factors. On failure they return no partial output. A failed estimator fit may
have completed earlier iterations; discard that fit rather than interpreting
its attributes as a completed result. Catching an error does not make invalid
inputs numerically meaningful. These guarantees concern C++ exceptions, not
external process termination or hardware faults.
