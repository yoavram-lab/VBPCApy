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

## Memory budget

`max_dense_bytes` currently guards dense observation-mask handling for sparse
input. It does not bound all dense allocations or peak memory:

```python
# Set the dense observation-mask limit for sparse input
model = VBPCA(n_components=5, max_dense_bytes=2 * 1024**3)
```

Full reconstruction and uncertainty diagnostics are dense even when training
input is sparse. With `return_diagnostics=True` (the default), the fit returns
reconstruction, latent variance and predictive variance, each with the full
feature-by-sample shape. For 5,846 features and 2,504 samples, three float64
outputs alone require about 335 MiB, before factors, covariance arrays and
temporary working storage. These outputs are not protected by the mask limit.

Use `return_diagnostics=False` when those fit-time outputs are unnecessary.
The fitted factors remain available. Later operations that reconstruct the
whole matrix can still allocate dense arrays; this option does not make
inference or prediction out of core.

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
