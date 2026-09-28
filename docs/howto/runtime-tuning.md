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

# No autotuning — use num_cpu for all kernels
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

## Memory budget

For large sparse matrices, VBPCApy avoids unintended densification. Control the
budget with `max_dense_bytes`:

```python
# Allow up to 2 GB of dense arrays
model = VBPCA(n_components=5, max_dense_bytes=2 * 1024**3)
```

If a dense operation would exceed this budget, it raises an error instead of
silently allocating.

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

## Covariance writeback modes

The `cov_writeback_mode` option controls how posterior covariances are written back
after each update:

| Mode | Description |
|------|-------------|
| `"python"` | Pure-Python writeback (slowest, most portable) |
| `"bulk"` | Batch writeback via NumPy (good default) |
| `"kernel"` | C++ kernel writeback (fastest on supported platforms) |

When `runtime_tuning` is `"safe"` or `"aggressive"`, the writeback mode is
benchmarked and selected automatically.
