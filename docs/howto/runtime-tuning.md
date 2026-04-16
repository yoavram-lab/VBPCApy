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

Or use environment variables for per-kernel control:

```bash
export VBPCA_NUM_THREADS=8          # global override
export VBPCA_SCORE_THREADS=4        # score update kernel
export VBPCA_LOADINGS_THREADS=4     # loadings update kernel
export VBPCA_NOISE_THREADS=2        # noise update kernel
export VBPCA_RMS_THREADS=4          # RMS computation kernel
```

Environment variables take precedence over the `num_cpu` parameter.

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

## Covariance writeback modes

The `cov_writeback` option controls how posterior covariances are written back
after each update:

| Mode | Description |
|------|-------------|
| `"python"` | Pure-Python writeback (slowest, most portable) |
| `"bulk"` | Batch writeback via NumPy (good default) |
| `"kernel"` | C++ kernel writeback (fastest on supported platforms) |

When `runtime_tuning` is `"safe"` or `"aggressive"`, the writeback mode is
benchmarked and selected automatically.
