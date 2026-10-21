"""Report resolved native worker pools separately from numerical-library pools."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from . import (
    dense_update_kernels,
    errpca_pt,
    noise_update_kernels,
    sparse_update_kernels,
    subtract_mu_from_sparse,
)
from ._runtime_fingerprint import blas_pools

if TYPE_CHECKING:
    from ._runtime_policy import RuntimeThreadConfig, RuntimeWorkloadProfile


def execution_pools(  # noqa: PLR0913
    threads: RuntimeThreadConfig,
    workload: RuntimeWorkloadProfile,
    *,
    patterns: bool,
    loading_covariances: bool,
    score_covariances: bool,
    bias: bool,
    missing_count: int | None = None,
) -> dict[str, object]:
    """Resolve active native pool sizes using the compiled kernels' own rules.

    Zero denotes a path without that native pool; Eigen/NumPy/SciPy can still
    use independent BLAS pools. These are resolved worker counts, not sampled
    operating-system CPU utilization. Conditional noise fallback is explicit.

    Returns:
        Requested settings, active effective worker counts and loaded BLAS pools.
    """
    p, n, q = workload.n_features, workload.n_samples, workload.n_components
    observed = workload.n_observed
    complete = observed == p * n
    effective: dict[str, int] = dict.fromkeys(asdict(threads), 0)
    if q > 0:
        if workload.is_sparse and not patterns:
            effective["score_update_sparse"] = (
                sparse_update_kernels.effective_thread_count(
                    threads.score_update_sparse, n, observed * max(1, q)
                )
            )
            effective["loadings_update_sparse"] = (
                sparse_update_kernels.effective_thread_count(
                    threads.loadings_update_sparse, p, observed * max(1, q)
                )
            )
        elif not workload.is_sparse:
            # The no-covariance complete-data kernels have no native worker pool.
            if not patterns and (loading_covariances or not complete):
                effective["score_update_dense"] = (
                    dense_update_kernels.effective_thread_count(
                        threads.score_update_dense, n
                    )
                )
            if not patterns and (score_covariances or not complete):
                effective["loadings_update_dense"] = (
                    dense_update_kernels.effective_thread_count(
                        threads.loadings_update_dense, p
                    )
                )
        if not complete:
            effective["noise_sxv_sum"] = noise_update_kernels.effective_thread_count(
                threads.noise_sxv_sum, observed
            )
        if workload.is_sparse:
            effective["rms"] = errpca_pt.effective_thread_count(threads.rms, p)
    noise: dict[str, object] = {"observed": effective["noise_sxv_sum"]}
    if missing_count is not None and not complete and q > 0:
        noise["complement"] = noise_update_kernels.effective_thread_count(
            threads.noise_sxv_sum, missing_count
        )
        noise["fallback"] = "observed"
    return {
        "requested": asdict(threads),
        "effective": effective,
        "noise_branches": noise,
        "sparse_centering_effective": subtract_mu_from_sparse.effective_thread_count(
            p, observed
        )
        if workload.is_sparse and bias and q > 0
        else 0,
        "rotation_native_workers": 0,
        "blas_pools": blas_pools(),
        "interpretation": (
            "Resolved native worker pools; zero means this pool is unused. "
            "Independent BLAS pools and conditional noise branches "
            "are reported separately."
        ),
    }
