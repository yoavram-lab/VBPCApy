"""Native failure boundaries must preserve the Python process and input arrays."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("kind", ["score", "loading"])
@pytest.mark.parametrize(
    "failure",
    ["noise", "factorization", "sparse_factorization", "shape", "sparse_pointer"],
)
def test_native_failure_is_catchable(threads, kind, failure) -> None:
    """Invalid input and worker failures raise without aborts, hangs or mutation."""
    script = (
        textwrap
        .dedent("""
        import numpy as np
        from vbpca_py import dense_update_kernels as dense
        from vbpca_py import sparse_update_kernels as sparse
        threads, kind, failure = THREADS, KIND, FAILURE
        x = np.ones((2, 4))
        mask = np.ones_like(x)
        factors = np.zeros((2, 1)) if kind == "score" else np.zeros((1, 4))
        before = (x.copy(), factors.copy())
        noise = -1.0 if failure == "noise" else 1.0
        try:
            if failure in {"sparse_pointer", "sparse_factorization"}:
                args = dict(x_data=np.array([1., 2.]),
                    x_indices=np.array([0, 1], dtype=np.int32),
                    x_indptr=np.array([0, 2, 1, 2] if failure == "sparse_pointer"
                        else [0, 1, 1, 1, 2], dtype=np.int32),
                    noise_var=1., num_cpu=threads)
                if kind == "score":
                    sparse.score_update_sparse_nopattern(loadings=factors,
                        loading_covariances=np.full((2, 1, 1), -10.), **args)
                else:
                    sparse.loadings_update_sparse_nopattern(
                        scores=np.zeros((1, 2)), prior_prec=np.eye(1),
                        score_covariances=np.full((2, 1, 1), -10.), **args)
            elif kind == "score":
                cov = np.full((2, 1, 1), -10.) if failure == "factorization" else None
                dense.score_update_dense_masked_nopattern(
                    x_data=x, mask=mask[:, :2] if failure == "shape" else mask,
                    loadings=factors, loading_covariances=cov,
                    noise_var=noise, num_cpu=threads)
            else:
                cov = np.full((4, 1, 1), -10.) if failure == "factorization" else None
                dense.loadings_update_dense_masked_nopattern(
                    x_data=x, mask=mask[:, :2] if failure == "shape" else mask,
                    scores=factors, score_covariances=cov,
                    prior_prec=np.eye(1), noise_var=noise, num_cpu=threads)
        except (ValueError, RuntimeError):
            np.testing.assert_array_equal(x, before[0])
            np.testing.assert_array_equal(factors, before[1])
            print("caught")
        else:
            raise AssertionError("Expected a catchable native exception")
    """)
        .replace("THREADS", repr(threads))
        .replace("KIND", repr(kind))
        .replace("FAILURE", repr(failure))
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "caught"


@pytest.mark.parametrize(
    "noise", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_update_entry_points_reject_invalid_noise(noise) -> None:
    """Dense fast/masked and sparse updates share the finite-positive contract."""
    import numpy as np

    from vbpca_py import dense_update_kernels as dense
    from vbpca_py import sparse_update_kernels as sparse

    x, a, s = np.ones((2, 4)), np.ones((2, 1)), np.ones((1, 4))
    data, indices, indptr = np.ones(2), np.array([0, 1]), np.array([0, 1, 2])
    calls = [
        lambda: dense.score_update_dense_no_av(x, a, noise),
        lambda: dense.loadings_update_dense_no_sv(x, s, np.eye(1), noise),
        lambda: dense.score_update_dense_masked_nopattern(x, x, a, noise_var=noise),
        lambda: dense.loadings_update_dense_masked_nopattern(
            x, x, s, prior_prec=np.eye(1), noise_var=noise
        ),
        lambda: sparse.score_update_sparse_nopattern(
            data, indices, indptr, a, noise_var=noise
        ),
        lambda: sparse.loadings_update_sparse_nopattern(
            data, indices, indptr, s[:, :2], prior_prec=np.eye(1), noise_var=noise
        ),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="noise_var must be finite and positive"):
            call()


@pytest.mark.parametrize("pointers", [[0, -1, 2], [0, 3, 2], [0, 2, 1, 2]])
@pytest.mark.parametrize("kind", ["score", "loading"])
def test_sparse_pointer_validation(pointers, kind) -> None:
    """Reject negative, out-of-bounds and descending pointers before iteration."""
    import numpy as np

    from vbpca_py import sparse_update_kernels as sparse

    args = (np.ones(2), np.array([0, 1]), np.array(pointers))
    if kind == "score":
        with pytest.raises(ValueError, match="indptr must be nondecreasing"):
            sparse.score_update_sparse_nopattern(*args, np.ones((2, 1)), noise_var=1.0)
    else:
        with pytest.raises(ValueError, match="indptr must be nondecreasing"):
            sparse.loadings_update_sparse_nopattern(
                *args, np.ones((1, 2)), prior_prec=np.eye(1), noise_var=1.0
            )
