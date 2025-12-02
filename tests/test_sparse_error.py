"""Tests for the sparse reconstruction error wrapper around the C++ extension."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

# Skip this whole module if the extension is not available
pytest.importorskip("vbpca_py.errpca_pt")

from vbpca_py._sparse_error import sparse_reconstruction_error


def _make_test_matrices() -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Small helper to build a consistent X, loadings, scores triple.

    We choose loadings and scores to be all zeros so that loadings @ scores == 0,
    and the reconstruction error on non-zero entries of X is just X itself.
    """
    x_dense = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    x_csr = sp.csr_matrix(x_dense)

    n_rows, n_cols = x_dense.shape
    n_components = 2

    loadings = np.zeros((n_rows, n_components), dtype=float)
    scores = np.zeros((n_components, n_cols), dtype=float)

    return x_csr, loadings, scores, x_dense


def test_sparse_reconstruction_error_basic_single_thread() -> None:
    """Basic correctness and sparsity structure with num_cpu=1."""
    x_csr, loadings, scores, x_dense = _make_test_matrices()

    err = sparse_reconstruction_error(x_csr, loadings, scores, num_cpu=1)

    assert sp.isspmatrix_csr(err)
    assert err.shape == x_csr.shape

    # Because loadings @ scores == 0, the error on non-zero entries is just X itself.
    # The C++ code only iterates over non-zeros of X, so the sparsity
    # structure should match exactly in this case.
    np.testing.assert_array_equal(err.indptr, x_csr.indptr)
    np.testing.assert_array_equal(err.indices, x_csr.indices)
    np.testing.assert_array_almost_equal(err.data, x_csr.data)

    # And the dense reconstruction matches on those entries.
    err_dense = err.toarray()
    np.testing.assert_array_almost_equal(err_dense, x_dense)


@pytest.mark.parametrize("num_cpu", [1, 2, 4])
def test_sparse_reconstruction_error_multi_thread(num_cpu: int) -> None:
    """Check that different num_cpu values produce the same result."""
    x_csr, loadings, scores, x_dense = _make_test_matrices()

    err = sparse_reconstruction_error(x_csr, loadings, scores, num_cpu=num_cpu)

    assert sp.isspmatrix_csr(err)
    assert err.shape == x_csr.shape

    err_dense = err.toarray()
    np.testing.assert_array_almost_equal(err_dense, x_dense)


def test_sparse_reconstruction_error_non_csr_raises() -> None:
    """Non-CSR inputs should raise a TypeError."""
    x_csc = sp.csc_matrix([[1.0]])  # wrong format
    loadings = np.zeros((1, 1), dtype=float)
    scores = np.zeros((1, 1), dtype=float)

    with pytest.raises(TypeError, match="csr_matrix"):
        sparse_reconstruction_error(x_csc, loadings, scores)
