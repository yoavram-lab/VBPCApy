# tests/test_rms.py
"""Tests for the _rms module functionality."""

import numpy as np
import pytest
import scipy.sparse as sp
from vbpca_py._rms import compute_rms


def _close(a: float, b: float, tol=1e-12) -> bool:
    """Helper to compare floating point values."""
    return abs(a - b) < tol


def test_rms_dense_exact_zero() -> None:
    """RMS should be zero when X == A @ S exactly, with full mask."""
    A = np.array([[1.0, 0.0], [0.0, 2.0]])
    S = np.array([[2.0, 3.0], [1.0, 1.0]])
    X = A @ S
    M = np.ones_like(X)
    n_data = X.size

    rms, err = compute_rms(X, A, S, M, n_data)

    assert _close(rms, 0.0)
    assert isinstance(err, np.ndarray)
    assert np.allclose(err, 0.0)


def test_rms_dense_masked() -> None:
    """RMS should be computed correctly with a masked input."""
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    S = np.array([[2.0, 4.0], [1.0, 3.0]])
    X = np.array([[3.0, 5.0], [1.0, 2.0]])

    # Residual = X - A@S
    residual = X - A @ S

    # Mask only the first column
    M = np.array([[1.0, 0.0], [1.0, 0.0]])

    # Observed entries:
    observed_sq_sum = residual[:, 0] ** 2
    expected_rms = np.sqrt(np.sum(observed_sq_sum) / 2)

    rms, err = compute_rms(X, A, S, M, n_data=2)

    assert _close(rms, expected_rms)
    assert err.shape == X.shape
    assert np.allclose(err[:, 1], 0.0)  # second column masked to zero


def test_rms_sparse_X() -> None:
    """RMS computation with sparse X input."""
    # Small CSR matrix with only two nonzeros
    data = np.array([10.0, -4.0])
    indices = np.array([0, 1])
    indptr = np.array([0, 1, 2])
    X = sp.csr_matrix((data, indices, indptr), shape=(2, 2))

    # Choose A, S so that A@S = zero matrix => err = X
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    S = np.array([[0.0, 0.0], [0.0, 0.0]])

    # Mask all observed entries (dense ones)
    M = np.ones((2, 2))

    n_data = X.nnz  # two observed entries

    out = []
    for cpu in [1, 2]:
        rms, err = compute_rms(X, A, S, M, n_data, num_cpu=cpu)
        out.append((rms, err))

        # Error must be CSR
        assert sp.isspmatrix_csr(err)
        assert np.array_equal(err.data, data)

    # RMS must match across cpu settings
    assert _close(out[0][0], out[1][0])
    expected_rms = np.sqrt((10.0**2 + (-4.0) ** 2) / 2)
    assert _close(out[0][0], expected_rms)


def test_rms_empty_X() -> None:
    """Empty X input should yield NaN RMS and empty error matrix."""
    X = np.empty((0, 0))
    A = np.empty((0, 1))
    S = np.empty((1, 0))
    M = np.empty((0, 0))

    rms, err = compute_rms(X, A, S, M, n_data=0)

    assert np.isnan(rms)
    assert isinstance(err, np.ndarray)
    assert err.shape == (0, 0)


def test_rms_shape_errors() -> None:
    """Mismatched shapes should raise ValueErrors."""
    X = np.zeros((3, 4))
    A = np.zeros((2, 2))  # wrong row count
    S = np.zeros((2, 4))
    M = np.ones_like(X)

    with pytest.raises(ValueError, match="A has"):
        compute_rms(X, A, S, M, n_data=12)

    # Second shape mismatch
    A = np.zeros((3, 2))
    S = np.zeros((3, 4))  # wrong k dimension
    with pytest.raises(ValueError, match="Incompatible latent"):
        compute_rms(X, A, S, M, n_data=12)

    # Third mismatch: S columns
    A = np.zeros((3, 2))
    S = np.zeros((2, 3))  # X has 4 samples
    with pytest.raises(ValueError, match="S has"):
        compute_rms(X, A, S, M, n_data=12)
