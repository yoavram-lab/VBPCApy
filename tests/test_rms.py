# tests/test_rms.py
"""Tests for the _rms module functionality."""

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py._rms import RmsConfig, compute_rms


def _close(a: float, b: float, tol: float = 1e-12) -> bool:
    """Helper to compare floating point values."""
    return abs(a - b) < tol


def test_rms_dense_exact_zero() -> None:
    """RMS should be zero when data == loadings @ scores exactly, with full mask."""
    loadings = np.array([[1.0, 0.0], [0.0, 2.0]])
    scores = np.array([[2.0, 3.0], [1.0, 1.0]])
    data = loadings @ scores
    mask = np.ones_like(data)

    config = RmsConfig(n_observed=data.size, num_cpu=1)

    rms, err = compute_rms(data, loadings, scores, mask, config)

    assert _close(rms, 0.0)
    assert isinstance(err, np.ndarray)
    assert np.allclose(err, 0.0)


def test_rms_dense_masked() -> None:
    """RMS should be computed correctly with a masked input."""
    loadings = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = np.array([[2.0, 4.0], [1.0, 3.0]])
    data = np.array([[3.0, 5.0], [1.0, 2.0]])

    residual = data - loadings @ scores

    # Mask only the first column
    mask = np.array([[1.0, 0.0], [1.0, 0.0]])

    observed_sq_sum = residual[:, 0] ** 2
    expected_rms = np.sqrt(np.sum(observed_sq_sum) / 2)

    config = RmsConfig(n_observed=2, num_cpu=1)

    rms, err = compute_rms(data, loadings, scores, mask, config)

    assert _close(rms, expected_rms)
    assert err.shape == data.shape
    assert np.allclose(err[:, 1], 0.0)  # second column masked to zero


def test_rms_sparse_data() -> None:
    """RMS computation with sparse data input."""
    # Small CSR matrix with only two nonzeros
    data_vals = np.array([10.0, -4.0])
    indices = np.array([0, 1])
    indptr = np.array([0, 1, 2])
    data = sp.csr_matrix((data_vals, indices, indptr), shape=(2, 2))

    # Choose loadings, scores so that loadings @ scores = zero matrix => err = data
    loadings = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = np.array([[0.0, 0.0], [0.0, 0.0]])

    # Mask all observed entries (dense ones)
    mask = np.ones((2, 2))

    n_observed = data.nnz  # two observed entries

    results = []
    for cpu in [1, 2]:
        config = RmsConfig(n_observed=n_observed, num_cpu=cpu)
        rms, err = compute_rms(data, loadings, scores, mask, config)
        results.append((rms, err))

        # Error must be CSR
        assert sp.isspmatrix_csr(err)
        assert np.array_equal(err.data, data_vals)

    # RMS must match across cpu settings
    assert _close(results[0][0], results[1][0])
    expected_rms = np.sqrt((10.0**2 + (-4.0) ** 2) / 2)
    assert _close(results[0][0], expected_rms)


def test_rms_empty_data() -> None:
    """Empty data input should yield NaN RMS and empty error matrix."""
    data = np.empty((0, 0))
    loadings = np.empty((0, 1))
    scores = np.empty((1, 0))
    mask = np.empty((0, 0))

    config = RmsConfig(n_observed=0, num_cpu=1)

    rms, err = compute_rms(data, loadings, scores, mask, config)

    assert np.isnan(rms)
    assert isinstance(err, np.ndarray)
    assert err.shape == (0, 0)


def test_rms_shape_errors() -> None:
    """Mismatched shapes should raise ValueErrors."""
    data = np.zeros((3, 4))
    mask = np.ones_like(data)
    config = RmsConfig(n_observed=12, num_cpu=1)

    # First mismatch: loadings rows vs data rows
    loadings = np.zeros((2, 2))  # wrong row count
    scores = np.zeros((2, 4))
    with pytest.raises(ValueError, match="loadings has"):
        compute_rms(data, loadings, scores, mask, config)

    # Second mismatch: latent dimension
    loadings = np.zeros((3, 2))
    scores = np.zeros((3, 4))  # wrong k dimension
    with pytest.raises(ValueError, match="Incompatible latent"):
        compute_rms(data, loadings, scores, mask, config)

    # Third mismatch: scores columns vs data columns
    loadings = np.zeros((3, 2))
    scores = np.zeros((2, 3))  # data has 4 samples
    with pytest.raises(ValueError, match="scores has"):
        compute_rms(data, loadings, scores, mask, config)
