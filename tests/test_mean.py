"""Tests for vbpca_py._mean.subtract_mu."""

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py._mean import subtract_mu


def _close(a: float, b: float, tol: float = 1e-12) -> bool:
    """Return True if |a - b| < tol."""
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Dense cases
# ---------------------------------------------------------------------------


def test_subtract_mu_dense_full_mask() -> None:
    """Dense x with full mask: subtract mu from every entry in each row."""
    x = np.array([[2.0, 3.0], [20.0, 40.0]])
    mu = np.array([1.0, 10.0])
    mask = np.ones_like(x)

    x_out, x_probe_out = subtract_mu(
        mu,
        x,
        mask,
        update_bias=True,
    )

    expected = np.array([[1.0, 2.0], [10.0, 30.0]])
    assert isinstance(x_out, np.ndarray)
    assert np.allclose(x_out, expected)
    assert x_probe_out is None


def test_subtract_mu_dense_with_masking() -> None:
    """Dense x with partial mask: only masked entries are shifted."""
    x = np.array([[2.0, 3.0], [20.0, 40.0]])
    mu = np.array([1.0, 10.0])
    # Only first column observed
    mask = np.array([[1.0, 0.0], [1.0, 0.0]])

    x_out, _ = subtract_mu(
        mu,
        x,
        mask,
        update_bias=True,
    )

    # First column: subtract mu; second column unchanged.
    expected = np.array([[1.0, 3.0], [10.0, 40.0]])
    assert np.allclose(x_out, expected)


def test_subtract_mu_dense_with_probe() -> None:
    """Dense x and x_probe both updated using their own masks."""
    x = np.array([[2.0, 4.0], [10.0, 20.0]])
    x_probe = np.array([[1.0, 2.0], [5.0, 8.0]])
    mu = np.array([1.0, 3.0])

    mask = np.ones_like(x)
    mask_probe = np.array([[1.0, 0.0], [0.0, 1.0]])

    x_out, x_probe_out = subtract_mu(
        mu,
        x,
        mask,
        x_probe=x_probe,
        mask_probe=mask_probe,
        update_bias=True,
    )

    expected_x = x - mu[:, None] * mask
    expected_probe = x_probe - mu[:, None] * mask_probe

    assert isinstance(x_out, np.ndarray)
    assert isinstance(x_probe_out, np.ndarray | type(None))
    assert x_probe_out is not None
    assert np.allclose(x_out, expected_x)
    assert np.allclose(x_probe_out, expected_probe)


def test_subtract_mu_update_bias_false_returns_unchanged() -> None:
    """When update_bias is False, x and x_probe are returned unchanged."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    x_probe = np.array([[5.0, 6.0], [7.0, 8.0]])
    mu = np.array([0.5, 1.0])
    mask = np.ones_like(x)

    x_out, x_probe_out = subtract_mu(
        mu,
        x,
        mask,
        x_probe=x_probe,
        mask_probe=mask,
        update_bias=False,
    )

    # Identity is a nice sanity check, but content is what matters.
    assert x_out is x
    assert x_probe_out is x_probe
    assert np.allclose(x_out, x)
    assert np.allclose(x_probe_out, x_probe)


# ---------------------------------------------------------------------------
# Sparse cases
# ---------------------------------------------------------------------------


def test_subtract_mu_sparse_basic() -> None:
    """Sparse x: subtract mu row-wise using the C++ helper."""
    # 2x3 CSR:
    # [1 0 2]
    # [0 3 0]
    data = np.array([1.0, 2.0, 3.0])
    indices = np.array([0, 2, 1])
    indptr = np.array([0, 2, 3])
    x = sp.csr_matrix((data, indices, indptr), shape=(2, 3))

    mu = np.array([1.0, 1.0])
    # Mask is ignored for sparse, but we must pass something.
    mask = np.ones((2, 3))

    x_out, x_probe_out = subtract_mu(
        mu,
        x,
        mask,
        update_bias=True,
    )

    assert sp.isspmatrix_csr(x_out)
    assert x_probe_out is None

    dense = x_out.toarray()
    # Expected: subtract 1 from each stored entry in its row
    expected = np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    # Zero entries may become EPS, so compare with tolerance.
    assert np.allclose(dense, expected, atol=1e-14)


def test_subtract_mu_sparse_zero_becomes_eps() -> None:
    """If subtraction yields exact zero, C++ helper replaces it with EPS."""
    # Single nonzero that becomes zero after subtracting mu
    data = np.array([1.0])
    indices = np.array([0])
    indptr = np.array([0, 1])
    x = sp.csr_matrix((data, indices, indptr), shape=(1, 1))

    mu = np.array([1.0])
    mask = np.ones((1, 1))

    x_out, _ = subtract_mu(
        mu,
        x,
        mask,
        update_bias=True,
    )

    assert sp.isspmatrix_csr(x_out)
    assert x_out.nnz == 1
    val = float(x_out.data[0])
    assert val != 0.0
    assert abs(val) <= 1e-14


def test_subtract_mu_sparse_with_probe() -> None:
    """Sparse x and sparse x_probe both updated."""
    # 2x2 CSR:
    # [1 0]
    # [0 2]
    data = np.array([1.0, 2.0])
    indices = np.array([0, 1])
    indptr = np.array([0, 1, 2])
    x = sp.csr_matrix((data, indices, indptr), shape=(2, 2))
    x_probe = x.copy()

    mu = np.array([1.0, 1.0])
    mask = np.ones((2, 2))  # ignored for sparse
    mask_probe = np.ones((2, 2))

    x_out, x_probe_out = subtract_mu(
        mu,
        x,
        mask,
        x_probe=x_probe,
        mask_probe=mask_probe,
        update_bias=True,
    )

    assert sp.isspmatrix_csr(x_out)
    assert sp.isspmatrix_csr(x_probe_out)

    dense = x_out.toarray()
    dense_probe = x_probe_out.toarray()
    expected = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert np.allclose(dense, expected, atol=1e-14)
    assert np.allclose(dense_probe, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_subtract_mu_shape_errors() -> None:
    """Shape mismatches should raise ValueError with informative messages."""
    x = np.zeros((2, 3))
    mu_wrong = np.array([1.0])  # wrong length
    mask = np.ones_like(x)

    with pytest.raises(ValueError, match="mu must have shape"):
        subtract_mu(
            mu_wrong,
            x,
            mask,
            update_bias=True,
        )

    mu = np.array([0.0, 0.0])
    bad_mask = np.ones((3, 3))

    with pytest.raises(ValueError, match="mask must have the same shape"):
        subtract_mu(
            mu,
            x,
            bad_mask,
            update_bias=True,
        )

    # Probe mask required if x_probe is given (dense case)
    x_probe = np.ones_like(x)
    with pytest.raises(ValueError, match="mask_probe must be provided"):
        subtract_mu(
            mu,
            x,
            mask,
            x_probe=x_probe,
            mask_probe=None,
            update_bias=True,
        )

    # Probe rows mismatch (dense)
    x_probe2 = np.ones((3, 3))
    mask_probe2 = np.ones_like(x_probe2)
    with pytest.raises(ValueError, match="x_probe must have the same number of rows"):
        subtract_mu(
            mu,
            x,
            mask,
            x_probe=x_probe2,
            mask_probe=mask_probe2,
            update_bias=True,
        )

    # Probe rows mismatch (sparse)
    xs = sp.csr_matrix(x)
    x_probe_sparse = sp.csr_matrix(np.ones((3, 3)))
    with pytest.raises(ValueError, match="x_probe must have the same number of rows"):
        subtract_mu(
            mu,
            xs,
            mask,
            x_probe=x_probe_sparse,
            mask_probe=mask_probe2,
            update_bias=True,
        )
