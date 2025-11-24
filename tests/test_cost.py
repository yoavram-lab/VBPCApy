"""Tests for the compute_full_cost function in vbpca_py._cost."""

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py._cost import CostParams, compute_full_cost


def _close(a: float, b: float, tol: float = 1e-10) -> bool:
    """Return True if |a - b| < tol."""
    return abs(a - b) < tol


def test_cost_basic_dense_no_priors() -> None:
    """Basic dense test with no priors and no covariance structure."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = np.array([[1.0, 2.0], [3.0, 4.0]])
    mu = np.zeros(2)
    noise_variance = 1.0

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        loading_priors=None,
    )

    cost, cost_x, cost_a, cost_mu, cost_s = compute_full_cost(x, a, s, params)

    assert np.isfinite(cost)
    assert np.isfinite(cost_x)
    assert np.isfinite(cost_a)
    assert np.isfinite(cost_mu)
    assert np.isfinite(cost_s)

    # Reference cost_x formula must mirror _compute_sxv:
    # s_xv = sum residual^2 + sum over mask of a_i @ I @ a_i^T
    residual = x - a @ s
    sse = float(np.sum(residual**2))

    mask = np.ones_like(x, dtype=bool)
    row_idx, _ = np.where(mask)
    row_norm_sq = np.sum(a**2, axis=1)
    extra = float(np.sum(row_norm_sq[row_idx]))

    n_data = mask.size
    expected_s_xv = sse + extra
    expected_cost_x = 0.5 * expected_s_xv / noise_variance + 0.5 * n_data * np.log(
        2.0 * np.pi * noise_variance
    )

    assert _close(cost_x, expected_cost_x, tol=1e-10)


def test_cost_sxv_equivalence() -> None:
    """Supplying s_xv should match the internally computed data term."""
    x = np.array([[2.0, 1.0], [0.0, 3.0]])
    a = np.eye(2)
    s = x.copy()  # perfect reconstruction: residual term is zero
    mu = np.zeros(2)
    noise_variance = 1.0
    mask = np.ones_like(x, dtype=bool)

    # First call: let compute_full_cost derive s_xv internally
    params1 = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        mask=mask,
        loading_priors=None,
    )

    cost1, cost_x1, *_ = compute_full_cost(x, a, s, params1)

    # Infer the effective n_data and implied s_xv from cost_x1:
    # cost_x = 0.5 * s_xv / V + 0.5 * n_data * log(2*pi*V)
    n_data = mask.size
    implied_s_xv = (
        2.0
        * noise_variance
        * (cost_x1 - 0.5 * n_data * np.log(2.0 * np.pi * noise_variance))
    )

    # Second call: pass s_xv explicitly and confirm cost_x is identical
    params2 = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        mask=mask,
        s_xv=implied_s_xv,
        n_data=n_data,
        loading_priors=None,
    )

    cost2, cost_x2, *_ = compute_full_cost(x, a, s, params2)

    assert _close(cost_x1, cost_x2, tol=1e-12)
    assert _close(cost1, cost2, tol=1e-12)


def test_cost_sparse_x_basic() -> None:
    """Sparse X path with exact reconstruction and dense mask."""
    data = np.array([5.0, -3.0])
    indices = np.array([0, 1])
    indptr = np.array([0, 1, 2])
    x = sp.csr_matrix((data, indices, indptr), shape=(2, 2))

    a = np.eye(2)
    s = np.array([[5.0, 0.0], [0.0, -3.0]])  # A @ S exactly equals X
    mu = np.zeros(2)
    noise_variance = 1.0
    mask = np.ones((2, 2), dtype=bool)

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        mask=mask,
        loading_priors=None,
    )

    _, cost_x, *_ = compute_full_cost(x, a, s, params)

    # residual term is zero; s_xv is purely from the Sv=I contributions:
    #   s_xv = sum over mask of a_i @ I @ a_i^T
    row_idx, _ = np.where(mask)
    row_norm_sq = np.sum(a**2, axis=1)
    extra = float(np.sum(row_norm_sq[row_idx]))

    n_data = mask.size
    expected_s_xv = extra
    expected_cost_x = 0.5 * expected_s_xv / noise_variance + 0.5 * n_data * np.log(
        2.0 * np.pi * noise_variance
    )

    assert _close(cost_x, expected_cost_x, tol=1e-12)


def test_cost_mean_prior_with_mu_variances() -> None:
    """Mean prior branch with finite mu_prior_variance and mu_variances."""
    x = np.array([[1.0, 1.0], [1.0, 1.0]])
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = np.zeros((2, 2))
    mu = np.array([0.5, -0.5])
    noise_variance = 1.0

    loading_priors = np.array([1.0, 1.0])
    mu_prior_variance = 2.0
    mu_variances = np.array([0.1, 0.2])
    mask = np.ones_like(x, dtype=bool)

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        loading_priors=loading_priors,
        mu_prior_variance=mu_prior_variance,
        mu_variances=mu_variances,
        mask=mask,
    )

    _, _, _, cost_mu, _ = compute_full_cost(x, a, s, params)

    n_features = x.shape[0]
    expected = (
        0.5 / mu_prior_variance * np.sum(mu**2 + mu_variances)
        - 0.5 * np.sum(np.log(mu_variances))
        + (n_features / 2.0) * np.log(mu_prior_variance)
        - (n_features / 2.0)
    )
    assert _close(cost_mu, float(expected), tol=1e-10)


def test_cost_loading_covariances_prior_branch() -> None:
    """Loading prior branch when Av is provided and Va is finite."""
    x = np.zeros((2, 2))
    a = np.eye(2)
    s = np.zeros((2, 2))
    mu = np.zeros(2)
    noise_variance = 1.0

    av0 = np.eye(2) * 0.5
    av1 = np.eye(2) * 0.25
    loading_covariances = [av0, av1]
    loading_priors = np.array([1.0, 2.0])

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        loading_covariances=loading_covariances,
        loading_priors=loading_priors,
    )

    _, _, cost_a, _, _ = compute_full_cost(x, a, s, params)

    assert np.isfinite(cost_a)
    assert cost_a > 0.0


def test_cost_score_covariances_pattern_index() -> None:
    """Score covariance pattern-indexed branch with Sv and score_pattern_index."""
    rng = np.random.default_rng(0)

    x = np.zeros((3, 4))
    a = rng.standard_normal((3, 2))
    s = rng.standard_normal((2, 4))
    mu = np.zeros(3)
    noise_variance = 1.0

    mask = np.ones_like(x, dtype=bool)
    sv0 = np.eye(2) * 0.3
    sv1 = np.eye(2) * 0.7
    score_covariances = [sv0, sv1]
    score_pattern_index = np.array([0, 1, 0, 1], dtype=int)

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        score_covariances=score_covariances,
        score_pattern_index=score_pattern_index,
        mask=mask,
        loading_priors=None,
    )

    cost, _, _, _, cost_s = compute_full_cost(x, a, s, params)

    assert np.isfinite(cost)
    assert np.isfinite(cost_s)


def test_cost_shape_errors() -> None:
    """Shape mismatches should raise ValueError with informative messages."""
    x = np.zeros((3, 4))
    a = np.zeros((2, 2))
    s = np.zeros((2, 4))
    mu = np.zeros(3)
    params = CostParams(mu=mu, noise_variance=1.0)

    with pytest.raises(ValueError, match=r"A rows must match X rows."):
        compute_full_cost(x, a, s, params)

    a = np.zeros((3, 2))
    s = np.zeros((3, 4))
    with pytest.raises(ValueError, match="latent dimensions are incompatible"):
        compute_full_cost(x, a, s, params)

    a = np.zeros((3, 2))
    s = np.zeros((2, 3))
    with pytest.raises(ValueError, match=r"S columns must match X columns."):
        compute_full_cost(x, a, s, params)


def test_cost_loading_nondiagonal_covariances() -> None:
    """Loading prior with non-diagonal Av to exercise full KL(A) branch."""
    rng = np.random.default_rng(123)

    # Small random A, no data contribution (x and s zero)
    x = np.zeros((3, 3))
    a = rng.standard_normal((3, 2))  # n_features = 3, n_components = 2
    s = np.zeros((2, 3))
    mu = np.zeros(3)
    noise_variance = 1.0

    # Non-diagonal, positive-definite covariance matrices for A
    base = np.array([[0.6, 0.2], [0.2, 0.4]])
    # One covariance per feature (row of A)
    av0 = base.copy()
    av1 = base * 1.5
    av2 = base * 2.0
    loading_covariances = [av0, av1, av2]

    # Simple finite loading priors (one per component)
    loading_priors = np.array([1.0, 0.5])

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        loading_covariances=loading_covariances,
        loading_priors=loading_priors,
    )

    _, _, cost_a, _, _ = compute_full_cost(x, a, s, params)

    # We don't assert an exact closed-form here (the KL is messy),
    # but we do expect:
    # - finiteness
    # - strict positivity (non-zero A, finite prior/Av => KL > 0)
    assert np.isfinite(cost_a)
    assert cost_a > 0.0


def test_cost_mask_ignores_unobserved_entries() -> None:
    """Changing X outside the mask should not affect the data term."""
    rng = np.random.default_rng(42)

    # Small dense X, random A and S
    x = rng.standard_normal((3, 4))
    a = rng.standard_normal((3, 2))
    s = rng.standard_normal((2, 4))
    mu = np.zeros(3)
    noise_variance = 1.0

    # Random mask: about half of entries observed
    mask = rng.random(x.shape) < 0.5

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        mask=mask,
        loading_priors=None,
    )

    # First cost with original X
    _, cost_x1, *_ = compute_full_cost(x, a, s, params)

    # Create a modified X that differs ONLY where mask is False
    x_modified = x.copy()
    x_modified[~mask] += 10.0  # big change in unobserved entries

    _, cost_x2, *_ = compute_full_cost(x_modified, a, s, params)

    # Data term must be invariant to changes outside mask
    assert _close(cost_x1, cost_x2, tol=1e-10)
