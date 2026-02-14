"""
End-to-end tests for compute_full_cost in vbpca_py._cost.

Includes:
- Python-only correctness/unit tests (including new "sharp edge" cases)
- Octave regression tests against tools/cf_full.m

Octave regression:
- Dense regression: requires octave on PATH.
- Sparse regression: optional; auto-compiles errpca_pt + subtract_mu MEX into tools/ using mkoctfile.

Repo assumptions:
- tools/cf_full.m exists
- tools/compute_rms.m exists (cf_full calls it)
- tools/errpca_pt.cpp exists (for sparse compute_rms branch)
- tools/subtract_mu.cpp exists (for sparse mean subtraction in regression script)
"""

from __future__ import annotations

import atexit
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.io import loadmat, savemat

from vbpca_py._cost import CostParams, compute_full_cost


def _close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(a - b) < tol


# ======================================================================================
# Python-only unit tests (existing + updates for mu-centered X semantics)
# ======================================================================================


def test_cost_basic_dense_no_priors() -> None:
    """Basic dense test with no priors and no covariance structure (mu=0)."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = np.array([[1.0, 2.0], [3.0, 4.0]])
    mu = np.zeros(2)
    noise_variance = 1.0

    params = CostParams(mu=mu, noise_variance=noise_variance, loading_priors=None)

    cost, cost_x, cost_a, cost_mu, cost_s = compute_full_cost(x, a, s, params)

    assert np.isfinite(cost)
    assert np.isfinite(cost_x)
    assert np.isfinite(cost_a)
    assert np.isfinite(cost_mu)
    assert np.isfinite(cost_s)

    # With mu=0 and full observation, x_centered == x.
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
    s = x.copy()
    mu = np.zeros(2)
    noise_variance = 1.0
    mask = np.ones_like(x, dtype=bool)

    params1 = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        mask=mask,
        loading_priors=None,
    )

    cost1, cost_x1, *_ = compute_full_cost(x, a, s, params1)

    n_data = int(np.sum(mask))
    implied_s_xv = (
        2.0
        * noise_variance
        * (cost_x1 - 0.5 * n_data * np.log(2.0 * np.pi * noise_variance))
    )

    params2 = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        mask=mask,
        s_xv=float(implied_s_xv),
        n_data=n_data,
        loading_priors=None,
    )

    cost2, cost_x2, *_ = compute_full_cost(x, a, s, params2)

    assert _close(cost_x1, cost_x2, tol=1e-12)
    assert _close(cost1, cost2, tol=1e-12)


def test_cost_sparse_x_basic() -> None:
    """Sparse X path with exact reconstruction and mask matching structure (mu=0)."""
    data = np.array([5.0, -3.0])
    indices = np.array([0, 1])
    indptr = np.array([0, 1, 2])
    x = sp.csr_matrix((data, indices, indptr), shape=(2, 2))

    a = np.eye(2)
    s = np.array([[5.0, 0.0], [0.0, -3.0]])  # A @ S exactly equals X
    mu = np.zeros(2)
    noise_variance = 1.0

    # Option A: mask must match the sparsity pattern of X (spones(X))
    mask = x.copy()
    mask.data[:] = 1.0  # values don't matter beyond nonzero-ness

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        mask=mask,
        loading_priors=None,
    )

    _, cost_x, *_ = compute_full_cost(x, a, s, params)

    # With MATLAB sparse semantics:
    # n_data = nnz(X) (NOT mask.size)
    n_data = int(x.nnz)

    # Residual term is zero on observed entries because A@S matches X exactly
    # Sv=I contribution: sum over observed entries of ||a_i||^2
    row_idx, _ = mask.nonzero()
    row_norm_sq = np.sum(a**2, axis=1)
    extra = float(np.sum(row_norm_sq[row_idx]))

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


def test_cost_requires_positive_noise_variance() -> None:
    x = np.zeros((2, 2))
    a = np.eye(2)
    s = np.zeros((2, 2))
    mu = np.zeros(2)

    with pytest.raises(ValueError, match="noise_variance must be positive"):
        compute_full_cost(x, a, s, CostParams(mu=mu, noise_variance=0.0))

    with pytest.raises(ValueError, match="noise_variance must be positive"):
        compute_full_cost(x, a, s, CostParams(mu=mu, noise_variance=-1.0))


def test_cost_loading_covariances_validation() -> None:
    x = np.zeros((2, 2))
    a = np.eye(2)
    s = np.zeros((2, 2))
    mu = np.zeros(2)

    bad_len = [np.eye(2)]  # length 1 instead of 2
    with pytest.raises(ValueError, match="loading_covariances length"):
        compute_full_cost(
            x,
            a,
            s,
            CostParams(mu=mu, noise_variance=1.0, loading_covariances=bad_len),
        )

    bad_shape = [np.eye(2), np.ones((2, 1))]
    with pytest.raises(ValueError, match="loading covariance must have shape"):
        compute_full_cost(
            x,
            a,
            s,
            CostParams(mu=mu, noise_variance=1.0, loading_covariances=bad_shape),
        )


def test_cost_score_covariances_validation() -> None:
    x = np.zeros((2, 3))
    a = np.eye(2)
    s = np.zeros((2, 3))
    mu = np.zeros(2)

    # Non-patterned: length mismatch
    sv_list_short = [np.eye(2)]
    with pytest.raises(ValueError, match="score_covariances length"):
        compute_full_cost(
            x,
            a,
            s,
            CostParams(
                mu=mu,
                noise_variance=1.0,
                score_covariances=sv_list_short,
                score_pattern_index=None,
            ),
        )

    # Patterned: list too short for pattern index
    sv_list = [np.eye(2)]
    pat = np.array([0, 1, 0], dtype=int)
    with pytest.raises(ValueError, match="out-of-range"):
        compute_full_cost(
            x,
            a,
            s,
            CostParams(
                mu=mu,
                noise_variance=1.0,
                score_covariances=sv_list,
                score_pattern_index=pat,
            ),
        )

    # Shape mismatch
    sv_bad_shape = [np.eye(2), np.ones((1, 1))]
    pat_ok = np.array([0, 1, 0], dtype=int)
    with pytest.raises(ValueError, match="score covariance must have shape"):
        compute_full_cost(
            x,
            a,
            s,
            CostParams(
                mu=mu,
                noise_variance=1.0,
                score_covariances=sv_bad_shape,
                score_pattern_index=pat_ok,
            ),
        )


def test_cost_nan_observed_with_mask_raises() -> None:
    x = np.array([[np.nan, 1.0], [2.0, 3.0]])
    a = np.eye(2)
    s = np.zeros((2, 2))
    mu = np.zeros(2)
    mask = np.ones_like(x, dtype=bool)

    with pytest.raises(ValueError, match="NaN on observed entries"):
        compute_full_cost(x, a, s, CostParams(mu=mu, noise_variance=1.0, mask=mask))


def test_cost_sxv_requires_positive_n_data() -> None:
    x = np.zeros((2, 2))
    a = np.eye(2)
    s = np.zeros((2, 2))
    mu = np.zeros(2)
    mask = np.zeros_like(x, dtype=bool)  # zero observed entries

    with pytest.raises(ValueError, match="n_data must be positive"):
        compute_full_cost(
            x,
            a,
            s,
            CostParams(mu=mu, noise_variance=1.0, mask=mask, s_xv=1.0),
        )


def test_cost_loading_nondiagonal_covariances() -> None:
    """Loading prior with non-diagonal Av to exercise full KL(A) branch."""
    rng = np.random.default_rng(123)
    x = np.zeros((3, 3))
    a = rng.standard_normal((3, 2))
    s = np.zeros((2, 3))
    mu = np.zeros(3)
    noise_variance = 1.0

    base = np.array([[0.6, 0.2], [0.2, 0.4]])
    loading_covariances = [base.copy(), base * 1.5, base * 2.0]
    loading_priors = np.array([1.0, 0.5])

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        loading_covariances=loading_covariances,
        loading_priors=loading_priors,
    )

    _, _, cost_a, _, _ = compute_full_cost(x, a, s, params)
    assert np.isfinite(cost_a)
    assert cost_a > 0.0


def test_cost_mask_ignores_unobserved_entries() -> None:
    """Changing X outside the mask should not affect the data term (with mu-centering)."""
    rng = np.random.default_rng(42)

    x = rng.standard_normal((3, 4))
    a = rng.standard_normal((3, 2))
    s = rng.standard_normal((2, 4))
    mu = rng.standard_normal(3)
    noise_variance = 1.0

    mask = rng.random(x.shape) < 0.5

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        mask=mask,
        loading_priors=None,
    )

    _, cost_x1, *_ = compute_full_cost(x, a, s, params)

    x_modified = x.copy()
    x_modified[~mask] += 10.0  # big change in unobserved entries

    _, cost_x2, *_ = compute_full_cost(x_modified, a, s, params)

    assert _close(cost_x1, cost_x2, tol=1e-10)


# ---------------------------------------------------------------------------
# Additional tests for s_xv refactor and branches
# (Updated where needed: residual uses mean-centered X)
# ---------------------------------------------------------------------------


def test_cost_sxv_identity_with_loading_covariances_matches_reference() -> None:
    """Identity Sv with Av present: cost_x matches explicit s_xv reference (mu=0)."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = np.array([[0.5, 1.0], [1.5, -0.5]])
    mu = np.zeros(2)
    noise_variance = 1.0

    av0 = np.diag([0.5, 0.2])
    av1 = np.diag([0.3, 0.7])
    loading_covariances = [av0, av1]

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        loading_covariances=loading_covariances,
        loading_priors=None,
    )

    _, cost_x, *_ = compute_full_cost(x, a, s, params)

    residual = x - a @ s
    sse = float(np.sum(residual**2))

    mask = np.ones_like(x, dtype=bool)
    row_idx, col_idx = np.where(mask)
    n_data = mask.size

    row_norm_sq = np.sum(a**2, axis=1)
    sv_contrib = float(np.sum(row_norm_sq[row_idx]))

    av_contrib = 0.0
    for i, j in zip(row_idx, col_idx, strict=True):
        av_i = loading_covariances[i]
        s_j = s[:, j]
        av_contrib += float(s_j.T @ av_i @ s_j)
        av_contrib += float(np.trace(av_i))

    expected_s_xv = sse + sv_contrib + av_contrib
    expected_cost_x = 0.5 * expected_s_xv / noise_variance + 0.5 * n_data * np.log(
        2.0 * np.pi * noise_variance
    )

    assert _close(cost_x, expected_cost_x, tol=1e-10)


def test_cost_sxv_patterned_sv_fastpath_matches_reference() -> None:
    """Pattern-index Sv fast path: cost_x matches explicit s_xv reference (mu=0)."""
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    s = np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]])
    x = a @ s
    mu = np.zeros(2)
    noise_variance = 1.0

    sv0 = np.array([[1.0, 0.0], [0.0, 2.0]])
    sv1 = np.array([[0.5, 0.1], [0.1, 1.5]])
    score_covariances = [sv0, sv1]
    score_pattern_index = np.array([0, 1, 0, 1], dtype=int)

    mask = np.ones_like(x, dtype=bool)

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        score_covariances=score_covariances,
        score_pattern_index=score_pattern_index,
        mask=mask,
        loading_priors=None,
    )

    _, cost_x, *_ = compute_full_cost(x, a, s, params)

    row_idx, col_idx = np.where(mask)
    n_data = mask.size

    sv_contrib = 0.0
    for i, j in zip(row_idx, col_idx, strict=True):
        sv_j = score_covariances[score_pattern_index[j]]
        a_i = a[i, :]
        sv_contrib += float(a_i @ sv_j @ a_i.T)

    expected_s_xv = sv_contrib
    expected_cost_x = 0.5 * expected_s_xv / noise_variance + 0.5 * n_data * np.log(
        2.0 * np.pi * noise_variance
    )

    assert _close(cost_x, expected_cost_x, tol=1e-10)


def test_cost_sxv_general_sv_and_av_matches_reference() -> None:
    """General per-observation Sv+Av branch matches explicit s_xv reference (mu=0)."""
    x = np.zeros((2, 2))
    a = np.array([[1.0, 0.5], [0.2, -0.3]])
    s = np.array([[0.5, -1.0], [1.0, 0.3]])
    mu = np.zeros(2)
    noise_variance = 1.0

    sv0 = np.array([[1.0, 0.2], [0.2, 0.5]])
    sv1 = np.array([[0.7, 0.1], [0.1, 0.9]])
    score_covariances = [sv0, sv1]

    av0 = np.array([[0.4, 0.0], [0.0, 0.6]])
    av1 = np.array([[0.3, 0.1], [0.1, 0.8]])
    loading_covariances = [av0, av1]

    mask = np.ones_like(x, dtype=bool)

    params = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        loading_covariances=loading_covariances,
        score_covariances=score_covariances,
        score_pattern_index=None,
        mask=mask,
        loading_priors=None,
    )

    _, cost_x, *_ = compute_full_cost(x, a, s, params)

    row_idx, col_idx = np.where(mask)
    n_data = mask.size

    residual = x - a @ s
    sse = float(np.sum(residual**2))

    var_contrib = 0.0
    for i, j in zip(row_idx, col_idx, strict=True):
        a_i = a[i, :]
        s_j = s[:, j]
        sv_j = score_covariances[j]
        av_i = loading_covariances[i]

        var_contrib += float(a_i @ sv_j @ a_i.T)
        var_contrib += float(s_j.T @ av_i @ s_j)
        var_contrib += float(np.sum(sv_j * av_i))

    expected_s_xv = sse + var_contrib
    expected_cost_x = 0.5 * expected_s_xv / noise_variance + 0.5 * n_data * np.log(
        2.0 * np.pi * noise_variance
    )

    assert _close(cost_x, expected_cost_x, tol=1e-10)


def test_cost_sxv_mu_variances_contribution_only() -> None:
    """mu_variances contribution to s_xv matches expected delta in cost_x."""
    x = np.zeros((2, 3))
    a = np.zeros((2, 1))
    s = np.zeros((1, 3))
    mu = np.zeros(2)
    noise_variance = 1.0

    mask = np.ones_like(x, dtype=bool)
    row_idx, _ = np.where(mask)

    mu_variances = np.array([0.5, 1.5])

    params_no_muvar = CostParams(
        mu=mu, noise_variance=noise_variance, mask=mask, loading_priors=None
    )
    _, cost_x_no_muvar, *_ = compute_full_cost(x, a, s, params_no_muvar)

    params_with_muvar = CostParams(
        mu=mu,
        noise_variance=noise_variance,
        mask=mask,
        mu_variances=mu_variances,
        loading_priors=None,
    )
    _, cost_x_with_muvar, *_ = compute_full_cost(x, a, s, params_with_muvar)

    delta_s_xv = float(np.sum(mu_variances[row_idx]))
    expected_delta_cost_x = 0.5 * delta_s_xv / noise_variance

    assert _close(cost_x_with_muvar - cost_x_no_muvar, expected_delta_cost_x, tol=1e-10)


def test_cost_sparse_num_cpu_invariance() -> None:
    """Changing num_cpu in CostParams should not change the cost."""
    data_vals = np.array([2.0, -1.0])
    indices = np.array([0, 1])
    indptr = np.array([0, 1, 2])
    x = sp.csr_matrix((data_vals, indices, indptr), shape=(2, 2))

    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = np.zeros((2, 2))
    mu = np.zeros(2)
    noise_variance = 1.0

    # For sparse, the canonical observed set is the structure of X; pass a matching mask.
    mask = x.copy()
    mask.data[:] = 1.0

    params_cpu1 = CostParams(
        mu=mu, noise_variance=noise_variance, mask=mask, loading_priors=None, num_cpu=1
    )
    params_cpu4 = CostParams(
        mu=mu, noise_variance=noise_variance, mask=mask, loading_priors=None, num_cpu=4
    )

    _, cost_x1, *_ = compute_full_cost(x, a, s, params_cpu1)
    _, cost_x4, *_ = compute_full_cost(x, a, s, params_cpu4)

    assert _close(cost_x1, cost_x4, tol=1e-12)


# ======================================================================================
# New "sharp edge" tests (mu-centering correctness)
# ======================================================================================


def test_cost_mu_affects_only_observed_entries_dense() -> None:
    """
    Changing mu should only affect cost_x via mean-centering on observed entries.
    Unobserved entries must not contribute.
    """
    x = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    a = np.zeros((2, 1))
    s = np.zeros((1, 3))
    noise_variance = 2.0

    # Observe only first and third columns in row 0, and only second column in row 1
    mask = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    mu0 = np.array([0.0, 0.0])
    mu1 = np.array([5.0, -1.0])

    p0 = CostParams(
        mu=mu0, noise_variance=noise_variance, mask=mask, loading_priors=None
    )
    p1 = CostParams(
        mu=mu1, noise_variance=noise_variance, mask=mask, loading_priors=None
    )

    _, cost_x0, *_ = compute_full_cost(x, a, s, p0)
    _, cost_x1, *_ = compute_full_cost(x, a, s, p1)

    # Reference delta: only observed entries get shifted by mu.
    m_bool = mask != 0
    x0c = x - mu0[:, None] * m_bool
    x1c = x - mu1[:, None] * m_bool

    # With A=S=0, residual == x_centered on observed set, and rms is over observed entries.
    # s_xv = sum( (x_centered ⊙ M)^2 ) + Sv=I term (but A=0 => Sv term is 0)
    err0 = x0c * m_bool
    err1 = x1c * m_bool
    ndata = int(np.sum(m_bool))
    s_xv0 = float(np.sum(err0**2))
    s_xv1 = float(np.sum(err1**2))

    expected_cost_x0 = 0.5 * s_xv0 / noise_variance + 0.5 * ndata * np.log(
        2.0 * np.pi * noise_variance
    )
    expected_cost_x1 = 0.5 * s_xv1 / noise_variance + 0.5 * ndata * np.log(
        2.0 * np.pi * noise_variance
    )

    assert _close(cost_x0, expected_cost_x0, tol=1e-10)
    assert _close(cost_x1, expected_cost_x1, tol=1e-10)


def test_cost_nan_mask_inference_dense_matches_explicit_mask() -> None:
    """When mask is None and X has NaNs, inferred mask should match explicit and centering should match."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 5))
    nan_mask = rng.random(x.shape) < 0.3
    x_nan = x.copy()
    x_nan[nan_mask] = np.nan

    a = rng.standard_normal((4, 2))
    s = rng.standard_normal((2, 5))
    mu = rng.standard_normal(4)
    noise_variance = 1.3

    # Case 1: mask inferred from NaNs
    p_infer = CostParams(
        mu=mu, noise_variance=noise_variance, mask=None, loading_priors=None
    )
    cost_infer, cost_x_infer, cost_a_infer, cost_mu_infer, cost_s_infer = (
        compute_full_cost(x_nan, a, s, p_infer)
    )

    # Case 2: explicit mask and NaNs already replaced by 0 like legacy preprocessing
    m_exp = ~nan_mask
    x_clean = x_nan.copy()
    x_clean[np.isnan(x_clean)] = 0.0

    p_exp = CostParams(
        mu=mu, noise_variance=noise_variance, mask=m_exp, loading_priors=None
    )
    cost_exp, cost_x_exp, cost_a_exp, cost_mu_exp, cost_s_exp = compute_full_cost(
        x_clean, a, s, p_exp
    )

    assert _close(cost_infer, cost_exp, tol=1e-10)
    assert _close(cost_x_infer, cost_x_exp, tol=1e-10)
    assert _close(cost_a_infer, cost_a_exp, tol=1e-10)
    assert _close(cost_mu_infer, cost_mu_exp, tol=1e-10)
    assert _close(cost_s_infer, cost_s_exp, tol=1e-10)


def test_cost_float_mask_treated_as_observed_nonzero() -> None:
    """Non-0 entries in a float mask must be treated as observed."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = np.zeros((2, 1))
    s = np.zeros((1, 2))
    mu = np.array([1.0, -2.0])
    noise_variance = 1.0

    mask_float = np.array([[2.0, 0.0], [-1.0, 0.0]])  # nonzero means observed

    p = CostParams(
        mu=mu, noise_variance=noise_variance, mask=mask_float, loading_priors=None
    )
    _, cost_x, *_ = compute_full_cost(x, a, s, p)

    m_bool = mask_float != 0
    x_centered = x - mu[:, None] * m_bool
    err = x_centered * m_bool
    ndata = int(np.sum(m_bool))
    expected_s_xv = float(np.sum(err**2))  # A=S=0 => no Sv term
    expected_cost_x = 0.5 * expected_s_xv + 0.5 * ndata * np.log(2.0 * np.pi)

    assert _close(cost_x, expected_cost_x, tol=1e-10)


def test_cost_pattern_index_validation_bounds() -> None:
    """Out-of-range score_pattern_index must raise."""
    x = np.zeros((2, 3))
    a = np.zeros((2, 1))
    s = np.zeros((1, 3))
    mu = np.zeros(2)

    sv0 = np.eye(1)
    score_covariances = [sv0]

    bad_index = np.array([0, 1, 0], dtype=int)  # '1' out of range for len=1

    p = CostParams(
        mu=mu,
        noise_variance=1.0,
        score_covariances=score_covariances,
        score_pattern_index=bad_index,
        loading_priors=None,
    )

    with pytest.raises(ValueError, match="out-of-range"):
        compute_full_cost(x, a, s, p)


# ======================================================================================
# Octave regression helpers
# ======================================================================================


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tools_dir() -> Path:
    return _repo_root() / "tools"


def _octave_available() -> bool:
    return shutil.which("octave") is not None


def _mkoctfile_available() -> bool:
    return shutil.which("mkoctfile") is not None


def _run_octave_eval(script: str) -> None:
    proc = subprocess.run(
        ["octave", "--quiet", "--eval", script],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Octave failed.\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
            f"--- script ---\n{script}\n"
        )


def _find_mex(tools: Path, stem: str) -> list[Path]:
    return sorted(
        [
            p
            for p in tools.glob(f"{stem}.*")
            if p.is_file() and p.suffix not in {".cpp", ".m", ".py", ".pyi"}
        ]
    )


@pytest.fixture(scope="session")
def octave_mex_errpca_and_subtract_mu_in_tools() -> bool:
    """
    Best-effort: compile tools/errpca_pt.cpp and tools/subtract_mu.cpp into MEX placed in tools/.
    Returns True if both are available to Octave, else False.
    """
    if not _octave_available() or not _mkoctfile_available():
        return False

    tools = _tools_dir()
    err_src = tools / "errpca_pt.cpp"
    sub_src = tools / "subtract_mu.cpp"

    if not err_src.exists() or not sub_src.exists():
        return False

    before_err = set(_find_mex(tools, "errpca_pt"))
    before_sub = set(_find_mex(tools, "subtract_mu"))

    # Compile errpca_pt (portable flags)
    cmd_err = [
        "mkoctfile",
        "--mex",
        "-O",
        "-DNOTHREADS",
        "-o",
        "errpca_pt",
        str(err_src),
    ]
    proc_err = subprocess.run(
        cmd_err,
        cwd=str(tools),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Compile subtract_mu (no threading)
    cmd_sub = ["mkoctfile", "--mex", "-O", "-o", "subtract_mu", str(sub_src)]
    proc_sub = subprocess.run(
        cmd_sub,
        cwd=str(tools),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    after_err = set(_find_mex(tools, "errpca_pt"))
    after_sub = set(_find_mex(tools, "subtract_mu"))

    created_err = sorted(after_err - before_err)
    created_sub = sorted(after_sub - before_sub)

    ok_build = (
        (proc_err.returncode == 0)
        and (proc_sub.returncode == 0)
        and bool(after_err)
        and bool(after_sub)
    )
    if not ok_build:
        return False

    # Verify Octave resolves them
    try:
        script = (
            "addpath('{tools}');"
            "w1 = which('errpca_pt');"
            "w2 = which('subtract_mu');"
            "if isempty(w1), error('errpca_pt not found'); end;"
            "if isempty(w2), error('subtract_mu not found'); end;"
        ).format(tools=str(tools).replace("'", "''"))
        _run_octave_eval(script)
    except Exception:
        return False

    def _cleanup() -> None:
        for p in created_err + created_sub:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    atexit.register(_cleanup)
    return True


def _as_cell_1xn(mats: list[np.ndarray]) -> np.ndarray:
    """Create a MATLAB-style 1xN cell array via dtype=object."""
    arr = np.empty((1, len(mats)), dtype=object)
    for j, m in enumerate(mats):
        arr[0, j] = np.asarray(m, dtype=np.float64)
    return arr


def _run_octave_cf_full_dense_convenience(mat_in: Path, mat_out: Path) -> None:
    """
    Call cf_full in its nargin==7 convenience mode:
      [cost,cx,ca,cmu,cs] = cf_full(X, A, S, Mu, V, cv, hp)
    where:
      cv.A  = Av cell or []
      cv.S  = Sv cell (per column or per pattern)
      cv.Isv = Isv (1-based) or []
      cv.Mu = Muv vector or []
      hp.Va = Va
      hp.Vmu = Vmu
    In this mode, cf_full will:
      - infer M from NaNs in X
      - set NaNs -> 0
      - subtract Mu .* M
      - compute sXv via compute_rms, etc.
    """
    tools = _tools_dir()
    script = (
        "addpath('{tools}');"
        "load('{infile}');"
        "cv = struct();"
        "cv.A = Av;"
        "cv.S = Sv;"
        "cv.Isv = Isv;"
        "cv.Mu = Muv;"
        "hp = struct();"
        "hp.Va = Va;"
        "hp.Vmu = Vmu;"
        "[cost,cost_x,cost_a,cost_mu,cost_s] = cf_full(X, A, S, Mu, V, cv, hp);"
        "save('-mat','{outfile}','cost','cost_x','cost_a','cost_mu','cost_s');"
    ).format(
        tools=str(tools).replace("'", "''"),
        infile=str(mat_in).replace("'", "''"),
        outfile=str(mat_out).replace("'", "''"),
    )
    _run_octave_eval(script)


def _run_octave_cf_full_sparse_fullargs(mat_in: Path, mat_out: Path) -> None:
    """
    Sparse regression using full argument list:
    - We pre-center X via subtract_mu(X, Mu) MEX (stored-entry semantics).
    - Then call cf_full with explicit M = spones(X_original), sXv=[], ndata=nnz(X_original).

    Signature:
      cf_full( Xc, A, S, Mu, V, Av, Sv, Isv, Muv, Va, Vmu, M, sXv, ndata )
    """
    tools = _tools_dir()
    script = (
        "addpath('{tools}');"
        "load('{infile}');"
        "M = spones(X);"
        "ndata = nnz(X);"
        "Xc = subtract_mu(X, Mu);"  # stored-entry subtraction (MEX)
        "% sXv empty so it recomputes using compute_rms (sparse branch uses errpca_pt)\n"
        "sXv = [];"
        "[cost,cost_x,cost_a,cost_mu,cost_s] = cf_full(Xc, A, S, Mu, V, Av, Sv, Isv, Muv, Va, Vmu, M, sXv, ndata);"
        "save('-mat','{outfile}','cost','cost_x','cost_a','cost_mu','cost_s');"
    ).format(
        tools=str(tools).replace("'", "''"),
        infile=str(mat_in).replace("'", "''"),
        outfile=str(mat_out).replace("'", "''"),
    )
    _run_octave_eval(script)


def _load_cost_mat(mat_out: Path) -> tuple[float, float, float, float, float]:
    out = loadmat(mat_out)
    cost = float(np.asarray(out["cost"]).squeeze())
    cost_x = float(np.asarray(out["cost_x"]).squeeze())
    cost_a = float(np.asarray(out["cost_a"]).squeeze())
    cost_mu = float(np.asarray(out["cost_mu"]).squeeze())
    cost_s = float(np.asarray(out["cost_s"]).squeeze())
    return cost, cost_x, cost_a, cost_mu, cost_s


# ======================================================================================
# Octave regression tests
# ======================================================================================

pytestmark = pytest.mark.skipif(
    not _octave_available(),
    reason="Octave not available on PATH; skipping Octave regression tests.",
)


def test_cost_regression_dense_cf_full_octave(tmp_path: Path) -> None:
    """
    Dense regression vs tools/cf_full.m (convenience mode).
    End-to-end through cf_full's internal:
    - NaN -> mask inference
    - X(isnan)=0
    - mean-centering by Mu on observed set
    - compute_rms + Sv contributions
    - cost decomposition
    """
    rng = np.random.default_rng(123)
    n1, n2, k = 6, 7, 3

    X = rng.standard_normal((n1, n2)).astype(np.float64)
    nan_mask = rng.random((n1, n2)) < 0.25
    X[nan_mask] = np.nan

    A = rng.standard_normal((n1, k)).astype(np.float64)
    S = rng.standard_normal((k, n2)).astype(np.float64)

    Mu = rng.standard_normal(n1).astype(np.float64)
    V = 0.8

    # --- IMPORTANT: cf_full.m requires Sv non-empty if it recomputes sXv ---
    # Make Sv a 1 x n2 cell array of eye(k).
    Sv = np.empty((1, n2), dtype=object)
    for j in range(n2):
        Sv[0, j] = np.eye(k, dtype=np.float64)

    # No patterns
    Isv = np.array([], dtype=np.int64)

    # No Av / Muv (empty in MATLAB sense)
    Av = np.empty((0, 0), dtype=object)
    Muv = np.array([], dtype=np.float64)

    # Priors disabled (Va = inf => no prior)
    Va = np.array([np.inf] * k, dtype=np.float64)
    Vmu = 0.0

    # Python: to match cf_full precisely, provide explicit Sv=I per column
    params_py = CostParams(
        mu=Mu,
        noise_variance=V,
        mask=None,  # infer from NaNs
        loading_priors=None,  # no A prior
        loading_covariances=None,  # Av empty
        score_covariances=[np.eye(k, dtype=np.float64) for _ in range(n2)],
        score_pattern_index=None,
        mu_variances=None,
        mu_prior_variance=None,
    )
    cost_py = compute_full_cost(X, A, S, params_py)

    mat_in = tmp_path / "in_dense.mat"
    mat_out = tmp_path / "out_dense.mat"
    savemat(
        mat_in,
        {
            "X": X,
            "A": A,
            "S": S,
            "Mu": Mu.reshape(-1, 1),
            "V": np.array([[V]], dtype=np.float64),
            "Av": Av,
            "Sv": Sv,
            "Isv": Isv,
            "Muv": Muv,
            "Va": Va.reshape(1, -1),
            "Vmu": np.array([[Vmu]], dtype=np.float64),
        },
    )

    _run_octave_cf_full_dense_convenience(mat_in, mat_out)
    oc = loadmat(mat_out)

    cost_oc = float(np.asarray(oc["cost"]).squeeze())
    cost_x_oc = float(np.asarray(oc["cost_x"]).squeeze())
    cost_a_oc = float(np.asarray(oc["cost_a"]).squeeze())
    cost_mu_oc = float(np.asarray(oc["cost_mu"]).squeeze())
    cost_s_oc = float(np.asarray(oc["cost_s"]).squeeze())

    cost_py_tot, cost_py_x, cost_py_a, cost_py_mu, cost_py_s = cost_py

    np.testing.assert_allclose(cost_py_tot, cost_oc, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cost_py_x, cost_x_oc, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cost_py_a, cost_a_oc, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cost_py_mu, cost_mu_oc, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cost_py_s, cost_s_oc, rtol=1e-10, atol=1e-10)


def test_cost_regression_dense_patterned_sv_cf_full_octave(tmp_path: Path) -> None:
    """
    Dense regression with patterned Sv + Isv (pattern index mapping).
    Confirms:
    - Python 0-based score_pattern_index matches Octave 1-based Isv in regression fixture
    - cost_s and sXv Sv contributions align
    """
    rng = np.random.default_rng(5)
    n1, n2, k = 5, 6, 2
    n_patterns = 3

    X = rng.standard_normal((n1, n2)).astype(np.float64)
    nan_mask = rng.random((n1, n2)) < 0.2
    X[nan_mask] = np.nan

    A = rng.standard_normal((n1, k)).astype(np.float64)
    S = rng.standard_normal((k, n2)).astype(np.float64)
    Mu = rng.standard_normal(n1).astype(np.float64)
    V = 1.1

    # Patterned Sv
    Sv_list = []
    for _ in range(n_patterns):
        # Make SPD matrices
        B = rng.standard_normal((k, k))
        Sv_list.append((B @ B.T) + np.eye(k) * 0.1)

    score_pattern_index_py = rng.integers(0, n_patterns, size=n2, dtype=np.int64)
    Isv_oct = (score_pattern_index_py + 1).astype(np.int64)  # 1-based for Octave

    params_py = CostParams(
        mu=Mu,
        noise_variance=V,
        mask=None,
        loading_priors=None,
        score_covariances=Sv_list,
        score_pattern_index=score_pattern_index_py,
    )
    cost_py = compute_full_cost(X, A, S, params_py)

    # Octave convenience mode expects Sv as 1xN cell (patterns) and Isv mapping
    Sv_cell = _as_cell_1xn(Sv_list)

    Va = np.array([np.inf] * k, dtype=np.float64)
    Vmu = 0.0
    Av = np.array([], dtype=object)
    Muv = np.array([], dtype=np.float64)

    mat_in = tmp_path / "in_dense_pat.mat"
    mat_out = tmp_path / "out_dense_pat.mat"
    savemat(
        mat_in,
        {
            "X": X,
            "A": A,
            "S": S,
            "Mu": Mu.reshape(-1, 1),
            "V": np.array([[V]], dtype=np.float64),
            "Av": Av,
            "Sv": Sv_cell,
            "Isv": Isv_oct.reshape(-1, 1),
            "Muv": Muv,
            "Va": Va.reshape(1, -1),
            "Vmu": np.array([[Vmu]], dtype=np.float64),
        },
    )

    _run_octave_cf_full_dense_convenience(mat_in, mat_out)
    cost_oc = _load_cost_mat(mat_out)

    np.testing.assert_allclose(cost_py, cost_oc, rtol=1e-8, atol=1e-8)


def test_cost_regression_sparse_cf_full_octave_optional(
    tmp_path: Path,
    octave_mex_errpca_and_subtract_mu_in_tools: bool,
) -> None:
    """
    Optional sparse regression end-to-end through tools/cf_full.m:
    - sparse branches (compute_rms uses errpca_pt MEX)
    - mean-centering for sparse done via subtract_mu MEX
    - cf_full full-args path recomputes sXv (so Sv must be provided)
    """
    if not octave_mex_errpca_and_subtract_mu_in_tools:
        pytest.skip(
            "Could not compile/use errpca_pt and subtract_mu MEX in tools/; skipping sparse regression."
        )

    rng = np.random.default_rng(99)
    n1, n2, k = 6, 7, 3

    obs = rng.random((n1, n2)) > 0.6
    vals = rng.standard_normal((n1, n2)).astype(np.float64)

    eps = np.finfo(np.float64).eps
    Xdense = np.zeros((n1, n2), dtype=np.float64)
    Xdense[obs] = vals[obs]
    Xdense[(obs) & (Xdense == 0.0)] = eps  # legacy "observed zeros stored"

    X = sp.csr_matrix(Xdense)

    A = rng.standard_normal((n1, k)).astype(np.float64)
    S = rng.standard_normal((k, n2)).astype(np.float64)
    Mu = rng.standard_normal(n1).astype(np.float64)
    V = 0.9

    # For Python sparse semantics: mask must match structure (spones(X))
    M = X.copy()
    M.data[:] = 1.0

    # --- IMPORTANT: cf_full.m requires Sv non-empty if it recomputes sXv ---
    Sv = np.empty((1, n2), dtype=object)
    for j in range(n2):
        Sv[0, j] = np.eye(k, dtype=np.float64)

    Isv = np.array([], dtype=np.int64)
    Av = np.empty((0, 0), dtype=object)
    Muv = np.array([], dtype=np.float64)
    Va = np.array([np.inf] * k, dtype=np.float64)
    Vmu = 0.0

    # Python: match cf_full exactly by providing explicit Sv=I per column
    params_py = CostParams(
        mu=Mu,
        noise_variance=V,
        mask=M,
        loading_priors=None,
        loading_covariances=None,
        score_covariances=[np.eye(k, dtype=np.float64) for _ in range(n2)],
        score_pattern_index=None,
        mu_variances=None,
        mu_prior_variance=None,
        num_cpu=2,
    )
    cost_py = compute_full_cost(X, A, S, params_py)

    mat_in = tmp_path / "in_sparse.mat"
    mat_out = tmp_path / "out_sparse.mat"
    savemat(
        mat_in,
        {
            "X": X,
            "A": A,
            "S": S,
            "Mu": Mu.reshape(-1, 1),
            "V": np.array([[V]], dtype=np.float64),
            "Av": Av,
            "Sv": Sv,
            "Isv": Isv,
            "Muv": Muv,
            "Va": Va.reshape(1, -1),
            "Vmu": np.array([[Vmu]], dtype=np.float64),
        },
    )

    _run_octave_cf_full_sparse_fullargs(mat_in, mat_out)
    oc = loadmat(mat_out)

    cost_oc = float(np.asarray(oc["cost"]).squeeze())
    cost_x_oc = float(np.asarray(oc["cost_x"]).squeeze())
    cost_a_oc = float(np.asarray(oc["cost_a"]).squeeze())
    cost_mu_oc = float(np.asarray(oc["cost_mu"]).squeeze())
    cost_s_oc = float(np.asarray(oc["cost_s"]).squeeze())

    cost_py_tot, cost_py_x, cost_py_a, cost_py_mu, cost_py_s = cost_py

    np.testing.assert_allclose(cost_py_tot, cost_oc, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cost_py_x, cost_x_oc, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cost_py_a, cost_a_oc, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cost_py_mu, cost_mu_oc, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cost_py_s, cost_s_oc, rtol=1e-10, atol=1e-10)


def test_cost_mask_dtype_equivalence_dense() -> None:
    rng = np.random.default_rng(1337)
    x = rng.standard_normal((4, 5))
    a = rng.standard_normal((4, 2))
    s = rng.standard_normal((2, 5))
    mu = rng.standard_normal(4)

    mask_bool = (rng.random((4, 5)) > 0.2).astype(bool)
    mask_int = mask_bool.astype(int)
    mask_float = mask_bool.astype(float)

    p_bool = CostParams(mu=mu, noise_variance=1.0, mask=mask_bool)
    p_int = CostParams(mu=mu, noise_variance=1.0, mask=mask_int)
    p_float = CostParams(mu=mu, noise_variance=1.0, mask=mask_float)

    c_bool = compute_full_cost(x, a, s, p_bool)
    c_int = compute_full_cost(x, a, s, p_int)
    c_float = compute_full_cost(x, a, s, p_float)

    np.testing.assert_allclose(c_bool, c_int, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_bool, c_float, rtol=1e-12, atol=1e-12)


def test_cost_mu_prior_variance_zero_with_mu_variances_raises() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = np.eye(2)
    s = np.eye(2)
    params = CostParams(
        mu=np.array([0.2, -0.1]),
        noise_variance=1.0,
        mask=np.ones_like(x),
        loading_priors=np.array([1.0, 1.0]),
        mu_variances=np.array([0.1, 0.2]),
        mu_prior_variance=0.0,
    )
    with pytest.raises(ValueError, match="mu_prior_variance"):
        compute_full_cost(x, a, s, params)


def test_cost_raises_when_n_data_non_positive_with_sxv() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = np.eye(2)
    s = np.eye(2)
    params = CostParams(
        mu=np.zeros(2),
        noise_variance=1.0,
        mask=np.ones_like(x),
        s_xv=1.0,
        n_data=0,
    )
    with pytest.raises(ValueError, match="n_data must be positive"):
        compute_full_cost(x, a, s, params)
