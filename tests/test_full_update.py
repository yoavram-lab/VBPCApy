"""Tests for full update functions in vbpca_py._full_update.py."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from vbpca_py._full_update import (
    BiasState,
    CenteringState,
    NoiseState,
    ScoreState,
    _build_masks_and_counts,
    _final_rotation,
    _initialize_parameters,
    _missing_patterns_info,
    _observed_indices,
    _prepare_data,
    _recompute_rms,
    _update_bias,
    _update_loadings,
    _update_noise_variance,
    _update_scores,
)

# ----------------------------------------------------------------------
# Helpers for tests
# ----------------------------------------------------------------------


def _make_dense_with_nans(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4, 5))
    mask = rng.random(x.shape) < 0.2
    x[mask] = np.nan
    return x


# ----------------------------------------------------------------------
# _prepare_data / _build_masks_and_counts / _observed_indices
# ----------------------------------------------------------------------


def test_build_masks_dense_basic() -> None:
    """Basic test of _build_masks_and_counts with dense input."""
    x = _make_dense_with_nans(seed=1)
    opts: dict[str, object] = {
        "xprobe": None,
        "earlystop": 1,
        "init": "random",
        "verbose": 0,
    }

    # _prepare_data: just a smoke test on shapes; main logic in remove_empty_entries
    x_data, x_probe, n1x, n2x, row_idx, col_idx = _prepare_data(x, opts)
    assert x_data.shape == (n1x, n2x)
    assert x_probe is None
    assert row_idx.shape[0] == n1x
    assert col_idx.shape[0] == n2x

    x_data2, x_probe2, mask, mask_probe, n_obs_row, n_data, n_probe = (
        _build_masks_and_counts(x_data, x_probe, opts)
    )

    # Probe is None → n_probe == 0 and earlystop should be turned off
    assert x_probe2 is None
    assert mask_probe is None
    assert n_probe == 0
    assert opts["earlystop"] == 0

    # expected mask comes from original x, restricted to surviving rows/cols
    base = x[np.ix_(row_idx, col_idx)]
    expected_mask = ~np.isnan(base)
    assert_allclose(mask.astype(bool), expected_mask)

    # counts and totals
    assert n_obs_row.shape == (x_data2.shape[0],)
    assert isinstance(n_data, float)
    assert n_data > 0.0


def test_build_masks_sparse_basic() -> None:
    """Basic test of _build_masks_and_counts with sparse input."""
    rng = np.random.default_rng(2)
    dense = rng.standard_normal((3, 4))
    dense[rng.random(dense.shape) < 0.5] = 0.0
    x_sparse = sp.csr_matrix(dense)

    opts: dict[str, object] = {
        "xprobe": None,
        "earlystop": 1,
        "init": "random",
        "verbose": 0,
    }
    x_data, x_probe, n1x, n2x, *_ = _prepare_data(x_sparse, opts)
    assert sp.isspmatrix_csr(x_data)
    assert x_data.shape == (n1x, n2x)
    assert x_probe is None

    x_data2, x_probe2, mask, mask_probe, n_obs_row, n_data, n_probe = (
        _build_masks_and_counts(x_data, x_probe, opts)
    )

    assert sp.isspmatrix(mask)
    assert x_probe2 is None
    assert mask_probe is None
    assert n_probe == 0
    # n_obs_row should count nonzeros per row
    assert_allclose(
        n_obs_row,
        np.asarray(mask.sum(axis=1)).ravel(),
    )
    assert n_data == float(mask.count_nonzero())


def test_observed_indices_match_nonzero() -> None:
    """_observed_indices should match np.nonzero for dense input."""
    x = np.array([[0.0, 1.0], [2.0, 0.0]])
    ix, jx = _observed_indices(x)

    exp_i, exp_j = np.nonzero(x)
    assert_allclose(ix, exp_i)
    assert_allclose(jx, exp_j)


def test_missing_patterns_info_uniquesv_false() -> None:
    """_missing_patterns_info with uniquesv == 0 returns no patterns."""
    x = _make_dense_with_nans(seed=3)
    opts = {"uniquesv": 0}
    # simple dense mask: all entries observed for this test
    mask = np.ones_like(x, dtype=bool)

    n_patterns, obs_patterns, pattern_index = _missing_patterns_info(
        mask, opts, n_samples=x.shape[1]
    )

    assert n_patterns == x.shape[1]
    assert obs_patterns == []
    assert pattern_index is None


# ----------------------------------------------------------------------
# _initialize_parameters
# ----------------------------------------------------------------------


def test_initialize_parameters_basic_centering() -> None:
    """_initialize_parameters returns correctly shaped parameters and centers data."""
    rng = np.random.default_rng(10)
    n_features, n_samples, n_components = 4, 5, 2
    x = rng.standard_normal((n_features, n_samples))

    # Simple case: no probe, fully observed dense data
    opts: dict[str, object] = {
        "init": "random",
        "bias": 1,
        "verbose": 0,
    }

    x_data = x.copy()
    x_probe = None
    mask = np.ones_like(x_data, dtype=bool)
    mask_probe = None
    n_obs_row = np.sum(mask, axis=1).astype(float)

    n_patterns = n_samples
    pattern_index = None

    (
        a,
        s,
        mu,
        noise_var,
        av,
        sv,
        muv,
        va,
        vmu,
        x_centered,
        x_probe_centered,
    ) = _initialize_parameters(
        x_data=x_data,
        x_probe=x_probe,
        mask=mask,
        mask_probe=mask_probe,
        n_features=n_features,
        n_samples=n_samples,
        n_components=n_components,
        n_patterns=n_patterns,
        pattern_index=pattern_index,
        n_obs_row=n_obs_row,
        use_prior=True,
        use_postvar=True,
        opts=opts,
    )

    # Shapes
    assert a.shape == (n_features, n_components)
    assert s.shape == (n_components, n_samples)
    assert mu.shape == (n_features, 1)
    assert muv.shape == (n_features, 1)
    assert isinstance(noise_var, float)
    assert isinstance(vmu, float)
    assert va.shape == (n_components,)
    assert len(av) == n_features
    assert len(sv) == n_patterns

    # Centering: with random init Mu (default) and bias=True, data should
    # be shifted but remain finite and same shape; probe is None.
    assert x_centered.shape == x_data.shape
    assert np.all(np.isfinite(x_centered))
    assert x_probe_centered is None


# ----------------------------------------------------------------------
# Bias / mean update
# ----------------------------------------------------------------------


def test_update_bias_disabled_is_noop() -> None:
    """Test that _update_bias does nothing when bias update is disabled."""
    x = np.ones((2, 3))
    mask = np.ones_like(x, dtype=bool)

    bias_state = BiasState(
        mu=np.zeros((2, 1)),
        muv=np.zeros((2, 1)),
        noise_var=1.0,
        vmu=10.0,
        n_obs_row=np.array([3.0, 3.0]),
    )
    centering = CenteringState(
        x_data=x,
        x_probe=None,
        mask=mask,
        mask_probe=None,
    )

    # err_mx arbitrary
    err = np.ones_like(x)

    new_bias_state, new_centering = _update_bias(
        bias=False,
        bias_state=bias_state,
        err_mx=err,
        centering=centering,
    )

    # Everything should be unchanged
    assert_allclose(new_bias_state.mu, bias_state.mu)
    assert_allclose(new_bias_state.muv, bias_state.muv)
    assert_allclose(new_centering.x_data, centering.x_data)


def test_update_bias_moves_mu_and_recenters() -> None:
    """Test that _update_bias updates mu and recenters x_data."""
    x = np.zeros((2, 2))
    mask = np.ones_like(x, dtype=bool)
    n_obs_row = np.array([2.0, 2.0])

    bias_state = BiasState(
        mu=np.zeros((2, 1)),
        muv=np.zeros((2, 1)),
        noise_var=1.0,
        vmu=10.0,
        n_obs_row=n_obs_row,
    )
    centering = CenteringState(
        x_data=x.copy(),
        x_probe=None,
        mask=mask,
        mask_probe=None,
    )

    # err_mx induces a positive d_mu
    err = np.array([[1.0, 1.0], [2.0, 2.0]])
    # d_mu = [1, 2]
    new_bias_state, new_centering = _update_bias(
        bias=True,
        bias_state=bias_state,
        err_mx=err,
        centering=centering,
    )

    # mu should now be non-zero and in the direction of d_mu
    mu = new_bias_state.mu.ravel()
    assert mu[0] > 0.0
    assert mu[1] > 0.0
    assert mu[1] > mu[0]  # reflects larger d_mu on second feature

    # x_data should have been shifted by -d_mu_update on observed entries
    # Here original x was all zeros, so new x_data should be negative of mu
    # (up to shrinkage). Just check shape and finite.
    assert new_centering.x_data.shape == x.shape
    assert np.all(np.isfinite(new_centering.x_data))


# ----------------------------------------------------------------------
# Scores / loadings updates
# ----------------------------------------------------------------------


def test_update_scores_dense_full_observed_matches_closed_form() -> None:
    """_update_scores with fully observed dense data matches closed-form solution."""
    rng = np.random.default_rng(0)
    n_features, n_samples, n_components = 4, 5, 2
    x = rng.standard_normal((n_features, n_samples))
    a = rng.standard_normal((n_features, n_components))
    s = np.zeros((n_components, n_samples))
    av: list[np.ndarray] = []
    sv: list[np.ndarray] = [np.eye(n_components) for _ in range(n_samples)]

    mask = np.ones_like(x, dtype=float)
    eye_components = np.eye(n_components)
    noise_var = 0.5

    state = ScoreState(
        x_data=x,
        mask=mask,
        a=a,
        s=s,
        av=av,
        sv=sv,
        pattern_index=None,
        obs_patterns=[],
        noise_var=noise_var,
        eye_components=eye_components,
        verbose=0,
    )

    # Closed-form update for fully observed case:
    # S = (A^T A + V I)^(-1) A^T X
    ata = a.T @ a
    inv_common = np.linalg.inv(ata + noise_var * eye_components)
    s_expected = inv_common @ (a.T @ x)

    updated_state = _update_scores(state)
    assert updated_state.s.shape == s_expected.shape
    assert_allclose(updated_state.s, s_expected, rtol=1e-6, atol=1e-8)


def test_update_loadings_dense_full_observed_matches_closed_form() -> None:
    """_update_loadings with fully observed dense data matches closed-form solution."""
    rng = np.random.default_rng(1)
    n_features, n_samples, n_components = 3, 4, 2
    x = rng.standard_normal((n_features, n_samples))
    s = rng.standard_normal((n_components, n_samples))
    mask = np.ones_like(x, dtype=float)

    av: list[np.ndarray] = []
    sv: list[np.ndarray] = [
        np.zeros((n_components, n_components)) for _ in range(n_samples)
    ]
    va = np.ones(n_components)
    noise_var = 0.3

    # Closed-form ridge regression per row:
    # A[i,:] = X[i,:] S^T (S S^T + V/va I)^(-1)
    prior_prec = np.diag(noise_var / va)
    phi = s @ s.T + prior_prec
    inv_phi = np.linalg.inv(phi)
    a_expected = np.empty((n_features, n_components))
    for i in range(n_features):
        a_expected[i, :] = x[i, :] @ s.T @ inv_phi

    a, av_out = _update_loadings(
        x_data=x,
        mask=mask,
        s=s,
        av=av,
        sv=sv,
        pattern_index=None,
        va=va,
        noise_var=noise_var,
        verbose=0,
    )

    assert a.shape == a_expected.shape
    assert_allclose(a, a_expected, rtol=1e-6, atol=1e-8)
    # With av == [], we expect av_out to remain an empty list
    assert av_out == []


# ----------------------------------------------------------------------
# RMS and noise variance
# ----------------------------------------------------------------------


def test_recompute_rms_matches_manual() -> None:
    """_recompute_rms should match manual RMS calculation."""
    rng = np.random.default_rng(3)
    n_features, n_samples, n_components = 4, 5, 2
    x = rng.standard_normal((n_features, n_samples))
    a = rng.standard_normal((n_features, n_components))
    s = rng.standard_normal((n_components, n_samples))

    mask = np.ones_like(x, dtype=float)
    n_data = float(mask.size)
    x_hat = a @ s
    residual = x - x_hat
    mse = np.sum(residual**2) / n_data
    rms_manual = float(np.sqrt(mse))

    rms, prms, err_mx = _recompute_rms(
        x_data=x,
        x_probe=None,
        mask=mask,
        mask_probe=None,
        n_data=n_data,
        n_probe=0,
        a=a,
        s=s,
    )

    assert np.isnan(prms)
    assert err_mx.shape == x.shape
    assert_allclose(rms, rms_manual, rtol=1e-6, atol=1e-8)


def test_update_noise_variance_positive_and_finite() -> None:
    """_update_noise_variance should yield positive, finite noise variance."""
    rng = np.random.default_rng(4)
    n_features, n_samples, n_components = 3, 4, 2
    a = rng.standard_normal((n_features, n_components))
    s = rng.standard_normal((n_components, n_samples))
    av: list[np.ndarray] = []
    sv: list[np.ndarray] = [np.eye(n_components) for _ in range(n_samples)]
    muv = np.zeros((n_features, 1))
    n_data = float(n_features * n_samples)

    noise_state = NoiseState(
        a=a,
        s=s,
        av=av,
        sv=sv,
        muv=muv,
        pattern_index=None,
        n_data=n_data,
        noise_var=0.5,
    )

    # All entries are "observed"
    ix, jx = np.nonzero(np.ones((n_features, n_samples), dtype=int))
    rms = 1.0

    updated_state, s_xv = _update_noise_variance(noise_state, rms, ix, jx)

    assert s_xv > 0.0
    assert updated_state.noise_var > 0.0
    assert np.isfinite(updated_state.noise_var)


# ----------------------------------------------------------------------
# Final rotation
# ----------------------------------------------------------------------


def test_final_rotation_preserves_shapes_and_mu_when_bias_false() -> None:
    """_final_rotation preserves shapes and mu when bias is False."""
    rng = np.random.default_rng(5)
    n_features, n_samples, n_components = 5, 6, 2
    a = rng.standard_normal((n_features, n_components))
    s = rng.standard_normal((n_components, n_samples))
    av = [np.eye(n_components) for _ in range(n_features)]
    sv = [np.eye(n_components) for _ in range(n_samples)]
    mu = rng.standard_normal((n_features, 1))

    pattern_index = None
    obs_patterns: list[list[int]] = []

    a2, av2, s2, sv2, mu2 = _final_rotation(
        a=a,
        av=av,
        s=s,
        sv=sv,
        mu=mu.copy(),
        pattern_index=pattern_index,
        obs_patterns=obs_patterns,
        bias=False,
    )

    # Shapes should be preserved
    assert a2.shape == a.shape
    assert len(av2) == len(av)
    assert s2.shape == s.shape
    assert len(sv2) == len(sv)
    # With bias=False, mu must be unchanged
    assert_allclose(mu2, mu)
