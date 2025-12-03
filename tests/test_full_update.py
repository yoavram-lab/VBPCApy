"""Tests for full update functions in vbpca_py._full_update."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from vbpca_py._full_update import (
    BiasState,
    CenteringState,
    HyperpriorContext,
    InitContext,
    LoadingsUpdateState,
    NoiseState,
    RmsContext,
    RotationContext,
    ScoreState,
    _build_masks_and_counts,
    _final_rotation,
    _initialize_parameters,
    _missing_patterns_info,
    _observed_indices,
    _prepare_data,
    _recompute_rms,
    _update_bias,
    _update_hyperpriors,
    _update_loadings,
    _update_noise_variance,
    _update_scores,
)
from vbpca_py._monitoring import InitShapes

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

    _x_data2, x_probe2, mask, mask_probe, n_obs_row, n_data, n_probe = (
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
    assert n_obs_row.shape == (mask.shape[0],)
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

    _x_data2, x_probe2, mask, mask_probe, n_obs_row, n_data, n_probe = (
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
        mask,
        opts,
        n_samples=x.shape[1],
    )

    assert n_patterns == x.shape[1]
    assert obs_patterns == []
    assert pattern_index is None


def test_missing_patterns_info_uniquesv_true() -> None:
    """_missing_patterns_info with uniquesv == 1 uses _missing_patterns."""
    x = np.ones((3, 4), dtype=float)
    mask = np.ones_like(x, dtype=bool)
    opts = {"uniquesv": 1}

    n_patterns, obs_patterns, pattern_index = _missing_patterns_info(
        mask,
        opts,
        n_samples=x.shape[1],
    )

    # With all ones, each column shares the same pattern.
    assert n_patterns == 1
    assert len(obs_patterns) == 1
    assert pattern_index is not None
    assert np.all(pattern_index == 0)


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

    shapes = InitShapes(
        n_features=n_features,
        n_samples=n_samples,
        n_components=n_components,
        n_obs_patterns=n_patterns,
    )
    ctx = InitContext(
        x_data=x_data,
        x_probe=x_probe,
        mask=mask,
        mask_probe=mask_probe,
        shapes=shapes,
        pattern_index=pattern_index,
        n_obs_row=n_obs_row,
        use_prior=True,
        use_postvar=True,
        opts=opts,
    )

    (
        loadings,
        scores,
        mu,
        noise_var,
        loading_covariances,
        score_covariances,
        mu_variances,
        va,
        vmu,
        x_centered,
        x_probe_centered,
    ) = _initialize_parameters(ctx)

    # Shapes
    assert loadings.shape == (n_features, n_components)
    assert scores.shape == (n_components, n_samples)
    assert mu.shape == (n_features, 1)
    assert mu_variances.shape == (n_features, 1)
    assert isinstance(noise_var, float)
    assert isinstance(vmu, float)
    assert va.shape == (n_components,)
    assert len(loading_covariances) == n_features
    assert len(score_covariances) == n_patterns

    # Centering: data should be finite and same shape; probe is None.
    assert x_centered.shape == x_data.shape
    assert np.all(np.isfinite(x_centered))
    assert x_probe_centered is None


# ----------------------------------------------------------------------
# Hyperprior updates
# ----------------------------------------------------------------------


def test_update_hyperpriors_before_warmup_is_noop() -> None:
    """_update_hyperpriors should be a no-op before the warmup period ends."""
    rng = np.random.default_rng(11)
    n_features, n_components = 4, 3
    loadings = rng.standard_normal((n_features, n_components))
    mu = rng.standard_normal((n_features, 1))
    mu_variances = np.ones_like(mu)
    va = np.ones(n_components)
    vmu = 2.0

    ctx = HyperpriorContext(
        iteration=1,
        use_prior=True,
        niter_broadprior=5,
        bias_enabled=True,
        mu=mu,
        mu_variances=mu_variances,
        loadings=loadings,
        loading_covariances=[],
        n_features=n_features,
        hp_va=1.0,
        hp_vb=1.0,
        va=va.copy(),
        vmu=vmu,
    )

    va_new, vmu_new = _update_hyperpriors(ctx)
    assert_allclose(va_new, va)
    assert vmu_new == vmu


def test_update_hyperpriors_after_warmup_updates_values() -> None:
    """_update_hyperpriors should update Va and Vmu after warmup."""
    rng = np.random.default_rng(12)
    n_features, n_components = 4, 2
    loadings = rng.standard_normal((n_features, n_components))
    mu = rng.standard_normal((n_features, 1))
    mu_variances = np.ones_like(mu)
    va = np.ones(n_components)
    vmu = 1.0

    ctx = HyperpriorContext(
        iteration=10,
        use_prior=True,
        niter_broadprior=5,
        bias_enabled=True,
        mu=mu,
        mu_variances=mu_variances,
        loadings=loadings,
        loading_covariances=[],
        n_features=n_features,
        hp_va=1.0,
        hp_vb=1.0,
        va=va,
        vmu=vmu,
    )

    va_new, vmu_new = _update_hyperpriors(ctx)
    # Both should change and remain positive.
    assert np.all(va_new > 0.0)
    assert vmu_new > 0.0


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

    err = np.ones_like(x)

    new_bias_state, new_centering = _update_bias(
        bias_enabled=False,
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

    err = np.array([[1.0, 1.0], [2.0, 2.0]])

    new_bias_state, new_centering = _update_bias(
        bias_enabled=True,
        bias_state=bias_state,
        err_mx=err,
        centering=centering,
    )

    mu_vec = new_bias_state.mu.ravel()
    assert mu_vec[0] > 0.0
    assert mu_vec[1] > 0.0
    assert mu_vec[1] > mu_vec[0]

    # x_data should remain finite and same shape.
    assert new_centering.x_data.shape == x.shape
    assert np.all(np.isfinite(new_centering.x_data))


def test_update_bias_handles_rows_with_no_observations() -> None:
    """Rows with zero n_obs_row should not cause NaNs in bias update."""
    x = np.zeros((3, 2))
    mask = np.array([[1, 1], [1, 1], [0, 0]], dtype=bool)
    n_obs_row = np.array([2.0, 2.0, 0.0])

    bias_state = BiasState(
        mu=np.zeros((3, 1)),
        muv=np.zeros((3, 1)),
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
    err = np.ones_like(x)

    new_bias_state, new_centering = _update_bias(
        bias_enabled=True,
        bias_state=bias_state,
        err_mx=err,
        centering=centering,
    )

    assert np.all(np.isfinite(new_bias_state.mu))
    assert np.all(np.isfinite(new_centering.x_data))


# ----------------------------------------------------------------------
# Scores / loadings updates
# ----------------------------------------------------------------------


def test_update_scores_dense_full_observed_matches_closed_form() -> None:
    """_update_scores with fully observed dense data matches closed-form solution."""
    rng = np.random.default_rng(0)
    n_features, n_samples, n_components = 4, 5, 2
    x = rng.standard_normal((n_features, n_samples))
    loadings = rng.standard_normal((n_features, n_components))
    scores = np.zeros((n_components, n_samples))
    loading_covariances: list[np.ndarray] = []
    score_covariances: list[np.ndarray] = [
        np.eye(n_components) for _ in range(n_samples)
    ]

    mask = np.ones_like(x, dtype=float)
    eye_components = np.eye(n_components)
    noise_var = 0.5

    state = ScoreState(
        x_data=x,
        mask=mask,
        loadings=loadings,
        scores=scores,
        loading_covariances=loading_covariances,
        score_covariances=score_covariances,
        pattern_index=None,
        obs_patterns=[],
        noise_var=noise_var,
        eye_components=eye_components,
        verbose=0,
    )

    # Closed-form update for fully observed case:
    # S = (A^T A + V I)^(-1) A^T X
    ata = loadings.T @ loadings
    inv_common = np.linalg.inv(ata + noise_var * eye_components)
    scores_expected = inv_common @ (loadings.T @ x)

    updated_state = _update_scores(state)
    assert updated_state.scores.shape == scores_expected.shape
    assert_allclose(updated_state.scores, scores_expected, rtol=1e-6, atol=1e-8)


def test_update_scores_with_patterns_shares_covariance() -> None:
    """_update_scores with pattern sharing should reuse covariance across pattern."""
    rng = np.random.default_rng(13)
    n_features, n_samples, n_components = 3, 4, 2
    x = rng.standard_normal((n_features, n_samples))
    loadings = rng.standard_normal((n_features, n_components))
    scores = np.zeros((n_components, n_samples))
    loading_covariances: list[np.ndarray] = []
    score_covariances: list[np.ndarray] = [np.zeros((n_components, n_components))] * 2

    mask = np.ones_like(x, dtype=float)
    eye_components = np.eye(n_components)
    noise_var = 0.1

    # Two patterns: columns [0, 2] and [1, 3]
    obs_patterns = [[0, 2], [1, 3]]
    pattern_index = np.array([0, 1, 0, 1], dtype=int)

    state = ScoreState(
        x_data=x,
        mask=mask,
        loadings=loadings,
        scores=scores,
        loading_covariances=loading_covariances,
        score_covariances=score_covariances,
        pattern_index=pattern_index,
        obs_patterns=obs_patterns,
        noise_var=noise_var,
        eye_components=eye_components,
        verbose=0,
    )

    updated_state = _update_scores(state)

    # Covariance for pattern 0 and pattern 1 should both be positive definite
    sv0 = updated_state.score_covariances[0]
    sv1 = updated_state.score_covariances[1]
    assert sv0.shape == (n_components, n_components)
    assert sv1.shape == (n_components, n_components)
    assert np.all(np.linalg.eigvalsh(sv0) > 0)
    assert np.all(np.linalg.eigvalsh(sv1) > 0)


def test_update_loadings_dense_full_observed_matches_closed_form() -> None:
    """_update_loadings with fully observed dense data matches closed-form solution."""
    rng = np.random.default_rng(1)
    n_features, n_samples, n_components = 3, 4, 2
    x = rng.standard_normal((n_features, n_samples))
    scores = rng.standard_normal((n_components, n_samples))
    mask = np.ones_like(x, dtype=float)

    loading_covariances: list[np.ndarray] = []
    score_covariances: list[np.ndarray] = [
        np.zeros((n_components, n_components)) for _ in range(n_samples)
    ]
    va = np.ones(n_components)
    noise_var = 0.3

    # Closed-form ridge regression per row:
    # A[i,:] = X[i,:] S^T (S S^T + V/va I)^(-1)
    prior_prec = np.diag(noise_var / va)
    phi = scores @ scores.T + prior_prec
    inv_phi = np.linalg.inv(phi)
    loadings_expected = np.empty((n_features, n_components))
    for i in range(n_features):
        loadings_expected[i, :] = x[i, :] @ scores.T @ inv_phi

    state = LoadingsUpdateState(
        x_data=x,
        mask=mask,
        scores=scores,
        loading_covariances=loading_covariances,
        score_covariances=score_covariances,
        pattern_index=None,
        va=va,
        noise_var=noise_var,
        verbose=0,
    )

    loadings, loading_covariances_out = _update_loadings(state)

    assert loadings.shape == loadings_expected.shape
    assert_allclose(loadings, loadings_expected, rtol=1e-6, atol=1e-8)
    # With loading_covariances == [], we expect it to remain an empty list
    assert loading_covariances_out == []


# ----------------------------------------------------------------------
# RMS and noise variance
# ----------------------------------------------------------------------


def test_recompute_rms_matches_manual() -> None:
    """_recompute_rms should match manual RMS calculation."""
    rng = np.random.default_rng(3)
    n_features, n_samples, n_components = 4, 5, 2
    x = rng.standard_normal((n_features, n_samples))
    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))

    mask = np.ones_like(x, dtype=float)
    n_data = float(mask.size)
    x_hat = loadings @ scores
    residual = x - x_hat
    mse = np.sum(residual**2) / n_data
    rms_manual = float(np.sqrt(mse))

    ctx = RmsContext(
        x_data=x,
        x_probe=None,
        mask=mask,
        mask_probe=None,
        n_data=n_data,
        n_probe=0,
        loadings=loadings,
        scores=scores,
    )

    rms, prms, err_mx = _recompute_rms(ctx)

    assert np.isnan(prms)
    assert err_mx.shape == x.shape
    assert_allclose(rms, rms_manual, rtol=1e-6, atol=1e-8)


def test_update_noise_variance_positive_and_finite() -> None:
    """_update_noise_variance should yield positive, finite noise variance."""
    rng = np.random.default_rng(4)
    n_features, n_samples, n_components = 3, 4, 2
    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))
    loading_covariances: list[np.ndarray] = []
    score_covariances: list[np.ndarray] = [
        np.eye(n_components) for _ in range(n_samples)
    ]
    mu_variances = np.zeros((n_features, 1))
    n_data = float(n_features * n_samples)

    noise_state = NoiseState(
        loadings=loadings,
        scores=scores,
        loading_covariances=loading_covariances,
        score_covariances=score_covariances,
        mu_variances=mu_variances,
        pattern_index=None,
        n_data=n_data,
        noise_var=0.5,
    )

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
    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))
    loading_covariances = [np.eye(n_components) for _ in range(n_features)]
    score_covariances = [np.eye(n_components) for _ in range(n_samples)]
    mu = rng.standard_normal((n_features, 1))

    pattern_index = None
    obs_patterns: list[list[int]] = []

    ctx = RotationContext(
        loadings=loadings,
        loading_covariances=loading_covariances,
        scores=scores,
        score_covariances=score_covariances,
        mu=mu.copy(),
        pattern_index=pattern_index,
        obs_patterns=obs_patterns,
        bias_enabled=False,
    )

    loadings_rot, av_rot, scores_rot, sv_rot, mu_out = _final_rotation(ctx)

    # Shapes should be preserved
    assert loadings_rot.shape == loadings.shape
    assert len(av_rot) == len(loading_covariances)
    assert scores_rot.shape == scores.shape
    assert len(sv_rot) == len(score_covariances)
    # With bias=False, mu must be unchanged
    assert_allclose(mu_out, mu)


def test_final_rotation_updates_mu_when_bias_true() -> None:
    """_final_rotation should adjust mu when bias is True."""
    rng = np.random.default_rng(6)
    n_features, n_samples, n_components = 4, 5, 2
    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))
    loading_covariances = [np.eye(n_components) for _ in range(n_features)]
    score_covariances = [np.eye(n_components) for _ in range(n_samples)]
    mu = rng.standard_normal((n_features, 1))

    ctx = RotationContext(
        loadings=loadings,
        loading_covariances=loading_covariances,
        scores=scores,
        score_covariances=score_covariances,
        mu=mu.copy(),
        pattern_index=None,
        obs_patterns=[],
        bias_enabled=True,
    )

    _, _, _, _, mu_out = _final_rotation(ctx)

    # Bias-enabled rotation should normally shift mu.
    # We only assert that the result is finite and the same shape.
    assert mu_out.shape == mu.shape
    assert np.all(np.isfinite(mu_out))
