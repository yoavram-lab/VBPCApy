"""Tests for full update helper functions in vbpca_py._full_update.

This suite covers:
- Data prep: empty removal, mask construction, observation counting
- Init: parameter initialization + centering
- Core updates: bias, scores, loadings, hyperpriors, noise variance
- Rotation: final PCA rotation wiring
- Diagnostics: equivalence tests between fast/general branches and pattern/expanded branches
- Semantics checks: observed-index definition consistency (mask vs nonzero), dense vs sparse behavior

Important diagnostic note
-------------------------
The dense-data path normalizes data by:
- mask = ~isnan(x)
- replacing exact zeros with eps
- replacing NaNs with 0

That means "observed entries" must be defined consistently via the mask,
NOT via np.nonzero(x_data) after normalization. Several tests below are
explicitly designed to catch any accidental reliance on nonzeros.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

import vbpca_py._pca_full as pca_mod

# Import internal branch functions for equivalence tests.
# These are intentionally private but stable enough for diagnostic testing.
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
    _loadings_update_fast_dense_no_sv,
    _loadings_update_general,
    _missing_patterns_info,
    _observed_indices,
    _prepare_data,
    _recompute_rms,
    _score_update_fast_dense_no_av,
    _score_update_general_no_patterns,
    _update_bias,
    _update_hyperpriors,
    _update_loadings,
    _update_noise_variance,
    _update_scores,
)
from vbpca_py._mean import subtract_mu
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


def _set_of_pairs(ix: np.ndarray, jx: np.ndarray) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in zip(ix, jx, strict=True)}


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

    # _prepare_data: smoke test on shapes; main logic in remove_empty_entries
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


def test_observed_indices_match_nonzero_dense_raw() -> None:
    """_observed_indices should match np.nonzero for dense input (raw semantics)."""
    x = np.array([[0.0, 1.0], [2.0, 0.0]])
    ix, jx = _observed_indices(x)

    exp_i, exp_j = np.nonzero(x)
    assert_allclose(ix, exp_i)
    assert_allclose(jx, exp_j)


def test_missing_patterns_info_uniquesv_false() -> None:
    """_missing_patterns_info with uniquesv == 0 returns no patterns."""
    x = _make_dense_with_nans(seed=3)
    opts = {"uniquesv": 0}
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


def test_iteration_order_scores_rotate_loadings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure iteration keeps MATLAB ordering: scores -> rotate -> loadings."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((3, 4))

    init = {
        "A": rng.standard_normal((3, 2)),
        "S": rng.standard_normal((2, 4)),
        "Mu": np.zeros(3, dtype=float),
        "V": 1.0,
    }

    opts = pca_mod._build_options(
        {
            "init": init,
            "maxiters": 1,
            "bias": 1,
            "verbose": 0,
            "autosave": 0,
            "rotate2pca": 1,
        }
    )

    prepared = pca_mod._prepare_problem(x, opts)
    training = pca_mod._initialize_model(
        prepared=prepared,
        n_components=2,
        use_prior=True,
        use_postvar=True,
        opts=opts,
    )

    order: list[str] = []

    orig_scores = pca_mod._update_scores
    orig_rotate = pca_mod._rotate_towards_pca
    orig_loadings = pca_mod._update_loadings

    def patched_scores(state):
        order.append("scores")
        return orig_scores(state)

    def patched_rotate(**kwargs):
        order.append("rotate")
        return orig_rotate(**kwargs)

    def patched_loadings(state):
        order.append("loadings")
        return orig_loadings(state)

    monkeypatch.setattr(pca_mod, "_update_scores", patched_scores)
    monkeypatch.setattr(pca_mod, "_rotate_towards_pca", patched_rotate)
    monkeypatch.setattr(pca_mod, "_update_loadings", patched_loadings)

    _ = pca_mod._run_training_loop(
        prepared=prepared,
        training=training,
        use_prior=True,
        opts=opts,
    )

    assert order == ["scores", "rotate", "loadings"]


# ----------------------------------------------------------------------
# NEW: Observed-index semantics diagnostics
# ----------------------------------------------------------------------


def test_dense_observed_indices_should_match_mask_post_normalization() -> None:
    """Diagnostic: after dense normalization, observed set should be mask-defined.

    If this fails, it indicates a mismatch between:
      - mask semantics (True = observed)
      - _observed_indices semantics (nonzero after eps/NaN normalization)

    This is a common source of pca_full drift via noise variance update.
    """
    x = np.array(
        [
            [0.0, 1.0, np.nan, 0.0],
            [2.0, 0.0, 3.0, np.nan],
            [0.0, 0.0, 0.0, 4.0],
        ],
        dtype=float,
    )
    opts: dict[str, object] = {
        "xprobe": None,
        "earlystop": 1,
        "init": "random",
        "verbose": 0,
    }

    x_data, x_probe, *_ = _prepare_data(x, opts)
    x_norm, _x_probe2, mask, _mask_probe, _n_obs_row, _n_data, _n_probe = (
        _build_masks_and_counts(x_data, x_probe, opts)
    )

    ix_nz, jx_nz = _observed_indices(x_norm)
    ix_m, jx_m = np.nonzero(np.asarray(mask, dtype=bool))

    set_nz = _set_of_pairs(ix_nz, jx_nz)
    set_m = _set_of_pairs(ix_m, jx_m)

    # This is the desired invariant. If it fails, it pinpoints a semantics mismatch.
    assert set_nz == set_m


def test_dense_mask_and_observed_indices_agree_after_normalization() -> None:
    """
    In dense mode, _build_masks_dense normalizes the data so that:
      - observed zeros are replaced by eps (=> become nonzero),
      - missing NaNs are replaced by 0.0 (=> remain zero),
    so np.nonzero(x_data_normalized) should match the observed mask entries.

    This is a deterministic invariant test (no skip).
    """
    rng = np.random.default_rng(123)

    # Construct dense data with BOTH:
    #  - observed zeros (mask True, value == 0.0)
    #  - missing entries (mask False, value == NaN)
    x = rng.standard_normal((6, 7))
    x[0, 0] = 0.0
    x[1, 3] = 0.0
    x[2, 5] = 0.0
    x[3, 1] = np.nan
    x[4, 6] = np.nan
    x[5, 2] = np.nan

    opts: dict[str, object] = {
        "xprobe": None,
        "earlystop": 1,
        "init": "random",
        "verbose": 0,
    }

    # Note: remove_empty_entries may drop rows/cols with no observations.
    # Our construction ensures every row/col has at least one observed entry.
    x_data, x_probe, _n1x, _n2x, row_idx, col_idx = _prepare_data(x, opts)
    assert x_probe is None

    x_norm, _x_probe2, mask, _mask_probe, _n_obs_row, _n_data, _n_probe = (
        _build_masks_and_counts(x_data, x_probe, opts)
    )

    # Dense contract checks
    x_norm_arr = np.asarray(x_norm, dtype=float)
    mask_arr = np.asarray(mask, dtype=bool)

    # (A) Missing entries become exactly 0.0
    assert np.all(x_norm_arr[~mask_arr] == 0.0)

    # (B) Observed entries must be nonzero:
    #     - originally nonzero values stay nonzero
    #     - originally observed zeros are replaced by eps
    assert np.all(x_norm_arr[mask_arr] != 0.0)

    # (C) Therefore, observed index sets agree:
    ix_nz, jx_nz = _observed_indices(x_norm_arr)
    obs_nz = set(zip(ix_nz.tolist(), jx_nz.tolist(), strict=False))

    ix_m, jx_m = np.nonzero(mask_arr)
    obs_mask = set(zip(ix_m.tolist(), jx_m.tolist(), strict=False))

    assert obs_nz == obs_mask


# ----------------------------------------------------------------------
# _initialize_parameters
# ----------------------------------------------------------------------


def test_initialize_parameters_basic_centering() -> None:
    """_initialize_parameters returns correctly shaped parameters and centers data."""
    rng = np.random.default_rng(10)
    n_features, n_samples, n_components = 4, 5, 2
    x = rng.standard_normal((n_features, n_samples))

    opts: dict[str, object] = {"init": "random", "bias": 1, "verbose": 0}

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

    assert loadings.shape == (n_features, n_components)
    assert scores.shape == (n_components, n_samples)
    assert mu.shape == (n_features, 1)
    assert mu_variances.shape == (n_features, 1)
    assert isinstance(noise_var, float)
    assert isinstance(vmu, float)
    assert va.shape == (n_components,)
    assert len(loading_covariances) == n_features
    assert len(score_covariances) == n_patterns

    assert x_centered.shape == x_data.shape
    assert np.all(np.isfinite(x_centered))
    assert x_probe_centered is None


def test_initialize_parameters_respects_provided_init_without_rng() -> None:
    """When init is provided, parameters should pass through unchanged."""
    n_features, n_samples, n_components = 3, 4, 2
    x = np.ones((n_features, n_samples), dtype=float)
    mask = np.ones_like(x, dtype=bool)
    mask_probe = None
    n_obs_row = np.sum(mask, axis=1).astype(float)

    init = {
        "A": np.full((n_features, n_components), 2.0, dtype=float),
        "S": np.full((n_components, n_samples), -1.0, dtype=float),
        "Mu": np.full(n_features, 0.5, dtype=float),
        "V": 0.7,
        "Av": [np.eye(n_components, dtype=float) * 0.1 for _ in range(n_features)],
        "Sv": [np.eye(n_components, dtype=float) * 0.2 for _ in range(n_samples)],
    }

    opts: dict[str, object] = {"init": init, "bias": 1, "verbose": 0}
    shapes = InitShapes(
        n_features=n_features,
        n_samples=n_samples,
        n_components=n_components,
        n_obs_patterns=n_samples,
    )
    ctx = InitContext(
        x_data=x,
        x_probe=None,
        mask=mask,
        mask_probe=mask_probe,
        shapes=shapes,
        pattern_index=None,
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

    assert_allclose(loadings, init["A"])
    assert_allclose(scores, init["S"])
    assert_allclose(mu, np.full((n_features, 1), 0.5), atol=0.0)
    assert noise_var == pytest.approx(0.7)
    assert len(loading_covariances) == n_features
    assert len(score_covariances) == n_samples
    assert_allclose(mu_variances, np.ones((n_features, 1)))
    assert np.all(np.isfinite(va))
    assert np.isfinite(vmu)
    assert x_probe_centered is None

    # Centering should match subtract_mu with the provided Mu.
    expected_centered, _ = subtract_mu(
        mu,
        x,
        mask,
        probe=None,
        update_bias=True,
    )
    assert_allclose(x_centered, expected_centered)


def test_initialize_model_centers_prepared_data_once() -> None:
    """PreparedProblem should carry the initially centered data into the loop."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    init = {
        "A": np.ones((2, 1), dtype=float),
        "S": np.ones((1, 2), dtype=float),
        "Mu": np.array([0.25, -0.5], dtype=float),
        "V": 1.0,
    }

    opts = pca_mod._build_options(
        {
            "init": init,
            "maxiters": 1,
            "bias": 1,
            "verbose": 0,
            "autosave": 0,
            "rotate2pca": 0,
        }
    )

    prepared = pca_mod._prepare_problem(x, opts)
    x_uncentered = np.array(prepared.x_data, copy=True)

    _ = pca_mod._initialize_model(
        prepared=prepared,
        n_components=1,
        use_prior=True,
        use_postvar=True,
        opts=opts,
    )

    expected_centered, _ = subtract_mu(
        init["Mu"].reshape(-1, 1),
        x_uncentered,
        prepared.mask,
        probe=None,
        update_bias=True,
    )

    assert_allclose(prepared.x_data, expected_centered)


# ----------------------------------------------------------------------
# Hyperprior updates
# ----------------------------------------------------------------------


def test_update_hyperpriors_before_warmup_is_noop() -> None:
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
    assert np.all(va_new > 0.0)
    assert vmu_new > 0.0


# ----------------------------------------------------------------------
# Bias / mean update
# ----------------------------------------------------------------------


def test_update_bias_disabled_is_noop() -> None:
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

    assert_allclose(new_bias_state.mu, bias_state.mu)
    assert_allclose(new_bias_state.muv, bias_state.muv)
    assert_allclose(new_centering.x_data, centering.x_data)


def test_update_bias_moves_mu_and_recenters() -> None:
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

    assert new_centering.x_data.shape == x.shape
    assert np.all(np.isfinite(new_centering.x_data))


def test_update_bias_handles_rows_with_no_observations() -> None:
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


def test_update_bias_incremental_recentering_matches_full_recentering() -> None:
    """Diagnostic: incremental centering using d_mu_update should match full recenter.

    If this fails, pca_full can drift because the centered X used in subsequent
    updates isn't equivalent to subtracting the new mu from the original data.
    """
    rng = np.random.default_rng(202)
    n_features, n_samples = 5, 6

    # Raw data with missingness
    x_raw = rng.standard_normal((n_features, n_samples))
    miss = rng.random((n_features, n_samples)) < 0.25
    x_raw[miss] = np.nan

    mask = ~np.isnan(x_raw)
    n_obs_row = np.sum(mask, axis=1).astype(float)

    mu_old = rng.standard_normal((n_features, 1))
    x_centered_old, _ = subtract_mu(mu_old, x_raw, mask, probe=None, update_bias=True)

    # err_mx should be zero at missing entries (common pca_full convention)
    err_mx = rng.standard_normal((n_features, n_samples)) * mask.astype(float)

    bias_state = BiasState(
        mu=mu_old.copy(),
        muv=np.zeros((n_features, 1)),
        noise_var=0.5,
        vmu=10.0,
        n_obs_row=n_obs_row,
    )
    centering = CenteringState(
        x_data=np.array(x_centered_old, dtype=float, copy=True),
        x_probe=None,
        mask=mask,
        mask_probe=None,
    )

    new_bias_state, new_centering = _update_bias(
        bias_enabled=True,
        bias_state=bias_state,
        err_mx=err_mx,
        centering=centering,
    )

    # Full recenter from the original raw data using mu_new
    x_centered_full, _ = subtract_mu(
        new_bias_state.mu, x_raw, mask, probe=None, update_bias=True
    )

    assert_allclose(
        np.array(new_centering.x_data, dtype=float),
        np.array(x_centered_full, dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )


# ----------------------------------------------------------------------
# Scores / loadings updates
# ----------------------------------------------------------------------


def test_update_scores_dense_full_observed_matches_closed_form() -> None:
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

    ata = loadings.T @ loadings
    inv_common = np.linalg.inv(ata + noise_var * eye_components)
    scores_expected = inv_common @ (loadings.T @ x)

    updated_state = _update_scores(state)
    assert updated_state.scores.shape == scores_expected.shape
    assert_allclose(updated_state.scores, scores_expected, rtol=1e-6, atol=1e-8)


def test_update_scores_with_patterns_shares_covariance() -> None:
    rng = np.random.default_rng(13)
    n_features, n_samples, n_components = 3, 4, 2
    x = rng.standard_normal((n_features, n_samples))
    loadings = rng.standard_normal((n_features, n_components))
    scores = np.zeros((n_components, n_samples))
    loading_covariances: list[np.ndarray] = []
    score_covariances: list[np.ndarray] = [
        np.zeros((n_components, n_components)) for _ in range(2)
    ]

    mask = np.ones_like(x, dtype=float)
    eye_components = np.eye(n_components)
    noise_var = 0.1

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

    sv0 = updated_state.score_covariances[0]
    sv1 = updated_state.score_covariances[1]
    assert sv0.shape == (n_components, n_components)
    assert sv1.shape == (n_components, n_components)
    assert np.all(np.linalg.eigvalsh(sv0) > 0)
    assert np.all(np.linalg.eigvalsh(sv1) > 0)


def test_update_loadings_dense_full_observed_matches_closed_form() -> None:
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
    assert loading_covariances_out == []


# ----------------------------------------------------------------------
# NEW: Equivalence tests (fast vs general, patterns vs expanded)
# ----------------------------------------------------------------------


def test_scores_fast_path_equals_general_path_fully_observed_no_av() -> None:
    """Fast score update should equal general update when fully observed and Av empty."""
    rng = np.random.default_rng(300)
    n_features, n_samples, n_components = 5, 7, 3

    x = rng.standard_normal((n_features, n_samples))
    mask = np.ones_like(x, dtype=float)

    loadings = rng.standard_normal((n_features, n_components))
    eye = np.eye(n_components)
    noise_var = 0.25

    # State for fast path
    state_fast = ScoreState(
        x_data=x,
        mask=mask,
        loadings=loadings,
        scores=np.zeros((n_components, n_samples)),
        loading_covariances=[],
        score_covariances=[
            np.zeros((n_components, n_components)) for _ in range(n_samples)
        ],
        pattern_index=None,
        obs_patterns=[],
        noise_var=noise_var,
        eye_components=eye,
        verbose=0,
    )

    # State for general path
    state_gen = ScoreState(
        x_data=x,
        mask=mask,
        loadings=loadings,
        scores=np.zeros((n_components, n_samples)),
        loading_covariances=[],
        score_covariances=[
            np.zeros((n_components, n_components)) for _ in range(n_samples)
        ],
        pattern_index=None,
        obs_patterns=[],
        noise_var=noise_var,
        eye_components=eye,
        verbose=0,
    )

    out_fast = _score_update_fast_dense_no_av(state_fast)
    out_gen = _score_update_general_no_patterns(state_gen)

    assert_allclose(out_fast.scores, out_gen.scores, rtol=1e-10, atol=1e-12)
    for j in range(n_samples):
        assert_allclose(
            out_fast.score_covariances[j],
            out_gen.score_covariances[j],
            rtol=1e-10,
            atol=1e-12,
        )


def test_loadings_fast_path_equals_general_path_when_sv_disabled() -> None:
    """Fast loadings update should equal general update when Sv contributes nothing."""
    rng = np.random.default_rng(301)
    n_features, n_samples, n_components = 6, 8, 3

    x = rng.standard_normal((n_features, n_samples))
    mask = np.ones_like(x, dtype=float)
    scores = rng.standard_normal((n_components, n_samples))

    va = np.full(n_components, 2.0)
    noise_var = 0.4

    # With score_covariances == [], _update_loadings chooses the fast path.
    state_fast = LoadingsUpdateState(
        x_data=x,
        mask=mask,
        scores=scores,
        loading_covariances=[],
        score_covariances=[],  # <- disable sv contribution
        pattern_index=None,
        va=va,
        noise_var=noise_var,
        verbose=0,
    )

    # Compare the fast impl directly against the general impl with empty Sv.
    a_fast, av_fast = _loadings_update_fast_dense_no_sv(state_fast)

    state_gen = LoadingsUpdateState(
        x_data=x,
        mask=mask,
        scores=scores,
        loading_covariances=[],
        score_covariances=[],  # <- still empty
        pattern_index=None,
        va=va,
        noise_var=noise_var,
        verbose=0,
    )
    a_gen, av_gen = _loadings_update_general(state_gen)

    assert_allclose(a_fast, a_gen, rtol=1e-10, atol=1e-12)
    assert av_fast == av_gen == []


def test_scores_pattern_mode_equals_expanded_mode() -> None:
    """Pattern-mode score updates should match expanded per-sample mode."""
    rng = np.random.default_rng(400)
    n_features, n_samples, n_components = 5, 4, 2

    x = rng.standard_normal((n_features, n_samples))

    # Two distinct missingness patterns:
    # - pattern 0 observed rows: [0,1,2]
    # - pattern 1 observed rows: [0,3,4]
    mask = np.zeros((n_features, n_samples), dtype=float)
    # cols 0 and 2 -> pattern 0
    mask[[0, 1, 2], 0] = 1.0
    mask[[0, 1, 2], 2] = 1.0
    # cols 1 and 3 -> pattern 1
    mask[[0, 3, 4], 1] = 1.0
    mask[[0, 3, 4], 3] = 1.0

    obs_patterns = [[0, 2], [1, 3]]
    pattern_index = np.array([0, 1, 0, 1], dtype=int)

    loadings = rng.standard_normal((n_features, n_components))
    eye = np.eye(n_components)
    noise_var = 0.15

    # Pattern-mode
    state_pat = ScoreState(
        x_data=x,
        mask=mask,
        loadings=loadings,
        scores=np.zeros((n_components, n_samples)),
        loading_covariances=[],
        score_covariances=[
            np.zeros((n_components, n_components)) for _ in range(len(obs_patterns))
        ],
        pattern_index=pattern_index,
        obs_patterns=obs_patterns,
        noise_var=noise_var,
        eye_components=eye,
        verbose=0,
    )
    out_pat = _update_scores(state_pat)

    # Expanded-mode (no patterns): one covariance per sample
    state_exp = ScoreState(
        x_data=x,
        mask=mask,
        loadings=loadings,
        scores=np.zeros((n_components, n_samples)),
        loading_covariances=[],
        score_covariances=[
            np.zeros((n_components, n_components)) for _ in range(n_samples)
        ],
        pattern_index=None,
        obs_patterns=[],
        noise_var=noise_var,
        eye_components=eye,
        verbose=0,
    )
    out_exp = _score_update_general_no_patterns(state_exp)

    # Scores should match columnwise
    assert_allclose(out_pat.scores, out_exp.scores, rtol=1e-10, atol=1e-12)

    # Expanded Sv[j] should match its pattern Sv[pattern_index[j]]
    for j in range(n_samples):
        assert_allclose(
            out_exp.score_covariances[j],
            out_pat.score_covariances[pattern_index[j]],
            rtol=1e-10,
            atol=1e-12,
        )


def test_loadings_pattern_mode_equals_expanded_mode() -> None:
    """Pattern-mode loadings updates should match expanded per-sample Sv mode."""
    rng = np.random.default_rng(401)
    n_features, n_samples, n_components = 6, 4, 2

    x = rng.standard_normal((n_features, n_samples))

    mask = np.zeros((n_features, n_samples), dtype=float)
    mask[[0, 1, 2], 0] = 1.0
    mask[[0, 1, 2], 2] = 1.0
    mask[[0, 3, 4, 5], 1] = 1.0
    mask[[0, 3, 4, 5], 3] = 1.0

    obs_patterns = [[0, 2], [1, 3]]
    pattern_index = np.array([0, 1, 0, 1], dtype=int)

    scores = rng.standard_normal((n_components, n_samples))
    va = np.full(n_components, 5.0)
    noise_var = 0.2

    # Build a plausible Sv per pattern (SPD)
    sv0 = np.array([[0.5, 0.1], [0.1, 0.4]])
    sv1 = np.array([[0.3, -0.05], [-0.05, 0.6]])
    sv_patterns = [sv0, sv1]
    sv_expanded = [sv_patterns[pattern_index[j]] for j in range(n_samples)]

    # Pattern-mode loadings update
    state_pat = LoadingsUpdateState(
        x_data=x,
        mask=mask,
        scores=scores,
        loading_covariances=[],
        score_covariances=[sv.copy() for sv in sv_patterns],
        pattern_index=pattern_index,
        va=va,
        noise_var=noise_var,
        verbose=0,
    )
    a_pat, av_pat = _update_loadings(state_pat)

    # Expanded-mode loadings update (pattern_index=None, Sv per sample)
    state_exp = LoadingsUpdateState(
        x_data=x,
        mask=mask,
        scores=scores,
        loading_covariances=[],
        score_covariances=[sv.copy() for sv in sv_expanded],
        pattern_index=None,
        va=va,
        noise_var=noise_var,
        verbose=0,
    )
    a_exp, av_exp = _update_loadings(state_exp)

    assert_allclose(a_pat, a_exp, rtol=1e-10, atol=1e-12)
    assert av_pat == av_exp == []


def test_sparse_vs_dense_semantics_differ_when_zeros_present() -> None:
    """Informational: sparse treats zeros as missing, dense (mask-all-ones) treats them observed."""
    rng = np.random.default_rng(500)
    n_features, n_samples, n_components = 5, 6, 2

    x_dense = rng.standard_normal((n_features, n_samples))
    x_dense[0, :] = 0.0  # explicit observed zeros

    # Dense path: treat all entries as observed
    mask_dense = np.ones_like(x_dense, dtype=float)

    # Sparse path: structural zeros are absent -> treated missing by sparse mask (x != 0)
    x_sparse = sp.csr_matrix(x_dense)

    loadings = rng.standard_normal((n_features, n_components))
    eye = np.eye(n_components)
    noise_var = 0.3

    state_dense = ScoreState(
        x_data=x_dense,
        mask=mask_dense,
        loadings=loadings,
        scores=np.zeros((n_components, n_samples)),
        loading_covariances=[],
        score_covariances=[
            np.zeros((n_components, n_components)) for _ in range(n_samples)
        ],
        pattern_index=None,
        obs_patterns=[],
        noise_var=noise_var,
        eye_components=eye,
        verbose=0,
    )
    out_dense = _update_scores(state_dense)

    mask_sparse = (x_sparse != 0).astype(float)
    state_sparse = ScoreState(
        x_data=x_sparse,
        mask=mask_sparse,
        loadings=loadings,
        scores=np.zeros((n_components, n_samples)),
        loading_covariances=[],
        score_covariances=[
            np.zeros((n_components, n_components)) for _ in range(n_samples)
        ],
        pattern_index=None,
        obs_patterns=[],
        noise_var=noise_var,
        eye_components=eye,
        verbose=0,
    )
    out_sparse = _update_scores(state_sparse)

    assert not np.allclose(out_dense.scores, out_sparse.scores)


# ----------------------------------------------------------------------
# RMS and noise variance
# ----------------------------------------------------------------------


def test_recompute_rms_matches_manual() -> None:
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
    rng = np.random.default_rng(5)
    n_features, n_samples, n_components = 5, 6, 2
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
        bias_enabled=False,
    )

    loadings_rot, av_rot, scores_rot, sv_rot, mu_out = _final_rotation(ctx)

    assert loadings_rot.shape == loadings.shape
    assert len(av_rot) == len(loading_covariances)
    assert scores_rot.shape == scores.shape
    assert len(sv_rot) == len(score_covariances)
    assert_allclose(mu_out, mu)


def test_final_rotation_updates_mu_when_bias_true() -> None:
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

    assert mu_out.shape == mu.shape
    assert np.all(np.isfinite(mu_out))
