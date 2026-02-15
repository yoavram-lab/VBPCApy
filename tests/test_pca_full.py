"""Tests for the pca_full implementation.

These tests cover:

- Basic shape and type correctness
- Handling of missing values (NaNs) in dense inputs
- Dense vs. sparse inputs
- Different algorithms ("vb", "map", "ppca")
- uniquesv pattern-sharing path
- Cost computation when cfstop is given
- Consistency between lc["rms"] and an explicit reconstruction RMS
- Smoke tests for rotate2pca on/off and bias on/off
- Optional regression tests against MATLAB fixtures (if present)

Notes on RMS consistency:
-------------------------
In strict_legacy mode, the implementation normalizes dense inputs by:
- treating NaNs as missing (mask = ~np.isnan(x)),
- setting NaNs to 0.0 in the internal centered matrix, and
- replacing exact zeros with eps to avoid ambiguity with sparse "missing" zeros.

The helper `_compute_masked_rms_dense_like_impl` mirrors strict_legacy behavior
so explicit RMS comparisons match lc["rms"] in those tests.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.io import loadmat

from vbpca_py._pca_full import _explained_variance, pca_full

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_toy_dense_with_nans(
    n_features: int = 6,
    n_samples: int = 8,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_features, n_samples))

    # Introduce some NaNs as missing values
    mask = rng.random(x.shape) < 0.2
    x[mask] = np.nan
    return x


def _dense_preprocess_like_impl(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the dense preprocessing in _full_update._build_masks_dense.

    Args:
        x: Dense matrix with possible NaN entries.

    Returns:
        Tuple ``(x_proc, mask)`` where ``x_proc`` has NaNs replaced by
        zeros and exact zeros replaced by ``eps``, and ``mask`` is the
        observed boolean mask (``~np.isnan(x)``).
    """
    x_proc = np.array(x, dtype=float, copy=True)
    mask = ~np.isnan(x_proc)

    eps = np.finfo(float).eps
    x_proc[x_proc == 0.0] = eps
    x_proc[np.isnan(x_proc)] = 0.0
    return x_proc, mask


def _compute_masked_rms_dense_like_impl(
    x_original: np.ndarray,
    loadings: np.ndarray,
    scores: np.ndarray,
    mean: np.ndarray,
) -> float:
    """Compute RMS error on observed entries to match compute_rms behavior."""
    x_proc, mask = _dense_preprocess_like_impl(x_original)
    n_obs = int(np.count_nonzero(mask))
    if n_obs == 0:
        return float("nan")

    recon = loadings @ scores + mean
    diff = (recon - x_proc) * mask
    mse = float(np.sum(diff**2) / float(n_obs))
    return float(np.sqrt(mse))


def _fixture_path(name: str) -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.joinpath("data").joinpath(name)


# ----------------------------------------------------------------------
# Basic dense test, algorithm="vb"
# ----------------------------------------------------------------------


def test_pca_full_dense_vb_basic() -> None:
    x = _make_toy_dense_with_nans(n_features=5, n_samples=7, seed=1)

    result = pca_full(
        x,
        n_components=2,
        maxiters=5,
        algorithm="vb",
        autosave=0,
        display=0,
        verbose=0,
    )

    # Basic keys
    for key in (
        "A",
        "S",
        "Mu",
        "V",
        "Av",
        "Sv",
        "Isv",
        "Muv",
        "Va",
        "Vmu",
        "lc",
        "cv",
        "hp",
    ):
        assert key in result

    loadings = result["A"]
    scores = result["S"]
    mean = result["Mu"]
    noise_var = result["V"]
    lc = result["lc"]

    # Shapes
    assert loadings.shape == (5, 2)
    assert scores.shape == (2, 7)
    assert mean.shape == (5, 1)
    assert np.isscalar(noise_var) or np.shape(noise_var) == ()

    # Learning curves: keys and consistent lengths
    for k in ("rms", "prms", "time", "cost"):
        assert k in lc
    n = len(lc["rms"])
    assert n >= 1
    assert len(lc["prms"]) == n
    assert len(lc["time"]) == n
    assert len(lc["cost"]) == n

    # RMS should be non-negative (or NaN)
    assert all(r >= 0.0 or np.isnan(r) for r in lc["rms"])

    # Explicit RMS reconstruction on observed entries should roughly
    # match last lc["rms"].
    last_rms = float(lc["rms"][-1])
    explicit_rms = _compute_masked_rms_dense_like_impl(x, loadings, scores, mean)
    if not np.isnan(last_rms) and not np.isnan(explicit_rms):
        assert_allclose(explicit_rms, last_rms, rtol=1e-4, atol=1e-6)


# ----------------------------------------------------------------------
# Sparse input, algorithm="map"
# ----------------------------------------------------------------------


def test_pca_full_sparse_map_basic() -> None:
    rng = np.random.default_rng(2)
    dense = rng.standard_normal((4, 6))
    dense[rng.random(dense.shape) < 0.3] = 0.0  # many zeros -> will be sparse
    x_sparse = sp.csr_matrix(dense)

    result = pca_full(
        x_sparse,
        n_components=2,
        maxiters=5,
        algorithm="map",
        autosave=0,
        display=0,
        verbose=0,
    )

    loadings = result["A"]
    scores = result["S"]
    mean = result["Mu"]
    av = result["Av"]
    muv = result["Muv"]
    va = result["Va"]
    vmu = result["Vmu"]

    assert loadings.shape == (4, 2)
    assert scores.shape == (2, 6)
    assert mean.shape == (4, 1)

    # For MAP: use_postvar=False, so Av and Muv are empty
    assert isinstance(av, list)
    assert len(av) == 0
    assert isinstance(muv, np.ndarray)
    assert muv.size == 0

    # Va and Vmu should be finite (priors enabled in MAP)
    assert np.all(np.isfinite(va))
    assert np.isfinite(vmu)


def test_pca_full_invalid_compat_mode_raises() -> None:
    """pca_full should reject unsupported compatibility modes."""
    x = _make_toy_dense_with_nans(n_features=4, n_samples=5, seed=17)
    with pytest.raises(ValueError, match="compat_mode"):
        _ = pca_full(
            x,
            n_components=2,
            maxiters=2,
            autosave=0,
            display=0,
            verbose=0,
            compat_mode="legacy",
        )


# ----------------------------------------------------------------------
# Algorithm="ppca" behavior
# ----------------------------------------------------------------------


def test_pca_full_ppca_no_posterior_variances() -> None:
    x = _make_toy_dense_with_nans(n_features=5, n_samples=6, seed=3)

    result = pca_full(
        x,
        n_components=2,
        maxiters=5,
        algorithm="ppca",
        autosave=0,
        display=0,
        verbose=0,
    )

    av = result["Av"]
    muv = result["Muv"]
    va = result["Va"]
    vmu = result["Vmu"]

    # In ppca: no priors, no posterior variances
    assert isinstance(av, list)
    assert len(av) == 0
    assert isinstance(muv, np.ndarray)
    assert muv.size == 0

    assert np.all(np.isinf(va))
    assert np.isinf(vmu)


# ----------------------------------------------------------------------
# uniquesv=True: pattern-sharing path
# ----------------------------------------------------------------------


def test_pca_full_uniquesv_pattern_mode() -> None:
    # Columns 0 and 2 share pattern; columns 1 and 3 share another.
    x = np.array(
        [
            [1.0, np.nan, 3.0, np.nan],
            [2.0, np.nan, 4.0, np.nan],
            [np.nan, 5.0, np.nan, 7.0],
            [np.nan, 6.0, np.nan, 8.0],
        ]
    )

    result = pca_full(
        x,
        n_components=2,
        maxiters=5,
        algorithm="vb",
        uniquesv=1,
        autosave=0,
        display=0,
        verbose=0,
    )

    sv = result["Sv"]
    pattern_index = result["Isv"]
    lc = result["lc"]

    assert pattern_index is not None
    pattern_index = np.asarray(pattern_index, dtype=int)

    n_samples = x.shape[1]
    assert isinstance(sv, list)
    assert 1 <= len(sv) <= n_samples

    assert np.all((pattern_index >= 0) & (pattern_index < len(sv)))
    assert len(lc["rms"]) >= 1


# ----------------------------------------------------------------------
# cfstop: cost computation path
# ----------------------------------------------------------------------


def test_pca_full_cost_computation_cfstop() -> None:
    x = _make_toy_dense_with_nans(n_features=4, n_samples=5, seed=4)

    result = pca_full(
        x,
        n_components=2,
        maxiters=4,
        algorithm="vb",
        cfstop=np.array([10, 1e-4, 1e-3]),
        autosave=0,
        display=0,
        verbose=0,
    )

    lc = result["lc"]
    costs = np.asarray(lc["cost"], dtype=float)
    rms_series = np.asarray(lc["rms"], dtype=float)

    assert costs.shape[0] == rms_series.shape[0]

    # At least one finite cost value should be present when cfstop is enabled
    assert int(np.isfinite(costs).sum()) >= 1


# ----------------------------------------------------------------------
# Smoke test with rotate2pca off
# ----------------------------------------------------------------------


def test_pca_full_rotate2pca_off() -> None:
    x = _make_toy_dense_with_nans(n_features=5, n_samples=7, seed=5)

    result = pca_full(
        x,
        n_components=3,
        maxiters=5,
        algorithm="vb",
        rotate2pca=0,
        autosave=0,
        display=0,
        verbose=0,
    )

    assert result["A"].shape == (5, 3)
    assert result["S"].shape == (3, 7)
    assert result["Mu"].shape == (5, 1)


def test_pca_full_return_diagnostics_default_includes_arrays() -> None:
    x = _make_toy_dense_with_nans(n_features=5, n_samples=7, seed=15)

    result = pca_full(
        x,
        n_components=2,
        maxiters=4,
        algorithm="vb",
        autosave=0,
        display=0,
        verbose=0,
    )

    assert result["Xrec"] is not None
    assert result["Vr"] is not None
    assert result["ExplainedVar"] is not None
    assert result["ExplainedVarRatio"] is not None


def test_pca_full_return_diagnostics_disabled_skips_heavy_outputs() -> None:
    x = _make_toy_dense_with_nans(n_features=5, n_samples=7, seed=16)

    result = pca_full(
        x,
        n_components=2,
        maxiters=4,
        algorithm="vb",
        autosave=0,
        display=0,
        verbose=0,
        return_diagnostics=0,
    )

    assert result["Xrec"] is None
    assert result["Vr"] is None
    assert result["ExplainedVar"] is None
    assert result["ExplainedVarRatio"] is None


# ----------------------------------------------------------------------
# Bias / mean behaviour
# ----------------------------------------------------------------------


def test_pca_full_bias_disabled_gives_zero_mean() -> None:
    x = _make_toy_dense_with_nans(n_features=4, n_samples=6, seed=10)

    result = pca_full(
        x,
        n_components=2,
        maxiters=5,
        algorithm="vb",
        bias=0,
        autosave=0,
        display=0,
        verbose=0,
    )

    mean = result["Mu"]
    assert_allclose(mean, np.zeros_like(mean), atol=1e-8)


def test_pca_full_respects_provided_mu_when_bias_disabled() -> None:
    """When bias is off, an init Mu should pass through unchanged."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    init_mu = np.array([0.5, -0.25], dtype=float)
    init = {
        "A": np.eye(2, dtype=float),
        "S": np.ones((2, 2), dtype=float),
        "Mu": init_mu,
        "V": 1.0,
    }

    result = pca_full(
        x,
        n_components=2,
        maxiters=3,
        algorithm="vb",
        bias=0,
        init=init,
        autosave=0,
        display=0,
        verbose=0,
    )

    assert_allclose(result["Mu"].ravel(), init_mu)


def test_pca_full_tiny_problem_smoke() -> None:
    rng = np.random.default_rng(123)
    x = rng.standard_normal((2, 2))

    result = pca_full(
        x,
        n_components=1,
        maxiters=20,
        algorithm="vb",
        autosave=0,
        display=0,
        verbose=0,
    )

    assert result["A"].shape == (2, 1)
    assert result["S"].shape == (1, 2)
    assert result["Mu"].shape == (2, 1)
    assert np.isfinite(result["V"])
    assert all(np.isfinite(r) for r in result["lc"]["rms"] if not np.isnan(r))


def test_pca_full_respects_maxiters_cap() -> None:
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((6, 8))
    result = pca_full(
        x,
        n_components=2,
        maxiters=2,
        algorithm="vb",
        autosave=0,
        display=0,
        verbose=0,
    )
    assert len(result["lc"]["rms"]) <= 3


def test_pca_full_earlystop_with_no_probe_is_safe() -> None:
    rng = np.random.default_rng(202)
    x = rng.standard_normal((5, 7))
    result = pca_full(
        x,
        n_components=2,
        maxiters=6,
        earlystop=1,
        algorithm="vb",
        autosave=0,
        display=0,
        verbose=0,
    )
    assert len(result["lc"]["rms"]) >= 1
    assert np.all(np.isnan(np.asarray(result["lc"]["prms"], dtype=float)))


def test_pca_full_rmsstop_and_cfstop_both_provided() -> None:
    rng = np.random.default_rng(555)
    x = rng.standard_normal((5, 9))
    result = pca_full(
        x,
        n_components=2,
        maxiters=8,
        rmsstop=np.array([2, 1e-8, 1e-8]),
        cfstop=np.array([2, 1e-8, 1e-8]),
        algorithm="vb",
        autosave=0,
        display=0,
        verbose=0,
    )
    rms = np.asarray(result["lc"]["rms"], dtype=float)
    cost = np.asarray(result["lc"]["cost"], dtype=float)
    assert rms.size >= 1
    assert rms.size == cost.size
    assert np.isfinite(cost).any()


def test_pca_full_angle_every_option_smoke() -> None:
    rng = np.random.default_rng(912)
    x = rng.standard_normal((6, 10))

    result = pca_full(
        x,
        n_components=3,
        maxiters=6,
        algorithm="vb",
        angle_every=2,
        autosave=0,
        display=0,
        verbose=0,
    )

    assert result["A"].shape == (6, 3)
    assert result["S"].shape == (3, 10)
    assert len(result["lc"]["rms"]) >= 1


def test_explained_variance_tall_svd_matches_cov_reference() -> None:
    rng = np.random.default_rng(913)
    xrec = rng.standard_normal((80, 20))

    ev_fast, evr_fast = _explained_variance(xrec, n_components=10)

    cov = np.cov(np.asarray(xrec, dtype=float))
    ev_ref = np.flip(np.sort(np.real(np.linalg.eigvalsh(cov))))[:10]
    total_ref = float(np.sum(np.flip(np.sort(np.real(np.linalg.eigvalsh(cov))))))
    evr_ref = np.zeros_like(ev_ref) if total_ref <= 0.0 else ev_ref / total_ref

    assert_allclose(ev_fast, ev_ref, rtol=1e-10, atol=1e-12)
    assert_allclose(evr_fast, evr_ref, rtol=1e-10, atol=1e-12)


def test_pca_full_uniquesv_equivalence_when_fully_observed() -> None:
    rng = np.random.default_rng(808)
    x = rng.standard_normal((5, 7))
    init = {
        "A": rng.standard_normal((5, 2)),
        "S": rng.standard_normal((2, 7)),
        "Mu": np.zeros((5, 1), dtype=float),
        "V": 1.0,
    }

    out_no = pca_full(
        x,
        n_components=2,
        maxiters=4,
        uniquesv=0,
        init=init,
        rotate2pca=0,
        autosave=0,
        display=0,
        verbose=0,
    )
    out_yes = pca_full(
        x,
        n_components=2,
        maxiters=4,
        uniquesv=1,
        init=init,
        rotate2pca=0,
        autosave=0,
        display=0,
        verbose=0,
    )

    rms_no = float(np.asarray(out_no["lc"]["rms"], dtype=float)[-1])
    rms_yes = float(np.asarray(out_yes["lc"]["rms"], dtype=float)[-1])

    assert np.isfinite(rms_no)
    assert np.isfinite(rms_yes)
    assert abs(rms_no - rms_yes) < 0.5


def test_pca_full_bias_toggle_mean_shift_effect() -> None:
    rng = np.random.default_rng(1200)
    x = rng.standard_normal((4, 10)) + 3.0  # strong global shift

    out_bias = pca_full(
        x,
        n_components=2,
        maxiters=5,
        bias=1,
        rotate2pca=0,
        autosave=0,
        display=0,
        verbose=0,
    )
    out_nobias = pca_full(
        x,
        n_components=2,
        maxiters=5,
        bias=0,
        rotate2pca=0,
        autosave=0,
        display=0,
        verbose=0,
    )

    assert np.linalg.norm(out_bias["Mu"]) > np.linalg.norm(out_nobias["Mu"])


# ----------------------------------------------------------------------
# Optional MATLAB regression tests
# ----------------------------------------------------------------------


def test_pca_full_matches_matlab_dense_fixture() -> None:
    fixture = _fixture_path("legacy_pca_full_dense.mat")
    if not fixture.exists():
        pytest.skip("MATLAB dense fixture not available")

    mat = loadmat(fixture, squeeze_me=True, struct_as_record=False)

    x = np.asarray(mat["x"], dtype=float)
    k = int(mat["k"])
    mat_res = mat["result"]

    result = pca_full(
        x,
        n_components=k,
        algorithm="vb",
        maxiters=int(getattr(mat_res, "maxiters", 200)),
        bias=int(getattr(mat_res, "bias", 1)),
        uniquesv=int(getattr(mat_res, "uniquesv", 0)),
        init=mat_res,
        autosave=0,
        display=0,
        verbose=0,
    )

    A_py = result["A"]
    S_py = result["S"]
    Mu_py = result["Mu"]
    V_py = float(result["V"])

    A_mat = np.asarray(mat_res.A, dtype=float)
    S_mat = np.asarray(mat_res.S, dtype=float)
    Mu_mat = np.asarray(mat_res.Mu, dtype=float)
    if Mu_mat.ndim == 1:
        Mu_mat = Mu_mat[:, None]
    V_mat = float(mat_res.V)

    x_rec_py = A_py @ S_py + Mu_py
    x_rec_mat = A_mat @ S_mat + Mu_mat

    assert_allclose(x_rec_py, x_rec_mat, rtol=1e-5, atol=1e-7)
    assert_allclose(V_py, V_mat, rtol=1e-3, atol=1e-6)


def test_pca_full_matches_matlab_missing_fixture() -> None:
    fixture = _fixture_path("legacy_pca_full_missing.mat")
    if not fixture.exists():
        pytest.skip("MATLAB missing-data fixture not available")

    mat = loadmat(fixture, squeeze_me=True, struct_as_record=False)

    x = np.asarray(mat["x"], dtype=float)
    k = int(mat["k"])
    mat_res = mat["result"]

    result = pca_full(
        x,
        n_components=k,
        algorithm="vb",
        maxiters=int(getattr(mat_res, "maxiters", 200)),
        bias=int(getattr(mat_res, "bias", 1)),
        uniquesv=int(getattr(mat_res, "uniquesv", 0)),
        init=mat_res,
        autosave=0,
        display=0,
        verbose=0,
        rotate2pca=1,
    )

    A_py = result["A"]
    S_py = result["S"]
    Mu_py = result["Mu"]
    V_py = float(result["V"])

    A_mat = np.asarray(mat_res.A, dtype=float)
    S_mat = np.asarray(mat_res.S, dtype=float)
    Mu_mat = np.asarray(mat_res.Mu, dtype=float)
    if Mu_mat.ndim == 1:
        Mu_mat = Mu_mat[:, None]
    V_mat = float(mat_res.V)

    x_rec_py = A_py @ S_py + Mu_py
    x_rec_mat = A_mat @ S_mat + Mu_mat

    mask = ~np.isnan(x)
    diff = (x_rec_py - x_rec_mat) * mask
    mse = np.sum(diff**2) / float(np.count_nonzero(mask))
    rms = float(np.sqrt(mse))

    assert rms <= 1e-2
    assert_allclose(V_py, V_mat, rtol=1e-3, atol=1e-6)


def test_pca_full_dense_fixture_smoke_map_and_ppca() -> None:
    fixture = _fixture_path("legacy_pca_full_dense.mat")
    if not fixture.exists():
        pytest.skip("MATLAB dense fixture not available")

    mat = loadmat(fixture, squeeze_me=True, struct_as_record=False)
    x = np.asarray(mat["x"], dtype=float)
    k = int(mat["k"])

    res_map = pca_full(
        x,
        n_components=k,
        algorithm="map",
        maxiters=30,
        autosave=0,
        display=0,
        verbose=0,
    )
    res_ppca = pca_full(
        x,
        n_components=k,
        algorithm="ppca",
        maxiters=30,
        autosave=0,
        display=0,
        verbose=0,
    )

    assert np.isfinite(float(res_map["V"]))
    assert np.isfinite(float(res_ppca["V"]))
    assert len(res_map["lc"]["rms"]) >= 1
    assert len(res_ppca["lc"]["rms"]) >= 1


def test_pca_full_missing_fixture_option_surface_smoke() -> None:
    fixture = _fixture_path("legacy_pca_full_missing.mat")
    if not fixture.exists():
        pytest.skip("MATLAB missing-data fixture not available")

    mat = loadmat(fixture, squeeze_me=True, struct_as_record=False)
    x = np.asarray(mat["x"], dtype=float)
    k = int(mat["k"])

    out = pca_full(
        x,
        n_components=k,
        algorithm="vb",
        maxiters=40,
        uniquesv=1,
        rotate2pca=1,
        cfstop=np.array([5, 1e-6, 1e-6]),
        autosave=0,
        display=0,
        verbose=0,
    )

    assert len(out["lc"]["rms"]) == len(out["lc"]["cost"])
    assert np.asarray(out["Isv"]).shape[0] == x.shape[1]


@pytest.mark.parametrize("rotate2pca", [0, 1])
def test_pca_full_missing_fixture_matches_matlab_across_rotate_modes(
    rotate2pca: int,
) -> None:
    fixture = _fixture_path("legacy_pca_full_missing.mat")
    if not fixture.exists():
        pytest.skip("MATLAB missing-data fixture not available")

    mat = loadmat(fixture, squeeze_me=True, struct_as_record=False)
    x = np.asarray(mat["x"], dtype=float)
    k = int(mat["k"])
    mat_res = mat["result"]

    result = pca_full(
        x,
        n_components=k,
        algorithm="vb",
        maxiters=int(getattr(mat_res, "maxiters", 200)),
        bias=int(getattr(mat_res, "bias", 1)),
        uniquesv=int(getattr(mat_res, "uniquesv", 0)),
        init=mat_res,
        rotate2pca=rotate2pca,
        autosave=0,
        display=0,
        verbose=0,
    )

    A_py = np.asarray(result["A"], dtype=float)
    S_py = np.asarray(result["S"], dtype=float)
    Mu_py = np.asarray(result["Mu"], dtype=float)
    if Mu_py.ndim == 1:
        Mu_py = Mu_py[:, None]

    A_mat = np.asarray(mat_res.A, dtype=float)
    S_mat = np.asarray(mat_res.S, dtype=float)
    Mu_mat = np.asarray(mat_res.Mu, dtype=float)
    if Mu_mat.ndim == 1:
        Mu_mat = Mu_mat[:, None]

    x_rec_py = A_py @ S_py + Mu_py
    x_rec_mat = A_mat @ S_mat + Mu_mat
    mask = ~np.isnan(x)
    diff = (x_rec_py - x_rec_mat) * mask
    rms = float(np.sqrt(np.sum(diff**2) / float(np.count_nonzero(mask))))

    assert rms <= 2e-2
    assert_allclose(float(result["V"]), float(mat_res.V), rtol=2e-2, atol=1e-6)
