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
The implementation normalizes dense inputs by:
- treating NaNs as missing (mask = ~np.isnan(x)),
- setting NaNs to 0.0 in the internal centered matrix, and
- replacing exact zeros with eps to avoid ambiguity with sparse "missing" zeros.

The helper `_compute_masked_rms_dense_like_impl` mirrors that behavior so that
explicit RMS comparisons match lc["rms"].
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.io import loadmat

from vbpca_py.pca_full import pca_full

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

    Returns:
    -------
    x_proc : ndarray
        Copy of x with NaNs replaced by 0 and exact zeros replaced by eps.
    mask : ndarray[bool]
        Observed mask (~np.isnan(x)).
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

    # Explicit RMS reconstruction on observed entries should roughly match last lc["rms"]
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
