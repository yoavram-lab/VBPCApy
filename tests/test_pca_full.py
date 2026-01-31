"""Tests for the pca_full implementation.

These tests focus on:

- Basic shape and type correctness
- Handling of missing values (NaNs)
- Dense vs. sparse inputs
- Different algorithms ("vb", "map", "ppca")
- uniquesv pattern-sharing path
- Cost computation when cfstop is given
- Consistency between lc["rms"] and explicit reconstruction error
- Optional regression tests against the original MATLAB implementation
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.io import loadmat
from vbpca_py.pca_full import pca_full


def _make_toy_dense_with_nans(
    n_features: int = 6,
    n_samples: int = 8,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_features, n_samples))
    mask = rng.random(x.shape) < 0.2
    x[mask] = np.nan
    return x


def _compute_masked_rms(
    x_original: np.ndarray,
    loadings: np.ndarray,
    scores: np.ndarray,
    mean: np.ndarray,
) -> float:
    recon = loadings @ scores + mean
    mask = ~np.isnan(x_original)
    ndata = int(np.count_nonzero(mask))
    if ndata == 0:
        return float("nan")
    diff = (recon - np.nan_to_num(x_original, copy=True)) * mask
    mse = np.sum(diff**2) / float(ndata)
    return float(np.sqrt(mse))


def _fixture_path(name: str) -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.joinpath("data").joinpath(name)


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

    assert loadings.shape == (5, 2)
    assert scores.shape == (2, 7)
    assert mean.shape == (5, 1)
    assert np.isscalar(noise_var) or np.shape(noise_var) == ()

    assert len(lc["rms"]) >= 1
    assert len(lc["prms"]) == len(lc["rms"])
    assert len(lc["time"]) == len(lc["rms"])
    assert len(lc["cost"]) == len(lc["rms"])

    # Optional initial-error curve (kept for backward compatibility)
    last_rms = float(lc["rms"][-1])
    explicit_rms = _compute_masked_rms(x, loadings, scores, mean)
    if not np.isnan(last_rms) and not np.isnan(explicit_rms):
        assert_allclose(explicit_rms, last_rms, rtol=1e-4, atol=1e-6)


def test_pca_full_sparse_map_basic() -> None:
    rng = np.random.default_rng(2)
    dense = rng.standard_normal((4, 6))
    dense[rng.random(dense.shape) < 0.3] = 0.0
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

    assert isinstance(av, list)
    assert len(av) == 0
    assert muv.size == 0

    assert np.all(np.isfinite(va))
    assert np.isfinite(vmu)


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

    assert isinstance(av, list)
    assert len(av) == 0
    assert muv.size == 0

    assert np.all(np.isinf(va))
    assert np.isinf(vmu)


def test_pca_full_uniquesv_pattern_mode() -> None:
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

    assert costs.shape[0] == len(lc["rms"])
    assert int(np.isfinite(costs).sum()) >= 1


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

    assert_allclose(result["Mu"], np.zeros_like(result["Mu"]), atol=1e-8)


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
        maxiters=mat_res.maxiters if hasattr(mat_res, "maxiters") else 200,
        bias=int(getattr(mat_res, "bias", 1)),
        uniquesv=int(getattr(mat_res, "uniquesv", 0)),
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
    V_mat = float(mat_res.V)

    assert_allclose(A_py @ S_py + Mu_py, A_mat @ S_mat + Mu_mat, rtol=1e-5, atol=1e-7)
    assert_allclose(V_py, V_mat, rtol=1e-5, atol=1e-7)


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
        maxiters=mat_res.maxiters if hasattr(mat_res, "maxiters") else 200,
        bias=int(getattr(mat_res, "bias", 1)),
        uniquesv=int(getattr(mat_res, "uniquesv", 0)),
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
    V_mat = float(mat_res.V)

    x_rec_py = A_py @ S_py + Mu_py
    x_rec_mat = A_mat @ S_mat + Mu_mat

    mask = ~np.isnan(x)
    diff = (x_rec_py - x_rec_mat) * mask
    mse = np.sum(diff**2) / float(np.count_nonzero(mask))
    rms = float(np.sqrt(mse))

    assert rms <= 1e-4
    assert_allclose(V_py, V_mat, rtol=1e-5, atol=1e-7)
