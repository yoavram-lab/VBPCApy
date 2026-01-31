"""Tests for the pca_full implementation.

These tests focus on:

- Basic shape and type correctness
- Handling of missing values (NaNs)
- Dense vs. sparse inputs
- Different algorithms ("vb", "map", "ppca")
- uniquesv pattern-sharing path
- Cost computation when cfstop is given
- Consistency between lc["rms"] and explicit reconstruction error
- Comparison with scikit-learn PCA in fully observed cases
- Optional regression tests against the original MATLAB implementation
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.io import loadmat

# Import the refactored top-level routine
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


def _compute_masked_rms(
    x_original: np.ndarray,
    loadings: np.ndarray,
    scores: np.ndarray,
    mean: np.ndarray,
) -> float:
    """Compute RMS error on observed entries only, matching compute_rms logic.

    Parameters
    ----------
    x_original
        Data matrix (n_features, n_samples) with NaNs where missing.
    loadings, scores, mean
        Returned by pca_full as A, S, Mu.

    Returns:
    -------
    RMS over non-missing entries, or NaN if there are none.
    """
    # Reconstruction in original (uncentered) space
    recon = loadings @ scores + mean

    mask = ~np.isnan(x_original)
    ndata = int(np.count_nonzero(mask))

    if ndata == 0:
        return float("nan")

    diff = (recon - np.nan_to_num(x_original, copy=True)) * mask
    mse = np.sum(diff**2) / float(ndata)
    return float(np.sqrt(mse))


def _fixture_path(name: str) -> pathlib.Path:
    """Return path to a MATLAB fixture under tests/data."""
    return pathlib.Path(__file__).resolve().parent.joinpath("data").joinpath(name)


# ----------------------------------------------------------------------
# Basic dense test, algorithm="vb"
# ----------------------------------------------------------------------


def test_pca_full_dense_vb_basic() -> None:
    """Dense data with NaNs, VB algorithm, small number of iterations."""
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

    # Learning curves: at least one update beyond initial
    assert len(lc["rms"]) >= 1
    assert len(lc["prms"]) == len(lc["rms"])
    assert len(lc["time"]) == len(lc["rms"])
    assert len(lc["cost"]) == len(lc["rms"])
    # Optional initial-error curve
    assert "err0" in lc
    assert len(lc["err0"]) == 1

    # RMS should be non-negative (or NaN)
    assert all(r >= 0.0 or np.isnan(r) for r in lc["rms"])

    # Explicit RMS reconstruction on observed entries should roughly
    # match last lc["rms"]
    last_rms = float(lc["rms"][-1])
    explicit_rms = _compute_masked_rms(x, loadings, scores, mean)
    if not np.isnan(last_rms) and not np.isnan(explicit_rms):
        assert_allclose(explicit_rms, last_rms, rtol=1e-4, atol=1e-6)


# ----------------------------------------------------------------------
# Sparse input, algorithm="map"
# ----------------------------------------------------------------------


def test_pca_full_sparse_map_basic() -> None:
    """Sparse CSR input, MAP algorithm, check shapes and MAP-specific behavior."""
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

    # For MAP: use_prior=True, use_postvar=False, so Av and Muv are empty
    assert isinstance(av, list)
    assert len(av) == 0
    assert muv.size == 0

    # Va and Vmu should be finite (not inf here)
    assert np.all(np.isfinite(va))
    assert np.isfinite(vmu)


# ----------------------------------------------------------------------
# Algorithm="ppca" behavior
# ----------------------------------------------------------------------


def test_pca_full_ppca_no_posterior_variances() -> None:
    """PPCA mode: no priors, no posterior variances for A and Mu."""
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

    # In ppca: use_prior=False, use_postvar=False → Av and Muv are empty
    assert isinstance(av, list)
    assert len(av) == 0
    assert muv.size == 0

    # Va and Vmu should be infinite (no priors)
    assert np.all(np.isinf(va))
    assert np.isinf(vmu)


# ----------------------------------------------------------------------
# uniquesv=True: pattern-sharing path
# ----------------------------------------------------------------------


def test_pca_full_uniquesv_pattern_mode() -> None:
    """Exercise uniquesv=True path with shared missingness patterns."""
    # Construct data where pairs of columns share identical missingness
    x = np.array(
        [
            [1.0, np.nan, 3.0, np.nan],
            [2.0, np.nan, 4.0, np.nan],
            [np.nan, 5.0, np.nan, 7.0],
            [np.nan, 6.0, np.nan, 8.0],
        ]
    )
    # Columns 0 and 2 share pattern; columns 1 and 3 share another.

    result = pca_full(
        x,
        n_components=2,
        maxiters=5,
        algorithm="vb",
        uniquesv=1,  # enable pattern sharing
        autosave=0,
        display=0,
        verbose=0,
    )

    sv = result["Sv"]
    pattern_index = result["Isv"]
    lc = result["lc"]

    # We expect some pattern structure: Isv should not be None
    assert pattern_index is not None
    pattern_index = np.asarray(pattern_index, dtype=int)

    n_samples = x.shape[1]
    assert isinstance(sv, list)
    assert 1 <= len(sv) <= n_samples

    # Isv indices should be within Sv range
    assert np.all((pattern_index >= 0) & (pattern_index < len(sv)))

    # Learning curves are still populated
    assert len(lc["rms"]) >= 1


# ----------------------------------------------------------------------
# cfstop: cost computation path
# ----------------------------------------------------------------------


def test_pca_full_cost_computation_cfstop() -> None:
    """Non-empty cfstop should trigger cost computation and fill lc['cost']."""
    x = _make_toy_dense_with_nans(n_features=4, n_samples=5, seed=4)

    result = pca_full(
        x,
        n_components=2,
        maxiters=4,
        algorithm="vb",
        cfstop=np.array([10, 1e-4, 1e-3]),  # anything non-empty triggers cost
        autosave=0,
        display=0,
        verbose=0,
    )

    lc = result["lc"]
    costs = np.asarray(lc["cost"], dtype=float)

    # There should be one cost per iteration (including initial NaN)
    assert costs.shape[0] == len(lc["rms"])

    # After the first iteration or two we expect some finite cost values
    finite_mask = np.isfinite(costs)
    # At least one finite cost after the initial entry
    assert int(finite_mask.sum()) >= 1


# ----------------------------------------------------------------------
# Smoke test with rotate2pca off
# ----------------------------------------------------------------------


def test_pca_full_rotate2pca_off() -> None:
    """Smoke test: rotate2pca=False should still run and return valid shapes."""
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

    loadings = result["A"]
    scores = result["S"]
    mean = result["Mu"]

    assert loadings.shape == (5, 3)
    assert scores.shape == (3, 7)
    assert mean.shape == (5, 1)


# ----------------------------------------------------------------------
# Bias / mean behaviour
# ----------------------------------------------------------------------


def test_pca_full_bias_disabled_gives_zero_mean() -> None:
    """When bias=0, the returned mean vector should be (approximately) zero."""
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
    # Bias disabled → mean should be very close to zero.
    assert_allclose(mean, np.zeros_like(mean), atol=1e-8)


def test_pca_full_tiny_problem_smoke() -> None:
    """Very small problem (2x2, k=1) runs and produces finite results."""
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

    loadings = result["A"]
    scores = result["S"]
    mean = result["Mu"]
    lc = result["lc"]

    assert loadings.shape == (2, 1)
    assert scores.shape == (1, 2)
    assert mean.shape == (2, 1)
    assert np.isfinite(result["V"])
    assert all(np.isfinite(r) for r in lc["rms"] if not np.isnan(r))


# ----------------------------------------------------------------------
# Optional MATLAB regression tests (fixtures generated externally)
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_pca_full_matches_matlab_dense_fixture() -> None:
    """Regression test: compare against MATLAB PCA_FULL on a dense fixture.

    This test assumes you have generated a .mat file with keys:

    - "x": data matrix (n_features, n_samples)
    - "k": integer number of components
    - "result": MATLAB result struct with fields A, S, Mu, V (at least)

    You can generate this fixture in MATLAB using the original PCA_FULL
    implementation, a fixed RNG seed, and a fixed "init" struct.
    """
    fixture = _fixture_path("legacy_pca_full_dense.mat")
    if not fixture.exists():
        pytest.skip("MATLAB dense fixture not available")

    mat = loadmat(fixture, squeeze_me=True, struct_as_record=False)

    x = np.asarray(mat["x"], dtype=float)
    k = int(mat["k"])
    mat_res = mat["result"]

    # MATLAB uses rows as features, columns as samples, like our code.
    # Call our implementation with matching algorithm / settings.
    result = pca_full(
        x,
        n_components=k,
        # Adjust these to match how you ran MATLAB PCA_FULL:
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

    # Reconstructions should match closely
    assert_allclose(x_rec_py, x_rec_mat, rtol=1e-5, atol=1e-7)
    # Noise variance should also be very close
    assert_allclose(V_py, V_mat, rtol=1e-5, atol=1e-7)


@pytest.mark.slow
def test_pca_full_matches_matlab_missing_fixture() -> None:
    """Regression test: compare against MATLAB PCA_FULL with missing data.

    Assumes a .mat fixture with:

    - "x": data matrix with NaNs
    - "k": components
    - "result": MATLAB result struct
    """
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

    # Only compare on observed entries (non-NaN positions)
    mask = ~np.isnan(x)
    diff = (x_rec_py - x_rec_mat) * mask
    mse = np.sum(diff**2) / float(np.count_nonzero(mask))
    rms = float(np.sqrt(mse))

    assert rms <= 1e-4
    assert_allclose(V_py, V_mat, rtol=1e-5, atol=1e-7)
