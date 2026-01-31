"""Tests for the pca_full implementation.

These tests focus on:
- basic shape and type correctness
- handling of missing values (NaNs)
- dense vs. sparse inputs
- different algorithms ('vb', 'map', 'ppca')
- uniquesv pattern-sharing path
- cost computation when cfstop is given
- consistency between lc['rms'] and explicit reconstruction error
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

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
    A: np.ndarray,
    S: np.ndarray,
    Mu: np.ndarray,
) -> float:
    """Compute RMS error on observed entries only, matching compute_rms logic.

    Parameters
    ----------
    x_original : (n_features, n_samples) with NaNs where missing.
    A, S, Mu : returned by pca_full.

    Returns:
    -------
    float
        Root mean squared error over non-missing entries.
    """
    # Reconstruction in original (uncentered) space
    recon = A @ S + Mu

    mask = ~np.isnan(x_original)
    ndata = np.count_nonzero(mask)

    # If there are no observed entries, define RMS as nan
    if ndata == 0:
        return float("nan")

    diff = (recon - np.nan_to_num(x_original, copy=True)) * mask
    mse = np.sum(diff**2) / float(ndata)
    return float(np.sqrt(mse))


# ----------------------------------------------------------------------
# Basic dense test, algorithm='vb'
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
    for key in ("A", "S", "Mu", "V", "Av", "Sv", "Isv", "Muv", "Va", "Vmu", "lc"):
        assert key in result

    A = result["A"]
    S = result["S"]
    Mu = result["Mu"]
    V = result["V"]
    lc = result["lc"]

    # Shapes
    assert A.shape == (5, 2)
    assert S.shape == (2, 7)
    # Mu may be (n_features, 1) or (n_features,), but your impl uses (n_features, 1)
    assert Mu.shape == (5, 1)
    assert np.isscalar(V) or np.shape(V) == ()

    # Learning curves: at least one update beyond initial
    assert len(lc["rms"]) >= 2
    assert len(lc["prms"]) == len(lc["rms"])
    assert len(lc["time"]) == len(lc["rms"])
    assert len(lc["cost"]) == len(lc["rms"])

    # RMS should be non-negative
    assert all(r >= 0.0 or np.isnan(r) for r in lc["rms"])

    # Explicit RMS reconstruction on observed entries should match last lc["rms"]
    last_rms = lc["rms"][-1]
    explicit_rms = _compute_masked_rms(x, A, S, Mu)
    if not np.isnan(last_rms) and not np.isnan(explicit_rms):
        assert_allclose(explicit_rms, last_rms, rtol=1e-5, atol=1e-7)


# ----------------------------------------------------------------------
# Sparse input, algorithm='map'
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

    A = result["A"]
    S = result["S"]
    Mu = result["Mu"]
    Av = result["Av"]
    Muv = result["Muv"]
    Va = result["Va"]
    Vmu = result["Vmu"]

    assert A.shape == (4, 2)
    assert S.shape == (2, 6)
    assert Mu.shape == (4, 1)

    # For MAP: use_prior=True, use_postvar=False, so Av and Muv are empty
    assert Av == [] or (isinstance(Av, list) and len(Av) == 0)
    assert Muv.size == 0

    # Va and Vmu should be finite (not inf here)
    assert np.all(np.isfinite(Va))
    assert np.isfinite(Vmu)


# ----------------------------------------------------------------------
# Algorithm='ppca' behavior
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

    Av = result["Av"]
    Muv = result["Muv"]
    Va = result["Va"]
    Vmu = result["Vmu"]

    # In ppca: use_prior=False, use_postvar=False → Av and Muv are empty
    assert Av == [] or (isinstance(Av, list) and len(Av) == 0)
    assert Muv.size == 0

    # Va and Vmu should be infinite (no priors)
    assert np.all(np.isinf(Va))
    assert np.isinf(Vmu)


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
    # Columns 0 and 2 share pattern, columns 1 and 3 share another

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

    Sv = result["Sv"]
    Isv = result["Isv"]
    lc = result["lc"]

    # We expect some pattern structure: Isv should not be None
    assert Isv is not None
    Isv = np.asarray(Isv, dtype=int)

    # There should be at most n_samples distinct Sv entries
    n_samples = x.shape[1]
    assert isinstance(Sv, list)
    assert 1 <= len(Sv) <= n_samples

    # Isv indices should be within Sv range
    assert np.all((Isv >= 0) & (Isv < len(Sv)))

    # Learning curves are still populated
    assert len(lc["rms"]) >= 2


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
    assert finite_mask.sum() >= 1


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

    A = result["A"]
    S = result["S"]
    Mu = result["Mu"]

    assert A.shape == (5, 3)
    assert S.shape == (3, 7)
    assert Mu.shape == (5, 1)
