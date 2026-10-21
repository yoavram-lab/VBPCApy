"""Check analysis-only prototypes against independent observed-cell sums."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

import vbpca_py._full_update as updates
import vbpca_py._pca_full as fitting

_PROBES = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "analysis" / "review_backend_costs.py")
)


@pytest.mark.parametrize("rank", [1, 3, 7])
@pytest.mark.parametrize("pattern", ["complete", "random", "structured", "one_cell"])
def test_full_covariance_rearrangements(rank: int, pattern: str) -> None:
    """Keep off-diagonal terms, bias uncertainty and explicit observation masks."""
    rng = np.random.default_rng(814)
    features, samples = 9, 11
    a = rng.normal(size=(features, rank))
    s = rng.normal(size=(rank, samples))
    ra = rng.normal(size=(features, rank, rank))
    rs = rng.normal(size=(samples, rank, rank))
    av = ra @ ra.transpose(0, 2, 1) + 0.01 * np.eye(rank)
    sv = rs @ rs.transpose(0, 2, 1) + 0.02 * np.eye(rank)
    muv = rng.uniform(0.01, 0.2, size=(features, 1))
    expected = np.empty((features, samples))
    for i in range(features):
        for j in range(samples):
            expected[i, j] = (
                a[i] @ sv[j] @ a[i]
                + s[:, j] @ av[i] @ s[:, j]
                + np.trace(av[i] @ sv[j])
                + muv[i, 0]
            )
    actual = _PROBES["contractions"](a, s, av, sv, muv)
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    mask = np.ones((features, samples), dtype=bool)
    if pattern == "random":
        mask = rng.random(mask.shape) > 0.3
    elif pattern == "structured":
        mask[2:7, 3:8] = False
    elif pattern == "one_cell":
        mask[:] = False
        mask[4, 5] = True
    state = SimpleNamespace(
        loadings=a,
        scores=s,
        loading_covariances=av,
        score_covariances=sv,
        mu_variances=muv,
    )
    total = _PROBES["complete_minus_missing"](state, mask)
    assert_allclose(total, expected[mask].sum(), rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("mode", ["observed", "complement"])
def test_noise_prototype_fixed_iteration_fit(monkeypatch, mode: str) -> None:
    """A bounded full-VB check retains RMS, hyperpriors, floors and cost traces."""
    rng = np.random.default_rng(815)
    x = rng.normal(size=(19, 23))
    mask = rng.random(x.shape) > 0.15
    mask[0, :] = True
    mask[:, 0] = True
    x[0, 0] = 0.0  # An observed zero must contribute to sufficient statistics.
    opts = {
        **_PROBES["OPTIONS"],
        "maxiters": 6,
        "record_cost": True,
        "rotate2pca": False,
        "dense_sufficient_statistics": mode,
    }
    reference = fitting.pca_full(x, 3, mask=mask, **opts)

    def prototype(state, rms, ix, jx, hp_v=0.001):
        assert state.pattern_index is None
        np.testing.assert_array_equal(np.nonzero(mask), (ix, jx))
        s_xv = _PROBES["complete_minus_missing"](state, mask)
        s_xv += rms**2 * state.n_data
        return updates._finalize_noise_state(state, s_xv, hp_v)

    monkeypatch.setattr(fitting, "_update_noise_variance", prototype)
    candidate = fitting.pca_full(x, 3, mask=mask, **opts)
    for key in ("A", "S", "Mu", "Va", "Muv", "V", "Av", "Sv"):
        assert_allclose(candidate[key], reference[key], rtol=1e-10, atol=1e-11)
    for key in ("rms", "cost"):
        assert_allclose(
            candidate["lc"][key], reference["lc"][key], rtol=1e-10, atol=1e-11
        )
