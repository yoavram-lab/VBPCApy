"""Tests for the exposed ARD pruning state (#251)."""

from __future__ import annotations

import numpy as np
import pytest

from vbpca_py import VBPCA
from vbpca_py._pca_full import pca_full


def _planted(rank: int = 2, n_features: int = 30, n_samples: int = 80) -> np.ndarray:
    rng = np.random.default_rng(7)
    loadings = rng.standard_normal((n_features, rank)) * 3.0
    scores = rng.standard_normal((rank, n_samples))
    return loadings @ scores + 0.1 * rng.standard_normal((n_features, n_samples))


def _fit(x: np.ndarray, **options: object) -> VBPCA:
    model = VBPCA(
        8,
        maxiters=400,
        hp_va=1e-3,
        hp_vb=1e-3,
        niter_broadprior=20,
        random_state=3,
        **options,
    )
    return model.fit(x)


def test_prior_variances_match_pca_full() -> None:
    x = _planted()
    model = _fit(x)
    result = pca_full(x, 8, **model.get_options())

    assert model.prior_variances_ is not None
    np.testing.assert_allclose(model.prior_variances_, np.ravel(result["Va"]))
    assert model.bias_prior_variance_ == pytest.approx(float(result["Vmu"]))


def test_relevance_sums_to_one_and_follows_components() -> None:
    model = _fit(_planted())

    relevance = model.component_relevance_
    assert relevance is not None
    assert relevance.shape == (8,)
    assert relevance.sum() == pytest.approx(1.0)
    energy = np.linalg.norm(model.components_, axis=0) * np.linalg.norm(
        model.scores_, axis=1
    )
    np.testing.assert_allclose(relevance, energy / energy.sum())


def test_strong_prior_prunes_to_the_planted_rank() -> None:
    model = _fit(_planted(rank=2))

    assert model.effective_rank() == 2
    assert model.effective_rank(threshold=0.0) >= 2


def test_prior_trace_is_recorded_on_request() -> None:
    model = _fit(_planted(), record_prior_trace=True)

    trace = model.prior_trace_
    assert trace is not None
    assert trace.shape == (model.n_iter_, 8)
    assert np.all(trace[0] == trace[0][0])


def test_prior_trace_is_off_by_default() -> None:
    assert _fit(_planted()).prior_trace_ is None


def test_effective_rank_requires_a_fit() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        VBPCA(3).effective_rank()
