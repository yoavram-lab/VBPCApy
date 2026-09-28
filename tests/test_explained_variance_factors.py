"""Tests for exact explained variance from fitted low-rank factors."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from vbpca_py._pca_full import (
    _explained_variance,
    _explained_variance_from_factors,
    _reconstruct_data,
)


@pytest.mark.parametrize(
    ("n_features", "n_samples", "n_components"),
    [(80, 30, 7), (30, 80, 7), (7, 11, 7), (5, 9, 8)],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_factor_variance_matches_direct_svd(
    n_features: int,
    n_samples: int,
    n_components: int,
    order: str,
) -> None:
    rng = np.random.default_rng(212 + n_features + n_samples)
    loadings = np.array(
        rng.normal(size=(n_features, n_components)),
        order=order,
    )
    scores = np.array(
        rng.normal(size=(n_components, n_samples)),
        order=order,
    )
    mean = rng.normal(size=n_features)
    reconstruction = _reconstruct_data(loadings, scores, mean)

    expected_ev, expected_evr = _explained_variance(
        reconstruction,
        n_components,
        solver="svd",
    )
    actual_ev, actual_evr = _explained_variance_from_factors(
        loadings,
        scores,
        n_components,
    )

    assert_allclose(actual_ev, expected_ev, rtol=1e-11, atol=1e-12)
    assert_allclose(actual_evr, expected_evr, rtol=1e-11, atol=1e-12)


def test_factor_variance_matches_rank_deficient_reconstruction() -> None:
    rng = np.random.default_rng(213)
    base = rng.normal(size=(20, 3))
    loadings = np.column_stack((base, base[:, :2], np.zeros(20)))
    scores = rng.normal(size=(6, 35))
    reconstruction = loadings @ scores

    expected_ev, expected_evr = _explained_variance(
        reconstruction,
        6,
        solver="svd",
    )
    actual_ev, actual_evr = _explained_variance_from_factors(loadings, scores, 6)

    assert_allclose(actual_ev, expected_ev, rtol=1e-11, atol=1e-12)
    assert_allclose(actual_evr, expected_evr, rtol=1e-11, atol=1e-12)


def test_factor_variance_handles_rank_zero_and_constant_scores() -> None:
    empty_ev, empty_evr = _explained_variance_from_factors(
        np.empty((8, 0)),
        np.empty((0, 12)),
        0,
    )
    assert empty_ev.size == 0
    assert empty_evr.size == 0

    zero_ev, zero_evr = _explained_variance_from_factors(
        np.ones((8, 3)),
        np.ones((3, 12)),
        3,
    )
    assert_allclose(zero_ev, 0.0, atol=1e-28)
    assert_allclose(zero_evr, 0.0, atol=0.0)


def test_factor_variance_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="compatible 2-D"):
        _explained_variance_from_factors(np.ones((5, 3)), np.ones((2, 7)), 2)
