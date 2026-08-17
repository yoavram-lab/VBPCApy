"""Tests for regime-aware default configuration recommendations."""

import numpy as np
import pytest

from vbpca_py import VBPCA, recommend_config


def test_recommend_config_buckets_by_p() -> None:
    """Different feature counts map to distinct recommendation buckets."""
    small = recommend_config(n=100, p=20)
    trans = recommend_config(n=100, p=50)
    large = recommend_config(n=200, p=150)

    # All expose the dominant ARD lever with a strong loadings prior.
    for cfg in (small, trans, large):
        assert cfg["hp_va"] >= 0.5

    # Buckets are not all identical (p drives the recommendation).
    assert not (small == trans == large)


def test_recommend_config_priority_presets() -> None:
    """Priority presets adjust the iteration budget as documented."""
    balanced = recommend_config(n=100, p=100, priority="balanced")
    speed = recommend_config(n=100, p=100, priority="speed")
    accuracy = recommend_config(n=100, p=100, priority="accuracy")

    assert speed["maxiters"] < balanced["maxiters"]
    assert accuracy["maxiters"] > balanced["maxiters"]
    assert speed["niter_broadprior"] == 0


def test_recommend_config_validates_inputs() -> None:
    """Non-positive sizes and unknown priorities raise ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        recommend_config(n=0, p=10)
    with pytest.raises(ValueError, match="unknown priority"):
        recommend_config(n=10, p=10, priority="fast")  # type: ignore[arg-type]


def test_recommended_config_fits() -> None:
    """A recommended config is accepted by the estimator and fits."""
    rng = np.random.default_rng(0)
    p, n, k = 50, 100, 5
    x = rng.standard_normal((p, k)) @ rng.standard_normal((k, n))
    x += 0.3 * rng.standard_normal((p, n))

    cfg = recommend_config(n=n, p=p)
    model = VBPCA(n_components=k, verbose=0, **cfg)
    model.fit(x)
    assert model.components_ is not None
    assert model.components_.shape == (p, k)
