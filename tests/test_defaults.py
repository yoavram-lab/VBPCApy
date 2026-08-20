"""Tests for regime-aware default configuration recommendations."""

import warnings

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


def test_recommend_config_default_missingness_does_not_warn() -> None:
    """The default missingness="auto" is a documented no-op and stays silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        recommend_config(n=100, p=20)


def test_recommend_config_explicit_missingness_warns() -> None:
    """Passing a non-default missingness warns that it is not yet branched on."""
    with pytest.warns(UserWarning, match="does not yet branch on missingness"):
        recommend_config(n=100, p=20, missingness="mcar")


def test_recommend_config_warns_beyond_validated_p() -> None:
    """p past the trade study's max (200) warns it's an extrapolation."""
    with pytest.warns(UserWarning, match="validated region"):
        recommend_config(n=200, p=2000)


def test_recommend_config_warns_beyond_validated_aspect_ratio() -> None:
    """p/n past the trade study's max (2.0) warns even if p itself is small."""
    with pytest.warns(UserWarning, match="validated region"):
        recommend_config(n=30, p=100)


def test_recommend_config_within_validated_region_does_not_warn() -> None:
    """p and p/n within the trade study's grid stay silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        recommend_config(n=100, p=200)
        recommend_config(n=200, p=150)


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
