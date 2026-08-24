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
    """p past the trade study's max (200) routes to wide_moderate and warns."""
    with pytest.warns(UserWarning, match="'wide_moderate'"):
        recommend_config(n=200, p=2000)


def test_recommend_config_warns_beyond_validated_aspect_ratio() -> None:
    """p/n past the trade study's max (2.0) warns even if p itself is small."""
    with pytest.warns(UserWarning, match="'wide_moderate'"):
        recommend_config(n=30, p=100)


def test_recommend_config_warns_wide_extreme() -> None:
    """p/n past the wide-extreme threshold (10.0) routes to wide_extreme."""
    with pytest.warns(UserWarning, match="'wide_extreme'"):
        recommend_config(n=30, p=2000)


def test_recommend_config_warns_tall_moderate() -> None:
    """n/p past the trade study's max (15.0) routes to tall_moderate."""
    with pytest.warns(UserWarning, match="'tall_moderate'"):
        recommend_config(n=1000, p=50)


def test_recommend_config_warns_tall_extreme() -> None:
    """n/p past the tall-extreme threshold (50.0) routes to tall_extreme."""
    with pytest.warns(UserWarning, match="'tall_extreme'"):
        recommend_config(n=3000, p=30)


def test_recommend_config_within_validated_region_does_not_warn() -> None:
    """p and p/n within the trade study's grid stay silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        recommend_config(n=100, p=200)
        recommend_config(n=200, p=150)


def test_recommend_config_boundary_ratios_do_not_warn() -> None:
    """Ratios exactly at a threshold stay within the wider (non-extreme) bucket."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # p/n == 10.0 exactly: not > _WIDE_EXTREME_RATIO, so wide_moderate
        # (which itself warns) is avoided only if p/n and p are also within
        # the original grid -- pick n/p == 15.0 exactly instead, which
        # stays within the original "large" bucket (p=70 is its own edge).
        recommend_config(n=1050, p=70)


def test_recommend_config_extreme_buckets_differ_from_moderate() -> None:
    """wide_extreme/tall_extreme configs differ from their moderate siblings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        wide_extreme = recommend_config(n=30, p=2000)
        wide_moderate = recommend_config(n=30, p=100)
        tall_extreme = recommend_config(n=3000, p=30)
        tall_moderate = recommend_config(n=1000, p=50)

    assert wide_extreme != wide_moderate
    assert tall_extreme != tall_moderate


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


@pytest.mark.parametrize(("n", "p"), [(30, 2000), (30, 100), (3000, 30), (1000, 50)])
def test_aspect_ratio_bucket_configs_fit(n: int, p: int) -> None:
    """The four aspect-ratio buckets' configs are valid, splat-ready VBPCA kwargs.

    Regression test for #120's initial wiring: criterion_order/
    active_criteria were shipped as trade-study preset *names* rather
    than the list/dict VBPCA's constructor actually expects, and
    rmsstop_window/atol/rtol were shipped as three separate keys instead
    of VBPCA's single compound ``rmsstop`` kwarg -- both raised a
    TypeError immediately on `VBPCA(**cfg).fit()`.
    """
    rng = np.random.default_rng(0)
    pp, nn, k = 20, 15, 2
    x = rng.standard_normal((pp, k)) @ rng.standard_normal((k, nn))
    x += 0.3 * rng.standard_normal((pp, nn))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = recommend_config(n=n, p=p)
    model = VBPCA(n_components=k, verbose=0, **cfg)
    model.fit(x)
    assert model.components_ is not None


def test_aspect_ratio_bucket_configs_not_mutated_by_caller() -> None:
    """Returned criterion_order/convergence_criteria/rmsstop aren't shared refs.

    dict(_BUCKET_CONFIGS[bucket]) is a shallow copy -- without a deep
    copy, mutating a returned config's nested list/dict would corrupt
    the shared module-level bucket for every future call.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg1 = recommend_config(n=30, p=2000)
    cfg1["rmsstop"].append(999)
    cfg1["convergence_criteria"]["angle"] = False
    cfg1["criterion_order"].append("bogus")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg2 = recommend_config(n=30, p=2000)
    assert 999 not in cfg2["rmsstop"]
    assert cfg2["convergence_criteria"]["angle"] is True
    assert "bogus" not in cfg2["criterion_order"]
