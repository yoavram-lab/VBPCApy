"""Tests for the ARD/noise variance update order (#253)."""

from __future__ import annotations

import numpy as np
import pytest

from vbpca_py import VBPCA
from vbpca_py._pca_full import pca_full

HP = 1e-3


def _data(missing: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    loadings = rng.standard_normal((40, 3)) * np.array([4.0, 2.0, 1.0])
    scores = rng.standard_normal((3, 120))
    x = loadings @ scores + 0.3 * rng.standard_normal((40, 120)) + 2.0
    if missing:
        x[rng.random(x.shape) < missing] = np.nan
    return x


def _fit(x: np.ndarray, order: str, **options: object) -> dict[str, object]:
    return pca_full(
        x,
        8,
        maxiters=300,
        hp_va=HP,
        hp_vb=HP,
        niter_broadprior=20,
        random_state=3,
        verbose=0,
        variance_update_order=order,
        **options,
    )


def _direct_prior_variances(result: dict[str, object], x: np.ndarray) -> np.ndarray:
    loadings = np.asarray(result["A"])
    posterior = sum(np.diag(np.asarray(cov)) for cov in result["Av"])  # type: ignore[union-attr]
    observed = np.isfinite(x).mean()
    return (np.sum(loadings**2, axis=0) + posterior + 2 * HP) / (
        (loadings.shape[0] + 2 * HP) / observed
    )


def test_invalid_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="variance_update_order"):
        _fit(_data(0.0), "sideways")


@pytest.mark.parametrize(
    ("compat_mode", "expected"),
    [("strict_legacy", "legacy"), ("modern", "post_rotation")],
)
def test_auto_follows_the_compatibility_mode(compat_mode: str, expected: str) -> None:
    result = _fit(_data(0.0), "auto", compat_mode=compat_mode, runtime_report=True)

    report = result["RuntimeReport"]
    assert isinstance(report, dict)
    assert report["variance_update_order"] == expected


@pytest.mark.parametrize("missing", [0.0, 0.2, 0.5])
def test_post_rotation_returns_prior_variances_of_the_returned_components(
    missing: float,
) -> None:
    x = _data(missing)
    result = _fit(x, "post_rotation")

    np.testing.assert_allclose(
        np.ravel(result["Va"]), _direct_prior_variances(result, x), rtol=1e-8
    )


@pytest.mark.parametrize("missing", [0.0, 0.2, 0.5])
def test_only_legacy_lags_while_the_rotation_still_moves_components(
    missing: float,
) -> None:
    x = _data(missing)
    off = dict.fromkeys(
        ("angle", "earlystop", "rms_plateau", "cost", "composite", "slowing_down"),
        False,
    )
    early = {"convergence_criteria": off}

    legacy = pca_full(
        x,
        8,
        maxiters=12,
        hp_va=HP,
        hp_vb=HP,
        niter_broadprior=2,
        va_init=10.0,
        random_state=3,
        verbose=0,
        variance_update_order="legacy",
        **early,
    )
    post = pca_full(
        x,
        8,
        maxiters=12,
        hp_va=HP,
        hp_vb=HP,
        niter_broadprior=2,
        va_init=10.0,
        random_state=3,
        verbose=0,
        variance_update_order="post_rotation",
        **early,
    )

    assert not np.allclose(np.ravel(legacy["Va"]), _direct_prior_variances(legacy, x))
    np.testing.assert_allclose(
        np.ravel(post["Va"]), _direct_prior_variances(post, x), rtol=1e-8
    )


@pytest.mark.parametrize("order", ["legacy", "post_rotation"])
@pytest.mark.parametrize("missing", [0.0, 0.2, 0.5])
def test_cost_does_not_increase_after_warmup(order: str, missing: float) -> None:
    result = _fit(_data(missing), order, record_cost=True)

    cost = np.asarray(result["lc"]["cost"])[21:]  # type: ignore[index]
    assert np.all(np.diff(cost) <= 1e-8 * np.abs(cost[:-1]))


def test_orders_agree_without_rotation() -> None:
    x = _data(0.2)
    common = {"rotate2pca": False, "bias_update_order": "post_factor"}

    legacy = _fit(x, "legacy", **common)
    post = _fit(x, "post_rotation", **common)

    np.testing.assert_allclose(legacy["Xrec"], post["Xrec"], rtol=1e-10, atol=1e-12)


def test_estimator_reports_aligned_prior_variances() -> None:
    x = _data(0.2)
    model = VBPCA(
        8,
        maxiters=300,
        hp_va=HP,
        hp_vb=HP,
        niter_broadprior=20,
        random_state=3,
        variance_update_order="post_rotation",
    ).fit(x)

    assert model.prior_variances_ is not None
    assert model.component_relevance_ is not None
    order_by_prior = np.argsort(-model.prior_variances_)
    order_by_relevance = np.argsort(-model.component_relevance_)
    np.testing.assert_array_equal(order_by_prior[:3], order_by_relevance[:3])
