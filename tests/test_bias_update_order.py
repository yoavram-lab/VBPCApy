"""Regression tests for statistically coherent bias/RMS update ordering."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from vbpca_py._pca_full import _build_options, pca_full
from vbpca_py.model_selection import SelectionConfig, select_n_components

_CONVERGENCE_OFF = {
    "angle": False,
    "earlystop": False,
    "rms_plateau": False,
    "cost": False,
    "composite": False,
    "slowing_down": False,
}


def _offset_problem(missing_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(914)
    left = rng.normal(size=(40, 3))
    right = rng.normal(size=(3, 60))
    x = (
        left @ right
        + 0.15 * rng.normal(size=(40, 60))
        + np.linspace(-20.0, 20.0, 40)[:, np.newaxis]
    )
    mask = rng.random(x.shape) >= missing_fraction
    mask[:, 0] = True
    mask[0, :] = True
    return x, mask


def _fixed_fit(
    x: np.ndarray,
    mask: np.ndarray,
    order: str,
    *,
    maxiters: int = 15,
) -> dict[str, object]:
    return pca_full(
        x,
        mask=mask,
        n_components=5,
        maxiters=maxiters,
        niter_broadprior=maxiters,
        convergence_criteria=_CONVERGENCE_OFF,
        rotate2pca=True,
        runtime_tuning="off",
        compat_mode="modern",
        bias_update_order=order,
        random_state=914,
        display=0,
        verbose=0,
        bias=True,
    )


@pytest.mark.filterwarnings("ignore:maxiters=.*does not exceed niter_broadprior")
@pytest.mark.parametrize("missing_fraction", [0.0, 0.2, 0.5])
@pytest.mark.parametrize("order", ["legacy", "post_factor"])
def test_bias_state_persists_and_rms_matches_returned_model(
    missing_fraction: float,
    order: str,
) -> None:
    x, mask = _offset_problem(missing_fraction)
    result = _fixed_fit(x, mask, order)
    rms = np.asarray(result["lc"]["rms"][1:], dtype=float)

    assert np.all(np.isfinite(rms))
    assert np.count_nonzero(np.diff(rms) > 1e-12) == 0

    reconstruction = np.asarray(result["A"]) @ np.asarray(result["S"]) + np.asarray(
        result["Mu"]
    )
    manual_rms = np.sqrt(np.sum(((x - reconstruction) * mask) ** 2) / mask.sum())
    assert_allclose(rms[-1], manual_rms, rtol=1e-11, atol=1e-12)


@pytest.mark.filterwarnings("ignore:maxiters=.*does not exceed niter_broadprior")
def test_post_factor_order_preserves_selection_and_objective() -> None:
    rng = np.random.default_rng(222)
    x = (
        rng.normal(size=(24, 3)) @ rng.normal(size=(3, 32))
        + 0.15 * rng.normal(size=(24, 32))
        + np.linspace(-8.0, 8.0, 24)[:, np.newaxis]
    )
    mask = rng.random(x.shape) >= 0.2
    mask[:, 0] = True
    mask[0, :] = True
    common: dict[str, object] = {
        "mask": mask,
        "components": [1, 2, 3, 4, 5],
        "config": SelectionConfig(
            metric="cost",
            compute_explained_variance=False,
        ),
        "maxiters": 25,
        "niter_broadprior": 25,
        "convergence_criteria": _CONVERGENCE_OFF,
        "rotate2pca": True,
        "runtime_tuning": "off",
        "compat_mode": "modern",
        "random_state": 222,
        "verbose": 0,
    }
    legacy_k, _, legacy_trace, _ = select_n_components(
        x, bias_update_order="legacy", **common
    )
    post_k, _, post_trace, _ = select_n_components(
        x, bias_update_order="post_factor", **common
    )

    assert legacy_k == post_k == 3
    assert [row["k"] for row in legacy_trace] == [row["k"] for row in post_trace]
    legacy_costs = np.asarray([row["cost"] for row in legacy_trace])
    post_costs = np.asarray([row["cost"] for row in post_trace])
    assert_allclose(
        post_costs,
        legacy_costs,
        rtol=2e-4,
        atol=1e-6,
    )


@pytest.mark.filterwarnings("ignore:maxiters=.*does not exceed niter_broadprior")
def test_post_factor_probe_rms_matches_returned_model() -> None:
    rng = np.random.default_rng(223)
    x = rng.normal(size=(12, 16)) + np.linspace(-5.0, 5.0, 12)[:, np.newaxis]
    xprobe = np.full_like(x, np.nan)
    probe_rows = np.arange(12)
    probe_cols = (3 * probe_rows) % x.shape[1]
    xprobe[probe_rows, probe_cols] = x[probe_rows, probe_cols]

    out = pca_full(
        x,
        n_components=3,
        xprobe=xprobe,
        maxiters=10,
        niter_broadprior=10,
        convergence_criteria=_CONVERGENCE_OFF,
        compat_mode="modern",
        bias_update_order="post_factor",
        runtime_tuning="off",
        random_state=223,
        verbose=0,
    )
    reconstruction = np.asarray(out["A"]) @ np.asarray(out["S"]) + np.asarray(out["Mu"])
    probe_mask = np.isfinite(xprobe)
    manual = np.sqrt(
        np.sum((xprobe[probe_mask] - reconstruction[probe_mask]) ** 2)
        / probe_mask.sum()
    )
    assert_allclose(out["lc"]["prms"][-1], manual, rtol=1e-11, atol=1e-12)


def test_bias_update_order_resolution_and_reporting() -> None:
    assert _build_options({"compat_mode": "modern"})["bias_update_order"] == (
        "post_factor"
    )
    assert (
        _build_options({"compat_mode": "strict_legacy"})["bias_update_order"]
        == "legacy"
    )
    with pytest.raises(ValueError, match="bias_update_order"):
        _build_options({"bias_update_order": "unknown"})

    out = pca_full(
        np.arange(24, dtype=float).reshape(4, 6),
        n_components=2,
        maxiters=1,
        niter_broadprior=0,
        compat_mode="modern",
        runtime_tuning="off",
        runtime_report=True,
        verbose=0,
    )
    assert out["RuntimeReport"]["bias_update_order"] == "post_factor"
