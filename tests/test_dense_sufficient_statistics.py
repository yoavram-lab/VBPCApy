"""Dispatch tests for density-adaptive dense sufficient statistics."""

from __future__ import annotations

import numpy as np
import pytest

from vbpca_py._pca_full import pca_full

_CONVERGENCE_OFF = {
    "angle": False,
    "earlystop": False,
    "rms_plateau": False,
    "cost": False,
    "composite": False,
    "slowing_down": False,
}


def _fit_with_report(
    *,
    size: int,
    density: float,
    components: int,
    mode: str = "auto",
) -> dict[str, object]:
    rng = np.random.default_rng(212)
    x = rng.normal(size=(size, size))
    mask = rng.random(x.shape) < density
    mask[0, :] = True
    mask[:, 0] = True
    result = pca_full(
        x,
        n_components=components,
        mask=mask,
        maxiters=1,
        niter_broadprior=0,
        convergence_criteria=_CONVERGENCE_OFF,
        rotate2pca=False,
        return_diagnostics=False,
        runtime_tuning="off",
        runtime_report=True,
        dense_sufficient_statistics=mode,
        compat_mode="modern",
        random_state=212,
        display=0,
        verbose=0,
    )
    report = result["RuntimeReport"]
    assert isinstance(report, dict)
    dense_report = report["dense_sufficient_statistics"]
    assert isinstance(dense_report, dict)
    return dense_report


def test_auto_uses_complement_above_calibrated_thresholds() -> None:
    report = _fit_with_report(size=320, density=0.95, components=10)

    assert report["score_mode"] == "complement"
    assert report["loading_mode"] == "complement"
    assert report["source"] == "density_heuristic"
    assert float(report["observed_fraction"]) >= 0.92


def test_auto_keeps_study_density_on_observed_path() -> None:
    report = _fit_with_report(size=320, density=0.82, components=10)

    assert report["score_mode"] == "observed"
    assert report["loading_mode"] == "observed"


def test_auto_retains_observed_path_below_crossover() -> None:
    report = _fit_with_report(size=320, density=0.50, components=10)

    assert report["score_mode"] == "observed"
    assert report["loading_mode"] == "observed"


@pytest.mark.parametrize("mode", ["observed", "complement"])
def test_user_mode_overrides_small_problem_guard(mode: str) -> None:
    report = _fit_with_report(size=20, density=0.85, components=3, mode=mode)

    assert report["score_mode"] == mode
    assert report["loading_mode"] == mode
    assert report["source"] == "user"


def test_invalid_dense_sufficient_statistics_mode_raises() -> None:
    with pytest.raises(ValueError, match="dense_sufficient_statistics"):
        pca_full(
            np.ones((4, 5)),
            n_components=2,
            dense_sufficient_statistics="unknown",
            verbose=0,
        )
