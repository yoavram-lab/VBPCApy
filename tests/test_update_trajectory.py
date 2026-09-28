"""Fixed-iteration trajectory guards for backend optimization."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from vbpca_py._pca_full import pca_full

_CONVERGENCE_OFF = {
    "angle": False,
    "earlystop": False,
    "rms_plateau": False,
    "cost": False,
    "composite": False,
    "slowing_down": False,
}


def _fit_fixed_trajectory(
    x: np.ndarray,
    mask: np.ndarray,
    *,
    num_cpu: int,
) -> dict[str, object]:
    return pca_full(
        x,
        n_components=3,
        mask=mask,
        algorithm="vb",
        maxiters=6,
        convergence_criteria=_CONVERGENCE_OFF,
        niter_broadprior=0,
        rotate2pca=False,
        record_cost=True,
        return_diagnostics=False,
        runtime_tuning="off",
        num_cpu=num_cpu,
        compat_mode="modern",
        random_state=207,
        display=0,
        verbose=0,
    )


def _covariance_stack(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.stack(value)


@pytest.fixture(scope="module")
def fixed_problem() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(208)
    x = rng.normal(size=(11, 13))
    mask = rng.random(x.shape) > 0.25
    mask[0, :] = True
    mask[:, 0] = True
    return x, mask


@pytest.fixture(scope="module")
def trajectory_reference(
    fixed_problem: tuple[np.ndarray, np.ndarray],
) -> dict[str, object]:
    x, mask = fixed_problem
    return _fit_fixed_trajectory(
        np.array(x, order="C"),
        np.array(mask, order="C"),
        num_cpu=1,
    )


@pytest.mark.parametrize(("order", "num_cpu"), [("C", 4), ("F", 1), ("F", 4)])
def test_fixed_trajectory_is_layout_and_thread_stable(
    fixed_problem: tuple[np.ndarray, np.ndarray],
    trajectory_reference: dict[str, object],
    order: str,
    num_cpu: int,
) -> None:
    """Six VB iterations should be invariant to layout and worker count."""
    x, mask = fixed_problem
    result = _fit_fixed_trajectory(
        np.array(x, order=order),
        np.array(mask, order=order),
        num_cpu=num_cpu,
    )

    assert len(result["lc"]["rms"]) == 7
    assert result["lc"]["convergence_reason"] == "maxiters"
    for key in ("A", "S", "Mu", "Va", "Muv"):
        assert_allclose(
            result[key],
            trajectory_reference[key],
            rtol=1e-11,
            atol=1e-12,
        )
    for key in ("Av", "Sv"):
        assert_allclose(
            _covariance_stack(result[key]),
            _covariance_stack(trajectory_reference[key]),
            rtol=1e-11,
            atol=1e-12,
        )
    assert_allclose(result["V"], trajectory_reference["V"], rtol=1e-11)
    assert_allclose(
        result["lc"]["rms"],
        trajectory_reference["lc"]["rms"],
        rtol=1e-11,
        atol=1e-12,
    )
    assert_allclose(
        result["lc"]["cost"],
        trajectory_reference["lc"]["cost"],
        rtol=1e-11,
        atol=1e-12,
    )
