"""Fixed-iteration trajectory guards for backend optimization."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from vbpca_py._pca_full import pca_full
from vbpca_py.model_selection import SelectionConfig, select_n_components

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
    dense_sufficient_statistics: str = "observed",
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


def _mechanism_mask(
    mechanism: str, x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    if mechanism == "complete":
        mask = np.ones_like(x, dtype=bool)
    elif mechanism == "mcar":
        mask = rng.random(x.shape) >= 0.2
    elif mechanism == "mar":
        standardized = (x[0] - np.mean(x[0])) / np.std(x[0])
        missing_probability = 0.05 + 0.35 / (1.0 + np.exp(-standardized))
        mask = rng.random(x.shape) >= missing_probability[np.newaxis, :]
    else:
        mask = np.ones_like(x, dtype=bool)
        mask[3:10, 6:17] = False
        mask[::4, 2::5] = False

    mask[0, :] = True
    mask[:, 0] = True
    return mask


@pytest.mark.parametrize("mechanism", ["complete", "mcar", "mar", "structured"])
def test_complement_trajectory_matches_observed_cells(mechanism: str) -> None:
    rng = np.random.default_rng(210)
    x = rng.normal(size=(19, 23))
    mask = _mechanism_mask(mechanism, x, rng)
    observed = _fit_fixed_trajectory(
        x,
        mask,
        num_cpu=4,
        dense_sufficient_statistics="observed",
    )
    complement = _fit_fixed_trajectory(
        x,
        mask,
        num_cpu=4,
        dense_sufficient_statistics="complement",
    )

    for key in ("A", "S", "Mu", "Va", "Muv"):
        assert_allclose(complement[key], observed[key], rtol=5e-9, atol=5e-11)
    for key in ("Av", "Sv"):
        assert_allclose(
            _covariance_stack(complement[key]),
            _covariance_stack(observed[key]),
            rtol=5e-9,
            atol=5e-11,
        )
    assert_allclose(complement["V"], observed["V"], rtol=5e-9, atol=5e-11)
    assert_allclose(
        complement["lc"]["rms"],
        observed["lc"]["rms"],
        rtol=5e-9,
        atol=5e-11,
    )


def test_complement_mode_preserves_component_selection() -> None:
    rng = np.random.default_rng(211)
    left = rng.normal(size=(28, 3))
    right = rng.normal(size=(3, 31))
    x = left @ right + 0.2 * rng.normal(size=(28, 31))
    mask = rng.random(x.shape) >= 0.15
    mask[0, :] = True
    mask[:, 0] = True
    config = SelectionConfig(metric="cost", compute_explained_variance=False)
    common: dict[str, object] = {
        "maxiters": 8,
        "niter_broadprior": 0,
        "convergence_criteria": _CONVERGENCE_OFF,
        "runtime_tuning": "off",
        "compat_mode": "modern",
        "rotate2pca": False,
        "random_state": 211,
        "verbose": 0,
    }

    observed_k, _, observed_trace, _ = select_n_components(
        x,
        mask=mask,
        components=[1, 2, 3, 4],
        config=config,
        dense_sufficient_statistics="observed",
        **common,
    )
    complement_k, _, complement_trace, _ = select_n_components(
        x,
        mask=mask,
        components=[1, 2, 3, 4],
        config=config,
        dense_sufficient_statistics="complement",
        **common,
    )

    assert complement_k == observed_k
    assert_allclose(
        [entry["cost"] for entry in complement_trace],
        [entry["cost"] for entry in observed_trace],
        rtol=5e-9,
        atol=5e-11,
    )
