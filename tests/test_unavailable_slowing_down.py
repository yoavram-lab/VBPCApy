"""The live fitter cannot advertise a legacy event counter that it never supplies."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py import recommend_config
from vbpca_py._converge import DEFAULT_CRITERION_ORDER, convergence_check
from vbpca_py._pca_full import _build_options, pca_full
from vbpca_py.legacy import VBPCA


def test_live_options_disable_unavailable_rule_without_mutating_input() -> None:
    """Requested and effective controls differ explicitly, with no caller mutation."""
    supplied = {"angle": False, "slowing_down": True}
    with pytest.warns(UserWarning, match="no event counter"):
        options = _build_options({"convergence_criteria": supplied})
    assert supplied == {"angle": False, "slowing_down": True}
    assert options["convergence_criteria"] == {"angle": False, "slowing_down": False}
    assert _build_options({})["convergence_criteria"] == {"slowing_down": False}


@pytest.mark.parametrize(
    ("n", "p"), [(30, 2000), (3000, 30), (1000, 1000), (30, 10000)]
)
def test_recommendations_do_not_enable_an_unavailable_counter(n: int, p: int) -> None:
    """Extremal recommendation buckets expose only supported stopping controls."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        config = recommend_config(n=n, p=p)
    assert config.get("convergence_criteria", {}).get("slowing_down", False) is False


@pytest.mark.parametrize("sparse", [False, True])
def test_real_fit_has_no_slowing_event_and_preserves_existing_endpoints(
    *, sparse: bool
) -> None:
    """Removing the inert rule cannot change the computed dense/CSR fit."""
    matrix = np.random.default_rng(53).normal(size=(12, 20))
    data = sp.csr_matrix(matrix) if sparse else matrix
    controls = dict.fromkeys(DEFAULT_CRITERION_ORDER, False)
    options = {
        "algorithm": "ppca",
        "random_state": 7,
        "verbose": 0,
        "maxiters": 12,
        "niter_broadprior": 0,
        "runtime_tuning": "off",
        "num_cpu": 1,
    }
    disabled = pca_full(data, 2, convergence_criteria=controls, **options)
    requested = controls | {"slowing_down": True}
    with pytest.warns(UserWarning, match="no event counter"):
        unavailable = pca_full(data, 2, convergence_criteria=requested, **options)
    assert requested["slowing_down"] is True
    for name in ("A", "S", "Mu", "V", "Xrec", "Vpred"):
        np.testing.assert_array_equal(disabled[name], unavailable[name])
    for name in ("rms", "angle", "criterion_satisfied_slowing_down"):
        np.testing.assert_array_equal(disabled["lc"][name], unavailable["lc"][name])
    assert unavailable["lc"]["convergence_reason"] == "maxiters"
    assert set(unavailable["lc"]["criterion_available_slowing_down"]) == {0.0}
    assert len(unavailable["lc"]["rms"]) == 13


def test_legacy_counter_diagnostic_requires_an_actual_counter() -> None:
    """Direct legacy-helper use stays distinct from the unavailable live path."""
    controls = dict.fromkeys(DEFAULT_CRITERION_ORDER, False) | {"slowing_down": True}
    curve = {"rms": [1.0, 1.0], "cost": [], "prms": []}
    message = convergence_check(
        {"convergence_criteria": controls}, curve, 1.0, sd_iter=40
    )
    assert "Slowing-down" in message
    assert curve["criterion_available_slowing_down"][-1] == pytest.approx(1.0)


@pytest.mark.parametrize("flag", ["False", "true", 0, 1, None, np.nan, [False]])
def test_nonboolean_flags_fail_before_public_fit(flag: object) -> None:
    """Invalid configuration cannot silently enable an accepted stop."""
    values = np.random.default_rng(2).normal(size=(10, 12))
    model = VBPCA(
        n_components=2,
        maxiters=3,
        verbose=0,
        convergence_criteria={"angle": flag},  # type: ignore[dict-item]
    )
    with pytest.raises(TypeError, match="values must be booleans"):
        model.fit(values)


def test_numpy_boolean_controls_are_normalized_without_mutating_caller() -> None:
    """Supported native boolean scalars retain their meaning and input ownership."""
    supplied = {"angle": np.bool_(0), "cost": np.bool_(1)}
    options = _build_options({"convergence_criteria": supplied})
    assert options["convergence_criteria"] == {
        "angle": False,
        "cost": True,
        "slowing_down": False,
    }
    assert isinstance(supplied["angle"], np.bool_)
