"""Regression tests for estimator kwargs used by analysis trade studies."""

from __future__ import annotations

import copy
from pathlib import Path
from runpy import run_path

import pytest

from vbpca_py import recommend_config

build_vbpca_kwargs = run_path(
    str(Path(__file__).parents[1] / "analysis" / "trade_study" / "_vbpca_kwargs.py")
)["build_vbpca_kwargs"]

ORDER_LEVELS = {"named": ["cost", "angle"]}
CRITERIA_PRESETS = {"named": {"cost": True, "angle": False}}


def _resolve(config: dict[str, object]) -> dict[str, object]:
    return build_vbpca_kwargs(
        config,
        random_state=123,
        criterion_order_levels=ORDER_LEVELS,
        active_criteria_presets=CRITERIA_PRESETS,
    )


def test_public_recommendation_compound_options_are_forwarded() -> None:
    with pytest.warns(UserWarning, match="wide_moderate"):
        config = recommend_config(n=50, p=300)
    expected_rmsstop = copy.deepcopy(config["rmsstop"])
    expected_criteria = copy.deepcopy(config["convergence_criteria"])

    kwargs = _resolve(config)

    assert kwargs["rmsstop"] == expected_rmsstop
    assert kwargs["convergence_criteria"] == expected_criteria
    assert kwargs["criterion_order"] == config["criterion_order"]
    assert kwargs["random_state"] == 123


def test_factorized_search_options_resolve_to_estimator_options() -> None:
    kwargs = _resolve({
        "rmsstop_window": 50,
        "rmsstop_atol": 0.01,
        "rmsstop_rtol": 0.02,
        "criterion_order": "named",
        "active_criteria": "named",
    })

    assert kwargs["rmsstop"] == [50, 0.01, 0.02]
    assert kwargs["criterion_order"] == ORDER_LEVELS["named"]
    assert kwargs["convergence_criteria"] == CRITERIA_PRESETS["named"]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"rmsstop": [10, 0.1, 0.2], "rmsstop_window": 10},
            "either rmsstop or rmsstop_window",
        ),
        (
            {"convergence_criteria": {}, "active_criteria": "named"},
            "either convergence_criteria or active_criteria",
        ),
        ({"rmsstop_atol": 0.1}, "require rmsstop_window"),
    ],
)
def test_ambiguous_or_incomplete_compound_options_fail(
    config: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve(config)
