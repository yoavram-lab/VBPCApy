"""Tests for offline convergence-policy replay."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis.trade_study.convergence_detector import (
    DetectorPolicy,
    collapse_equivalent_policies,
    policy_from_options,
    replay_policy,
)
from vbpca_py._converge import DEFAULT_CRITERION_ORDER


def _curve(
    *,
    rms: list[float],
    angle: list[float] | None = None,
    cost: list[float] | None = None,
    prms: list[float] | None = None,
) -> dict[str, list[float]]:
    length = len(rms)
    return {
        "rms": rms,
        "prms": prms if prms is not None else [np.nan] * length,
        "cost": cost if cost is not None else [np.nan] * length,
        "angle": angle if angle is not None else [np.nan] * length,
    }


def test_replay_resets_patience_after_warmup() -> None:
    policy = DetectorPolicy(
        name="angle",
        enabled=("angle",),
        minangle=0.1,
        patience=2,
        warmup=2,
    )
    result = replay_policy(
        _curve(rms=[5, 4, 3, 2, 1], angle=[np.nan, 0.05, 0.04, 0.03, 0.02]),
        policy,
    )

    assert result.stop_iteration == 4
    assert result.reason == "angle"


def test_disabled_raw_hit_cannot_stop_replay() -> None:
    policy = DetectorPolicy(
        name="rms_only",
        enabled=("rms_plateau",),
        minangle=0.1,
        rmsstop=(1, 0.01, 0.01),
    )
    result = replay_policy(
        _curve(rms=[5, 4, 3], angle=[np.nan, 0.01, 0.001]),
        policy,
    )

    assert result.stop_iteration is None
    assert result.reason == "maxiters"


def test_criterion_order_changes_reason_but_not_stop_iteration() -> None:
    base = {
        "enabled": ("angle", "rms_plateau"),
        "minangle": 0.1,
        "rmsstop": (1, 0.1, 0.1),
    }
    angle_first = DetectorPolicy(name="angle_first", **base)
    rms_order = (
        "rms_plateau",
        "angle",
        "earlystop",
        "cost",
        "composite",
        "slowing_down",
    )
    rms_first = DetectorPolicy(name="rms_first", criterion_order=rms_order, **base)
    learning_curve = _curve(rms=[1.0, 0.95], angle=[np.nan, 0.05])

    result_angle = replay_policy(learning_curve, angle_first)
    result_rms = replay_policy(learning_curve, rms_first)

    assert result_angle.stop_iteration == result_rms.stop_iteration == 1
    assert result_angle.reason == "angle"
    assert result_rms.reason == "rms_plateau"


def test_policy_collapse_ignores_reason_attribution_order() -> None:
    policy_a = DetectorPolicy(
        name="a",
        enabled=("angle", "rms_plateau"),
        minangle=1e-5,
        rmsstop=(10, 1e-4, 1e-3),
    )
    policy_b = DetectorPolicy(
        name="b",
        enabled=("angle", "rms_plateau"),
        minangle=1e-5,
        rmsstop=(10, 1e-4, 1e-3),
        criterion_order=tuple(reversed(DEFAULT_CRITERION_ORDER)),
    )

    representatives, aliases = collapse_equivalent_policies([policy_a, policy_b])

    assert representatives == [policy_a]
    assert aliases == {"a": "a", "b": "a"}


def test_relative_cost_rule_replays_with_patience() -> None:
    policy = DetectorPolicy(
        name="relative_cost",
        enabled=("cost",),
        cfstop_rel=0.02,
        patience=2,
    )
    result = replay_policy(
        _curve(rms=[4, 3, 2, 1], cost=[100, 90, 89, 88]),
        policy,
    )

    assert result.stop_iteration == 3
    assert result.reason == "cfstop_rel"


def test_policy_from_options_keeps_only_effectively_configured_criteria() -> None:
    options = {
        "criterion_order": list(DEFAULT_CRITERION_ORDER),
        "convergence_criteria": dict.fromkeys(DEFAULT_CRITERION_ORDER, True),
        "minangle": 1e-4,
        "earlystop": False,
        "rmsstop": np.array([50, 1e-6, 1e-3]),
        "cfstop": np.array([]),
        "cfstop_rel": 1e-5,
        "cfstop_curv": None,
        "composite_stop": None,
        "patience": 2,
        "niter_broadprior": 100,
    }

    policy = policy_from_options("production", options)
    without_cost = policy_from_options(
        "production_without_cost",
        options,
        disabled=("cost",),
    )

    assert policy.enabled == (
        "angle",
        "rms_plateau",
        "cost",
        "slowing_down",
    )
    assert policy.patience == 2
    assert policy.warmup == 100
    assert policy.rmsstop == (50, 1e-6, 1e-3)
    assert np.isclose(policy.cfstop_rel, 1e-5)
    assert "cost" not in without_cost.enabled
    assert without_cost.cfstop_rel is None


def test_policy_from_options_rejects_unknown_disabled_criterion() -> None:
    with pytest.raises(ValueError, match="unknown disabled"):
        policy_from_options(
            "invalid",
            {"minangle": 1e-4},
            disabled=("not-a-criterion",),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"enabled": ("angle",)}, "minangle"),
        ({"enabled": ("cost",)}, "cost detector"),
        ({"enabled": ("rms_plateau",)}, "rmsstop"),
        ({"enabled": ("earlystop",)}, "detector flag"),
    ],
)
def test_policy_rejects_incomplete_enabled_criteria(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        DetectorPolicy(name="invalid", **kwargs)  # type: ignore[arg-type]
