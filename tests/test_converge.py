# tests/test_converge.py
"""Tests for convergence checking functionality.

This module contains unit tests for the convergence_check function,
verifying angle-based stopping, early stopping, RMS plateau detection,
cost plateau detection, and slowing-down iteration criteria.
"""

from __future__ import annotations

from typing import Any

import pytest

from vbpca_py._converge import DEFAULT_CRITERION_ORDER, convergence_check


def _lc(
    *,
    rms: list[float],
    prms: list[float],
    cost: list[float],
) -> dict[str, Any]:
    """Helper to build a minimal lc dict.

    Returns:
        Dictionary containing ``rms``, ``prms``, ``cost``, and ``time``
        sequences with matching lengths.
    """
    n = len(rms)
    return {
        "rms": list(rms),
        "prms": list(prms),
        "cost": list(cost),
        # time just needs to be the same length as rms/prms for PrintStep logic
        "time": list(range(n)),
    }


# --------------------------------------------------------------------------
# Basic angle / early stop behaviour
# --------------------------------------------------------------------------


def test_angle_convergence_triggers() -> None:
    """If angleA < minangle, we should stop with an angle-based message."""
    opts = {
        "minangle": 1e-3,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
    }
    lc = _lc(
        rms=[1.0, 0.9],
        prms=[1.0, 0.9],
        cost=[10.0, 9.0],
    )

    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert "angle" in msg.lower()


def test_no_angle_no_earlystop_no_plateau() -> None:
    """If no criterion is met, we should return an empty message."""
    opts = {
        "minangle": 0.0,  # angle always bigger
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
    }
    lc = _lc(
        rms=[1.0, 0.9, 0.8],
        prms=[0.9, 0.8, 0.7],
        cost=[10.0, 9.5, 9.0],
    )

    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert not msg


def test_early_stop_triggers_when_prms_increases() -> None:
    """Early stopping triggers when probe RMS increases."""
    opts = {
        "minangle": 0.0,
        "earlystop": True,
        "rmsstop": None,
        "cfstop": None,
    }
    lc = _lc(
        rms=[1.0, 0.9, 0.8],
        prms=[0.5, 0.4, 0.45],  # last > previous
        cost=[10.0, 9.5, 9.0],
    )

    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert "early" in msg.lower()


# --------------------------------------------------------------------------
# RMS plateau behaviour
# --------------------------------------------------------------------------


def test_rms_plateau_requires_enough_iterations() -> None:
    """RMS plateau is not checked until we have more than 'window' lag."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": [3, 1e-3, 1e-3],  # window=3
    }
    # len(rms) = 4 => len(rms)-1 = 3, which is NOT > window=3
    lc = _lc(
        rms=[0.5, 0.4, 0.35, 0.34],
        prms=[0.9, 0.8, 0.75, 0.74],
        cost=[10.0, 9.0, 8.5, 8.2],
    )

    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    # Not enough history; should not stop on RMS plateau yet.
    assert not msg


def test_rms_plateau_absolute_tolerance() -> None:
    """RMS plateau stop should trigger when absolute change < abs_tol."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        # window=3, abs_tol=1e-3, rel_tol=0.0
        "rmsstop": [3, 1e-3, 0.0],
    }
    # We need len(rms) - 1 > window => len(rms) >= 5
    # window = 3 => compare rms[-5+?] (-(3+1) = -4) with rms[-1].
    # older = rms[-4] = 0.3002, newer = rms[-1] = 0.3006
    # delta = 4e-4 < 1e-3 -> plateau.
    lc = _lc(
        rms=[0.3000, 0.3002, 0.3004, 0.3005, 0.3006],
        prms=[0.9, 0.8, 0.75, 0.74, 0.74],
        cost=[10.0, 9.0, 8.5, 8.3, 8.2],
    )

    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert "rms" in msg.lower()


def test_rms_plateau_relative_tolerance() -> None:
    """RMS plateau should also work via relative tolerance."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        # window=2, tiny abs_tol so we trigger via relative tolerance
        "rmsstop": [2, 1e-12, 1e-3],
    }
    # len(rms) = 4 => len(rms)-1 = 3 > window=2, so check plateau.
    # older = rms[-3] = 0.100, newer = rms[-1] = 0.10006
    # delta = 6e-05; rel ≈ 6e-04 < 1e-3 -> plateau.
    lc = _lc(
        rms=[0.15, 0.100, 0.1000, 0.10006],
        prms=[0.9, 0.8, 0.7, 0.7],
        cost=[10.0, 9.0, 8.5, 8.4],
    )

    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert "rms" in msg.lower()


# --------------------------------------------------------------------------
# Cost plateau behaviour
# --------------------------------------------------------------------------


def test_cost_plateau_requires_enough_iterations() -> None:
    """Cost plateau is not checked until history length > window."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "cfstop": [3, 1e-3, 1e-3],  # window=3
    }
    # len(cost) = 4 => len(cost)-1 = 3, not > 3
    lc = _lc(
        rms=[0.5, 0.4, 0.35, 0.34],
        prms=[0.9, 0.8, 0.75, 0.74],
        cost=[10.0, 9.0, 8.5, 8.3],
    )

    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert not msg


def test_cost_plateau_after_rms() -> None:
    """Cost plateau is checked (after angle/earlystop/RMS) when enabled."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        # window=2, abs_tol=1e-3, rel_tol=1e-3
        "cfstop": [2, 1e-3, 1e-3],
    }
    # len(cost) = 4 => len(cost)-1 = 3 > 2, so we check plateau.
    # older = cost[-3] = 8.0, newer = 8.0005 -> delta = 5e-4 < 1e-3
    lc = _lc(
        rms=[0.4, 0.3, 0.25, 0.24],
        prms=[0.9, 0.8, 0.7, 0.7],
        cost=[10.0, 8.0, 8.0, 8.0005],
    )

    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert "cost" in msg.lower()


# --------------------------------------------------------------------------
# Slowing-down / sd_iter behaviour
# --------------------------------------------------------------------------


def test_slowing_down_stop_triggers_at_40() -> None:
    """Slowing-down stop criterion triggers exactly when sd_iter == 40."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
    }
    lc = _lc(
        rms=[1.0, 0.9],
        prms=[1.0, 0.9],
        cost=[10.0, 9.0],
    )

    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=40)
    assert "slowing-down" in msg.lower() or "slowing" in msg.lower()


def test_slowing_down_stop_not_triggered_before_40() -> None:
    """sd_iter < 40 should not trigger the slowing-down criterion."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
    }
    lc = _lc(
        rms=[1.0, 0.9],
        prms=[1.0, 0.9],
        cost=[10.0, 9.0],
    )

    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=10)
    assert msg == ""


# --------------------------------------------------------------------------
# Priority / interaction tests
# --------------------------------------------------------------------------


def test_angle_has_priority_over_rms_and_cost() -> None:
    """If angle criterion is met, it should short-circuit other criteria."""
    opts = {
        "minangle": 1e-3,
        "earlystop": True,
        "rmsstop": [2, 1e-3, 1e-3],
        "cfstop": [2, 1e-3, 1e-3],
    }
    lc = _lc(
        rms=[0.5, 0.4, 0.39, 0.389],  # plateau-ish
        prms=[0.5, 0.4, 0.41, 0.42],  # would trigger earlystop
        cost=[10.0, 9.0, 8.0, 8.0],  # plateau-ish
    )

    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    # We expect an angle-based message, not an earlystop/RMS/cost plateau one.
    assert "angle" in msg.lower()


def test_earlystop_has_priority_over_rms_and_cost() -> None:
    """If earlystop is triggered, RMS/cost plateaus should not override it."""
    opts = {
        "minangle": 0.0,
        "earlystop": True,
        "rmsstop": [2, 1e-3, 1e-3],
        "cfstop": [2, 1e-3, 1e-3],
    }
    lc = _lc(
        rms=[0.5, 0.4, 0.39, 0.389],  # plateau-ish
        prms=[0.5, 0.4, 0.39, 0.40],  # increases -> earlystop
        cost=[10.0, 9.0, 8.0, 8.0],  # plateau-ish
    )

    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert "early" in msg.lower()


def test_rms_plateau_message_contains_window_info() -> None:
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": [2, 1e-3, 1e-3],
        "cfstop": None,
    }
    lc = _lc(
        rms=[0.5, 0.4, 0.4001, 0.4002],
        prms=[1.0, 0.9, 0.8, 0.8],
        cost=[10.0, 9.0, 8.5, 8.4],
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert "RMS" in msg
    assert "over 2 iterations" in msg


def test_cost_plateau_message_contains_window_info() -> None:
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": [2, 1e-3, 1e-3],
    }
    lc = _lc(
        rms=[0.5, 0.4, 0.3, 0.2],
        prms=[1.0, 0.9, 0.8, 0.7],
        cost=[8.0, 7.0, 7.0004, 7.0006],
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert "cost" in msg
    assert "over 2 iterations" in msg


# --------------------------------------------------------------------------
# Relative ELBO decrease (cfstop_rel) behaviour
# --------------------------------------------------------------------------


def test_cfstop_rel_triggers_when_change_small() -> None:
    """Relative ELBO stop fires when |ΔELBO|/|ELBO| < threshold."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": 1e-3,
    }
    # |8.0001 - 8.0| / |8.0001| ≈ 1.25e-5 < 1e-3 -> trigger
    lc = _lc(
        rms=[0.5, 0.4],
        prms=[1.0, 0.9],
        cost=[8.0, 8.0001],
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert "relative ELBO" in msg.lower() or "cfstop_rel" in msg


def test_cfstop_rel_does_not_trigger_when_change_large() -> None:
    """Relative ELBO stop should not fire when change is above threshold."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": 1e-6,
    }
    # |9.0 - 10.0| / |9.0| ≈ 0.111 >> 1e-6
    lc = _lc(
        rms=[0.5, 0.4],
        prms=[1.0, 0.9],
        cost=[10.0, 9.0],
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert msg == ""


def test_cfstop_rel_needs_two_cost_values() -> None:
    """Cannot compute relative change with only one cost value."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": 1e-3,
    }
    lc = _lc(
        rms=[0.5],
        prms=[1.0],
        cost=[8.0],
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert msg == ""


def test_cfstop_rel_disabled_when_none() -> None:
    """cfstop_rel=None should not trigger any stop."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
    }
    lc = _lc(
        rms=[0.5, 0.4],
        prms=[1.0, 0.9],
        cost=[8.0, 8.0],  # zero change, but disabled
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert msg == ""


def test_cfstop_rel_message_format() -> None:
    """The stop message should include the relative change and threshold."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": 1e-4,
    }
    lc = _lc(
        rms=[0.5, 0.4],
        prms=[1.0, 0.9],
        cost=[100.0, 100.000001],
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert "cfstop_rel" in msg
    assert "1.000e-04" in msg


# --------------------------------------------------------------------------
# ELBO curvature / 2nd difference (cfstop_curv) behaviour
# --------------------------------------------------------------------------


def test_cfstop_curv_triggers_when_curvature_small() -> None:
    """ELBO curvature stop fires when |Δ²ELBO| < threshold."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": 1e-3,
    }
    # Δ = [-1.0, -0.9999] -> curvature = |(-0.9999) - (-1.0)| = 1e-4 < 1e-3
    lc = _lc(
        rms=[0.5, 0.4, 0.3],
        prms=[1.0, 0.9, 0.8],
        cost=[10.0, 9.0, 8.0001],
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert "curvature" in msg.lower() or "cfstop_curv" in msg


def test_cfstop_curv_does_not_trigger_when_curvature_large() -> None:
    """ELBO curvature stop should not fire when curvature is above threshold."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": 1e-6,
    }
    # Δ = [-1.0, -0.5] -> curvature = |(-0.5) - (-1.0)| = 0.5 >> 1e-6
    lc = _lc(
        rms=[0.5, 0.4, 0.3],
        prms=[1.0, 0.9, 0.8],
        cost=[10.0, 9.0, 8.5],
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert msg == ""


def test_cfstop_curv_needs_three_cost_values() -> None:
    """Cannot compute 2nd difference with fewer than 3 cost values."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": 1e-3,
    }
    lc = _lc(
        rms=[0.5, 0.4],
        prms=[1.0, 0.9],
        cost=[10.0, 9.0],
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert msg == ""


def test_cfstop_curv_disabled_when_none() -> None:
    """cfstop_curv=None should not trigger any stop."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": None,
    }
    lc = _lc(
        rms=[0.5, 0.4, 0.3],
        prms=[1.0, 0.9, 0.8],
        cost=[10.0, 10.0, 10.0],  # zero curvature, but disabled
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert msg == ""


def test_cfstop_curv_message_format() -> None:
    """The stop message should include the curvature value and threshold."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": 1e-2,
    }
    # Δ = [-1.0, -1.001] -> curvature = 0.001 < 0.01
    lc = _lc(
        rms=[0.5, 0.4, 0.3],
        prms=[1.0, 0.9, 0.8],
        cost=[10.0, 9.0, 7.999],
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert "cfstop_curv" in msg


# --------------------------------------------------------------------------
# Priority: cfstop_rel / cfstop_curv vs other criteria
# --------------------------------------------------------------------------


def test_cost_plateau_has_priority_over_cfstop_rel() -> None:
    """cfstop (plateau) should fire before cfstop_rel when both are met."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": [2, 1e-3, 1e-3],
        "cfstop_rel": 1e-3,
    }
    # Both cost plateau and relative ELBO would trigger
    lc = _lc(
        rms=[0.5, 0.4, 0.3, 0.2],
        prms=[1.0, 0.9, 0.8, 0.7],
        cost=[8.0, 7.0, 7.0004, 7.0006],
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    # Cost plateau has priority (checked first)
    assert "cost" in msg.lower()
    assert "over 2 iterations" in msg


def test_cfstop_rel_has_priority_over_cfstop_curv() -> None:
    """cfstop_rel should fire before cfstop_curv when both are met."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": 1e-3,
        "cfstop_curv": 1e-3,
    }
    # Both would trigger
    lc = _lc(
        rms=[0.5, 0.4, 0.3],
        prms=[1.0, 0.9, 0.8],
        cost=[10.0, 10.00001, 10.00002],
    )
    msg = convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert "cfstop_rel" in msg


# --------------------------------------------------------------------------
# Composite stop (composite_stop) behaviour
# --------------------------------------------------------------------------


def test_composite_stop_all_met() -> None:
    """Composite stop triggers when ALL sub-criteria are satisfied."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": None,
        "composite_stop": {"angle": 1e-3, "rms": 1e-3},
    }
    # angle_a=1e-4 < 1e-3 ✓; rms rel change ≈ 2.5e-4 < 1e-3 ✓
    lc = _lc(
        rms=[0.5, 0.4, 0.4001],
        prms=[1.0, 0.9, 0.8],
        cost=[10.0, 9.0, 8.0],
    )
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert "composite" in msg.lower()
    assert "angle" in msg.lower()
    assert "rms_rel" in msg.lower()


def test_composite_stop_partial_not_met() -> None:
    """Composite stop does NOT trigger when only some sub-criteria are met."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": None,
        "composite_stop": {"angle": 1e-3, "rms": 1e-6},
    }
    # angle met (1e-4 < 1e-3), but rms rel change ≈ 0.25 >> 1e-6
    lc = _lc(
        rms=[0.5, 0.4, 0.3],
        prms=[1.0, 0.9, 0.8],
        cost=[10.0, 9.0, 8.0],
    )
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert msg == ""


def test_composite_stop_with_elbo_rel() -> None:
    """Composite stop works with the elbo_rel sub-criterion."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": None,
        "composite_stop": {"angle": 1e-3, "rms": 1e-3, "elbo_rel": 1e-3},
    }
    # All three met
    lc = _lc(
        rms=[0.5, 0.4, 0.4001],
        prms=[1.0, 0.9, 0.8],
        cost=[8.0, 8.0001, 8.00015],
    )
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert "composite" in msg.lower()
    assert "elbo_rel" in msg.lower()


def test_composite_stop_elbo_rel_not_met() -> None:
    """Composite stop fails when elbo_rel sub-criterion is not met."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": None,
        "composite_stop": {"angle": 1e-3, "elbo_rel": 1e-8},
    }
    # angle met, but ELBO change is large
    lc = _lc(
        rms=[0.5, 0.4],
        prms=[1.0, 0.9],
        cost=[10.0, 9.0],
    )
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert msg == ""


def test_composite_stop_disabled_when_none() -> None:
    """composite_stop=None should not trigger any stop."""
    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": None,
        "composite_stop": None,
    }
    lc = _lc(
        rms=[0.5, 0.4, 0.4],
        prms=[1.0, 0.9, 0.8],
        cost=[8.0, 8.0, 8.0],
    )
    msg = convergence_check(opts, lc, angle_a=1e-10, sd_iter=None)
    assert msg == ""


def test_composite_stop_unknown_key_raises() -> None:
    """Unknown keys in composite_stop should raise ValueError."""
    import pytest

    opts = {
        "minangle": 0.0,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": None,
        "composite_stop": {"angle": 1e-3, "bogus": 0.1},
    }
    lc = _lc(
        rms=[0.5, 0.4],
        prms=[1.0, 0.9],
        cost=[10.0, 9.0],
    )
    with pytest.raises(ValueError, match="Unknown composite_stop key"):
        convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)


def test_individual_criteria_still_fire_with_composite_unset() -> None:
    """Individual criteria still work when composite_stop is None."""
    opts = {
        "minangle": 1e-3,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": None,
        "composite_stop": None,
    }
    lc = _lc(
        rms=[0.5, 0.4],
        prms=[1.0, 0.9],
        cost=[10.0, 9.0],
    )
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert "angle" in msg.lower()


# --------------------------------------------------------------------------
# Patience window behaviour
# --------------------------------------------------------------------------


def _opts_with_patience(patience: int) -> dict[str, object]:
    """Build opts with angle criterion and a patience window."""
    return {
        "minangle": 1e-3,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "cfstop_rel": None,
        "cfstop_curv": None,
        "composite_stop": None,
        "patience": patience,
    }


def test_patience_suppresses_first_trigger() -> None:
    """With patience=3, the first trigger should be suppressed."""
    opts = _opts_with_patience(3)
    lc = _lc(rms=[0.5, 0.4], prms=[1.0, 0.9], cost=[10.0, 9.0])
    lc["_patience"] = [0.0]

    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert msg == ""
    assert lc["_patience"][0] == pytest.approx(1.0)


def test_patience_fires_after_consecutive_triggers() -> None:
    """With patience=3, the third consecutive trigger emits the message."""
    opts = _opts_with_patience(3)
    lc = _lc(rms=[0.5, 0.4], prms=[1.0, 0.9], cost=[10.0, 9.0])
    lc["_patience"] = [0.0]

    # Triggers 1 and 2: suppressed
    convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert lc["_patience"][0] == pytest.approx(2.0)

    # Trigger 3: fires
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert "angle" in msg.lower()


def test_patience_resets_on_no_trigger() -> None:
    """If a non-triggering iteration breaks the streak, counter resets."""
    opts = _opts_with_patience(3)
    lc = _lc(rms=[0.5, 0.4], prms=[1.0, 0.9], cost=[10.0, 9.0])
    lc["_patience"] = [0.0]

    # Two triggers
    convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert lc["_patience"][0] == pytest.approx(2.0)

    # No trigger (angle too large)
    convergence_check(opts, lc, angle_a=1.0, sd_iter=None)
    assert lc["_patience"][0] == pytest.approx(0.0)

    # Need 3 more consecutive to fire
    convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert "angle" in msg.lower()


def test_patience_1_is_default_behaviour() -> None:
    """patience=1 should behave like no patience (immediate trigger)."""
    opts = _opts_with_patience(1)
    lc = _lc(rms=[0.5, 0.4], prms=[1.0, 0.9], cost=[10.0, 9.0])

    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert "angle" in msg.lower()


# --------------------------------------------------------------------------
# Criterion ordering (criterion_order)
# --------------------------------------------------------------------------


def test_default_criterion_order_matches_current_behaviour() -> None:
    """DEFAULT_CRITERION_ORDER reproduces the existing hardcoded priority."""
    assert DEFAULT_CRITERION_ORDER == [
        "angle",
        "earlystop",
        "rms_plateau",
        "cost",
        "composite",
        "slowing_down",
    ]


def test_criterion_order_changes_winner() -> None:
    """Reordering criteria lets a lower-priority criterion win."""
    # Both angle and rms_plateau would fire; default order → angle wins.
    opts_default: dict[str, Any] = {
        "minangle": 1e-3,
        "earlystop": False,
        "rmsstop": [2, 1e-1, 1e-1],
        "cfstop": None,
    }
    lc = _lc(rms=[1.0, 1.0, 1.0], prms=[1.0, 0.9, 0.8], cost=[10.0, 9.0, 8.0])

    msg = convergence_check(opts_default, lc, angle_a=1e-4, sd_iter=None)
    assert "angle" in msg.lower()

    # Now put rms_plateau first.
    opts_reordered = {
        **opts_default,
        "criterion_order": [
            "rms_plateau",
            "angle",
            "earlystop",
            "cost",
            "composite",
            "slowing_down",
        ],
    }
    lc2 = _lc(rms=[1.0, 1.0, 1.0], prms=[1.0, 0.9, 0.8], cost=[10.0, 9.0, 8.0])
    msg2 = convergence_check(opts_reordered, lc2, angle_a=1e-4, sd_iter=None)
    assert "rms" in msg2.lower()


def test_criterion_order_none_uses_default() -> None:
    """criterion_order=None reproduces default order."""
    opts: dict[str, Any] = {
        "minangle": 1e-3,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "criterion_order": None,
    }
    lc = _lc(rms=[1.0, 0.9], prms=[1.0, 0.9], cost=[10.0, 9.0])
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert "angle" in msg.lower()


# --------------------------------------------------------------------------
# Per-criterion enable/disable (convergence_criteria)
# --------------------------------------------------------------------------


def test_disabled_criterion_is_skipped() -> None:
    """Disabling the angle criterion lets a lower one fire instead."""
    opts: dict[str, Any] = {
        "minangle": 1e-3,
        "earlystop": False,
        "rmsstop": [2, 1e-1, 1e-1],
        "cfstop": None,
        "convergence_criteria": {"angle": False},
    }
    lc = _lc(rms=[1.0, 1.0, 1.0], prms=[1.0, 0.9, 0.8], cost=[10.0, 9.0, 8.0])
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    # angle is disabled, so rms_plateau should win.
    assert "rms" in msg.lower()
    assert "angle" not in msg.lower()


def test_disabled_criterion_no_stop() -> None:
    """Disabling the only triggering criterion results in no stop."""
    opts: dict[str, Any] = {
        "minangle": 1e-3,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "convergence_criteria": {"angle": False},
    }
    lc = _lc(rms=[1.0, 0.9], prms=[1.0, 0.9], cost=[10.0, 9.0])
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert not msg


def test_all_enabled_by_default() -> None:
    """Empty convergence_criteria dict means all criteria are enabled."""
    opts: dict[str, Any] = {
        "minangle": 1e-3,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "convergence_criteria": {},
    }
    lc = _lc(rms=[1.0, 0.9], prms=[1.0, 0.9], cost=[10.0, 9.0])
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert "angle" in msg.lower()


def test_convergence_criteria_none_enables_all() -> None:
    """convergence_criteria=None enables all criteria (default)."""
    opts: dict[str, Any] = {
        "minangle": 1e-3,
        "earlystop": False,
        "rmsstop": None,
        "cfstop": None,
        "convergence_criteria": None,
    }
    lc = _lc(rms=[1.0, 0.9], prms=[1.0, 0.9], cost=[10.0, 9.0])
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    assert "angle" in msg.lower()


def test_ordering_and_disable_combined() -> None:
    """criterion_order + convergence_criteria work together."""
    opts: dict[str, Any] = {
        "minangle": 1e-3,
        "earlystop": True,
        "rmsstop": [2, 1e-1, 1e-1],
        "cfstop": None,
        # Put earlystop first, but disable it.
        "criterion_order": [
            "earlystop",
            "rms_plateau",
            "angle",
            "cost",
            "composite",
            "slowing_down",
        ],
        "convergence_criteria": {"earlystop": False},
    }
    lc = _lc(rms=[1.0, 1.0, 1.0], prms=[0.5, 0.4, 0.45], cost=[10.0, 9.0, 8.0])
    msg = convergence_check(opts, lc, angle_a=1e-4, sd_iter=None)
    # earlystop is first but disabled; rms_plateau is next and fires.
    assert "rms" in msg.lower()
