# tests/test_converge.py
"""Tests for convergence checking functionality.

This module contains unit tests for the convergence_check function,
verifying angle-based stopping, early stopping, RMS plateau detection,
cost plateau detection, and slowing-down iteration criteria.
"""

from __future__ import annotations

from typing import Any

from vbpca_py._converge import convergence_check


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
