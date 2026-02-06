# tests/test_options.py
"""Tests for the _options helper in vbpca_py._options."""

from typing import Any

import pytest

from vbpca_py._options import _normalize_keys_to_lower, _options

# ---------------------------------------------------------------------------
# _normalize_keys_to_lower
# ---------------------------------------------------------------------------


def test_normalize_keys_to_lower_basic() -> None:
    """Keys are lowercased, values preserved."""
    src: dict[str, Any] = {"Alpha": 1, "beta": 2, "GAMMA": 3}
    out = _normalize_keys_to_lower(src)

    assert set(out.keys()) == {"alpha", "beta", "gamma"}
    assert out["alpha"] == 1
    assert out["beta"] == 2
    assert out["gamma"] == 3


def test_normalize_keys_to_lower_collision_last_wins() -> None:
    """If keys differ only by case, the last one wins."""
    src = {"Alpha": 1, "ALPHA": 2}
    out = _normalize_keys_to_lower(src)

    assert list(out.keys()) == ["alpha"]
    assert out["alpha"] == 2  # last value wins


# ---------------------------------------------------------------------------
# _options: happy paths
# ---------------------------------------------------------------------------


def test_options_overrides_defaults_case_insensitive() -> None:
    """User options override defaults, case-insensitively."""
    defopts = {"maxiter": 100, "tol": 1e-3}

    opts, wrnmsg = _options(defopts, MaxIter=200, TOL=1e-4)

    # Defaults overridden
    assert opts["maxiter"] == 200
    assert opts["tol"] == pytest.approx(1e-4)

    # No warning for known keys
    assert not wrnmsg

    # Original defopts not mutated
    assert defopts["maxiter"] == 100
    assert defopts["tol"] == 1e-3


def test_options_unknown_params_included_with_warning() -> None:
    """Unknown user parameters are added and listed in the warning."""
    defopts = {"maxiter": 100}

    opts, wrnmsg = _options(defopts, alpha=0.1, BETA=0.2)

    # Unknown keys are present, lowercased
    assert opts["alpha"] == 0.1
    assert opts["beta"] == 0.2

    # Defaults still present
    assert opts["maxiter"] == 100

    # Warning message lists unknown parameters in order
    assert wrnmsg == "Unknown parameter(s): alpha, beta"


def test_options_no_unknown_params_no_warning() -> None:
    """When all user keys are known, warning message is empty."""
    defopts = {"maxiter": 100, "tol": 1e-3}

    _, wrnmsg = _options(defopts, maxiter=50, tol=1e-4)

    assert not wrnmsg


def test_options_handles_empty_kwargs() -> None:
    """Empty kwargs returns defaults unchanged and empty warning."""
    defopts = {"maxiter": 100, "tol": 1e-3}

    opts, wrnmsg = _options(defopts)

    assert opts == defopts
    assert not wrnmsg


def test_options_mixed_known_and_unknown_keys() -> None:
    """Known keys override defaults; unknown keys added and warned."""
    defopts = {"maxiter": 100, "tol": 1e-3}

    opts, wrnmsg = _options(defopts, maxiter=200, foo=1, BAR=2)

    # Known override
    assert opts["maxiter"] == 200
    assert opts["tol"] == 1e-3

    # Unknown added with lowercased keys
    assert opts["foo"] == 1
    assert opts["bar"] == 2

    assert wrnmsg == "Unknown parameter(s): foo, bar"
