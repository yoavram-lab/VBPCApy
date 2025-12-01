"""Tests for the _monitoring module.

This module tests:
- init_params: parameter initialization with random and explicit dictionary modes
- Logging functions: log_first_step, log_step, log_progress
- Display helpers: display_init, display_progress (matplotlib optional)
- Shape validation for Av, Muv, and Sv parameters
- Pattern-sharing behavior for score covariances
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from numpy.testing import assert_allclose

from vbpca_py._monitoring import (
    ERR_AV_SHAPE,
    ERR_INIT_TYPE,
    ERR_MUV_SHAPE,
    ERR_SV_PATTERN_INDEX,
    InitShapes,
    display_init,
    display_progress,
    init_params,
    log_first_step,
    log_progress,
    log_step,
)

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


# ------------------------------------------------------------------------------
# init_params — basic random initialization
# ------------------------------------------------------------------------------


def test_init_params_random_basic() -> None:
    """Basic random initialization with no pattern indexing."""
    shapes = InitShapes(
        n_features=5,
        n_samples=4,
        n_components=3,
        n_obs_patterns=4,
    )

    result = init_params("random", shapes, score_pattern_index=None, rng=_rng())

    # Shapes
    assert result.a.shape == (5, 3)
    assert result.s.shape == (3, 4)
    assert result.mu.shape == (5,)
    assert result.muv.shape == (5, 1)
    assert len(result.av) == 5
    assert len(result.sv) == 4
    assert result.v == 1.0

    # AV default are identity
    for av_i in result.av:
        assert_allclose(av_i, np.eye(3))


# ------------------------------------------------------------------------------
# init_params — explicit dictionary path
# ------------------------------------------------------------------------------


def test_init_params_dict_provided_all() -> None:
    """Initialization from explicit dict with all parameters provided."""
    shapes = InitShapes(
        n_features=4,
        n_samples=3,
        n_components=2,
        n_obs_patterns=3,
    )

    init_dict = {
        "A": np.ones((4, 2)),
        "Av": np.ones((4, 2)),
        "Mu": np.arange(4),
        "Muv": np.ones((4, 1)) * 2.0,
        "V": 3.0,
        "S": np.ones((2, 3)),
        "Sv": np.eye(2).reshape(1, 2, 2).repeat(3, axis=0),
    }

    result = init_params(init_dict, shapes, score_pattern_index=None, rng=_rng())

    assert_allclose(result.a, np.ones((4, 2)))
    assert_allclose(result.s, np.ones((2, 3)))
    assert_allclose(result.mu, np.arange(4))
    assert_allclose(result.muv, np.ones((4, 1)) * 2.0)
    assert result.v == 3.0

    # Av (from 2D array) → diagonal matrices
    for i in range(4):
        assert_allclose(result.av[i], np.diag([1.0, 1.0]))

    # Sv from 3D array
    for sv_i in result.sv:
        assert_allclose(sv_i, np.eye(2))


# ------------------------------------------------------------------------------
# init_params — AV shape constraints
# ------------------------------------------------------------------------------


def test_av_shape_error() -> None:
    """Test that an invalid Av shape raises a ValueError."""
    shapes = InitShapes(5, 4, 3, 4)
    init_dict = {"Av": np.ones((5, 5))}  # invalid

    with pytest.raises(ValueError, match=re.escape(ERR_AV_SHAPE)):
        init_params(init_dict, shapes, score_pattern_index=None, rng=_rng())


# ------------------------------------------------------------------------------
# init_params — Muv shape constraints
# ------------------------------------------------------------------------------


def test_muv_shape_error() -> None:
    """Test that an invalid Muv shape raises a ValueError."""
    shapes = InitShapes(5, 4, 3, 4)
    init_dict = {"Muv": np.ones((5, 2))}  # invalid

    with pytest.raises(ValueError, match=re.escape(ERR_MUV_SHAPE)):
        init_params(init_dict, shapes, score_pattern_index=None, rng=_rng())


# ------------------------------------------------------------------------------
# init_params — invalid init type
# ------------------------------------------------------------------------------


def test_init_invalid_type() -> None:
    """Test that an invalid init type raises a ValueError."""
    shapes = InitShapes(5, 4, 3, 4)

    with pytest.raises(ValueError, match=re.escape(ERR_INIT_TYPE)):
        init_params(12345, shapes, score_pattern_index=None, rng=_rng())


# ------------------------------------------------------------------------------
# Sv pattern-sharing
# ------------------------------------------------------------------------------


def test_sv_pattern_sharing() -> None:
    """Test Sv pattern-sharing behavior."""
    shapes = InitShapes(
        n_features=5,
        n_samples=6,
        n_components=3,
        n_obs_patterns=2,  # < n_samples → pattern mode
    )

    # Two patterns: columns [0,2,4] and [1,3,5]
    isv = np.array([0, 1, 0, 1, 0, 1])
    # each component j stored variance vector for each column j
    sv_raw = np.vstack(
        [
            np.arange(6),  # component 0 variances
            np.arange(6) + 10,  # component 1 variances
            np.arange(6) + 20,  # component 2 variances
        ]
    )

    init_dict = {"Sv": sv_raw, "Isv": isv}

    result = init_params(init_dict, shapes, score_pattern_index=isv, rng=_rng())

    # Since n_obs_patterns=2, result.sv must have 2 covariance matrices
    assert len(result.sv) == 2

    # Pattern 0 → use first occurrence in isv = index 0
    sv0 = result.sv[0]
    assert_allclose(sv0, np.diag(sv_raw[:, 0]))

    # Pattern 1 → index 1
    sv1 = result.sv[1]
    assert_allclose(sv1, np.diag(sv_raw[:, 1]))


def test_sv_pattern_missing_isv_raises() -> None:
    """Test that missing Isv with multiple observation patterns raises a ValueError."""
    shapes = InitShapes(
        n_features=4,
        n_samples=5,
        n_components=2,
        n_obs_patterns=2,
    )

    init_dict = {"Sv": np.ones((2, 5))}

    with pytest.raises(ValueError, match=re.escape(ERR_SV_PATTERN_INDEX)):
        init_params(init_dict, shapes, score_pattern_index=None, rng=_rng())


# ------------------------------------------------------------------------------
# Logging tests (these do not require matplotlib)
# ------------------------------------------------------------------------------


def test_log_first_step(caplog: pytest.LogCaptureFixture) -> None:
    """Test log_first_step output."""
    caplog.set_level("INFO")
    log_first_step(1, rms=0.5, prms=0.25)
    assert "Step 0" in caplog.text
    assert "0.500000" in caplog.text
    assert "0.250000" in caplog.text


def test_log_step(caplog: pytest.LogCaptureFixture) -> None:
    """Test log_step output."""
    caplog.set_level("INFO")
    lc = {"cost": [1.23], "rms": [0.8], "prms": [0.7]}
    log_step(1, lc, angle_a=0.33)
    assert "Step 0" in caplog.text
    assert "cost = 1.230000" in caplog.text
    assert "rms = 0.800000" in caplog.text
    assert "prms = 0.700000" in caplog.text
    assert "angle(A) = 0.3300" in caplog.text


def test_log_progress(caplog: pytest.LogCaptureFixture) -> None:
    """Test log_progress output."""
    caplog.set_level("INFO")
    log_progress(2, current=3, total=10, phase="E-step")
    assert "E-step 3/10" in caplog.text


# ------------------------------------------------------------------------------
# Display helpers (matplotlib optional)
# ------------------------------------------------------------------------------


def test_display_init_no_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test display initialization when matplotlib is not available."""
    # Force matplotlib to be unavailable
    from vbpca_py import _monitoring as mon

    monkeypatch.setattr(mon, "plt", None)

    dsph = display_init(1, {"rms": [1, 2], "prms": [3, 4]})
    assert dsph == {"display": True}  # only display flag is returned


def test_display_init_with_matplotlib() -> None:
    """Test display initialization when matplotlib is available."""
    # Only assert that handles exist, *not* pixel values.
    try:
        import matplotlib.pyplot as plt  # noqa: F401, PLC0415
    except ImportError:
        pytest.skip("matplotlib not installed")

    lc = {"rms": [1, 2, 3], "prms": [2, 4, 6]}
    dsph = display_init(1, lc)

    assert dsph["display"] is True
    assert "fig" in dsph
    assert "rms" in dsph
    assert "prms" in dsph


def test_display_progress_smoke() -> None:
    """Test display progress when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401, PLC0415
    except ImportError:
        pytest.skip("matplotlib not installed")

    lc = {"rms": [1, 2, 3], "prms": [2, 4, 6]}
    dsph = display_init(1, lc)
    display_progress(dsph, lc)
