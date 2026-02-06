"""Tests for the vbpca_py._monitoring helpers (init, logging, plotting, RMS)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

import vbpca_py._monitoring as mon
from vbpca_py._monitoring import (
    ERR_SV_PATTERN_INDEX,
    ERR_SV_SHAPE,
    InitialMonitoringInputs,
    InitShapes,
    display_init,
    display_progress,
    init_params,
    log_first_step,
    log_progress,
    log_step,
)
from vbpca_py._rms import RmsConfig


def _rng() -> np.random.Generator:
    """Deterministic RNG for tests.

    Returns:
        A seeded NumPy random number generator.
    """
    return np.random.default_rng(1234)


# ------------------------------------------------------------------------------
# Sv / Isv initialisation tests
# ------------------------------------------------------------------------------


def test_sv_pattern_mode_from_array() -> None:
    """Check Sv pattern mode when Sv is provided as a 2D variance array."""
    shapes = InitShapes(
        n_features=5,
        n_samples=6,
        n_components=3,
        n_obs_patterns=2,
    )

    # Two patterns: columns [0,2,4] and [1,3,5]
    isv = np.array([0, 1, 0, 1, 0, 1])
    # each component j stored variance vector for each column j
    sv_raw = np.vstack(
        [
            np.arange(6),  # component 0 variances
            np.arange(6) + 10,  # component 1 variances
            np.arange(6) + 20,  # component 2 variances
        ],
    )

    init_dict: Mapping[str, object] = {"Sv": sv_raw, "Isv": isv}

    result = init_params(init_dict, shapes, score_pattern_index=isv, rng=_rng())

    # Since n_obs_patterns=2, result.sv must have 2 covariance matrices
    assert len(result.sv) == 2

    # Pattern 0 → use first occurrence in isv = index 0
    sv0 = result.sv[0]
    assert np.allclose(sv0, np.diag(sv_raw[:, 0]))

    # Pattern 1 → index 1
    sv1 = result.sv[1]
    assert np.allclose(sv1, np.diag(sv_raw[:, 1]))


def test_sv_pattern_missing_isv_raises() -> None:
    """Missing Isv with multiple obs patterns should raise a ValueError."""
    shapes = InitShapes(
        n_features=4,
        n_samples=5,
        n_components=2,
        n_obs_patterns=2,
    )

    init_dict: Mapping[str, object] = {"Sv": np.ones((2, 5))}

    with pytest.raises(ValueError, match=re.escape(ERR_SV_PATTERN_INDEX)):
        init_params(init_dict, shapes, score_pattern_index=None, rng=_rng())


def test_sv_invalid_array_shape_raises() -> None:
    """Unsupported Sv array shape should raise ERR_SV_SHAPE."""
    shapes = InitShapes(
        n_features=3,
        n_samples=4,
        n_components=2,
        n_obs_patterns=4,
    )

    # Shape does not match any supported Sv layout
    bad_sv = np.ones((5, 5))
    init_dict: Mapping[str, object] = {"Sv": bad_sv}

    with pytest.raises(ValueError, match=re.escape(ERR_SV_SHAPE)):
        init_params(init_dict, shapes, score_pattern_index=None, rng=_rng())


# ------------------------------------------------------------------------------
# Logging tests (these do not require matplotlib)
# ------------------------------------------------------------------------------


def test_log_first_step(caplog: pytest.LogCaptureFixture) -> None:
    """Test log_first_step output."""
    caplog.set_level("INFO")
    log_first_step(1, rms=0.5, prms=0.25)
    text = caplog.text
    assert "Step 0" in text
    assert "0.500000" in text
    assert "0.250000" in text


def test_log_step(caplog: pytest.LogCaptureFixture) -> None:
    """Test log_step output."""
    caplog.set_level("INFO")
    lc: Mapping[str, Sequence[float]] = {"cost": [1.23], "rms": [0.8], "prms": [0.7]}
    log_step(1, lc, angle_a=0.33)
    text = caplog.text
    assert "Step 0" in text
    assert "cost = 1.230000" in text
    assert "rms = 0.800000" in text
    assert "prms = 0.700000" in text
    assert "angle(A) = 0.3300" in text


def test_log_progress(caplog: pytest.LogCaptureFixture) -> None:
    """Test log_progress output."""
    caplog.set_level("INFO")
    log_progress(2, current=3, total=10, phase="E-step")
    assert "E-step 3/10" in caplog.text


# ------------------------------------------------------------------------------
# Display helpers (matplotlib optional)
# ------------------------------------------------------------------------------


def test_display_init_no_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """display_init should degrade gracefully when matplotlib is unavailable."""
    monkeypatch.setattr(mon, "plt", None, raising=False)

    dsph = display_init(1, {"rms": [1, 2], "prms": [3, 4]})
    assert dsph == {"display": True}  # only display flag is returned


def test_display_init_with_matplotlib() -> None:
    """display_init returns figure/lines when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401, PLC0415
    except ImportError:
        pytest.skip("matplotlib not installed")

    lc: Mapping[str, Sequence[float]] = {"rms": [1, 2, 3], "prms": [2, 4, 6]}
    dsph = display_init(1, lc)

    assert dsph["display"] is True
    assert "fig" in dsph
    assert "rms" in dsph
    assert "prms" in dsph


def test_display_progress_smoke() -> None:
    """display_progress should run without errors when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401, PLC0415
    except ImportError:
        pytest.skip("matplotlib not installed")

    lc: Mapping[str, Sequence[float]] = {"rms": [1, 2, 3], "prms": [2, 4, 6]}
    dsph = display_init(1, lc)
    display_progress(dsph, lc)


# ------------------------------------------------------------------------------
# Initial monitoring helper
# ------------------------------------------------------------------------------


def test_initial_monitoring_with_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_initial_monitoring computes RMS for data and probe and logs correctly."""
    x_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    x_probe = np.array([[5.0, 6.0], [7.0, 8.0]])
    mask = np.ones_like(x_data)

    calls: dict[str, list[object]] = {
        "compute": [],
        "display": [],
        "log": [],
    }

    def fake_compute_rms(
        x: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        mask_in: np.ndarray,
        config: RmsConfig,
    ) -> tuple[float, str]:
        calls["compute"].append((x, a, s, mask_in, config))
        if x is x_data:
            return 1.0, "err_data"
        if x is x_probe:
            return 2.0, "err_probe"
        pytest.fail("Unexpected matrix passed to compute_rms")

    def fake_display_init(
        *,
        display: int,
        lc: Mapping[str, Sequence[float]],
    ) -> dict:
        calls["display"].append((display, lc))
        # Minimal dsph dict
        return {"display": True, "marker": "ok"}

    def fake_log_first_step(
        *,
        verbose: int,
        rms: float,
        prms: float,
    ) -> None:
        calls["log"].append((verbose, rms, prms))

    monkeypatch.setattr(mon, "compute_rms", fake_compute_rms)
    monkeypatch.setattr(mon, "display_init", fake_display_init)
    monkeypatch.setattr(mon, "log_first_step", fake_log_first_step)

    inputs = InitialMonitoringInputs(
        x_data=x_data,
        x_probe=x_probe,
        mask=mask,
        n_data=float(x_data.size),
        n_probe=int(x_probe.size),
        a=np.eye(2),
        s=np.zeros((2, 2)),
        opts={"display": 1, "verbose": 2, "num_cpu": 3},
    )

    rms, err_matrix, prms, lc, dsph = mon._initial_monitoring(inputs)  # noqa: SLF001

    # Data path results
    assert rms == pytest.approx(1.0)
    assert err_matrix == "err_data"
    # Probe path results
    assert prms == pytest.approx(2.0)

    # Learning curves
    assert lc["rms"] == [pytest.approx(1.0)]
    assert lc["prms"] == [pytest.approx(2.0)]
    assert lc["time"] == [0.0]
    assert len(lc["cost"]) == 1
    assert np.isnan(lc["cost"][0])

    # display_init called with display=1 and same lc
    assert len(calls["display"]) == 1
    disp_display, disp_lc = calls["display"][0]
    assert disp_display == 1
    assert disp_lc is lc  # same object passed through
    assert dsph["display"] is True
    assert dsph["marker"] == "ok"

    # log_first_step called with verbose=2 and matching rms/prms
    assert len(calls["log"]) == 1
    verbose_val, rms_val, prms_val = calls["log"][0]
    assert verbose_val == 2
    assert rms_val == pytest.approx(1.0)
    assert prms_val == pytest.approx(2.0)

    # compute_rms called once for data and once for probe
    assert len(calls["compute"]) == 2

    # Check that num_cpu made it through into config
    _x0, _a0, _s0, _m0, cfg0 = calls["compute"][0]
    _x1, _a1, _s1, _m1, cfg1 = calls["compute"][1]
    assert isinstance(cfg0, RmsConfig)
    assert isinstance(cfg1, RmsConfig)
    assert cfg0.num_cpu == 3
    assert cfg1.num_cpu == 3


def test_initial_monitoring_without_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_initial_monitoring should skip probe RMS when n_probe == 0 or probe is None."""
    x_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.ones_like(x_data)

    calls: dict[str, list[object]] = {
        "compute": [],
        "display": [],
        "log": [],
    }

    def fake_compute_rms(
        x: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        mask_in: np.ndarray,
        config: RmsConfig,
    ) -> tuple[float, str]:
        calls["compute"].append((x, a, s, mask_in, config))
        # Only data matrix is expected
        assert x is x_data
        assert isinstance(config, RmsConfig)
        return 1.5, "err_data_only"

    def fake_display_init(
        *,
        display: int,
        lc: Mapping[str, Sequence[float]],
    ) -> dict:
        calls["display"].append((display, lc))
        return {"display": False}

    def fake_log_first_step(
        *,
        verbose: int,
        rms: float,
        prms: float,
    ) -> None:
        calls["log"].append((verbose, rms, prms))
        # prms should be NaN when probe is not used
        assert np.isnan(prms)

    monkeypatch.setattr(mon, "compute_rms", fake_compute_rms)
    monkeypatch.setattr(mon, "display_init", fake_display_init)
    monkeypatch.setattr(mon, "log_first_step", fake_log_first_step)

    inputs = InitialMonitoringInputs(
        x_data=x_data,
        x_probe=None,
        mask=mask,
        n_data=float(x_data.size),
        n_probe=0,
        a=np.eye(2),
        s=np.zeros((2, 2)),
        opts={"display": 1, "verbose": 1},
    )

    rms, err_matrix, prms, lc, dsph = mon._initial_monitoring(inputs)  # noqa: SLF001

    assert rms == pytest.approx(1.5)
    assert err_matrix == "err_data_only"
    assert np.isnan(prms)

    assert lc["rms"] == [pytest.approx(1.5)]
    assert len(lc["prms"]) == 1
    assert np.isnan(lc["prms"][0])
    assert lc["time"] == [0.0]
    assert len(lc["cost"]) == 1
    assert np.isnan(lc["cost"][0])

    # display_init still called once
    assert len(calls["display"]) == 1
    assert dsph == {"display": False}

    # log_first_step called once
    assert len(calls["log"]) == 1
    verbose_val, rms_val, prms_val = calls["log"][0]
    assert verbose_val == 1
    assert rms_val == pytest.approx(1.5)
    assert np.isnan(prms_val)

    # compute_rms called only once (for data)
    assert len(calls["compute"]) == 1
