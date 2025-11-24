# src/vbpca_py/_monitoring.py
"""
Initialization, logging, and (optional) progress display utilities for VBPCA.

This module collects helper functions that were previously scattered and
relied on ad-hoc print statements. The goal is to:

- keep initialization logic for A, S, mu, Av, Sv in one place;
- provide logging-based progress reporting instead of bare prints;
- keep optional plotting / display code isolated and lazily imported.

All functions here are pure helpers and can be used (or ignored) by the
high-level PCA FULL / PCA Diagnostic routines.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.linalg import orth

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]

__all__ = [
    "InitResult",
    "InitShapes",
    "display_init",
    "display_progress",
    "init_params",
    "log_first_step",
    "log_progress",
    "log_step",
]

# ---------------------------------------------------------------------
# Error messages as constants
# ---------------------------------------------------------------------

ERR_INIT_TYPE = "init must be a string, mapping, or None."
ERR_MUV_SHAPE = "Muv must have shape (n_features, 1)."
ERR_AV_SHAPE = "Unsupported Av array shape for initialization."
ERR_SV_PATTERN_INDEX = (
    "score_pattern_index or Isv is required when n_obs_patterns < n_samples."
)


# ---------------------------------------------------------------------
# Shapes and result containers
# ---------------------------------------------------------------------


@dataclass
class InitShapes:
    """Container for core shape parameters used during initialization."""

    n_features: int
    n_samples: int
    n_components: int
    n_obs_patterns: int


@dataclass
class InitResult:
    """Result of parameter initialization for VBPCA."""

    a: np.ndarray
    s: np.ndarray
    mu: np.ndarray
    v: float
    av: list[np.ndarray]
    sv: list[np.ndarray]
    muv: np.ndarray


# ---------------------------------------------------------------------
# Top-level init_params orchestrator
# ---------------------------------------------------------------------


def init_params(
    init: str | Mapping[str, Any] | None,
    shapes: InitShapes,
    score_pattern_index: np.ndarray | Sequence[int] | None,
    rng: np.random.Generator | None = None,
) -> InitResult:
    """Initialize A, S, mu, V, Av, Sv, and Muv.

    This is a modernized translation of the original MATLAB init_parms,
    refactored into smaller helpers for readability and testability.

    Parameters
    ----------
    init
        Either:
        - "random" (case-insensitive) to request random initialization,
        - a path to a .mat file containing a struct-like "init",
        - a Python mapping with keys such as "A", "Av", "Mu", "Muv",
          "V", "S", "Sv", "Isv".
    shapes
        InitShapes holding (n_features, n_samples, n_components, n_obs_patterns).
    score_pattern_index
        Pattern index per sample (column), equivalent to Isv in the
        original code, or None if there is no pattern sharing.
    rng
        Random number generator instance. If None, uses default_rng().

    Returns:
    -------
    InitResult
        Dataclass containing initialized A, S, mu, V, Av, Sv, Muv.
    """
    if rng is None:
        rng = np.random.default_rng()

    init_dict = _normalize_init(init)

    a, av = _init_a_av(init_dict, shapes, rng)
    mu, muv, v = _init_mu_muv_v(init_dict, shapes.n_features)
    s, sv = _init_s_sv(
        init_dict,
        shapes,
        score_pattern_index=score_pattern_index,
        rng=rng,
    )

    return InitResult(a=a, s=s, mu=mu, v=v, av=av, sv=sv, muv=muv)


# ---------------------------------------------------------------------
# Helpers: normalize init
# ---------------------------------------------------------------------


def _normalize_init(init: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize init into a plain dictionary."""
    if isinstance(init, str):
        if init.lower() == "random":
            return {}

        mat_data = loadmat(init)
        # If MATLAB stored a struct named "init", prefer that; otherwise use the whole dict.
        if "init" in mat_data:
            raw = mat_data["init"]
            # MATLAB structs come in with dtype.names
            if getattr(raw, "dtype", None) is not None and raw.dtype.names:
                # Convert MATLAB struct to dict[name -> value]
                out: dict[str, Any] = {}
                for name in raw.dtype.names:
                    out[name] = raw[name].squeeze()
                return out
            return dict(raw)

        return dict(mat_data)

    if init is None:
        return {}

    if isinstance(init, Mapping):
        return dict(init)

    raise ValueError(ERR_INIT_TYPE)


# ---------------------------------------------------------------------
# Helpers: A and Av
# ---------------------------------------------------------------------


def _init_a_av(
    init_dict: Mapping[str, Any],
    shapes: InitShapes,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Initialize loading matrix A and its covariances Av."""
    n_features = shapes.n_features
    n_components = shapes.n_components

    if "A" in init_dict:
        a = np.asarray(init_dict["A"], dtype=float)
    else:
        # Orthogonal random A, as in original MATLAB code.
        a = orth(rng.standard_normal((n_features, n_components)))

    av_raw = init_dict.get("Av")
    if av_raw is None or np.size(av_raw) == 0:
        av = [np.eye(n_components, dtype=float) for _ in range(n_features)]
        return a, av

    if isinstance(av_raw, list):
        av = [np.asarray(m, dtype=float) for m in av_raw]
        return a, av

    av_arr = np.asarray(av_raw, dtype=float)
    av = _expand_av_array(av_arr, n_features, n_components)
    return a, av


def _expand_av_array(
    av_arr: np.ndarray,
    n_features: int,
    n_components: int,
) -> list[np.ndarray]:
    """Convert an Av array into a list of covariance matrices."""
    if av_arr.ndim == 2 and av_arr.shape == (n_features, n_components):
        # Diagonal variances for each row of A.
        return [np.diag(av_arr[i, :]) for i in range(n_features)]
    if av_arr.ndim == 3 and av_arr.shape[0] == n_features:
        return [av_arr[i] for i in range(n_features)]
    raise ValueError(ERR_AV_SHAPE)


# ---------------------------------------------------------------------
# Helpers: Mu, Muv, V
# ---------------------------------------------------------------------


def _init_mu_muv_v(
    init_dict: Mapping[str, Any],
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Initialize mean vector mu, its variance muv, and noise variance v."""
    mu = np.asarray(
        init_dict.get("Mu", np.zeros(n_features)),
        dtype=float,
    ).reshape(n_features)

    muv = np.asarray(
        init_dict.get("Muv", np.ones((n_features, 1))),
        dtype=float,
    )
    if muv.shape != (n_features, 1):
        raise ValueError(ERR_MUV_SHAPE)

    v_raw = init_dict.get("V", 1.0)
    v = float(v_raw)
    return mu, muv, v


# ---------------------------------------------------------------------
# Helpers: S and Sv
# ---------------------------------------------------------------------


def _init_s_sv(
    init_dict: Mapping[str, Any],
    shapes: InitShapes,
    score_pattern_index: np.ndarray | Sequence[int] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Initialize score matrix S and its covariances Sv."""
    n_samples = shapes.n_samples
    n_components = shapes.n_components
    n_obs_patterns = shapes.n_obs_patterns

    if "S" in init_dict:
        s = np.asarray(init_dict["S"], dtype=float)
    else:
        s = rng.standard_normal((n_components, n_samples))

    sv_raw = init_dict.get("Sv")
    isv_init = init_dict.get("Isv")
    isv = _resolve_score_pattern_index(score_pattern_index, isv_init, n_samples)

    # No Sv provided: default to identity per pattern.
    if sv_raw is None or np.size(sv_raw) == 0:
        sv_default = [np.eye(n_components, dtype=float) for _ in range(n_obs_patterns)]
        return s, sv_default

    # Pattern-sharing mode.
    if n_obs_patterns < n_samples:
        if isv is None:
            raise ValueError(ERR_SV_PATTERN_INDEX)
        sv = _init_sv_pattern_mode(sv_raw, isv, n_obs_patterns, n_components)
        return s, sv

    # No pattern-sharing: one covariance per sample.
    if isinstance(sv_raw, list):
        if isv is not None and len(sv_raw) >= len(isv):
            sv = [np.asarray(sv_raw[idx], dtype=float) for idx in isv]
            return s, sv
        if len(sv_raw) == n_samples:
            sv = [np.asarray(m, dtype=float) for m in sv_raw]
            return s, sv
        return s, []

    sv_arr = np.asarray(sv_raw, dtype=float)
    sv = _expand_sv_array(sv_arr, n_samples, n_components)
    return s, sv


def _resolve_score_pattern_index(
    score_pattern_index: np.ndarray | Sequence[int] | None,
    isv_init: Any,
    n_samples: int,
) -> np.ndarray | None:
    """Resolve the score pattern index from explicit argument or init dict."""
    if score_pattern_index is not None:
        return np.asarray(score_pattern_index, dtype=int)
    if isv_init is not None:
        isv_arr = np.asarray(isv_init, dtype=int)
        if isv_arr.shape[0] == n_samples:
            return isv_arr
    return None


def _init_sv_pattern_mode(
    sv_raw: Any,
    isv: np.ndarray,
    n_obs_patterns: int,
    n_components: int,
) -> list[np.ndarray]:
    """Initialize Sv when multiple columns share covariance patterns."""
    _, first_idx = np.unique(isv, return_index=True)

    if not isinstance(sv_raw, list):
        sv_arr = np.asarray(sv_raw, dtype=float)
        # Expect shape (n_components, n_samples).
        sv_list: list[np.ndarray] = []
        for j in range(n_obs_patterns):
            col = isv[first_idx[j]]
            sv_list.append(np.diag(sv_arr[:, col]))
        return sv_list

    sv_list = []
    for j in range(n_obs_patterns):
        col = isv[first_idx[j]]
        sv_list.append(np.asarray(sv_raw[col], dtype=float))
    return sv_list


def _expand_sv_array(
    sv_arr: np.ndarray,
    n_samples: int,
    n_components: int,
) -> list[np.ndarray]:
    """Convert an Sv array into a list of covariance matrices."""
    if sv_arr.ndim == 2 and sv_arr.shape == (n_components, n_samples):
        return [np.diag(sv_arr[:, j]) for j in range(n_samples)]
    if sv_arr.ndim == 3 and sv_arr.shape[0] == n_samples:
        return [sv_arr[j] for j in range(n_samples)]
    # Fallback if shape is unexpected; caller can decide how to handle.
    return []


# ---------------------------------------------------------------------
# Logging helpers (replacing print_* functions)
# ---------------------------------------------------------------------


def log_first_step(
    verbose: int,
    rms: float,
    prms: float | np.ndarray,
) -> None:
    """Log the initial RMS and optional probe RMS."""
    if not verbose:
        return

    rms_val = float(rms)
    msg = f"Step 0: rms = {rms_val:.6f}"

    prms_arr = np.asarray(prms)
    if prms_arr.size > 0:
        prms_val = float(prms_arr.ravel()[0])
        if not np.isnan(prms_val):
            msg += f" (prms = {prms_val:.6f})"

    logger.info(msg)


def log_step(
    verbose: int,
    lc: Mapping[str, Sequence[float]],
    angle_a: float | None,
) -> None:
    """Log metrics for the current iteration."""
    if not verbose:
        return

    step_index = len(lc["rms"]) - 1
    if step_index < 0:
        return

    cost_val = float(np.asarray(lc["cost"][-1]).ravel()[0])
    rms_val = float(np.asarray(lc["rms"][-1]).ravel()[0])

    msg = f"Step {step_index}: cost = {cost_val:.6f}, rms = {rms_val:.6f}"

    if "prms" in lc and len(lc["prms"]) > 0:
        prms_val = float(np.asarray(lc["prms"][-1]).ravel()[0])
        if not np.isnan(prms_val):
            msg += f", prms = {prms_val:.6f}"

    if angle_a is not None:
        msg += f", angle(A) = {float(angle_a):.4f} rad"

    logger.info(msg)


def log_progress(
    verbose: int,
    *,
    current: int,
    total: int,
    phase: str,
) -> None:
    """Log coarse progress (e.g. 'E-step i/N') when verbose == 2."""
    if verbose == 2:
        logger.info("%s %d/%d", phase, current, total)


# ---------------------------------------------------------------------
# Optional plotting helpers
# ---------------------------------------------------------------------


def display_init(display: int, lc: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Initialize matplotlib plots for RMS training/test errors.

    Parameters
    ----------
    display
        If 0, returns a dict with {'display': False}.
        If non-zero and matplotlib is available, creates a figure with two
        subplots and returns handles.
    lc
        Log container with 'rms' and 'prms' sequences.

    Returns:
    -------
    dsph : dict
        A dictionary with:
        - 'display': bool
        - 'fig': matplotlib Figure (if display is non-zero and plt is available)
        - 'rms': Line2D handle for training RMS
        - 'prms': Line2D handle for test RMS
    """
    dsph: dict[str, Any] = {"display": bool(display)}

    if not dsph["display"] or plt is None:
        return dsph

    rms_values = np.asarray(lc.get("rms", []), dtype=float)
    prms_values = np.asarray(lc.get("prms", []), dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Training RMS
    ax1 = axes[0]
    steps_rms = np.arange(rms_values.size)
    (line_rms,) = ax1.plot(steps_rms, rms_values, label="Training RMS")
    ax1.set_ylabel("RMS")
    ax1.legend()
    ax1.grid(True)

    # Probe / test RMS
    ax2 = axes[1]
    steps_prms = np.arange(prms_values.size)
    (line_prms,) = ax2.plot(steps_prms, prms_values, label="Probe RMS")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("RMS")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.draw()

    dsph["fig"] = fig
    dsph["rms"] = line_rms
    dsph["prms"] = line_prms

    return dsph


def display_progress(
    dsph: Mapping[str, Any], lc: Mapping[str, Sequence[float]]
) -> None:
    """Update RMS training/test plots if display is enabled."""
    if not dsph.get("display", False) or plt is None:
        return

    line_rms = dsph["rms"]
    line_prms = dsph["prms"]

    rms_values = np.asarray(lc.get("rms", []), dtype=float)
    prms_values = np.asarray(lc.get("prms", []), dtype=float)

    steps_rms = np.arange(rms_values.size)
    line_rms.set_xdata(steps_rms)
    line_rms.set_ydata(rms_values)

    steps_prms = np.arange(prms_values.size)
    line_prms.set_xdata(steps_prms)
    line_prms.set_ydata(prms_values)

    ax1 = line_rms.axes
    ax2 = line_prms.axes
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()

    plt.draw()
