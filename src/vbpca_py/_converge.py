from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

import time

import numpy as np
from scipy.linalg import subspace_angles
from scipy.sparse import issparse, spmatrix

from ._cost import CostParams, compute_full_cost
from ._monitoring import display_progress, log_step

Array = np.ndarray
Sparse = spmatrix
Matrix = Array | Sparse


@dataclass
class ConvergenceState:
    """State used when logging and checking convergence criteria.

    Naming is aligned with the helper modules (_full_update, etc.):

    - loadings: A
    - scores: S
    - mu: Mu
    - noise_var: V
    - loading_covariances: Av (optional)
    - score_covariances: Sv
    - mu_variances: Muv (optional)
    """

    opts: MutableMapping[str, object]
    x_data: Matrix
    loadings: np.ndarray
    scores: np.ndarray
    mu: np.ndarray
    noise_var: float
    va: np.ndarray
    loading_covariances: list[np.ndarray]
    vmu: float
    mu_variances: np.ndarray
    score_covariances: list[np.ndarray]
    pattern_index: np.ndarray | None
    mask: Matrix
    s_xv: float
    n_data: float
    time_start: float
    lc: dict[str, list[float]]
    loadings_old: np.ndarray
    dsph: dict[str, object]


# ---------------------------------------------------------------------------
# Internal helpers for convergence criteria
# ---------------------------------------------------------------------------


def _angle_stop_message(opts: Mapping[str, Any], angle_a: float) -> str | None:
    """Return an angle-based convergence message, or None if not triggered."""
    minangle = float(opts.get("minangle", np.inf))
    if np.isfinite(minangle) and angle_a < minangle:
        return (
            f"Convergence achieved: subspace angle {angle_a:.2e} "
            f"is below minangle = {minangle:.2e}."
        )
    return None


def _early_stop_message(
    opts: Mapping[str, Any],
    prms: np.ndarray,
) -> str | None:
    """Return an early-stopping message based on probe RMS, or None."""
    if not opts.get("earlystop"):
        return None
    if prms.size < 2:
        return None

    last = prms[-1]
    prev = prms[-2]

    if np.isfinite(last) and np.isfinite(prev) and last > prev:
        return "Early stopping: probe RMS increased."
    return None


def _plateau_stop(
    series: np.ndarray,
    stop_cfg: Sequence[float] | np.ndarray | None,
    label: str,
) -> str | None:
    """Generic plateau stop for RMS / cost series.

    Returns a message if the series is flat over the configured window,
    otherwise None.

    Returns:
        Either a human-readable stop message or ``None`` when no plateau
        condition is met.
    """
    if stop_cfg is None or series.size == 0:
        return None

    cfg = np.asarray(stop_cfg, dtype=float).ravel()
    if cfg.size < 2:
        return None

    window = int(cfg[0])
    abs_tol = float(cfg[1])
    rel_tol = float(cfg[2]) if cfg.size > 2 else np.nan

    # Need at least window+1 points so we can compare current vs. older
    if window <= 0 or series.size <= window:
        return None

    older = series[-window - 1]
    newer = series[-1]

    if not (np.isfinite(older) and np.isfinite(newer)):
        return None

    delta = abs(older - newer)
    rel = delta / (abs(newer) + np.finfo(float).eps)

    if delta < abs_tol or (np.isfinite(rel_tol) and rel < rel_tol):
        return (
            f"Stop: {label} changed by "
            f"{delta:.3e} (rel {rel:.3e}) over {window} iterations."
        )

    return None


def _slowing_down_message(sd_iter: int | None) -> str | None:
    """Return a slowing-down message if sd_iter hits the threshold."""
    if sd_iter is not None and sd_iter == 40:
        return (
            "Slowing-down stop: step size repeatedly reduced. "
            "Consider changing the gradient type or learning rates."
        )
    return None


# ---------------------------------------------------------------------------
# Public convergence check
# ---------------------------------------------------------------------------


def convergence_check(
    opts: Mapping[str, Any],
    lc: Mapping[str, Sequence[float]],
    angle_a: float,
    sd_iter: int | None = None,
) -> str:
    """Check convergence criteria and return a human-readable message.

    The following criteria are evaluated **in order**; the first one that
    triggers returns a non-empty message, otherwise an empty string:

    1. Subspace-angle stop (``minangle``).
    2. Early stopping based on probe RMS (``earlystop``).
    3. RMS plateau stop (``rmsstop = [window, abs_tol, rel_tol]``).
    4. Cost plateau stop (``cfstop = [window, abs_tol, rel_tol]``).
    5. “Slowing-down'' stop based on ``sd_iter`` (gradient backtracking).

    Returns:
        A non-empty convergence message when a criterion triggers,
        otherwise an empty string.
    """
    # 1. Angle-based stop
    angle_msg = _angle_stop_message(opts, angle_a)
    if angle_msg:
        return angle_msg

    rms = np.asarray(lc.get("rms", []), dtype=float)
    prms = np.asarray(lc.get("prms", []), dtype=float)
    cost = np.asarray(lc.get("cost", []), dtype=float)

    # 2. Early stopping on probe RMS
    early_msg = _early_stop_message(opts, prms)
    if early_msg:
        return early_msg

    # 3. RMS plateau
    rmsstop = opts.get("rmsstop")
    if rms.size >= 2 and rmsstop is not None:
        plateau_msg = _plateau_stop(rms, rmsstop, "RMS")
        if plateau_msg:
            return plateau_msg

    # 4. Cost plateau
    cfstop = opts.get("cfstop")
    if cost.size >= 2 and cfstop is not None:
        plateau_msg = _plateau_stop(cost, cfstop, "cost")
        if plateau_msg:
            return plateau_msg

    # 5. Slowing-down criterion
    slow_msg = _slowing_down_message(sd_iter)
    if slow_msg:
        return slow_msg

    return ""


# ---------------------------------------------------------------------------
# Logging + cost computation helper
# ---------------------------------------------------------------------------


def _log_and_check_convergence(
    state: ConvergenceState,
    rms: float,
    prms: float,
) -> tuple[dict[str, list[float]], float, str, np.ndarray]:
    """Update learning curves, compute cost (if requested), and check convergence.

    Returns:
        Updated learning-curve dict, latest subspace angle, convergence
        message, and a copy of ``state.loadings`` for the next iteration.
    """
    verbose = int(state.opts["verbose"])
    elapsed = time.time() - state.time_start

    state.lc["rms"].append(float(rms))
    state.lc["prms"].append(float(prms))
    state.lc["time"].append(float(elapsed))

    if np.size(state.opts.get("cfstop", [])) > 0:
        mask_arr = (
            state.mask.toarray().astype(bool)  # type: ignore[union-attr]
            if issparse(state.mask)
            else np.asarray(state.mask, dtype=bool)
        )

        params = CostParams(
            mu=state.mu.ravel(),
            noise_variance=float(state.noise_var),
            loading_priors=state.va,
            loading_covariances=state.loading_covariances or None,
            mu_prior_variance=float(state.vmu),
            mu_variances=state.mu_variances.ravel()
            if state.mu_variances.size > 0
            else None,
            score_covariances=state.score_covariances,
            score_pattern_index=state.pattern_index,
            mask=mask_arr,
            s_xv=float(state.s_xv),
            n_data=int(state.n_data),
        )
        cost, *_ = compute_full_cost(state.x_data, state.loadings, state.scores, params)
        state.lc["cost"].append(float(cost))
    else:
        state.lc["cost"].append(float("nan"))

    # UI / logging hooks
    display_progress(state.dsph, state.lc)

    angles = subspace_angles(state.loadings, state.loadings_old)
    angle_a = float(np.max(angles))
    log_step(verbose, state.lc, angle_a)

    convmsg = convergence_check(state.opts, state.lc, angle_a)
    loadings_old_new = state.loadings.copy()
    return state.lc, angle_a, convmsg, loadings_old_new
