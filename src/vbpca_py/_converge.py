from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, SupportsIndex, SupportsInt, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import subspace_angles
from scipy.sparse import spmatrix

from ._cost import CostParams, compute_full_cost
from ._monitoring import display_progress, log_step
from ._sparsity import validate_mask_compatibility

Array = np.ndarray
Sparse = spmatrix
Matrix = Array | Sparse

logger = logging.getLogger(__name__)


def _coerce_int(
    val: SupportsInt | SupportsIndex | str | bytes | bytearray | None,
    default: int = 0,
) -> int:
    """Safely coerce common int-like inputs with a fallback.

    Returns:
        Integer conversion of ``val`` when possible; ``default`` otherwise.
    """
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


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
    minangle_val = opts.get("minangle", np.inf)
    minangle = (
        float(minangle_val)
        if isinstance(minangle_val, (int, float, np.floating, str, bytes, bytearray))
        else float(np.inf)
    )
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


def _relative_elbo_stop(
    cost: np.ndarray,
    threshold: float | None,
) -> str | None:
    """Return a message if relative ELBO decrease is below *threshold*.

    Checks ``|ELBO[t] - ELBO[t-1]| / |ELBO[t]| < threshold``.  This is
    scale-invariant and more robust than an absolute plateau check.

    Returns:
        A human-readable stop message, or ``None``.
    """
    if threshold is None or threshold <= 0 or cost.size < 2:
        return None

    curr, prev = cost[-1], cost[-2]
    if not (np.isfinite(curr) and np.isfinite(prev)):
        return None

    rel_change = abs(curr - prev) / (abs(curr) + np.finfo(float).eps)
    if rel_change < threshold:
        return (
            f"Stop: relative ELBO change {rel_change:.3e} "
            f"is below cfstop_rel = {threshold:.3e}."
        )
    return None


def _elbo_curvature_stop(
    cost: np.ndarray,
    threshold: float | None,
) -> str | None:
    """Return a message if ELBO curvature (2nd difference) is below *threshold*.

    Checks ``|ΔELBO[t] - ΔELBO[t-1]| < threshold``, i.e. whether the
    *rate of improvement* has itself stabilised.

    Returns:
        A human-readable stop message, or ``None``.
    """
    if threshold is None or threshold <= 0 or cost.size < 3:
        return None

    d1 = cost[-1] - cost[-2]
    d0 = cost[-2] - cost[-3]

    if not (np.isfinite(d1) and np.isfinite(d0)):
        return None

    curvature = abs(d1 - d0)
    if curvature < threshold:
        return (
            f"Stop: ELBO curvature {curvature:.3e} "
            f"is below cfstop_curv = {threshold:.3e}."
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


def _cost_criteria(
    opts: Mapping[str, Any],
    cost: np.ndarray,
) -> str | None:
    """Evaluate all cost/ELBO-based stopping criteria in priority order.

    Returns:
        The first triggered message, or ``None``.
    """
    # Cost plateau
    cfstop = opts.get("cfstop")
    if cost.size >= 2 and cfstop is not None:
        plateau_msg = _plateau_stop(cost, cfstop, "cost")
        if plateau_msg:
            return plateau_msg

    # Relative ELBO decrease
    cfstop_rel = opts.get("cfstop_rel")
    if cfstop_rel is not None:
        rel_msg = _relative_elbo_stop(cost, float(cfstop_rel))
        if rel_msg:
            return rel_msg

    # ELBO curvature (2nd difference)
    cfstop_curv = opts.get("cfstop_curv")
    if cfstop_curv is not None:
        curv_msg = _elbo_curvature_stop(cost, float(cfstop_curv))
        if curv_msg:
            return curv_msg

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
    4. Cost / ELBO criteria (``cfstop``, ``cfstop_rel``, ``cfstop_curv``).
    5. "Slowing-down'' stop based on ``sd_iter`` (gradient backtracking).

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

    # 4. Cost / ELBO criteria
    cost_msg = _cost_criteria(opts, cost)
    if cost_msg:
        return cost_msg

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
    _runtime_state_guard(state)

    verbose_val = state.opts.get("verbose", 0)
    verbose = _coerce_int(
        verbose_val
        if isinstance(verbose_val, (int, str, bytes, bytearray, np.integer))
        else 0
    )
    elapsed = time.time() - state.time_start

    state.lc["rms"].append(float(rms))
    state.lc["prms"].append(float(prms))
    state.lc["time"].append(float(elapsed))

    _append_cost_value(state)

    # UI / logging hooks
    display_progress(state.dsph, state.lc)

    angle_a, angle_for_log = _angle_for_iteration(state)

    log_step(verbose, state.lc, angle_for_log)

    convmsg = convergence_check(state.opts, state.lc, angle_a)
    loadings_old_new = state.loadings.copy()
    return state.lc, angle_a, convmsg, loadings_old_new


def _runtime_state_guard(state: ConvergenceState) -> None:
    """Fail fast when runtime state violates dense/sparse contract.

    Raises:
        ValueError: When factors are sparse, mask/data shapes differ, or
            mask compatibility fails the legacy dense/sparse rules.
    """
    preflight_obj = state.opts.setdefault("_runtime_preflight", [])
    preflight = cast("list[dict[str, object]]", preflight_obj)

    if sp.issparse(state.loadings) or sp.issparse(state.scores):
        msg = "loadings and scores must be dense; sparse factors are unsupported"
        raise ValueError(msg)

    if state.x_data.shape != state.mask.shape:
        msg = "mask shape must match data shape in convergence state"
        raise ValueError(msg)

    validate_mask_compatibility(
        state.x_data,
        state.mask,
        allow_sparse_mask_for_dense=False,
        allow_dense_mask_for_sparse=False,
        context="convergence_check",
        preflight=preflight,
    )


def _append_cost_value(state: ConvergenceState) -> None:
    cfstop_opt = state.opts.get("cfstop", [])
    if np.size(np.asarray(cfstop_opt)) <= 0:
        state.lc["cost"].append(float("nan"))
        return

    preflight_obj = state.opts.setdefault("_runtime_preflight", [])
    preflight = cast("list[dict[str, object]]", preflight_obj)

    validate_mask_compatibility(
        state.x_data,
        state.mask,
        allow_sparse_mask_for_dense=False,
        allow_dense_mask_for_sparse=False,
        context="compute_full_cost",
        preflight=preflight,
    )

    if sp.issparse(state.mask):
        mask_csr = sp.csr_matrix(cast("Any", state.mask))
        mask_csr.data = np.asarray(mask_csr.data, dtype=bool)
        mask_arg: Matrix = mask_csr
    else:
        mask_arg = np.asarray(state.mask, dtype=bool)

    params = CostParams(
        mu=state.mu.ravel(),
        noise_variance=float(state.noise_var),
        loading_priors=state.va,
        loading_covariances=state.loading_covariances or None,
        mu_prior_variance=float(state.vmu),
        mu_variances=(
            state.mu_variances.ravel() if state.mu_variances.size > 0 else None
        ),
        score_covariances=state.score_covariances,
        score_pattern_index=state.pattern_index,
        mask=mask_arg,
        s_xv=float(state.s_xv),
        n_data=_coerce_int(state.n_data),
    )
    cost, *_ = compute_full_cost(state.x_data, state.loadings, state.scores, params)
    state.lc["cost"].append(float(cost))


def _angle_for_iteration(state: ConvergenceState) -> tuple[float, float | None]:
    angle_every = max(1, _coerce_int(cast("Any", state.opts.get("angle_every", 1)), 1))
    iter_index = len(state.lc["rms"])
    if iter_index % angle_every != 0:
        state.lc["angle"].append(float("nan"))
        return float("inf"), None

    angles = subspace_angles(state.loadings, state.loadings_old)
    angle_a = float(np.max(angles))
    state.lc["angle"].append(angle_a)
    return angle_a, angle_a
