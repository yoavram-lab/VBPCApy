from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, SupportsFloat, SupportsIndex, SupportsInt, cast

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from .results import LearningCurve, NumericTraceKey

import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import subspace_angles
from scipy.sparse import spmatrix

from ._cost import CostParams, compute_full_cost
from ._monitoring import display_progress, log_step
from ._sparsity import validate_mask_compatibility
from .options import Criterion

Array = np.ndarray
Sparse = spmatrix
Matrix = Array | Sparse

logger = logging.getLogger(__name__)

#: Default priority order used by :func:`convergence_check`.
DEFAULT_CRITERION_ORDER: list[str] = [criterion.value for criterion in Criterion]

_VALID_CRITERION_NAMES: frozenset[str] = frozenset(DEFAULT_CRITERION_ORDER)

_REASON_TAGS_BY_CRITERION: dict[str, tuple[str, ...]] = {
    "angle": ("angle",),
    "earlystop": ("earlystop",),
    "rms_plateau": ("rms_plateau",),
    "cost": ("cost_plateau", "cfstop_rel", "cfstop_curv"),
    "composite": ("composite",),
    "slowing_down": ("slowing_down",),
}


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
    lc: LearningCurve
    loadings_old: np.ndarray
    dsph: dict[str, object]


# ---------------------------------------------------------------------------
# Internal helpers for convergence criteria
# ---------------------------------------------------------------------------


def _angle_stop_message(opts: Mapping[str, object], angle_a: float) -> str | None:
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
    opts: Mapping[str, object],
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

    # RMS and variational free energy (negative ELBO) are both minimised.
    # A small worsening is not evidence of convergence, even when its absolute
    # magnitude falls within the configured plateau tolerance.
    if not (np.isfinite(older) and np.isfinite(newer)) or newer > older:
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
    """Return a message if relative free-energy improvement is below threshold.

    The recorded cost is variational free energy (negative ELBO), so lower is
    better. A worsening step never satisfies the criterion.

    Returns:
        A human-readable stop message, or ``None``.
    """
    if threshold is None or threshold <= 0 or cost.size < 2:
        return None

    curr, prev = cost[-1], cost[-2]
    if not (np.isfinite(curr) and np.isfinite(prev)):
        return None

    if curr > prev:
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
    """Return a message if free-energy slope and curvature are both small.

    The recorded cost is variational free energy (negative ELBO), so lower is
    better. Curvature alone is insufficient: a constant, steep descent has
    zero curvature but is not converged. This criterion therefore requires a
    non-worsening current step and both ``|Δcost[t]|`` and
    ``|Δcost[t] - Δcost[t-1]|`` below *threshold*.

    Returns:
        A human-readable stop message, or ``None``.
    """
    if threshold is None or threshold <= 0 or cost.size < 3:
        return None

    d1 = cost[-1] - cost[-2]
    d0 = cost[-2] - cost[-3]

    if not (np.isfinite(d1) and np.isfinite(d0)):
        return None

    if d1 > 0 or abs(d1) >= threshold:
        return None

    curvature = abs(d1 - d0)
    if curvature < threshold:
        return (
            f"Stop: free-energy slope {abs(d1):.3e} and curvature "
            f"{curvature:.3e} "
            f"is below cfstop_curv = {threshold:.3e}."
        )
    return None


def _slowing_down_message(sd_iter: int | None) -> str | None:
    """Interpret an explicitly supplied legacy counter; live fitting has none.

    Returns:
        A legacy terminal message at the threshold, otherwise no message.
    """
    if sd_iter is not None and sd_iter == 40:
        return (
            "Slowing-down stop: step size repeatedly reduced. "
            "Consider changing the gradient type or learning rates."
        )
    return None


def _cost_criteria(
    opts: Mapping[str, object],
    cost: np.ndarray,
) -> str | None:
    """Evaluate all cost/ELBO-based stopping criteria in priority order.

    Returns:
        The first triggered message, or ``None``.
    """
    _tag, msg = _cost_criteria_tagged(opts, cost)
    return msg


def _cost_criteria_tagged(
    opts: Mapping[str, object],
    cost: np.ndarray,
) -> tuple[str | None, str | None]:
    """Like :func:`_cost_criteria` but also returns a reason tag.

    Returns:
        ``(reason_tag, message)`` or ``(None, None)`` when nothing fires.
    """
    checks = _cost_subcriteria(opts, cost)
    for tag in ("cost_plateau", "cfstop_rel", "cfstop_curv"):
        msg = checks[tag]
        if msg:
            return tag, msg
    return None, None


def _cost_subcriteria(
    opts: Mapping[str, object],
    cost: np.ndarray,
) -> dict[str, str | None]:
    """Evaluate every cost subcriterion without applying winner priority.

    Returns:
        Mapping from public reason tag to its message or ``None``.
    """
    cfstop = opts.get("cfstop")
    plateau_msg = (
        _plateau_stop(cost, cast("Sequence[float] | np.ndarray", cfstop), "cost")
        if cost.size >= 2 and cfstop is not None
        else None
    )

    cfstop_rel = opts.get("cfstop_rel")
    rel_msg = (
        _relative_elbo_stop(
            cost, float(cast("str | SupportsFloat | SupportsIndex", cfstop_rel))
        )
        if cfstop_rel is not None
        else None
    )

    cfstop_curv = opts.get("cfstop_curv")
    curv_msg = (
        _elbo_curvature_stop(
            cost, float(cast("str | SupportsFloat | SupportsIndex", cfstop_curv))
        )
        if cfstop_curv is not None
        else None
    )
    return {
        "cost_plateau": plateau_msg,
        "cfstop_rel": rel_msg,
        "cfstop_curv": curv_msg,
    }


def _check_sub_criterion(
    key: str,
    threshold: float,
    angle_a: float,
    rms: np.ndarray,
    cost: np.ndarray,
) -> str | None:
    """Evaluate a single composite sub-criterion.

    Returns:
        A short summary string like ``"angle=1.2e-04<1.0e-03"`` when the
        criterion is satisfied, or ``None`` when it is not met.

    Raises:
        ValueError: If *key* is not a recognised sub-criterion name.
    """
    eps = np.finfo(float).eps

    if key == "angle":
        if np.isfinite(angle_a) and angle_a < threshold:
            return f"angle={angle_a:.2e}<{threshold:.2e}"
        return None

    if key == "rms":
        return _rel_change_check("rms_rel", rms, threshold, eps)

    if key == "elbo_rel":
        return _rel_change_check("elbo_rel", cost, threshold, eps)

    msg = f"Unknown composite_stop key: {key!r}"
    raise ValueError(msg)


def _rel_change_check(
    label: str,
    series: np.ndarray,
    threshold: float,
    eps: float,
) -> str | None:
    """Check relative change between last two elements of *series*.

    Returns:
        A summary string when change is below *threshold*, else ``None``.
    """
    if series.size < 2:
        return None
    curr, prev = series[-1], series[-2]
    if not (np.isfinite(curr) and np.isfinite(prev)):
        return None
    if curr > prev:
        return None
    rel = abs(curr - prev) / (abs(curr) + eps)
    if rel >= threshold:
        return None
    return f"{label}={rel:.2e}<{threshold:.2e}"


def _composite_stop(
    composite_cfg: Mapping[str, float],
    angle_a: float,
    rms: np.ndarray,
    cost: np.ndarray,
) -> str | None:
    """Check whether **all** sub-criteria in *composite_cfg* are satisfied.

    Supported keys (all optional, but at least one must be present):

    - ``"angle"``: subspace angle must be below this value.
    - ``"rms"``: relative RMS change over the last two iterations
      must be below this value.
    - ``"elbo_rel"``: relative ELBO change must be below this value.

    Returns:
        A stop message listing which sub-criteria were satisfied, or
        ``None`` if any sub-criterion is **not** met.
    """
    satisfied: list[str] = []
    for key, threshold in composite_cfg.items():
        result = _check_sub_criterion(key, threshold, angle_a, rms, cost)
        if result is None:
            return None
        satisfied.append(result)

    if not satisfied:
        return None

    detail = ", ".join(satisfied)
    return f"Composite stop: all criteria met ({detail})."


def _apply_criterion_patience(
    checks: Mapping[str, tuple[str | None, str | None]],
    order: Sequence[str],
    enabled: Mapping[str, bool],
    lc: LearningCurve,
    patience: int,
) -> tuple[str, str]:
    """Update per-criterion streaks and return the first eligible stop.

    A streak belongs to one criterion. Alternating satisfied criteria cannot
    jointly exhaust patience. Raw satisfaction is also appended to public
    numeric learning-curve traces so callers can replay counterfactual stop
    policies after a long-running fit.

    Returns:
        ``(reason_tag, message)`` for the first ordered criterion whose own
        streak reaches ``patience``, otherwise two empty strings.
    """
    counts = lc.setdefault("_criterion_patience", {})
    ordered = set(order)

    rms_size = len(lc.get("rms", []))
    for name, (tag, msg) in checks.items():
        history_key = cast("NumericTraceKey", f"criterion_satisfied_{name}")
        history = lc.setdefault(history_key, [0.0] * max(0, rms_size - 1))
        history.append(float(bool(msg)))

        eligible = name in ordered and enabled.get(name, True)
        streak_key = tag or name
        for reason_tag in _REASON_TAGS_BY_CRITERION[name]:
            if reason_tag != streak_key:
                counts[reason_tag] = 0.0
        counts[streak_key] = (
            counts.get(streak_key, 0.0) + 1.0 if eligible and msg else 0.0
        )

    required = max(1, patience)
    for name in order:
        if not enabled.get(name, True):
            continue
        tag, msg = checks[name]
        if msg and int(counts.get(tag or name, 0.0)) >= required:
            return tag or "", msg
    return "", ""


def _append_cost_diagnostics(
    lc: LearningCurve,
    checks: Mapping[str, str | None],
) -> None:
    """Append independent raw-satisfaction traces for each cost rule."""
    history_size = max(0, len(lc.get("rms", [])) - 1)
    for tag, message in checks.items():
        history_obj = lc.setdefault(
            cast("NumericTraceKey", f"criterion_satisfied_{tag}"), [0.0] * history_size
        )
        history_obj.append(float(bool(message)))


# ---------------------------------------------------------------------------
# Public convergence check
# ---------------------------------------------------------------------------


def convergence_check(
    opts: Mapping[str, object],
    lc: LearningCurve,
    angle_a: float,
    sd_iter: int | None = None,
) -> str:
    """Check convergence criteria and return a human-readable message.

    Criteria are evaluated in the order given by ``opts["criterion_order"]``
    (defaulting to :data:`DEFAULT_CRITERION_ORDER`).  The first **enabled**
    criterion that triggers returns a non-empty message.

    Individual criteria can be disabled via ``opts["convergence_criteria"]``,
    a ``dict[str, bool]``.  Disabled criteria are still evaluated (their
    results are available for diagnostics) but excluded from the stop
    decision.

    When ``patience`` is set (> 1), the winning criterion must fire for
    that many **consecutive** iterations before the message is returned.

    Returns:
        A non-empty convergence message when a criterion triggers,
        otherwise an empty string.
    """
    rms = np.asarray(lc.get("rms", []), dtype=float)
    prms = np.asarray(lc.get("prms", []), dtype=float)
    cost = np.asarray(lc.get("cost", []), dtype=float)

    rmsstop = opts.get("rmsstop")
    composite_cfg = opts.get("composite_stop")

    cost_subcriteria = _cost_subcriteria(opts, cost)
    cost_check: tuple[str | None, str | None] = (None, None)
    for tag in ("cost_plateau", "cfstop_rel", "cfstop_curv"):
        message = cost_subcriteria[tag]
        if message:
            cost_check = tag, message
            break

    # Build the full check registry — always evaluate all criteria.
    all_checks: dict[str, tuple[str | None, str | None]] = {
        "angle": ("angle", _angle_stop_message(opts, angle_a)),
        "earlystop": ("earlystop", _early_stop_message(opts, prms)),
        "rms_plateau": (
            "rms_plateau",
            _plateau_stop(rms, cast("Sequence[float] | np.ndarray", rmsstop), "RMS")
            if rms.size >= 2 and rmsstop is not None
            else None,
        ),
        "cost": cost_check,
        "composite": (
            "composite",
            _composite_stop(
                cast("Mapping[str, float]", composite_cfg), angle_a, rms, cost
            )
            if composite_cfg
            else None,
        ),
        "slowing_down": ("slowing_down", _slowing_down_message(sd_iter)),
    }

    # Determine ordering and enablement.
    order = cast(
        "Sequence[str]", opts.get("criterion_order") or DEFAULT_CRITERION_ORDER
    )
    enabled = cast("Mapping[str, bool]", opts.get("convergence_criteria") or {})

    # Preserve each cost rule independently in addition to the aggregate
    # ``criterion_satisfied_cost`` trace used by the public criterion order.
    _append_cost_diagnostics(lc, cost_subcriteria)
    _append_slowing_availability(lc, sd_iter)

    patience_val = opts.get("patience")
    patience = (
        int(cast("str | SupportsInt | SupportsIndex", patience_val))
        if patience_val is not None
        else 1
    )
    reason_tag, candidate = _apply_criterion_patience(
        all_checks,
        order,
        enabled,
        lc,
        patience,
    )

    # Store only a candidate tag here. The training loop may suppress a
    # criterion during broad-prior warmup; it promotes the tag only when the
    # stop is actually accepted.
    if candidate and reason_tag:
        lc["_candidate_convergence_reason"] = reason_tag
    else:
        lc.pop("_candidate_convergence_reason", None)

    return candidate


def _append_slowing_availability(lc: LearningCurve, sd_iter: int | None) -> None:
    """Keep a missing live event stream distinct from an unsatisfied criterion."""
    available = lc.setdefault(
        "criterion_available_slowing_down", [0.0] * max(0, len(lc.get("rms", [])) - 1)
    )
    available.append(float(sd_iter is not None))


# ---------------------------------------------------------------------------
# Logging + cost computation helper
# ---------------------------------------------------------------------------


def _log_and_check_convergence(
    state: ConvergenceState,
    rms: float,
    prms: float,
) -> tuple[LearningCurve, float, str, np.ndarray]:
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
    if not _cost_trace_required(state.opts):
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


def _cost_trace_required(opts: Mapping[str, object]) -> bool:
    """Return whether diagnostics or an enabled criterion consume cost.

    ``cfstop_rel`` and ``cfstop_curv`` are independent alternatives to the
    windowed ``cfstop`` criterion.  Likewise, a composite criterion may use
    ``elbo_rel`` without enabling any standalone cost stop.  Cost must be
    populated for all of those configurations, not only when ``cfstop`` is
    non-empty.

    Returns:
        ``True`` when recording was requested or an enabled stop criterion
        needs the cost trace.
    """
    if bool(opts.get("record_cost", False)):
        return True

    enabled_raw = opts.get("convergence_criteria")
    enabled = enabled_raw if isinstance(enabled_raw, Mapping) else {}

    cost_enabled = bool(enabled.get("cost", True))
    if cost_enabled:
        cfstop = opts.get("cfstop")
        if cfstop is not None and np.size(np.asarray(cfstop)) > 0:
            return True
        for name in ("cfstop_rel", "cfstop_curv"):
            value = opts.get(name)
            if (
                value is not None
                and float(cast("str | SupportsFloat | SupportsIndex", value)) > 0
            ):
                return True

    composite = opts.get("composite_stop")
    composite_enabled = bool(enabled.get("composite", True))
    return (
        composite_enabled and isinstance(composite, Mapping) and "elbo_rel" in composite
    )


def _angle_for_iteration(state: ConvergenceState) -> tuple[float, float | None]:
    angle_every = max(
        1,
        _coerce_int(
            cast(
                "str | SupportsInt | SupportsIndex | None",
                state.opts.get("angle_every", 1),
            ),
            1,
        ),
    )
    iter_index = len(state.lc["rms"])
    if iter_index % angle_every != 0:
        state.lc["angle"].append(float("nan"))
        return float("inf"), None

    angle_a = _max_subspace_angle(state.loadings, state.loadings_old)
    state.lc["angle"].append(angle_a)
    return angle_a, angle_a


def _max_subspace_angle(loadings: np.ndarray, loadings_old: np.ndarray) -> float:
    """Return a convergence-safe maximum principal angle.

    SciPy returns no principal angles when either loading matrix has zero
    columns. Two empty subspaces are stable and therefore have angle zero. A
    transition to or from an empty subspace is treated as maximally different,
    preventing angle convergence from firing on the pruning iteration.

    Returns:
        Maximum principal angle in radians.
    """
    angles = subspace_angles(loadings, loadings_old)
    if angles.size:
        return float(np.max(angles))
    if loadings.shape[1] == 0 and loadings_old.shape[1] == 0:
        return 0.0
    return float(np.pi / 2.0)
