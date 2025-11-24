from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

import numpy as np


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

    Parameters
    ----------
    opts
        Options dict. Relevant keys:

        - ``"minangle"`` : float, subspace angle threshold.
        - ``"earlystop"`` : bool/int, whether to use probe-based early stop.
        - ``"rmsstop"`` : array-like of length >= 2:
              [window_iters, abs_tol, (optional) rel_tol]
        - ``"cfstop"`` : array-like of length >= 2, same format as ``rmsstop``.

    lc
        Learning curves dict with keys:
        - ``"rms"`` : sequence of training RMS values.
        - ``"prms"`` : sequence of probe RMS values (may be empty).
        - ``"cost"`` : sequence of cost-function values (may be empty).

    angle_a
        Current subspace angle between successive A estimates.

    sd_iter
        Optional “slowing-down” iteration counter (for line-search style
        gradient methods). If equal to 40, triggers a special message.

    Returns:
    -------
    convmsg
        Empty string if no criterion is met; otherwise a diagnostic message.
    """
    # --- 1. Angle stop ----------------------------------------------------
    minangle = float(opts.get("minangle", np.inf))
    if np.isfinite(minangle) and angle_a < minangle:
        return (
            f"Convergence achieved: subspace angle {angle_a:.2e} "
            f"is below minangle = {minangle:.2e}."
        )

    rms = np.asarray(lc.get("rms", []), dtype=float)
    prms = np.asarray(lc.get("prms", []), dtype=float)
    cost = np.asarray(lc.get("cost", []), dtype=float)

    # Need at least two points for most criteria
    have_two_rms = rms.size >= 2
    have_two_prms = prms.size >= 2
    have_two_cost = cost.size >= 2

    # --- 2. Early stopping on probe RMS ----------------------------------
    if opts.get("earlystop") and have_two_prms:
        last, prev = prms[-1], prms[-2]
        if np.isfinite(last) and np.isfinite(prev) and last > prev:
            return "Early stopping: probe RMS increased."

    # Helper for plateau criteria
    def _plateau_stop(
        series: np.ndarray,
        stop_cfg: Sequence[float] | np.ndarray,
        label: str,
    ) -> str | None:
        if series.size == 0 or stop_cfg is None:
            return None

        cfg = np.asarray(stop_cfg, dtype=float).ravel()
        if cfg.size < 2:
            return None

        window = int(cfg[0])
        abs_tol = float(cfg[1])
        rel_tol = float(cfg[2]) if cfg.size > 2 else np.nan

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

    # --- 3. RMS plateau ---------------------------------------------------
    rmsstop = opts.get("rmsstop")
    if have_two_rms and rmsstop is not None:
        msg = _plateau_stop(rms, rmsstop, "RMS")
        if msg:
            return msg

    # --- 4. Cost plateau --------------------------------------------------
    cfstop = opts.get("cfstop")
    if have_two_cost and cfstop is not None:
        msg = _plateau_stop(cost, cfstop, "cost")
        if msg:
            return msg

    # --- 5. Slowing-down stop --------------------------------------------
    if sd_iter is not None and sd_iter == 40:
        return (
            "Slowing-down stop: step size repeatedly reduced. "
            "Consider changing the gradient type or learning rates."
        )

    return ""
