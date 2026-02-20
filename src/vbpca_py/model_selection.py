"""Model selection utilities for VBPCA.

Sweeps candidate component counts while reusing the VBPCA estimator and
convergence options. Stores only scalar endpoint metrics per candidate and
optionally retains the best-fit model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, SupportsFloat, SupportsIndex, cast

import numpy as np
import scipy.sparse as sp

from ._memory import exceeds_budget, format_bytes, resolve_max_dense_bytes
from ._pca_full import _explained_variance, _reconstruct_data

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping

    from ._pca_full import Matrix
    from .estimators import VBPCA

__all__ = ["SelectionConfig", "select_n_components"]

logger = logging.getLogger(__name__)

_Metric = Literal["prms", "rms", "cost"]
_AllowedFloat = (
    SupportsFloat
    | SupportsIndex
    | np.floating[Any]
    | np.integer[Any]
    | str
    | bytes
    | bytearray
)


@dataclass
class SelectionConfig:
    """Configuration for component selection."""

    metric: _Metric = "prms"
    stop_on_metric_reversal: bool = False
    patience: int | None = None
    max_trials: int | None = None
    compute_explained_variance: bool = True
    return_best_model: bool = False


@dataclass
class _SweepState:
    best_k: int
    best_val: float
    best_metrics: dict[str, object]
    best_model: VBPCA | None
    no_improve: int


def _normalize_components(
    components: Iterable[int] | None, n_features: int, n_samples: int
) -> list[int]:
    if components is None:
        max_k = max(1, min(n_features, n_samples))
        return list(range(1, max_k + 1))

    uniq: list[int] = []
    for val in components:
        k = int(val)
        if k <= 0:
            continue
        if k not in uniq:
            uniq.append(k)
    if not uniq:
        msg = "components must contain at least one positive integer"
        raise ValueError(msg)
    return uniq


def _to_float(val: object | None) -> float:
    try:
        return float(cast("_AllowedFloat", val))
    except (TypeError, ValueError):
        return float("nan")


def _metric_value(metric: _Metric, rms: float, prms: float, cost: float) -> float:
    if metric == "prms":
        return prms if np.isfinite(prms) else rms if np.isfinite(rms) else cost
    if metric == "rms":
        return rms if np.isfinite(rms) else prms if np.isfinite(prms) else cost
    return cost if np.isfinite(cost) else rms


def _metric_value_from_entry(metric: _Metric, entry: dict[str, object]) -> float:
    return _metric_value(
        metric,
        rms=cast("float", entry["rms"]),
        prms=cast("float", entry["prms"]),
        cost=cast("float", entry["cost"]),
    )


def _verbose_enabled(val: object) -> bool:
    try:
        return int(cast("SupportsIndex | str | bytes | bytearray", val)) > 0
    except (TypeError, ValueError):
        return bool(val)


def _is_metric_reversal(previous: float, current: float) -> bool:
    return (np.isfinite(previous) and not np.isfinite(current)) or (
        np.isfinite(previous)
        and np.isfinite(current)
        and current > previous
        and not np.isclose(current, previous, equal_nan=False)
    )


def _fit_candidate(
    k: int,
    x_arr: np.ndarray | sp.csr_matrix,
    mask: Matrix | None,
    cfg: SelectionConfig,
    opts: Mapping[str, object],
) -> tuple[dict[str, object], VBPCA]:
    from .estimators import VBPCA  # noqa: PLC0415 # Avoid circular dependency

    _ = cfg  # keep signature compatibility for injected stubs during tests

    est = VBPCA(k, **cast("dict[str, object]", dict(opts)))  # type: ignore[arg-type]
    est.fit(x_arr, mask=mask)

    rms = _to_float(est.rms_)
    prms = _to_float(est.prms_)
    cost = _to_float(est.cost_)

    entry: dict[str, object] = {
        "k": int(k),
        "rms": rms,
        "prms": prms,
        "cost": cost,
        "evr": None,
    }
    return entry, est


def _compute_evr_for_best(
    est: VBPCA,
    *,
    solver: str = "auto",
    gram_ratio: float = 4.0,
) -> np.ndarray | None:
    if est.components_ is None or est.scores_ is None or est.mean_ is None:
        return None
    xrec = _reconstruct_data(est.components_, est.scores_, est.mean_)
    ev, evr = _explained_variance(
        xrec,
        est.components_.shape[1],
        solver=solver,
        gram_ratio=gram_ratio,
    )
    est.variance_ = ev
    est.explained_variance_ = ev
    est.explained_variance_ratio_ = evr
    return evr


def select_n_components(  # noqa: C901, PLR0912, PLR0914, PLR0915
    x: Matrix,
    *,
    mask: Matrix | None = None,
    components: Iterable[int] | None = None,
    config: SelectionConfig | None = None,
    **opts: object,
) -> tuple[int, dict[str, object], list[dict[str, object]], VBPCA | None]:
    """Select n_components by sweeping candidates and tracking end metrics.

    Args:
        x: Data matrix (dense or sparse).
        mask: Optional boolean mask with the same shape as ``x``.
        components: Candidate component counts. Defaults to
            ``1..min(n_features, n_samples)``.
        config: Selection parameters controlling metric, stopping behavior,
            patience, trials, and whether to compute explained variance or
            retain the best model.
        **opts: Additional options forwarded to the ``VBPCA`` constructor and fit.

    Returns:
        Tuple ``(best_k, best_metrics, trace, best_model)`` where:
        - ``best_k``: chosen component count.
        - ``best_metrics``: scalar metrics for the best candidate.
        - ``trace``: list of per-k endpoint metrics.
        - ``best_model``: the best ``VBPCA`` instance, or None if not requested.

    Raises:
        ValueError: If ``metric`` is invalid or no valid ``components`` are provided.
    """
    cfg = config or SelectionConfig()
    if cfg.metric not in {"prms", "rms", "cost"}:
        msg = f"metric must be one of prms, rms, cost (got {cfg.metric!r})"
        raise ValueError(msg)

    is_sparse = sp.issparse(x)
    if not is_sparse and mask is not None and sp.issparse(mask):
        msg = "mask must be dense when x is dense"
        raise ValueError(msg)

    max_dense_bytes = resolve_max_dense_bytes(
        opts.get("max_dense_bytes", 2_000_000_000)
    )
    if is_sparse and mask is not None and not sp.issparse(mask):
        over, est_bytes = exceeds_budget(mask.shape, np.bool_, max_dense_bytes)
        if over:
            budget = 0 if max_dense_bytes is None else max_dense_bytes
            msg = (
                "Dense mask would exceed max_dense_bytes: "
                f"{format_bytes(est_bytes)} > {format_bytes(int(budget))}"
            )
            raise ValueError(msg)

        preflight_obj = opts.setdefault("_runtime_preflight", [])
        preflight = cast("list[dict[str, object]]", preflight_obj)
        preflight.append(
            {
                "check": "dense_mask_budget",
                "is_sparse_input": True,
                "mask_sparse": False,
                "estimate_bytes": int(est_bytes),
                "max_dense_bytes": max_dense_bytes,
                "over_budget": bool(over),
                "context": "model_selection",
            }
        )

    x_arr = sp.csr_matrix(x) if is_sparse else np.asarray(x)
    mask_arg: Matrix | None = None
    if mask is not None:
        mask_arg = (
            sp.csr_matrix(mask) if sp.issparse(mask) else np.asarray(mask, dtype=bool)
        )
    k_values = _normalize_components(components, x_arr.shape[0], x_arr.shape[1])

    trace: list[dict[str, object]] = []
    state = _SweepState(
        best_k=k_values[0],
        best_val=float("inf"),
        best_metrics={
            "rms": float("inf"),
            "prms": float("inf"),
            "cost": float("inf"),
            "evr": None,
        },
        best_model=None,
        no_improve=0,
    )

    fit_opts: dict[str, object] = dict(opts)
    selection_verbose_val = fit_opts.pop(
        "selection_verbose",
        fit_opts.get("verbose", 0),
    )
    verbose_enabled = _verbose_enabled(selection_verbose_val)
    # Avoid heavy diagnostics during sweeps; compute EVR once on the best.
    fit_opts.setdefault("return_diagnostics", False)
    prev_metric_val: float | None = None
    prev_entry: dict[str, object] | None = None
    best_est: VBPCA | None = None

    for idx, k in enumerate(k_values):
        if cfg.max_trials is not None and idx >= int(cfg.max_trials):
            break

        entry: dict[str, object]
        est: VBPCA
        entry, est = _fit_candidate(k, x_arr, mask_arg, cfg, fit_opts)
        trace.append(entry)

        metric_val = _metric_value_from_entry(cfg.metric, entry)

        if verbose_enabled:
            logger.info(
                (
                    "Model selection k=%d done: rms=%.6g prms=%.6g "
                    "cost=%.6g metric(=%s)=%.6g"
                ),
                k,
                cast("float", entry["rms"]),
                cast("float", entry["prms"]),
                cast("float", entry["cost"]),
                cfg.metric,
                metric_val,
            )

        # Optional local stopping rule: as soon as metric worsens at k compared
        # with k-1, select k-1 and stop.
        if (
            cfg.stop_on_metric_reversal
            and prev_metric_val is not None
            and _is_metric_reversal(prev_metric_val, metric_val)
            and prev_entry is not None
        ):
            state.best_k = int(cast("int", prev_entry["k"]))
            state.best_val = prev_metric_val
            state.best_metrics = prev_entry
            if cfg.return_best_model:
                state.best_model = best_est
            if verbose_enabled:
                logger.info(
                    (
                        "Model selection stopping on metric reversal at k=%d; "
                        "selecting previous k=%d"
                    ),
                    k,
                    state.best_k,
                )
            break

        if metric_val < state.best_val or (
            np.isclose(metric_val, state.best_val, equal_nan=False) and k < state.best_k
        ):
            state.best_k = int(k)
            state.best_val = metric_val
            state.best_metrics = entry
            state.no_improve = 0
            best_est = est
            if cfg.return_best_model:
                state.best_model = est
        else:
            state.no_improve += 1

        if cfg.patience is not None and state.no_improve > int(cfg.patience):
            if verbose_enabled:
                logger.info(
                    "Model selection stopping on patience at k=%d (best_k=%d)",
                    k,
                    state.best_k,
                )
            break

        prev_metric_val = metric_val
        prev_entry = entry
        # Keep reference to current best estimator for post-pass EVR computation.
        if best_est is None or state.best_k == int(k):
            best_est = est

    if verbose_enabled:
        logger.info("Model selection complete: best_k=%d", state.best_k)

    # Compute EV/EVR once on the best model if requested.
    if cfg.compute_explained_variance and best_est is not None:
        solver = str(fit_opts.get("explained_var_solver", "auto"))
        gram_ratio_val = fit_opts.get("explained_var_gram_ratio", 4.0)
        gram_ratio = float(cast("_AllowedFloat", gram_ratio_val))
        evr = _compute_evr_for_best(
            best_est,
            solver=solver,
            gram_ratio=gram_ratio,
        )
        state.best_metrics["evr"] = evr
        for entry in trace:
            if int(cast("int", entry.get("k", -1))) == state.best_k:
                entry["evr"] = evr
                break
        if cfg.return_best_model:
            state.best_model = best_est

    return state.best_k, state.best_metrics, trace, state.best_model
