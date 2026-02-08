"""Model selection utilities for VBPCA.

Sweeps candidate component counts while reusing the VBPCA estimator and
convergence options. Stores only scalar endpoint metrics per candidate and
optionally retains the best-fit model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, SupportsFloat, SupportsIndex, cast

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping

    from ._pca_full import Matrix
    from .estimators import VBPCA

Metric = Literal["prms", "rms", "cost"]
AllowedFloat = (
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

    metric: Metric = "prms"
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
        return float(cast("AllowedFloat", val))
    except (TypeError, ValueError):
        return float("nan")


def _metric_value(metric: Metric, rms: float, prms: float, cost: float) -> float:
    if metric == "prms":
        return prms if np.isfinite(prms) else rms if np.isfinite(rms) else cost
    if metric == "rms":
        return rms if np.isfinite(rms) else prms if np.isfinite(prms) else cost
    return cost if np.isfinite(cost) else rms


def _fit_candidate(
    k: int,
    x_arr: np.ndarray | sp.csr_matrix,
    mask: np.ndarray | None,
    cfg: SelectionConfig,
    opts: Mapping[str, object],
) -> tuple[dict[str, object], VBPCA]:
    from .estimators import VBPCA  # Import here to avoid circular dependency
    
    est = VBPCA(k, **cast("dict[str, object]", dict(opts)))  # type: ignore[arg-type]
    est.fit(x_arr, mask=mask)

    rms = _to_float(est.rms_)
    prms = _to_float(est.prms_)
    cost = _to_float(est.cost_)

    evr: np.ndarray | None = None
    if cfg.compute_explained_variance and est.explained_variance_ratio_ is not None:
        evr = np.asarray(est.explained_variance_ratio_, dtype=float)

    entry: dict[str, object] = {
        "k": int(k),
        "rms": rms,
        "prms": prms,
        "cost": cost,
        "evr": evr,
    }
    return entry, est


def select_n_components(
    x: Matrix,
    *,
    mask: np.ndarray | None = None,
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
        config: Selection parameters controlling metric, patience, trials, and
            whether to compute explained variance or retain the best model.
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

    x_arr = sp.csr_matrix(x) if sp.issparse(x) else np.asarray(x)
    shape = x_arr.shape[:2]

    k_values = _normalize_components(components, shape[0], shape[1])

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

    for idx, k in enumerate(k_values):
        if cfg.max_trials is not None and idx >= int(cfg.max_trials):
            break

        entry, est = _fit_candidate(k, x_arr, mask, cfg, fit_opts)
        trace.append(entry)

        rms_val = cast("float", entry["rms"])
        prms_val = cast("float", entry["prms"])
        cost_val = cast("float", entry["cost"])
        metric_val = _metric_value(
            cfg.metric, rms=rms_val, prms=prms_val, cost=cost_val
        )
        if metric_val < state.best_val or (
            np.isclose(metric_val, state.best_val, equal_nan=False) and k < state.best_k
        ):
            state.best_k = int(k)
            state.best_val = metric_val
            state.best_metrics = entry
            state.no_improve = 0
            if cfg.return_best_model:
                state.best_model = est
        else:
            state.no_improve += 1

        if cfg.patience is not None and state.no_improve > int(cfg.patience):
            break

    return state.best_k, state.best_metrics, trace, state.best_model
