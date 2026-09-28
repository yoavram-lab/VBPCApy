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
from ._missing import make_xprobe_mask
from ._pca_full import (
    _explained_variance,
    _explained_variance_from_factors,
    _marginal_variance,
    _reconstruct_data,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence

    from ._pca_full import Matrix
    from .estimators import VBPCA

__all__ = [
    "RANK_ZERO_SELECTION_SUPPORTED",
    "CVConfig",
    "SelectionConfig",
    "cross_validate_components",
    "select_n_components",
]

logger = logging.getLogger(__name__)

RANK_ZERO_SELECTION_SUPPORTED = True

_Metric = Literal["rms", "prms", "cost"]
_CVMetric = Literal["prms"]
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


_PROBE_FRACTION = 0.1
"""Fraction of observed entries held out for the probe set."""


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
        if k < 0:
            continue
        if k not in uniq:
            uniq.append(k)
    if not uniq:
        msg = "components must contain at least one non-negative integer"
        raise ValueError(msg)
    return uniq


def _to_float(val: object | None) -> float:
    try:
        return float(cast("_AllowedFloat", val))
    except (TypeError, ValueError):
        return float("nan")


def _metric_value(metric: _Metric, rms: float, prms: float, cost: float) -> float:
    if metric == "rms":
        return rms
    if metric == "prms":
        return prms
    return cost


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
) -> tuple[dict[str, object], VBPCA | None]:
    from .estimators import VBPCA  # noqa: PLC0415 # Avoid circular dependency

    _ = cfg  # keep signature compatibility for injected stubs during tests

    candidate_opts = dict(opts)
    xprobe = candidate_opts.pop("xprobe", None)
    if k == 0:
        return _fit_mean_only_candidate(
            x_arr,
            mask,
            cast("Matrix | None", xprobe),
            bias=bool(candidate_opts.get("bias", True)),
        )
    est = VBPCA(k, **candidate_opts)  # type: ignore[arg-type]
    est.fit(x_arr, mask=mask, xprobe=cast("Matrix | None", xprobe))

    rms = _to_float(est.rms_)
    prms = _to_float(est.prms_)
    cost = _to_float(est.cost_)

    entry: dict[str, object] = {
        "k": int(k),
        "rms": rms,
        "prms": prms,
        "cost": cost,
        "evr": None,
        "n_iter": est.n_iter_ if est.n_iter_ is not None else 0,
        "convergence_reason": est.convergence_reason_ or "maxiters",
        "converged": bool(est.converged_),
        "candidate_type": "low_rank",
    }
    return entry, est


def _fit_mean_only_candidate(
    x: np.ndarray | sp.csr_matrix,
    mask: Matrix | None,
    xprobe: Matrix | None,
    *,
    bias: bool,
) -> tuple[dict[str, object], None]:
    """Evaluate an explicit rank-zero (mean-only) candidate.

    Rank zero is a closed-form baseline rather than a degenerate call to the
    positive-rank VBPCA solver.  Its predictive error is therefore comparable
    with positive-rank candidates, while the variational cost is deliberately
    unavailable.

    Returns:
        Metric record and ``None`` because rank zero has no fitted estimator.

    Raises:
        ValueError: If an input is sparse, shapes differ, or a biased fit has a
            row without a training observation.
    """
    if sp.issparse(x) or sp.issparse(mask) or sp.issparse(xprobe):
        msg = "the rank-zero candidate currently supports dense input only"
        raise ValueError(msg)

    x_arr = np.asarray(x, dtype=float)
    observed = np.isfinite(x_arr)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != x_arr.shape:
            msg = "mask must have the same shape as x"
            raise ValueError(msg)
        observed &= mask_arr

    probe_arr: np.ndarray | None = None
    probe_observed = np.zeros(x_arr.shape, dtype=bool)
    if xprobe is not None:
        probe_arr = np.asarray(xprobe, dtype=float)
        if probe_arr.shape != x_arr.shape:
            msg = "xprobe must have the same shape as x"
            raise ValueError(msg)
        probe_observed = np.isfinite(probe_arr)
        observed &= ~probe_observed

    counts = np.sum(observed, axis=1)
    if bias and np.any(counts == 0):
        msg = "rank-zero mean estimation requires a training observation in each row"
        raise ValueError(msg)
    means = np.zeros(x_arr.shape[0], dtype=float)
    if bias:
        means = np.divide(
            np.sum(np.where(observed, x_arr, 0.0), axis=1),
            counts,
            out=means,
            where=counts > 0,
        )

    residual = x_arr - means[:, np.newaxis]
    rms = float(np.sqrt(np.mean(np.square(residual[observed]))))
    prms = float("nan")
    if probe_arr is not None and np.any(probe_observed):
        probe_residual = probe_arr - means[:, np.newaxis]
        prms = float(np.sqrt(np.mean(np.square(probe_residual[probe_observed]))))

    return (
        {
            "k": 0,
            "rms": rms,
            "prms": prms,
            "cost": float("nan"),
            "evr": None,
            "n_iter": 0,
            "convergence_reason": "closed_form_mean_only",
            "converged": True,
            "candidate_type": "mean_only",
        },
        None,
    )


def _compute_evr_for_best(
    est: VBPCA,
    *,
    solver: str = "auto",
    gram_ratio: float = 4.0,
) -> np.ndarray | None:
    if est.components_ is None or est.scores_ is None or est.mean_ is None:
        return None
    xrec = _reconstruct_data(est.components_, est.scores_, est.mean_)
    if solver == "auto":
        ev, evr = _explained_variance_from_factors(
            est.components_,
            est.scores_,
            est.components_.shape[1],
        )
    else:
        ev, evr = _explained_variance(
            xrec,
            est.components_.shape[1],
            solver=solver,
            gram_ratio=gram_ratio,
        )
    # Retain reconstruction for downstream consumers (e.g., posterior tests).
    est.reconstruction_ = xrec
    est.explained_variance_ = ev
    est.explained_variance_ratio_ = evr
    return evr


def _compute_variance_for_best(est: VBPCA) -> np.ndarray | None:
    if (
        est.components_ is None
        or est.scores_ is None
        or est._av is None  # noqa: SLF001
        or est._muv is None  # noqa: SLF001
    ):
        return None
    from ._pca_full import FinalState  # noqa: PLC0415

    final = FinalState(
        a=est.components_,
        s=est.scores_,
        mu=est.mean_ if est.mean_ is not None else np.zeros(est.components_.shape[0]),
        noise_var=est.noise_variance_ if est.noise_variance_ is not None else 0.0,
        av=est._av,  # noqa: SLF001
        sv=est._sv if est._sv is not None else [],  # noqa: SLF001
        pattern_index=est._pattern_index,  # noqa: SLF001
        muv=est._muv,  # noqa: SLF001
        va=np.zeros(est.components_.shape[1]),
        vmu=0.0,
        lc={},
        runtime_report=None,
    )
    vr = _marginal_variance(final)
    est.variance_ = vr
    if est.noise_variance_ is not None:
        est.predictive_variance_ = vr + float(est.noise_variance_)
    return vr


def _normalize_mask_for_selection(
    x: Matrix, mask: Matrix | None, max_dense_bytes: int | None, opts: dict[str, object]
) -> Matrix | None:
    """Normalize mask to dense bool or CSR, enforcing budget for sparse inputs.

    Returns:
        Normalized mask (dense bool or CSR) or ``None`` when absent.

    Raises:
        ValueError: If mask sparsity is incompatible or exceeds the dense budget.
    """
    if not sp.issparse(x):
        if mask is not None and sp.issparse(mask):
            msg = "mask must be dense when x is dense"
            raise ValueError(msg)
        return None if mask is None else np.asarray(mask, dtype=bool)

    if mask is None:
        return None
    if sp.issparse(mask):
        return sp.csr_matrix(mask)

    over, est_bytes = exceeds_budget(mask.shape, np.bool_, max_dense_bytes)
    if over:
        budget = 0 if max_dense_bytes is None else max_dense_bytes
        msg = (
            "Dense mask would exceed max_dense_bytes: "
            f"{format_bytes(est_bytes)} > {format_bytes(int(budget))}"
        )
        raise ValueError(msg)

    preflight = cast(
        "list[dict[str, object]]", opts.setdefault("_runtime_preflight", [])
    )
    preflight.append({
        "check": "dense_mask_budget",
        "is_sparse_input": True,
        "mask_sparse": False,
        "estimate_bytes": int(est_bytes),
        "max_dense_bytes": max_dense_bytes,
        "over_budget": bool(over),
        "context": "model_selection",
    })
    return np.asarray(mask, dtype=bool)


def _ensure_metric_opts(
    fit_opts: dict[str, object],
    x_arr: np.ndarray | sp.csr_matrix,
    mask: Matrix | None,
    cfg: SelectionConfig,
    seed: int | np.random.Generator | None = None,
) -> None:
    """Enable VBPCA options required by the chosen selection metric.

    Mutates *fit_opts* in place:

    * **record_cost** — enabled so endpoint diagnostics remain complete without
      enabling any cost-based convergence criterion.
    * **xprobe** — when no probe set has been supplied, a random hold-out
      of observed entries is created when the selection metric is ``"prms"``
      or the caller requested a positive ``xprobe_fraction``. The requested
      fraction is used when present; ``"prms"`` otherwise falls back to the
      historical 10 % default. Candidate fits receive the same explicit probe;
      the estimator removes those entries from its training data and mask.
    """
    fit_opts["record_cost"] = True

    # --- prms: ensure xprobe is populated -----------------------------------
    if fit_opts.get("xprobe") is not None:
        return  # user already supplied a probe set

    configured_fraction = _to_float(fit_opts.get("xprobe_fraction"))
    has_configured_fraction = bool(
        np.isfinite(configured_fraction) and configured_fraction > 0.0
    )
    if cfg.metric != "prms" and not has_configured_fraction:
        return
    probe_fraction = configured_fraction if has_configured_fraction else _PROBE_FRACTION

    _x_masked, xprobe = make_xprobe_mask(
        x_arr,
        fraction=probe_fraction,
        rng=np.random.default_rng(seed),
        mask=mask,
    )
    fit_opts["xprobe"] = xprobe


@dataclass(frozen=True)
class SweepInputs:
    cfg: SelectionConfig
    fit_opts: dict[str, object]
    verbose_enabled: bool


def _handle_metric_reversal(  # noqa: PLR0913
    *,
    state: _SweepState,
    cfg: SelectionConfig,
    prev_metric_val: float | None,
    metric_val: float,
    prev_entry: dict[str, object] | None,
    best_est: VBPCA | None,
    verbose_enabled: bool,
    k: int,
) -> bool:
    if not (
        cfg.stop_on_metric_reversal
        and prev_metric_val is not None
        and prev_entry is not None
        and _is_metric_reversal(prev_metric_val, metric_val)
    ):
        return False

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
    return True


def _handle_patience(
    *, state: _SweepState, cfg: SelectionConfig, k: int, verbose_enabled: bool
) -> bool:
    required = max(1, int(cfg.patience)) if cfg.patience is not None else None
    if required is None or state.no_improve < required:
        return False
    if verbose_enabled:
        logger.info(
            "Model selection stopping on patience at k=%d (best_k=%d)",
            k,
            state.best_k,
        )
    return True


def _sweep_components(
    k_values: Sequence[int],
    x_arr: Matrix,
    mask_arg: Matrix | None,
    inputs: SweepInputs,
) -> tuple[int, dict[str, object], list[dict[str, object]], VBPCA | None, VBPCA | None]:
    trace: list[dict[str, object]] = []
    cfg = inputs.cfg
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
    prev_metric_val: float | None = None
    prev_entry: dict[str, object] | None = None
    best_est: VBPCA | None = None

    for idx, k in enumerate(k_values):
        if cfg.max_trials is not None and idx >= int(cfg.max_trials):
            break

        entry, est = _fit_candidate(k, x_arr, mask_arg, cfg, inputs.fit_opts)
        trace.append(entry)

        metric_val = _metric_value_from_entry(cfg.metric, entry)
        if not np.isfinite(metric_val):
            msg = (
                f"selection metric {cfg.metric!r} is unavailable for k={k}; "
                "configure the requested metric instead of relying on a substitute"
            )
            raise ValueError(msg)

        if inputs.verbose_enabled:
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

        if _handle_metric_reversal(
            state=state,
            cfg=cfg,
            prev_metric_val=prev_metric_val,
            metric_val=metric_val,
            prev_entry=prev_entry,
            best_est=best_est,
            verbose_enabled=inputs.verbose_enabled,
            k=int(k),
        ):
            break

        better_metric = metric_val < state.best_val or (
            np.isclose(metric_val, state.best_val, equal_nan=False) and k < state.best_k
        )
        if better_metric:
            state.best_k = int(k)
            state.best_val = metric_val
            state.best_metrics = entry
            state.no_improve = 0
            best_est = est
            if cfg.return_best_model:
                state.best_model = est
        else:
            state.no_improve += 1

        if _handle_patience(
            state=state, cfg=cfg, k=int(k), verbose_enabled=inputs.verbose_enabled
        ):
            break

        prev_metric_val = metric_val
        prev_entry = entry

    return state.best_k, state.best_metrics, trace, state.best_model, best_est


def select_n_components(
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
        components: Candidate component counts. Include ``0`` to compare an
            explicit mean-only model. Defaults to positive ranks
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
        - ``best_model``: the best ``VBPCA`` instance, or ``None`` if it was
          not requested or the mean-only candidate won.

    Raises:
        ValueError: If ``metric`` is invalid or no valid ``components`` are provided.
    """
    cfg = config or SelectionConfig()
    if cfg.metric not in {"rms", "prms", "cost"}:
        msg = f"metric must be one of rms, prms, cost (got {cfg.metric!r})"
        raise ValueError(msg)
    if cfg.patience is not None and int(cfg.patience) < 0:
        msg = "patience must be a non-negative integer or None"
        raise ValueError(msg)
    x_arr: np.ndarray | sp.csr_matrix = (
        sp.csr_matrix(x, copy=True) if sp.issparse(x) else np.array(x, dtype=float)
    )
    mask_arg = _normalize_mask_for_selection(
        x,
        mask,
        resolve_max_dense_bytes(opts.get("max_dense_bytes", 2_000_000_000)),
        opts,
    )
    k_values = _normalize_components(components, x_arr.shape[0], x_arr.shape[1])
    if cfg.metric == "cost" and 0 in k_values:
        msg = "rank zero has no variational cost; select it with metric='rms' or 'prms'"
        raise ValueError(msg)

    fit_opts: dict[str, object] = dict(opts)
    verbose_enabled = _verbose_enabled(
        fit_opts.pop("selection_verbose", fit_opts.get("verbose", 0))
    )
    fit_opts.setdefault("return_diagnostics", False)

    # Record cost or prepare a probe set without changing convergence criteria.
    _ensure_metric_opts(
        fit_opts,
        x_arr,
        mask_arg,
        cfg,
        seed=cast("int | np.random.Generator | None", fit_opts.get("random_state")),
    )

    sweep_inputs = SweepInputs(
        cfg=cfg,
        fit_opts=fit_opts,
        verbose_enabled=verbose_enabled,
    )

    (
        best_k,
        best_metrics,
        trace,
        best_model,
        best_est,
    ) = _sweep_components(
        k_values,
        x_arr,
        mask_arg,
        sweep_inputs,
    )

    if verbose_enabled:
        logger.info("Model selection complete: best_k=%d", best_k)

    if cfg.compute_explained_variance and best_est is not None:
        evr = _compute_evr_for_best(
            best_est,
            solver=str(fit_opts.get("explained_var_solver", "auto")),
            gram_ratio=float(
                cast("_AllowedFloat", fit_opts.get("explained_var_gram_ratio", 4.0))
            ),
        )
        _compute_variance_for_best(best_est)
        best_metrics["evr"] = evr
        for entry in trace:
            if int(cast("int", entry.get("k", -1))) == best_k:
                entry["evr"] = evr
                break
        if cfg.return_best_model and best_model is None:
            best_model = best_est

    return best_k, best_metrics, trace, best_model


# ---------------------------------------------------------------------------
# K-fold cross-validated model selection
# ---------------------------------------------------------------------------


@dataclass
class CVConfig:
    """Configuration for K-fold cross-validated component selection.

    Attributes:
        metric: Held-out selection metric. Only ``"prms"`` is supported.
        n_splits: Number of cross-validation folds.
        one_se_rule: If ``True``, select the smallest *k* whose mean metric
            is within one standard error of the global minimum.
        seed: Random seed for fold partitioning and model fitting.
    """

    metric: _CVMetric = "prms"
    n_splits: int = 5
    one_se_rule: bool = True
    seed: int = 0


def _make_element_folds(
    x: np.ndarray,
    n_splits: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition observed entries of *x* into *n_splits* folds.

    Args:
        x: Data matrix ``(n_features, n_samples)`` with ``NaN`` for
            missing entries.
        n_splits: Number of folds.
        rng: NumPy random generator for reproducible shuffling.

    Returns:
        List of ``(probe_indices, train_indices)`` tuples where each
        element is a 1-D array of flat indices into the observed-entry
        array.

    Raises:
        ValueError: If there are too few observations or valid folds cannot
            preserve training coverage in every non-empty row and column.
    """
    observed = ~np.isnan(x)
    obs_rows, obs_cols = np.nonzero(observed)
    n_obs = len(obs_rows)
    support = _training_support_backbone(
        obs_rows,
        obs_cols,
        n_rows=x.shape[0],
        n_cols=x.shape[1],
        rng=rng,
    )
    eligible = np.flatnonzero(~support)
    if len(eligible) < n_splits:
        msg = (
            f"n_splits={n_splits} exceeds the {len(eligible)} validation-eligible "
            f"observed entries after reserving {int(np.sum(support))} entries "
            "that preserve training support in every non-empty row and column"
        )
        raise ValueError(msg)

    probe_folds = [
        np.asarray(part, dtype=int)
        for part in np.array_split(rng.permutation(eligible), n_splits)
    ]
    return [
        (
            probe_sel,
            np.setdiff1d(np.arange(n_obs), probe_sel, assume_unique=True),
        )
        for probe_sel in probe_folds
    ]


def _training_support_backbone(
    obs_rows: np.ndarray,
    obs_cols: np.ndarray,
    *,
    n_rows: int,
    n_cols: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Reserve a minimum edge cover of non-empty rows and columns.

    The returned boolean vector indexes the observed-entry arrays. Reserved
    entries remain in every training fold. A maximum bipartite matching is
    extended to an edge cover, which guarantees support without a retry loop
    and leaves as many observations as possible eligible for validation.
    Random row and column permutations avoid systematic tie preference.

    Returns:
        Boolean vector marking the permanent training-support backbone.

    Raises:
        ValueError: If the observed row and column index shapes differ.
    """
    if obs_rows.shape != obs_cols.shape:
        msg = "observed row and column index arrays must have matching shapes"
        raise ValueError(msg)
    support = np.zeros(len(obs_rows), dtype=bool)
    if len(obs_rows) == 0:
        return support

    row_order = rng.permutation(n_rows)
    col_order = rng.permutation(n_cols)
    row_position = np.empty(n_rows, dtype=int)
    col_position = np.empty(n_cols, dtype=int)
    row_position[row_order] = np.arange(n_rows)
    col_position[col_order] = np.arange(n_cols)
    graph = sp.csr_matrix(
        (
            np.ones(len(obs_rows), dtype=np.int8),
            (row_position[obs_rows], col_position[obs_cols]),
        ),
        shape=(n_rows, n_cols),
    )
    matched_columns = sp.csgraph.maximum_bipartite_matching(graph, perm_type="column")
    edge_index = {
        (int(row), int(column)): index
        for index, (row, column) in enumerate(zip(obs_rows, obs_cols, strict=True))
    }
    for permuted_row, permuted_column in enumerate(matched_columns):
        if permuted_column >= 0:
            row = int(row_order[permuted_row])
            column = int(col_order[permuted_column])
            support[edge_index[row, column]] = True

    row_covered = np.bincount(obs_rows[support], minlength=n_rows) > 0
    col_covered = np.bincount(obs_cols[support], minlength=n_cols) > 0
    row_counts = np.bincount(obs_rows, minlength=n_rows)
    col_counts = np.bincount(obs_cols, minlength=n_cols)
    for row in rng.permutation(np.flatnonzero((row_counts > 0) & ~row_covered)):
        candidates = np.flatnonzero(obs_rows == row)
        support[int(rng.choice(candidates))] = True
    col_covered = np.bincount(obs_cols[support], minlength=n_cols) > 0
    for column in rng.permutation(np.flatnonzero((col_counts > 0) & ~col_covered)):
        candidates = np.flatnonzero(obs_cols == column)
        support[int(rng.choice(candidates))] = True
    return support


def _fold_preserves_training_coverage(
    probe_sel: np.ndarray,
    obs_rows: np.ndarray,
    obs_cols: np.ndarray,
    row_counts: np.ndarray,
    col_counts: np.ndarray,
) -> bool:
    """Return whether one holdout leaves every observed row and column covered."""
    heldout_rows = np.bincount(obs_rows[probe_sel], minlength=len(row_counts))
    heldout_cols = np.bincount(obs_cols[probe_sel], minlength=len(col_counts))
    return bool(
        np.all(heldout_rows[row_counts > 0] < row_counts[row_counts > 0])
        and np.all(heldout_cols[col_counts > 0] < col_counts[col_counts > 0])
    )


_TRACKED_METRICS: tuple[str, ...] = ("rms", "prms", "cost")
"""Metrics recorded from each candidate fit for CV aggregation."""

_CONVERGENCE_FIELDS: tuple[str, ...] = (
    "n_iter",
    "converged",
    "convergence_reason",
)
"""Scalar convergence diagnostics retained for each candidate and fold."""


def _run_fold(  # noqa: PLR0913
    fold_i: int,
    x_base: np.ndarray,
    obs_rows: np.ndarray,
    obs_cols: np.ndarray,
    probe_sel: np.ndarray,
    *,
    k_list: list[int],
    metric: _CVMetric,
    opts: dict[str, object],
    seed: int,
    n_splits: int = 1,
    verbose: int = 0,
) -> dict[int, dict[str, object]]:
    """Run one fold: mask probe entries, sweep all *k* values, return metrics.

    Args:
        fold_i: Zero-based fold index (for logging).
        x_base: Original data matrix ``(n_features, n_samples)``.
        obs_rows: Row indices of all observed entries.
        obs_cols: Column indices of all observed entries.
        probe_sel: Indices into ``obs_rows``/``obs_cols`` for this fold's
            held-out probe entries.
        k_list: Candidate component counts to evaluate.
        metric: Held-out probe RMS (``"prms"``).
        opts: Options forwarded to ``select_n_components``.
        seed: Base seed used to derive a deterministic seed for this fold.
        n_splits: Total number of folds (for log messages).
        verbose: Verbosity level.

    Returns:
        Dict mapping each *k* to tracked metrics and convergence diagnostics.
    """
    if verbose:
        logger.info("  Fold %d/%d ...", fold_i + 1, n_splits)

    x_fold = x_base.copy()
    x_fold[obs_rows[probe_sel], obs_cols[probe_sel]] = np.nan

    xprobe = np.full(x_fold.shape, np.nan, dtype=float)
    xprobe[obs_rows[probe_sel], obs_cols[probe_sel]] = x_base[
        obs_rows[probe_sel], obs_cols[probe_sel]
    ]

    fold_opts = dict(opts)
    fold_opts["xprobe"] = xprobe
    fold_opts["random_state"] = seed + fold_i

    _best_k, _best_metrics, trace, _model = select_n_components(
        x_fold,
        components=k_list,
        config=SelectionConfig(
            metric=metric,
            patience=None,
            max_trials=len(k_list),
            compute_explained_variance=False,
            return_best_model=False,
        ),
        **fold_opts,  # type: ignore[arg-type]
    )

    return {
        int(cast("int", t["k"])): {
            **{m: float(cast("float", t[m])) for m in _TRACKED_METRICS},
            "n_iter": int(cast("int", t["n_iter"])),
            "converged": bool(t["converged"]),
            "convergence_reason": str(t["convergence_reason"]),
        }
        for t in trace
    }


def _add_cv_convergence_summary(
    entry: dict[str, object],
    k: int,
    fold_metrics: Sequence[dict[int, dict[str, object]]],
) -> None:
    """Add per-fold and aggregate convergence diagnostics for one candidate."""
    candidate_folds = [fold[k] for fold in fold_metrics if k in fold]
    iterations = [int(cast("int", fold["n_iter"])) for fold in candidate_folds]
    converged = [bool(fold["converged"]) for fold in candidate_folds]
    reasons = [str(fold["convergence_reason"]) for fold in candidate_folds]
    reason_counts = {reason: reasons.count(reason) for reason in sorted(set(reasons))}

    entry["mean_n_iter"] = float(np.mean(iterations)) if iterations else float("nan")
    entry["max_n_iter"] = max(iterations, default=0)
    entry["convergence_rate"] = float(np.mean(converged)) if converged else float("nan")
    entry["convergence_reason_counts"] = reason_counts
    for fold_index, fold in enumerate(candidate_folds, start=1):
        for field in _CONVERGENCE_FIELDS:
            entry[f"{field}_fold_{fold_index}"] = fold[field]


def _aggregate_cv_results(
    k_list: list[int],
    fold_metrics: list[dict[int, dict[str, object]]],
    selection_metric: _CVMetric,
) -> tuple[int, list[dict[str, object]]]:
    """Aggregate fold metrics and select *k* via the 1-SE rule.

    All tracked metrics (rms, prms, cost) are aggregated.  The 1-SE rule
    is applied to *selection_metric*.

    Args:
        k_list: Candidate component counts.
        fold_metrics: Per-fold dicts mapping *k* to a dict of all tracked
            metric values.
        selection_metric: The metric used for the 1-SE selection rule.

    Returns:
        Tuple ``(best_k, cv_results)`` where *cv_results* is a list of
        dicts with keys ``k``, per-metric ``mean_<m>``, ``std_<m>``,
        ``se_<m>`` columns, per-fold values, and convergence summaries.
    """
    cv_results: list[dict[str, object]] = []

    for k in k_list:
        entry: dict[str, object] = {"k": k}
        for m in _TRACKED_METRICS:
            vals = [
                float(cast("_AllowedFloat", fold[k][m]))
                for fold in fold_metrics
                if k in fold and m in fold[k]
            ]
            n = len(vals)
            std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            entry[f"mean_{m}"] = float(np.mean(vals)) if n > 0 else float("nan")
            entry[f"std_{m}"] = std_val
            entry[f"se_{m}"] = std_val / np.sqrt(n) if n > 1 else 0.0
            for i, fold in enumerate(fold_metrics):
                entry[f"{m}_fold_{i + 1}"] = fold.get(k, {}).get(m, float("nan"))
        _add_cv_convergence_summary(entry, k, fold_metrics)
        cv_results.append(entry)

    # Apply 1-SE rule on the selection metric
    means = np.array([
        float(cast("float", r[f"mean_{selection_metric}"])) for r in cv_results
    ])
    min_idx = int(np.argmin(means))
    threshold = means[min_idx] + float(
        cast("float", cv_results[min_idx][f"se_{selection_metric}"])
    )

    eligible = [
        r
        for r in cv_results
        if float(cast("float", r[f"mean_{selection_metric}"])) <= threshold
    ]
    if eligible:
        best = min(eligible, key=lambda r: int(cast("int", r["k"])))
        return int(cast("int", best["k"])), cv_results

    return int(cast("int", cv_results[min_idx]["k"])), cv_results


def cross_validate_components(
    x: Matrix,
    *,
    mask: Matrix | None = None,
    components: Iterable[int] | None = None,
    config: CVConfig | None = None,
    **opts: object,
) -> tuple[int, list[dict[str, object]]]:
    """K-fold cross-validated model selection for VBPCA.

    Partitions observed entries into *n_splits* folds.  For each fold
    the held-out entries become an xprobe set.  All candidate *k* values
    are evaluated on every fold via ``select_n_components``.  The final
    *k* is chosen by the **1-SE rule**: the smallest *k* whose mean
    metric across folds is within one standard error of the global
    minimum.

    All tracked metrics (rms, prms, cost) and convergence diagnostics are
    recorded per fold regardless of which metric is used for selection, so
    callers can compare selection criteria and audit fit quality without
    re-running.

    Args:
        x: Dense data matrix with shape ``(n_features, n_samples)``.
        mask: Optional boolean mask with the same shape as ``x``.
        components: Candidate component counts. Include ``0`` to compare an
            explicit mean-only model. Defaults to positive ranks
            ``1 .. min(n_features, n_samples)``.
        config: Cross-validation parameters.  Uses ``CVConfig()``
            defaults when ``None``.
        **opts: Additional options forwarded to ``select_n_components``
            and ultimately to the ``VBPCA`` constructor / fit.

    Returns:
        Tuple ``(best_k, cv_results)`` where:

        - ``best_k``: selected component count.
        - ``cv_results``: list of dicts (one per candidate *k*) with keys
          ``k``, ``mean_<m>``, ``std_<m>``, ``se_<m>`` for each metric,
          and ``<m>_fold_<i>`` per-fold values. Each entry also includes
          per-fold ``n_iter``, ``converged``, and ``convergence_reason`` plus
          candidate-level mean/max iterations, convergence rate, and reason
          counts.

    Raises:
        ValueError: If the input is sparse, ``metric`` is not ``"prms"``, no
            valid ``components`` are provided, folds cannot preserve row and
            column coverage, or ``n_splits < 2``.

    Example:
        >>> best_k, cv = cross_validate_components(
        ...     X, components=range(1, 6), config=CVConfig(n_splits=5)
        ... )
    """
    cv_cfg = config or CVConfig()

    if cv_cfg.metric != "prms":
        msg = (
            "cross-validation metric must be 'prms'; cost is a training "
            "objective, not a held-out fold metric"
        )
        raise ValueError(msg)
    if cv_cfg.n_splits < 2:
        msg = f"n_splits must be >= 2 (got {cv_cfg.n_splits})"
        raise ValueError(msg)

    if sp.issparse(x):
        msg = (
            "cross_validate_components currently supports dense input only; "
            "sparse input would lose structural missingness when materialized"
        )
        raise ValueError(msg)
    if mask is not None and sp.issparse(mask):
        msg = "mask must be dense when cross-validation input is dense"
        raise ValueError(msg)

    x_arr = np.array(x, dtype=float)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != x_arr.shape:
            msg = "mask must have the same shape as x"
            raise ValueError(msg)
        x_arr[~mask_arr] = np.nan

    k_list = _normalize_components(components, x_arr.shape[0], x_arr.shape[1])
    obs_rows, obs_cols = np.nonzero(~np.isnan(x_arr))
    folds = _make_element_folds(
        x_arr, cv_cfg.n_splits, np.random.default_rng(cv_cfg.seed)
    )

    fit_opts: dict[str, object] = dict(opts)
    verbose_level = int(
        cast(
            "SupportsIndex | str | bytes | bytearray",
            fit_opts.pop("selection_verbose", fit_opts.get("verbose", 0)),
        )
    )
    fit_opts["verbose"] = 0

    all_fold_metrics: list[dict[int, dict[str, object]]] = [
        _run_fold(
            fold_i=fold_i,
            x_base=x_arr,
            obs_rows=obs_rows,
            obs_cols=obs_cols,
            probe_sel=probe_sel,
            k_list=k_list,
            metric=cv_cfg.metric,
            opts=fit_opts,
            seed=cv_cfg.seed,
            n_splits=cv_cfg.n_splits,
            verbose=verbose_level,
        )
        for fold_i, (probe_sel, _train_sel) in enumerate(folds)
    ]

    best_k, cv_results = _aggregate_cv_results(k_list, all_fold_metrics, cv_cfg.metric)

    if not cv_cfg.one_se_rule:
        best_entry = min(
            cv_results,
            key=lambda r: (
                float(cast("float", r[f"mean_{cv_cfg.metric}"])),
                int(cast("int", r["k"])),
            ),
        )
        best_k = int(cast("int", best_entry["k"]))

    if verbose_level:
        min_entry = min(
            cv_results,
            key=lambda r: float(cast("float", r[f"mean_{cv_cfg.metric}"])),
        )
        logger.info(
            "CV result: best_k=%d (min mean %s=%.6g +/- %.6g at k=%d)",
            best_k,
            cv_cfg.metric,
            cast("float", min_entry[f"mean_{cv_cfg.metric}"]),
            cast("float", min_entry[f"se_{cv_cfg.metric}"]),
            cast("int", min_entry["k"]),
        )

    return best_k, cv_results
