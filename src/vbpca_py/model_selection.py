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
    from collections.abc import Iterable, Mapping, Sequence

    from ._pca_full import Matrix
    from .estimators import VBPCA

__all__ = [
    "CVConfig",
    "SelectionConfig",
    "cross_validate_components",
    "select_n_components",
]

logger = logging.getLogger(__name__)

_Metric = Literal["rms", "prms", "cost"]
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


_CFSTOP_DEFAULT = np.array([100, 1e-4, 1e-3])
"""Sensible default for cfstop=[window, abs_tol, rel_tol]."""

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
    if metric == "rms":
        return rms if np.isfinite(rms) else cost
    if metric == "prms":
        return prms if np.isfinite(prms) else cost
    return cost if np.isfinite(cost) else prms


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
    # Retain reconstruction for downstream consumers (e.g., posterior tests).
    est.reconstruction_ = xrec
    est.explained_variance_ = ev
    est.explained_variance_ratio_ = evr
    return evr


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


def _ensure_metric_opts(  # noqa: PLR0914
    fit_opts: dict[str, object],
    x_arr: np.ndarray | sp.csr_matrix,
    mask: Matrix | None,
    cfg: SelectionConfig,  # noqa: ARG001
    seed: int = 0,
) -> None:
    """Enable VBPCA options required by the chosen selection metric.

    Mutates *fit_opts* in place:

    * **cfstop** — always enabled so the cost learning-curve is populated.
    * **xprobe** — when the metric is ``"prms"`` and no probe set has been
      supplied, a random 10 % hold-out of observed entries is created.
      The corresponding entries are set to NaN in *x_arr* (dense) or
      removed from the CSR structure (sparse) so the main fit never sees
      them.
    """
    # --- cost: ensure cfstop is non-empty -----------------------------------
    cfstop_raw = fit_opts.get("cfstop")
    if cfstop_raw is None or np.size(np.asarray(cfstop_raw)) == 0:
        fit_opts["cfstop"] = _CFSTOP_DEFAULT

    # --- prms: ensure xprobe is populated -----------------------------------
    if fit_opts.get("xprobe") is not None:
        return  # user already supplied a probe set

    rng = np.random.default_rng(seed)

    if sp.issparse(x_arr):
        x_csr = sp.csr_matrix(x_arr)
        n_probe = max(1, round(x_csr.nnz * _PROBE_FRACTION))
        probe_idx = rng.choice(x_csr.nnz, size=n_probe, replace=False)

        # Build xprobe as a copy, then zero-out non-probe in probe
        # and zero-out probe in data.
        rows, cols = x_csr.nonzero()
        sp_rows = rows[probe_idx]
        sp_cols = cols[probe_idx]
        sp_vals = np.array(x_csr[sp_rows, sp_cols]).ravel()

        xprobe_sp = sp.lil_matrix(x_csr.shape, dtype=float)
        for r, c, v in zip(sp_rows, sp_cols, sp_vals, strict=True):
            xprobe_sp[r, c] = v
        fit_opts["xprobe"] = sp.csr_matrix(xprobe_sp)

        # Remove probe entries from training data
        for r, c in zip(sp_rows, sp_cols, strict=True):
            x_csr[r, c] = 0.0
        x_csr.eliminate_zeros()
        # Update x_arr in place (callers hold a reference to x_arr)
        x_csr.sort_indices()
    else:
        x_dense = np.asarray(x_arr)
        if mask is not None:
            obs_mask = np.asarray(mask, dtype=bool)
        else:
            obs_mask = ~np.isnan(x_dense)

        obs_rows, obs_cols = np.nonzero(obs_mask)
        n_probe = max(1, round(len(obs_rows) * _PROBE_FRACTION))
        probe_idx = rng.choice(len(obs_rows), size=n_probe, replace=False)

        probe_rows: np.ndarray = obs_rows[probe_idx]
        probe_cols: np.ndarray = obs_cols[probe_idx]

        xprobe_dense = np.full(x_dense.shape, np.nan, dtype=float)
        xprobe_dense[probe_rows, probe_cols] = x_dense[probe_rows, probe_cols]
        fit_opts["xprobe"] = xprobe_dense

        # Mark probe entries as missing in the training data
        x_dense[probe_rows, probe_cols] = np.nan


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
    if cfg.patience is None or state.no_improve <= int(cfg.patience):
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
        if best_est is None or state.best_k == int(k):
            best_est = est

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
    if cfg.metric not in {"prms", "cost"}:
        msg = f"metric must be one of prms, cost (got {cfg.metric!r})"
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

    fit_opts: dict[str, object] = dict(opts)
    verbose_enabled = _verbose_enabled(
        fit_opts.pop("selection_verbose", fit_opts.get("verbose", 0))
    )
    fit_opts.setdefault("return_diagnostics", False)

    # Enable cfstop / xprobe so that cost and prms metrics are populated.
    _ensure_metric_opts(fit_opts, x_arr, mask_arg, cfg)

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
        metric: Selection metric (``"prms"`` or ``"cost"``).
        n_splits: Number of cross-validation folds.
        one_se_rule: If ``True``, select the smallest *k* whose mean metric
            is within one standard error of the global minimum.
        seed: Random seed for fold partitioning and model fitting.
    """

    metric: _Metric = "prms"
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
    """
    obs_rows, _obs_cols = np.nonzero(~np.isnan(x))
    n_obs = len(obs_rows)
    perm = rng.permutation(n_obs)

    fold_size = n_obs // n_splits
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else n_obs
        test_sel = perm[start:end]
        train_sel = np.concatenate([perm[:start], perm[end:]])
        folds.append((
            test_sel,
            train_sel,
        ))
    return folds


_TRACKED_METRICS: tuple[str, ...] = ("rms", "prms", "cost")
"""Metrics recorded from each candidate fit for CV aggregation."""


def _run_fold(  # noqa: PLR0913
    fold_i: int,
    x_base: np.ndarray,
    obs_rows: np.ndarray,
    obs_cols: np.ndarray,
    probe_sel: np.ndarray,
    *,
    k_list: list[int],
    metric: _Metric,
    opts: dict[str, object],
    n_splits: int = 1,
    verbose: int = 0,
) -> dict[int, dict[str, float]]:
    """Run one fold: mask probe entries, sweep all *k* values, return metrics.

    Args:
        fold_i: Zero-based fold index (for logging).
        x_base: Original data matrix ``(n_features, n_samples)``.
        obs_rows: Row indices of all observed entries.
        obs_cols: Column indices of all observed entries.
        probe_sel: Indices into ``obs_rows``/``obs_cols`` for this fold's
            held-out probe entries.
        k_list: Candidate component counts to evaluate.
        metric: ``"prms"`` or ``"cost"``.
        opts: Options forwarded to ``select_n_components``.
        n_splits: Total number of folds (for log messages).
        verbose: Verbosity level.

    Returns:
        Dict mapping each *k* to a dict of all tracked metric values.
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
            m: float(cast("float", t[m])) for m in _TRACKED_METRICS
        }
        for t in trace
    }


def _aggregate_cv_results(
    k_list: list[int],
    fold_metrics: list[dict[int, dict[str, float]]],
    selection_metric: _Metric,
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
        ``se_<m>`` columns, and ``<m>_fold_<i>`` per-fold values.
    """
    cv_results: list[dict[str, object]] = []

    for k in k_list:
        entry: dict[str, object] = {"k": k}
        for m in _TRACKED_METRICS:
            vals = [fold[k][m] for fold in fold_metrics if k in fold and m in fold[k]]
            n = len(vals)
            std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            entry[f"mean_{m}"] = float(np.mean(vals)) if n > 0 else float("nan")
            entry[f"std_{m}"] = std_val
            entry[f"se_{m}"] = std_val / np.sqrt(n) if n > 1 else 0.0
            for i, fold in enumerate(fold_metrics):
                entry[f"{m}_fold_{i + 1}"] = fold.get(k, {}).get(m, float("nan"))
        cv_results.append(entry)

    # Apply 1-SE rule on the selection metric
    means = np.array([
        float(cast("float", r[f"mean_{selection_metric}"])) for r in cv_results
    ])
    min_idx = int(np.argmin(means))
    threshold = means[min_idx] + float(
        cast("float", cv_results[min_idx][f"se_{selection_metric}"])
    )

    for r in cv_results:
        if float(cast("float", r[f"mean_{selection_metric}"])) <= threshold:
            return int(cast("int", r["k"])), cv_results

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

    All tracked metrics (rms, prms, cost) are recorded per fold regardless
    of which metric is used for selection, so callers can compare selection
    criteria without re-running.

    Args:
        x: Data matrix (dense or sparse), shape
            ``(n_features, n_samples)``.
        mask: Optional boolean mask with the same shape as ``x``.
        components: Candidate component counts.  Defaults to
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
          and ``<m>_fold_<i>`` per-fold values.

    Raises:
        ValueError: If ``metric`` is invalid, no valid ``components``
            are provided, or ``n_splits < 2``.

    Example:
        >>> best_k, cv = cross_validate_components(
        ...     X, components=range(1, 6), config=CVConfig(n_splits=5)
        ... )
    """
    cv_cfg = config or CVConfig()

    if cv_cfg.metric not in {"prms", "cost"}:
        msg = f"metric must be one of prms, cost (got {cv_cfg.metric!r})"
        raise ValueError(msg)
    if cv_cfg.n_splits < 2:
        msg = f"n_splits must be >= 2 (got {cv_cfg.n_splits})"
        raise ValueError(msg)

    # Materialize dense array (sparse support for fold creation is future work).
    x_arr: np.ndarray = (
        np.asarray(x.toarray(), dtype=float)
        if sp.issparse(x)
        else np.array(x, dtype=float)
    )
    if mask is not None:
        x_arr[~np.asarray(mask, dtype=bool)] = np.nan

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

    all_fold_metrics: list[dict[int, dict[str, float]]] = [
        _run_fold(
            fold_i=fold_i,
            x_base=x_arr,
            obs_rows=obs_rows,
            obs_cols=obs_cols,
            probe_sel=probe_sel,
            k_list=k_list,
            metric=cv_cfg.metric,
            opts=fit_opts,
            n_splits=cv_cfg.n_splits,
            verbose=verbose_level,
        )
        for fold_i, (probe_sel, _train_sel) in enumerate(folds)
    ]

    best_k, cv_results = _aggregate_cv_results(k_list, all_fold_metrics, cv_cfg.metric)

    if not cv_cfg.one_se_rule:
        means = [float(cast("float", r[f"mean_{cv_cfg.metric}"])) for r in cv_results]
        best_k = int(cast("int", cv_results[int(np.argmin(means))]["k"]))

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
