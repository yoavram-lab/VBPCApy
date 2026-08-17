"""v4 regime-surrogate helpers.

Builds a :class:`trade_study.RegimeSurrogate` from per-(regime, config)
training rows produced by :mod:`analysis.trade_study.option_a_pipeline`,
and provides ``recommend_config`` for use as a drop-in replacement for the
hard-bucket :func:`analysis.trade_study.v3_compare._gate_v3_config`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from trade_study import (
    Factor,
    FactorType,
    RegimeSurrogate,
    ResultsTable,
    build_grid,
    fit_regime_surrogate,
)

from ._common import (
    REGIME_FEATURES,
    STAGE1_FACTORS,
    STAGE2_CONTINUOUS_FACTORS,
    STAGE2_DISCRETE_FACTORS,
)


def _design_factors() -> list[Factor]:
    """Return the union of Stage 1 + Stage 2 design factors.

    Note: regime features (``n``, ``p``, ``true_rank``, ``missingness``,
    ``noise_std``) are *not* included here — they are passed separately
    as the ``regime_factors`` argument to ``fit_regime_surrogate``.
    """
    by_name: dict[str, Factor] = {}
    for f in (*STAGE1_FACTORS, *STAGE2_DISCRETE_FACTORS, *STAGE2_CONTINUOUS_FACTORS):
        if f.name in {"n", "p", "true_rank", "missingness", "noise_std"}:
            continue
        by_name.setdefault(f.name, f)
    return list(by_name.values())


def _normalise_value(factor: Factor, val: Any) -> Any:
    if factor.factor_type == FactorType.CONTINUOUS:
        return float(val)
    if factor.factor_type == FactorType.DISCRETE:
        try:
            return int(val)
        except (TypeError, ValueError):
            return val
    return val


def _coerce_config(
    raw: dict[str, Any],
    all_factors: list[Factor],
) -> dict[str, Any]:
    """Project *raw* onto ``all_factors`` with type coercion.

    Missing keys default to the lower bound (continuous) or first level
    (discrete/categorical) so that the surrogate encoder always sees a
    valid value.
    """
    out: dict[str, Any] = {}
    for f in all_factors:
        if f.name in raw and raw[f.name] is not None:
            out[f.name] = _normalise_value(f, raw[f.name])
        elif f.factor_type == FactorType.CONTINUOUS and f.bounds is not None:
            out[f.name] = float(f.bounds[0])
        elif f.levels:
            out[f.name] = f.levels[0]
        else:
            out[f.name] = 0.0
    return out


def load_results_table(path: str | Path) -> ResultsTable:
    """Load per-trial rows produced by :mod:`v4_phase_surrogate_train`.

    The on-disk format is a JSON document with keys ``observable_names``
    and ``rows``, where each row is ``{"config": {...},
    "regime": {...}, "scores": {name: float, ...}}``.

    Returns a :class:`trade_study.ResultsTable` whose ``configs`` are
    the merged ``regime ∪ config`` dicts.
    """
    data = json.loads(Path(path).read_text())
    obs_names: list[str] = list(data["observable_names"])
    all_factors = [*REGIME_FEATURES, *_design_factors()]

    configs: list[dict[str, Any]] = []
    score_rows: list[list[float]] = []
    for row in data["rows"]:
        merged = {**row.get("regime", {}), **row.get("config", {})}
        cfg = _coerce_config(merged, all_factors)
        configs.append(cfg)
        score_rows.append([
            float(row["scores"].get(name, float("nan"))) for name in obs_names
        ])

    if not configs:
        msg = f"load_results_table: no rows in {path}"
        raise ValueError(msg)

    return ResultsTable(
        configs=configs,
        scores=np.asarray(score_rows, dtype=np.float64),
        observable_names=obs_names,
        metadata=[{} for _ in configs],
    )


def fit_v4_surrogate(
    results_path: str | Path,
    *,
    method: str = "rf",
    seed: int = 0,
) -> RegimeSurrogate:
    """Fit a :class:`RegimeSurrogate` from a saved per-trial JSON.

    ``method='rf'`` is the default because a Random Forest tolerates
    categorical missingness and handles the mixed factor space without
    the GP's stationarity assumptions. Use ``method='gp'`` to get
    calibrated ``uncertainty()`` predictions.
    """
    table = load_results_table(results_path)
    return fit_regime_surrogate(
        table,
        regime_factors=REGIME_FEATURES,
        factors=_design_factors(),
        method=method,
        seed=seed,
    )


def recommend_config(
    surrogate: RegimeSurrogate,
    regime: dict[str, Any],
    *,
    objective: str = "quality_composite_v4",
    n_candidates: int = 1024,
    seed: int = 0,
) -> dict[str, Any]:
    """Recommend a design-factor config at a query regime.

    Falls back to ``holdout_rmse`` if *objective* was not fitted.
    """
    objs = surrogate.inner.observable_names
    if objective not in objs:
        objective = "holdout_rmse" if "holdout_rmse" in objs else objs[0]
    return surrogate.recommend(
        regime=regime,
        objective=objective,
        mode="min",
        n_candidates=n_candidates,
        seed=seed,
    )


def recommend_ranked(
    surrogate: RegimeSurrogate,
    regime: dict[str, Any],
    *,
    primary: str = "rank_mae",
    guards: tuple[str, ...] = ("holdout_rmse", "holdout_mae"),
    guard_tol: float = 0.05,
    tiebreak: str = "total_iters",
    n_candidates: int = 2048,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Recommend a config minimising *primary* under quality guards.

    This implements the reframed objective (2026-07): ``rank_mae`` is the
    primary target, but a candidate is only feasible if its predicted
    ``guards`` (reconstruction error) are within ``guard_tol`` of the best
    value achievable in the candidate pool.  Among feasible candidates the
    one with the lowest predicted *primary* wins, ties broken by *tiebreak*.

    Args:
        surrogate: A fitted :class:`RegimeSurrogate`.
        regime: Mapping of regime-feature names to values.
        primary: Observable to minimise (default ``"rank_mae"``).
        guards: Observables that must not regress beyond ``guard_tol``
            relative to the pool-best (default RMSE and MAE).
        guard_tol: Relative tolerance for the guards (default 5%).
        tiebreak: Observable used to break ties (default ``"total_iters"``).
        n_candidates: Size of the Sobol candidate pool.
        seed: Seed for the Sobol sampler.

    Returns:
        Tuple ``(config, predicted_scores)`` for the recommended config.

    Raises:
        ValueError: If *primary* is not a fitted observable.
    """
    names = surrogate.inner.observable_names
    if primary not in names:
        msg = f"primary {primary!r} not in fitted observables: {names}"
        raise ValueError(msg)

    pool = build_grid(
        _design_factors(), method="sobol", n_samples=n_candidates, seed=seed
    )
    preds = surrogate.predict_batch(regime, pool)
    primary_arr = np.asarray(preds[primary], dtype=float)

    feasible = np.ones(len(pool), dtype=bool)
    for g in guards:
        if g in preds:
            g_arr = np.asarray(preds[g], dtype=float)
            g_best = float(np.nanmin(g_arr))
            feasible &= g_arr <= g_best * (1.0 + guard_tol)
    if not feasible.any():
        # No candidate satisfies all guards; relax to primary-only ranking.
        feasible = np.ones(len(pool), dtype=bool)

    tie_arr = (
        np.asarray(preds[tiebreak], dtype=float)
        if tiebreak in preds
        else np.zeros(len(pool), dtype=float)
    )
    idxs = np.where(feasible)[0]
    best = min(idxs, key=lambda i: (primary_arr[i], tie_arr[i]))
    scores = {name: float(np.asarray(preds[name])[best]) for name in preds}
    return dict(pool[best]), scores
