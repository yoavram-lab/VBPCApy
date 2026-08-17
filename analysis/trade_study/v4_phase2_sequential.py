#!/usr/bin/env python
"""v4 sequential retuning: halving/hyperband -> adaptive refinement.

This phase implements the retuning workflow requested after the Option A and
v3 analyses:

1. Multi-fidelity coarse search with ``run_successive_halving`` and
   ``run_hyperband`` using ``maxiters`` as the budget axis.
2. Build a narrowed factor space from top multi-fidelity survivors.
3. Run ``run_adaptive`` (Optuna NSGA-II) at full budget in the narrowed space.
4. Re-score finalists with the full score vector and apply rank-safe guards
   (RMSE/MAE within tolerance of the best observed).

Outputs
-------
- ``analysis/results/optionA/v4_sequential_retune.json``

Usage
-----
    python -m analysis.trade_study.v4_phase2_sequential
    python -m analysis.trade_study.v4_phase2_sequential --n-samples 96 --n-adaptive 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from trade_study import (
    Direction,
    Factor,
    FactorConstraint,
    FactorType,
    Observable,
    ResultsTable,
    build_grid,
    run_adaptive,
    run_hyperband,
    run_successive_halving,
)

from ._common import (
    QUALITY_COMPOSITE_WEIGHTS_V4,
    RNG_SEED,
    SMALLP_STUDY_REGIMES,
    STAGE1_FACTORS,
    STAGE2_CONTINUOUS_FACTORS,
    STAGE2_DISCRETE_FACTORS,
    STUDY_REGIMES,
    TRANS_FACTOR_CONSTRAINTS,
    TRANS_STUDY_REGIMES,
    round_discrete_factors,
)
from ._world import VBPCAScorer, VBPCASimulator

OUT = Path("analysis/results/optionA/v4_sequential_retune.json")


def _adaptive_available() -> bool:
    """Return whether trade-study adaptive dependencies are importable.

    ``run_adaptive`` in trade-study delegates to Optuna. We check availability
    up-front to provide a clear setup command instead of failing deep in the
    pipeline.
    """
    return importlib.util.find_spec("optuna") is not None


def _dedupe_factors(factors: list[Factor]) -> list[Factor]:
    by_name: dict[str, Factor] = {}
    for factor in factors:
        by_name.setdefault(factor.name, factor)
    return list(by_name.values())


def _design_factors() -> list[Factor]:
    """Union of Stage 1 + Stage 2 design factors excluding regime keys.

    ``maxiters`` is excluded because it is used as the explicit multi-fidelity
    budget axis in this phase.
    """
    factors = _dedupe_factors([
        *STAGE1_FACTORS,
        *STAGE2_DISCRETE_FACTORS,
        *STAGE2_CONTINUOUS_FACTORS,
    ])
    keep: list[Factor] = []
    for factor in factors:
        if factor.name in {"n", "p", "true_rank", "missingness", "noise_std"}:
            continue
        if factor.name == "maxiters":
            continue
        keep.append(factor)
    return keep


def _training_regimes() -> list[dict[str, Any]]:
    """Use the Option A multi-bucket regime union for retuning."""
    regimes = [
        *SMALLP_STUDY_REGIMES,
        *TRANS_STUDY_REGIMES,
        *STUDY_REGIMES,
    ]
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for regime in regimes:
        key = json.dumps(regime, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(regime)
    return deduped


def _quality_composite_v4(scores: dict[str, float]) -> float:
    rmse = scores.get("holdout_rmse", float("inf"))
    cov_gap = 1.0 - scores.get("coverage_95", 0.0)
    rank_pen = scores.get("rank_penalty", float("inf"))
    return float(
        QUALITY_COMPOSITE_WEIGHTS_V4["holdout_rmse"] * rmse
        + QUALITY_COMPOSITE_WEIGHTS_V4["coverage_95"] * cov_gap
        + QUALITY_COMPOSITE_WEIGHTS_V4["rank_penalty"] * rank_pen
    )


@dataclass
class _BudgetedEvaluator:
    """Adapter for multi-fidelity evaluators in ``trade_study``.

    Each trial is scored as the median across the training regime set.
    """

    regimes: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self._world = VBPCASimulator()
        self._scorer = VBPCAScorer()

    def evaluate(self, config: dict[str, Any], budget: float) -> dict[str, float]:
        cfg = dict(config)
        cfg["maxiters"] = int(round(budget))
        round_discrete_factors(cfg)

        aggregated: dict[str, list[float]] = {}
        for regime in self.regimes:
            full = {**cfg, **regime}
            truth, obs = self._world.generate(full)
            scores = self._scorer.score(truth, obs, full)
            for key, val in scores.items():
                aggregated.setdefault(key, []).append(float(val))

        out = {name: float(np.median(vals)) for name, vals in aggregated.items()}
        out["quality_composite_v4"] = _quality_composite_v4(out)
        return out


@dataclass
class _AdaptiveAggregateWorld:
    """Simulator wrapper that evaluates configs at a fixed maxiters budget."""

    evaluator: _BudgetedEvaluator
    budget: int
    selection_mode: str

    def generate(self, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        scores = self.evaluator.evaluate(
            {**config, "selection_mode": self.selection_mode},
            float(self.budget),
        )
        return {}, {"scores": scores}


class _AdaptivePassThroughScorer:
    """Scorer wrapper returning the pre-aggregated score dictionary."""

    def score(
        self,
        truth: Any,
        observations: Any,
        config: dict[str, Any],
    ) -> dict[str, float]:
        _ = (truth, config)
        return dict(observations["scores"])


def _top_records(
    results: ResultsTable,
    *,
    top_k: int,
    source: str,
    full_budget: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg, score_row, meta in zip(
        results.configs, results.scores, results.metadata, strict=True
    ):
        score_map = {
            name: float(score_row[j]) for j, name in enumerate(results.observable_names)
        }
        if "quality_composite_v4" not in score_map:
            score_map["quality_composite_v4"] = _quality_composite_v4(score_map)
        rows.append({
            "source": source,
            "config": cfg,
            "scores": score_map,
            "metadata": meta,
        })

    if full_budget is not None:
        budgeted = [
            row
            for row in rows
            if int(round(float(row["metadata"].get("budget", -1)))) == full_budget
        ]
        if budgeted:
            rows = budgeted

    rows.sort(key=lambda row: row["scores"]["quality_composite_v4"])
    return rows[:top_k]


def _narrow_factors(
    base: list[Factor], survivors: list[dict[str, Any]]
) -> list[Factor]:
    """Shrink factor space around survivor values for adaptive refinement."""
    narrowed: list[Factor] = []
    for factor in base:
        vals = [cfg.get(factor.name) for cfg in survivors if factor.name in cfg]
        vals = [v for v in vals if v is not None]
        if not vals:
            narrowed.append(factor)
            continue

        if factor.factor_type == FactorType.CONTINUOUS and factor.bounds is not None:
            lo_base, hi_base = factor.bounds
            arr = np.asarray(vals, dtype=float)
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            span = max(hi - lo, 1e-6)
            pad = 0.15 * span
            lo_n = max(float(lo_base), lo - pad)
            hi_n = min(float(hi_base), hi + pad)
            if hi_n <= lo_n:
                hi_n = min(float(hi_base), lo_n + 1e-5)
            narrowed.append(
                Factor(factor.name, FactorType.CONTINUOUS, bounds=(lo_n, hi_n))
            )
            continue

        if factor.levels:
            present = [v for v in factor.levels if v in vals]
            if present:
                narrowed.append(
                    Factor(
                        factor.name,
                        factor.factor_type,
                        levels=present,
                    )
                )
                continue

        narrowed.append(factor)

    return narrowed


def _rescore_candidates(
    evaluator: _BudgetedEvaluator,
    configs: list[dict[str, Any]],
    *,
    budget: int,
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        scores = evaluator.evaluate(cfg, float(budget))
        rows.append({"source": source, "config": cfg, "scores": scores})
    rows.sort(key=lambda row: row["scores"]["quality_composite_v4"])
    return rows


def _rank_safe_recommendations(
    rows: list[dict[str, Any]],
    *,
    guard_tol: float,
    top_k: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    rmse_best = min(row["scores"]["holdout_rmse"] for row in rows)
    mae_best = min(row["scores"]["holdout_mae"] for row in rows)

    feasible = [
        row
        for row in rows
        if row["scores"]["holdout_rmse"] <= rmse_best * (1.0 + guard_tol)
        and row["scores"]["holdout_mae"] <= mae_best * (1.0 + guard_tol)
    ]
    if not feasible:
        feasible = rows

    feasible.sort(
        key=lambda row: (
            row["scores"]["quality_composite_v4"],
            row["scores"].get("rank_under", float("inf")),
            row["scores"].get("rank_over", float("inf")),
            row["scores"].get("total_iters", float("inf")),
        )
    )
    return feasible[:top_k]


def run(
    *,
    n_samples: int = 128,
    n_adaptive: int = 200,
    top_k: int = 24,
    max_budget: int = 200,
    eta: float = 3.0,
    rungs: list[int] | None = None,
    guard_tol: float = 0.05,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    factors = _design_factors()
    regimes = _training_regimes()
    evaluator = _BudgetedEvaluator(regimes)

    constraints: list[FactorConstraint] = TRANS_FACTOR_CONSTRAINTS
    coarse = build_grid(
        factors,
        method="sobol",
        n_samples=n_samples,
        seed=seed,
        constraints=constraints,
    )
    for cfg in coarse:
        round_discrete_factors(cfg)

    rung_vals = rungs or [20, 40, 80, max_budget]

    print(
        f"Sequential retune: {len(coarse)} configs, {len(regimes)} regimes, "
        f"rungs={rung_vals}, adaptive_trials={n_adaptive}"
    )

    sh = run_successive_halving(
        trials=coarse,
        sim=evaluator,
        rungs=[float(r) for r in rung_vals],
        eta=eta,
        metric="quality_composite_v4",
        mode="min",
    )

    def _factory(bracket_idx: int, n: int) -> list[dict[str, Any]]:
        grid = build_grid(
            factors,
            method="sobol",
            n_samples=max(n, 2),
            seed=seed + bracket_idx + 1,
            constraints=constraints,
        )
        for cfg in grid:
            round_discrete_factors(cfg)
        return grid[:n]

    hb = run_hyperband(
        trial_factory=_factory,
        sim=evaluator,
        max_budget=float(max_budget),
        eta=eta,
        metric="quality_composite_v4",
        mode="min",
    )

    sh_top = _top_records(
        sh, top_k=top_k, source="successive_halving", full_budget=max_budget
    )
    hb_top = _top_records(hb, top_k=top_k, source="hyperband", full_budget=max_budget)

    survivor_cfgs = [row["config"] for row in [*sh_top, *hb_top]]
    if not survivor_cfgs:
        msg = "No survivors collected from multi-fidelity stages."
        raise RuntimeError(msg)

    narrowed_factors = _narrow_factors(factors, survivor_cfgs)

    adaptive_scorer = _AdaptivePassThroughScorer()
    adaptive_observables = [
        Observable("quality_composite_v4", Direction.MINIMIZE, weight=1.0),
        Observable("total_iters", Direction.MINIMIZE, weight=1.0),
    ]

    rescored_adaptive: list[dict[str, Any]] = []
    adaptive_error: str | None = None
    try:
        for i, selection_mode in enumerate(("prms", "cost", "cv_prms"), start=1):
            adaptive_world = _AdaptiveAggregateWorld(
                evaluator=evaluator,
                budget=max_budget,
                selection_mode=selection_mode,
            )
            adaptive = run_adaptive(
                world=adaptive_world,
                scorer=adaptive_scorer,
                factors=narrowed_factors,
                observables=adaptive_observables,
                n_trials=n_adaptive,
                seed=seed + i,
            )

            adaptive_cfgs = [
                {**cfg, "selection_mode": selection_mode} for cfg in adaptive.configs
            ]
            rescored_adaptive.extend(
                _rescore_candidates(
                    evaluator,
                    adaptive_cfgs,
                    budget=max_budget,
                    source=f"adaptive_{selection_mode}",
                )
            )
    except ModuleNotFoundError as exc:
        adaptive_error = (
            "Adaptive stage skipped because Optuna is not available in the "
            f"current environment: {exc}"
        )
        print(f"[warn] {adaptive_error}")

    combined = [
        *[
            {"source": row["source"], "config": row["config"], "scores": row["scores"]}
            for row in sh_top
        ],
        *[
            {"source": row["source"], "config": row["config"], "scores": row["scores"]}
            for row in hb_top
        ],
        *rescored_adaptive,
    ]

    dedup: dict[str, dict[str, Any]] = {}
    for row in combined:
        key = json.dumps(row["config"], sort_keys=True, default=str)
        best = dedup.get(key)
        if (
            best is None
            or row["scores"]["quality_composite_v4"]
            < best["scores"]["quality_composite_v4"]
        ):
            dedup[key] = row
    unique_rows = list(dedup.values())

    final_top = _rank_safe_recommendations(
        unique_rows,
        guard_tol=guard_tol,
        top_k=10,
    )

    out: dict[str, Any] = {
        "n_samples": n_samples,
        "n_adaptive": n_adaptive,
        "top_k": top_k,
        "max_budget": max_budget,
        "eta": eta,
        "rungs": rung_vals,
        "guard_tol": guard_tol,
        "n_regimes": len(regimes),
        "n_factors": len(factors),
        "n_narrowed_factors": len(narrowed_factors),
        "adaptive_error": adaptive_error,
        "successive_halving_top": sh_top,
        "hyperband_top": hb_top,
        "adaptive_top": rescored_adaptive[:top_k],
        "final_recommendations": final_top,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"Saved {OUT}")
    print("Top recommendations:")
    for idx, row in enumerate(final_top[:5], start=1):
        scores = row["scores"]
        print(
            f"  #{idx} [{row['source']}] qc={scores['quality_composite_v4']:.4f} "
            f"rmse={scores['holdout_rmse']:.4f} mae={scores['holdout_mae']:.4f} "
            f"under={scores.get('rank_under', float('nan')):.3f} "
            f"over={scores.get('rank_over', float('nan')):.3f} "
            f"iters={scores.get('total_iters', float('nan')):.1f}"
        )
    return out


def _parse_rungs(raw: str) -> list[int]:
    vals = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if len(vals) < 2:
        msg = "--rungs must include at least two comma-separated values"
        raise ValueError(msg)
    return sorted(set(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--n-adaptive", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--max-budget", type=int, default=200)
    parser.add_argument("--eta", type=float, default=3.0)
    parser.add_argument("--rungs", type=str, default="20,40,80,200")
    parser.add_argument("--guard-tol", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    if not _adaptive_available():
        print(
            "[warn] Optuna is not installed; adaptive refinement will be skipped.\n"
            "      Install with:\n"
            "      uv pip install -p .venv/bin/python "
            "-e /home/jcmacdo/Documents/GitHub/jcm-sci/trade-study[adaptive]"
        )

    rung_vals = _parse_rungs(args.rungs)
    if args.max_budget < rung_vals[-1]:
        msg = "--max-budget must be >= max(--rungs)"
        raise SystemExit(msg)

    run(
        n_samples=args.n_samples,
        n_adaptive=args.n_adaptive,
        top_k=args.top_k,
        max_budget=args.max_budget,
        eta=args.eta,
        rungs=rung_vals,
        guard_tol=args.guard_tol,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
