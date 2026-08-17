#!/usr/bin/env python
"""Option A — surrogate-first, ``rank_mae``-primary optimisation.

Pipeline (2026-07 reframe):

1. **Collect** ``(regime, config) -> scores`` over a Sobol design crossed
   with a regime grid spanning the small-p / transition / large buckets.
   Runs in parallel (``joblib``); exploits the fast convergence knee by
   letting ``maxiters`` vary as a design factor rather than fixing it high.
2. **Fit** a :class:`trade_study.RegimeSurrogate` over
   ``regime features x design factors -> observables``.
3. **Recommend** a config per regime that minimises ``rank_mae`` subject to
   RMSE/MAE guards (:func:`analysis.trade_study._surrogate.recommend_ranked`).

Coverage is *not* optimised: with ``predictive_variance_`` it is ~95%
regardless of config, so it is reported as a calibration check only.

Usage
-----
    python -m analysis.trade_study.option_a_pipeline collect --n-samples 96 --n-jobs -1
    python -m analysis.trade_study.option_a_pipeline recommend
    python -m analysis.trade_study.option_a_pipeline all --n-samples 96 --n-jobs -1
"""

from __future__ import annotations

import argparse
import json
import pathlib
from itertools import starmap
from typing import Any

from joblib import Parallel, delayed
from trade_study import build_grid, fit_regime_surrogate

from ._common import (
    REGIME_FEATURES,
    RNG_SEED,
    SMALLP_STUDY_REGIMES,
    STUDY_REGIMES,
    TRANS_STUDY_REGIMES,
    VALIDATION_REGIMES,
    round_discrete_factors,
)
from ._surrogate import _design_factors, load_results_table, recommend_ranked
from ._world import VBPCAScorer, VBPCASimulator

RESULTS_DIR = pathlib.Path("analysis/results/optionA")

_REGIME_KEYS = ("n", "p", "true_rank", "missingness", "noise_std")

# Training regimes span all three empirical p-buckets so the surrogate can
# interpolate across the (n, p) plane.
TRAINING_REGIMES: list[dict[str, Any]] = [
    *SMALLP_STUDY_REGIMES,
    *TRANS_STUDY_REGIMES,
    *STUDY_REGIMES,
]


def _one_trial(cfg: dict[str, Any], regime: dict[str, Any]) -> dict[str, Any] | None:
    """Run one (config, regime) trial; return a surrogate-training row."""
    world = VBPCASimulator()
    scorer = VBPCAScorer()
    full = {**cfg, **regime}
    try:
        truth, obs = world.generate(full)
        scores = scorer.score(truth, obs, full)
    except Exception:
        return None
    return {
        "regime": {k: regime[k] for k in _REGIME_KEYS},
        "config": {k: v for k, v in cfg.items() if k != "rmsstop"},
        "scores": {k: float(v) for k, v in scores.items()},
    }


def collect(
    n_samples: int = 96, seed: int = RNG_SEED, n_jobs: int = -1
) -> pathlib.Path:
    """Collect surrogate-training rows in parallel and save to JSON."""
    factors = _design_factors()
    grid = build_grid(factors, method="sobol", n_samples=n_samples, seed=seed)
    for cfg in grid:
        round_discrete_factors(cfg)

    tasks = [(cfg, regime) for regime in TRAINING_REGIMES for cfg in grid]
    print(f"Option A collection: {len(grid)} configs x {len(TRAINING_REGIMES)} regimes")
    print(f"  Total trials: {len(tasks)} | n_jobs={n_jobs}")

    rows_raw = Parallel(n_jobs=n_jobs, verbose=5)(starmap(delayed(_one_trial), tasks))
    rows = [r for r in rows_raw if r is not None]
    obs_names = list(rows[0]["scores"].keys()) if rows else []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "surrogate_train.json"
    out.write_text(json.dumps({"observable_names": obs_names, "rows": rows}, indent=2))
    print(f"\nSaved {len(rows)} rows ({len(rows_raw) - len(rows)} failed) -> {out}")
    return out


def recommend(train_path: pathlib.Path | None = None) -> pathlib.Path:
    """Fit the regime surrogate and recommend a config per regime."""
    train_path = train_path or (RESULTS_DIR / "surrogate_train.json")
    if not train_path.exists():
        msg = f"No training data at {train_path}; run `collect` first."
        raise SystemExit(msg)

    table = load_results_table(train_path)
    surrogate = fit_regime_surrogate(
        table,
        regime_factors=REGIME_FEATURES,
        factors=_design_factors(),
        method="rf",
        seed=RNG_SEED,
    )

    recommendations: list[dict[str, Any]] = []
    all_regimes = {
        "training": TRAINING_REGIMES,
        "validation": VALIDATION_REGIMES,
    }
    for split, regimes in all_regimes.items():
        for regime in regimes:
            reg = {k: regime[k] for k in _REGIME_KEYS}
            cfg, pred = recommend_ranked(surrogate, reg, primary="rank_mae")
            recommendations.append({
                "split": split,
                "regime": reg,
                "config": cfg,
                "predicted": pred,
            })

    out = RESULTS_DIR / "recommendations.json"
    out.write_text(json.dumps(recommendations, indent=2))

    print(f"\n{'regime':32s} {'rank_mae':>9s} {'rmse':>7s} {'iters':>7s}")
    print("-" * 60)
    for rec in recommendations:
        r = rec["regime"]
        tag = f"{r['n']}x{r['p']}_{r['missingness'][:6]}"
        p = rec["predicted"]
        print(
            f"{tag:32s} {p.get('rank_mae', float('nan')):9.3f} "
            f"{p.get('holdout_rmse', float('nan')):7.3f} "
            f"{p.get('total_iters', float('nan')):7.0f}"
        )
    print(f"\nSaved recommendations -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Collect surrogate-training data.")
    p_collect.add_argument("--n-samples", type=int, default=96)
    p_collect.add_argument("--seed", type=int, default=RNG_SEED)
    p_collect.add_argument("--n-jobs", type=int, default=-1)

    sub.add_parser("recommend", help="Fit surrogate and recommend per regime.")

    p_all = sub.add_parser("all", help="Collect then recommend.")
    p_all.add_argument("--n-samples", type=int, default=96)
    p_all.add_argument("--seed", type=int, default=RNG_SEED)
    p_all.add_argument("--n-jobs", type=int, default=-1)

    args = parser.parse_args()
    if args.cmd == "collect":
        collect(n_samples=args.n_samples, seed=args.seed, n_jobs=args.n_jobs)
    elif args.cmd == "recommend":
        recommend()
    elif args.cmd == "all":
        collect(n_samples=args.n_samples, seed=args.seed, n_jobs=args.n_jobs)
        recommend()


if __name__ == "__main__":
    main()
