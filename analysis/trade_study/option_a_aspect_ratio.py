#!/usr/bin/env python
"""Option A extension: extreme aspect-ratio (p/n) regimes (#116).

``recommend_config`` buckets purely on ``p``, with no upper bound and no
use of ``n`` at all -- the Option A trade study's original regime grid
(``TRAINING_REGIMES``/``VALIDATION_REGIMES`` in ``_common.py``) only
covers ``p`` up to 200 and ``p/n`` up to 2.0. Real data commonly falls far
outside that: genomics data (bulk RNA-seq/microarray, small cohorts with
thousands of features -- ``p/n`` of 50-1000x) and ecological/survey data
(many sites or respondents, few measured variables -- ``n/p`` of 20-100x)
sit at opposite extremes neither the original grid nor its RF surrogate
were trained on.

Rather than folding these into ``TRAINING_REGIMES`` and re-running the
existing Sobol-design + ``RegimeSurrogate`` pipeline (``option_a_pipeline.py``),
this uses ``trade_study.recommend_per_regime()``/``aggregate_bucketed_config()``
(jcm-sci/trade-study#123, split in #126):

1. The design space here is a genuine mixed continuous/discrete/categorical
   *optimization* problem (minimise ``rank_mae`` subject to a
   ``holdout_rmse`` guard), not a sensitivity-analysis problem -- Sobol's
   value is low-discrepancy coverage for variance decomposition, and 6 of
   ``_design_factors()``'s 15 factors are discrete/categorical, which get
   mapped onto the continuous Sobol sequence via binning (``round_discrete_factors``
   in ``_common.py``) -- lossy for unordered categoricals like
   ``criterion_order``, and buys nothing here since no Sobol indices are
   computed downstream. Optuna's NSGA-II (what ``run_adaptive`` uses under
   the hood) suggests each factor through its own native type, so nothing
   needs binning.
2. Only 3-5 regimes per extreme is too sparse to fit a meaningful
   ``RegimeSurrogate`` (RFs extrapolate poorly outside their training
   range, which every one of these regimes is relative to the original
   grid) -- direct per-regime optimization avoids relying on cross-regime
   interpolation this data can't support.

**Search vs aggregation are two separate steps** (``search`` then
``aggregate``), not one combined call: a first ``n_reps=1`` run's
2-bucket (wide/tall) aggregation performed far worse than the individual
per-regime optima that fed it (see git history), and re-grouping into
finer buckets needed to not mean repeating an hours-long search.
``search`` runs once and caches every regime's best config; ``aggregate``
re-groups from that cache for free, any number of times, with any
``--buckets`` grouping.

**Cost note:** ``maxiters`` is capped to ``[100, 200, 500]`` here rather
than the full Stage 1/2 range's ``[100, 200, 500, 1000]``. At
``p=2000, n=30`` a single ``maxiters=1000`` fit takes ~230-300s (measured);
30 NSGA-II trials sampling that range pushed a single regime's search past
4 hours with no interim progress signal. Fit cost is roughly linear in
``maxiters``, so dropping ``1000`` bounds worst-case per-trial cost to
~90-120s without meaningfully narrowing the search (300 already covers the
Option A "large" bucket's shipped default).

``VBPCASimulator(regime_defaults=...)`` (already built for exactly this --
see its docstring, "e.g. during adaptive refinement") fixes the regime so
the search only varies the tunable factors.

Usage
-----
    python -m analysis.trade_study.option_a_aspect_ratio search --n-trials 20 --n-reps 3
    python -m analysis.trade_study.option_a_aspect_ratio aggregate --buckets wide_tall
    python -m analysis.trade_study.option_a_aspect_ratio aggregate --buckets extremity
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import replace
from typing import Any

from trade_study import (
    Direction,
    FactorType,
    Observable,
    aggregate_bucketed_config,
    recommend_per_regime,
)

from ._common import RNG_SEED
from ._surrogate import _design_factors
from ._world import VBPCAScorer, VBPCASimulator

RESULTS_DIR = pathlib.Path("analysis/results/optionA")
PER_REGIME_PATH = RESULTS_DIR / "aspect_ratio_per_regime.json"

# Worst-case-bounding cap on maxiters for this screening -- see module
# docstring's "Cost note".
_CAPPED_MAXITERS_LEVELS = [100, 200, 500]

# p >> n: small cohort, many features.
WIDE_REGIMES: dict[str, dict[str, Any]] = {
    "bulk_rnaseq": {
        "n": 30,
        "p": 2000,
        "true_rank": 5,
        "missingness": "complete",
        "noise_std": 0.5,
    },
    "microbiome": {
        "n": 50,
        "p": 300,
        "true_rank": 2,
        "missingness": "mcar",
        "noise_std": 0.5,
    },
    "single_cell": {
        "n": 500,
        "p": 500,
        "true_rank": 10,
        "missingness": "complete",
        "noise_std": 0.5,
    },
}

# n >> p: many sites/respondents, few measured variables (ecological,
# survey/cultural data).
TALL_REGIMES: dict[str, dict[str, Any]] = {
    "ecological": {
        "n": 3000,
        "p": 30,
        "true_rank": 3,
        "missingness": "complete",
        "noise_std": 0.5,
    },
    "cultural": {
        "n": 1000,
        "p": 50,
        "true_rank": 5,
        "missingness": "complete",
        "noise_std": 0.5,
    },
}

ALL_REGIMES: dict[str, dict[str, Any]] = {**WIDE_REGIMES, **TALL_REGIMES}

# Aspect-ratio thresholds separating "moderate" from "extreme" within each
# direction -- chosen from the regime set itself: bulk_rnaseq (p/n=66.7) and
# ecological (n/p=100) are an order of magnitude past the next-most-skewed
# regime in their direction (microbiome p/n=6, cultural n/p=20).
_WIDE_EXTREME_RATIO = 10.0
_TALL_EXTREME_RATIO = 50.0


def _bucket_wide_tall(name: str, _regime: dict[str, Any]) -> str:
    """2-bucket grouping: wide vs tall.

    Returns:
        ``"wide"`` or ``"tall"``.
    """
    return "wide" if name in WIDE_REGIMES else "tall"


def _bucket_by_extremity(name: str, regime: dict[str, Any]) -> str:
    """4-bucket grouping: {wide,tall} x {moderate,extreme}.

    Returns:
        One of ``"wide_moderate"``, ``"wide_extreme"``, ``"tall_moderate"``,
        ``"tall_extreme"``.
    """
    n, p = regime["n"], regime["p"]
    if name in WIDE_REGIMES:
        return "wide_extreme" if p / n > _WIDE_EXTREME_RATIO else "wide_moderate"
    return "tall_extreme" if n / p > _TALL_EXTREME_RATIO else "tall_moderate"


_BUCKET_FNS = {
    "wide_tall": _bucket_wide_tall,
    "extremity": _bucket_by_extremity,
}


def _capped_factors() -> list[Any]:
    """``_design_factors()`` with ``maxiters`` capped -- see cost note.

    Returns:
        Factor list with every entry unchanged except ``maxiters``.
    """
    factors = _design_factors()
    return [
        replace(f, levels=_CAPPED_MAXITERS_LEVELS)
        if f.name == "maxiters" and f.factor_type == FactorType.DISCRETE
        else f
        for f in factors
    ]


class _ProgressScorer:
    """Wraps VBPCAScorer to print one line per trial for background visibility.

    Shared across every regime's ``run_adaptive`` call inside
    ``recommend_per_regime`` (it only takes one scorer, not per-regime),
    so the trial count is cumulative across all regimes rather than reset
    per regime -- still enough to confirm the run is alive and roughly
    how far through it is (regimes run in ``ALL_REGIMES`` order,
    ``n_trials`` trials each).
    """

    def __init__(self) -> None:
        self._inner = VBPCAScorer()
        self._count = 0
        self._t0 = time.perf_counter()

    def score(
        self, truth: Any, observations: Any, config: dict[str, Any]
    ) -> dict[str, float]:
        """Score via the wrapped scorer, printing progress as a side effect.

        Returns:
            Same dict ``VBPCAScorer.score`` returns.
        """
        self._count += 1
        result = self._inner.score(truth, observations, config)
        elapsed = time.perf_counter() - self._t0
        print(
            f"  trial {self._count} done "
            f"(elapsed {elapsed:.0f}s, rank_mae={result.get('rank_mae')})"
        )
        return result


def search(n_trials: int = 20, n_reps: int = 3, seed: int = RNG_SEED) -> pathlib.Path:
    """Find each regime's best config and cache it -- the expensive step.

    ``n_reps>1`` (jcm-sci/trade-study#122) averages each trial's scores
    over that many data draws before NSGA-II sees them -- a first pass at
    ``n_reps=1`` found the (then directly-aggregated) bucket configs
    performed far worse than the individual per-regime "best" trials that
    fed them (rank_mae back to 4-5 for most regimes despite trials
    hitting rank_mae=0 during search), consistent with those single-draw
    "best" picks being lucky draws rather than robust optima.

    Returns:
        Path to the saved per-regime results JSON.
    """
    factors = _capped_factors()
    observables = [
        Observable("rank_mae", Direction.MINIMIZE, weight=1.0),
        Observable("holdout_rmse", Direction.MINIMIZE, weight=0.2),
    ]

    per_regime = recommend_per_regime(
        ALL_REGIMES,
        world_factory=lambda r: VBPCASimulator(regime_defaults=r),
        scorer=_ProgressScorer(),
        factors=factors,
        observables=observables,
        primary="rank_mae",
        n_trials=n_trials,
        n_reps=n_reps,
        seed=seed,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PER_REGIME_PATH.write_text(json.dumps(per_regime, indent=2))
    print(f"\nSaved -> {PER_REGIME_PATH}")
    for name, cfg in per_regime.items():
        print(f"{name}: {cfg}")
    return PER_REGIME_PATH


def aggregate(bucket_grouping: str) -> pathlib.Path:
    """Re-group ``search()``'s cached per-regime results -- the cheap step.

    Returns:
        Path to the saved bucket-config JSON.

    Raises:
        FileNotFoundError: If ``search()`` hasn't been run yet.
    """
    if not PER_REGIME_PATH.exists():
        msg = f"Run `search` first -- missing {PER_REGIME_PATH}"
        raise FileNotFoundError(msg)
    per_regime = json.loads(PER_REGIME_PATH.read_text())

    buckets = aggregate_bucketed_config(
        per_regime,
        ALL_REGIMES,
        bucket_fn=_BUCKET_FNS[bucket_grouping],
        factors=_capped_factors(),
    )

    out = RESULTS_DIR / f"aspect_ratio_recommendations_{bucket_grouping}.json"
    out.write_text(json.dumps(buckets, indent=2))
    print(f"Saved -> {out}")
    for bucket, cfg in buckets.items():
        print(f"{bucket}: {cfg}")
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="Find each regime's best config.")
    p_search.add_argument("--n-trials", type=int, default=20)
    p_search.add_argument("--n-reps", type=int, default=3)
    p_search.add_argument("--seed", type=int, default=RNG_SEED)

    p_agg = sub.add_parser("aggregate", help="Re-group cached per-regime results.")
    p_agg.add_argument("--buckets", choices=sorted(_BUCKET_FNS), default="wide_tall")

    args = parser.parse_args()
    if args.cmd == "search":
        search(n_trials=args.n_trials, n_reps=args.n_reps, seed=args.seed)
    elif args.cmd == "aggregate":
        aggregate(args.buckets)


if __name__ == "__main__":
    main()
