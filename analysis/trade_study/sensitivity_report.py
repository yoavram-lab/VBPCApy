#!/usr/bin/env python
"""Sensitivity report for VBPCA hyperparameters under the Option A design.

Uses ``trade_study.sensitivity_from_table()`` (jcm-sci/trade-study#113)
against the existing surrogate-training table
(``optionA/surrogate_train.json``) to compute proper Sobol-via-surrogate
sensitivity, per p-bucket, for the 9 continuous VBPCA hyperparameters:
``hp_va``, ``hp_vb``, ``hp_v``, ``va_init``, ``xprobe_fraction``,
``minangle``, ``cfstop_rel``, ``rmsstop_atol``, ``rmsstop_rtol``.

This replaces the marginal Spearman correlation this script previously
computed for every factor. Unlike Spearman, this correctly detects
non-monotonic (e.g. U-shaped) effects -- exactly the kind ``defaults.py``'s
own docstring already documents for ``hp_va`` (a real nonlinear U-shape a
monotonic correlation understates, discovered by the same marginal-
correlation method this migration retires). ``surrogate_cv_r2`` is
reported alongside each bucket's indices so a poorly-fit surrogate's
sensitivity numbers aren't mistaken for a reliable finding.

Sobol/Morris both require continuous factors (``sensitivity_from_table``
silently drops anything else, matching ``screen()``'s own contract), so
the 4 *discrete* VBPCA hyperparameters (``niter_broadprior``,
``maxiters``, ``patience``, ``rmsstop_window``) keep the original
marginal-Spearman treatment -- clearly labeled as such below, since it's
outside what Sobol/Morris can cover for non-continuous factors.

Outputs:
- ``analysis/results/optionA/sensitivity_summary.json``
- ``analysis/results/figures/optionA/figure_sensitivity_*.png``

The intent is to decide whether to revisit the tuned settings and which
knobs should enter a second-stage retuning (Hyperband / Optuna / Pareto)
rather than blindly re-optimizing all factors.
"""

from __future__ import annotations

import argparse
import json
import operator
import pathlib
from itertools import pairwise
from typing import Any

import numpy as np
from trade_study import FactorType, sensitivity_from_table
from trade_study.protocols import ResultsTable

from ._common import RNG_SEED
from ._surrogate import _design_factors, load_results_table

RESULTS = pathlib.Path("analysis/results/optionA/surrogate_train.json")
OUT_JSON = pathlib.Path("analysis/results/optionA/sensitivity_summary.json")
FIG_DIR = pathlib.Path("analysis/results/figures/optionA")

OUTCOMES = ["rank_mae", "rank_under", "rank_over", "holdout_rmse", "total_iters"]
BUCKETS = ("smallp", "trans", "large")

# Below this many rows, neither a surrogate fit nor a Spearman correlation
# is trustworthy enough to report.
_MIN_ROWS = 8

_CONTINUOUS_FACTORS = [
    f for f in _design_factors() if f.factor_type == FactorType.CONTINUOUS
]
_DISCRETE_FACTOR_NAMES = [
    f.name for f in _design_factors() if f.factor_type == FactorType.DISCRETE
]


def _bucket(p: int) -> str:
    if p <= 30:
        return "smallp"
    if p <= 70:
        return "trans"
    return "large"


def _load_table() -> ResultsTable:
    if not RESULTS.exists():
        msg = f"Missing Option A results: {RESULTS}"
        raise SystemExit(msg)
    return load_results_table(RESULTS)


def _bucket_table(table: ResultsTable, bucket: str) -> ResultsTable:
    """Return the subset of ``table`` whose configs fall in ``bucket``.

    Returns:
        A new :class:`ResultsTable` restricted to rows in ``bucket``.
    """
    mask = [_bucket(int(cfg["p"])) == bucket for cfg in table.configs]
    idx = [i for i, keep in enumerate(mask) if keep]
    return ResultsTable(
        configs=[table.configs[i] for i in idx],
        scores=table.scores[idx],
        observable_names=table.observable_names,
    )


def _finite_pairs(
    table: ResultsTable, factor: str, outcome: str
) -> tuple[np.ndarray, np.ndarray]:
    if outcome not in table.observable_names:
        return np.array([]), np.array([])
    outcome_col = table.observable_names.index(outcome)
    xs, ys = [], []
    for cfg, row in zip(table.configs, table.scores, strict=True):
        xv = cfg.get(factor)
        yv = row[outcome_col]
        if xv is None:
            continue
        try:
            x = float(xv)
            y = float(yv)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _quantile_bins(x: np.ndarray) -> np.ndarray:
    q = np.quantile(x, [0.0, 1 / 3, 2 / 3, 1.0])
    # Ensure strictly non-decreasing for repeated levels.
    q[1] = max(q[1], q[0])
    q[2] = max(q[2], q[1])
    q[3] = max(q[3], q[2])
    return q


def _discrete_spearman(table: ResultsTable, factor: str) -> dict[str, Any]:
    """Marginal Spearman sensitivity for one discrete factor.

    Returns:
        Per-outcome dict of Spearman rho/p-value and tercile means, empty
        for outcomes without enough finite data.
    """
    from scipy.stats import spearmanr

    factor_out: dict[str, Any] = {}
    for outcome in OUTCOMES:
        x, y = _finite_pairs(table, factor, outcome)
        if x.size < _MIN_ROWS:
            continue
        rho, pval = spearmanr(x, y)
        if np.isnan(rho):
            continue
        bins = _quantile_bins(x)
        tercile_means = []
        for lo, hi in pairwise(bins):
            mask = (x >= lo) & (x <= hi if hi == bins[-1] else x < hi)
            tercile_means.append(
                float(np.mean(y[mask])) if mask.any() else float("nan")
            )
        factor_out[outcome] = {
            "method": "spearman_marginal",
            "spearman_rho": float(rho),
            "spearman_p": float(pval),
            "q0": float(bins[0]),
            "q33": float(bins[1]),
            "q66": float(bins[2]),
            "q100": float(bins[3]),
            "tercile_means": tercile_means,
        }
    return factor_out


def summarize(seed: int = RNG_SEED) -> dict[str, Any]:
    """Compute per-bucket sensitivity: Sobol for continuous, Spearman for discrete.

    Returns:
        Summary dict with one entry per bucket.
    """
    table = _load_table()
    summary: dict[str, Any] = {"buckets": {}}

    for bucket in BUCKETS:
        bucket_table = _bucket_table(table, bucket)
        n_rows = len(bucket_table.configs)
        bucket_out: dict[str, Any] = {"n_rows": n_rows, "factors": {}}

        if n_rows >= _MIN_ROWS:
            ts = sensitivity_from_table(
                bucket_table,
                _CONTINUOUS_FACTORS,
                method="sobol",
                seed=seed,
                warn_below_r2=None,
            )
            bucket_out["surrogate_cv_r2"] = ts.surrogate_cv_r2
            for outcome, s1_per_factor in ts.importance.items():
                for factor, s1 in zip(_CONTINUOUS_FACTORS, s1_per_factor, strict=True):
                    bucket_out["factors"].setdefault(factor.name, {})[outcome] = {
                        "method": "sobol_via_surrogate",
                        "sobol_s1": float(s1),
                    }

        for factor_name in _DISCRETE_FACTOR_NAMES:
            factor_out = _discrete_spearman(bucket_table, factor_name)
            if factor_out:
                bucket_out["factors"].setdefault(factor_name, {}).update(factor_out)

        summary["buckets"][bucket] = bucket_out
    return summary


def _plot_sobol(summary: dict[str, Any], fmt: str) -> None:
    import matplotlib.pyplot as plt

    continuous_names = [f.name for f in _CONTINUOUS_FACTORS]
    for outcome in ["rank_mae", "rank_under", "rank_over", "total_iters"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, bucket in zip(axes, BUCKETS, strict=True):
            facs = summary["buckets"][bucket]["factors"]
            items = [
                (name, facs[name][outcome]["sobol_s1"])
                for name in continuous_names
                if name in facs and outcome in facs[name]
            ]
            items.sort(key=operator.itemgetter(1), reverse=True)
            labels = [k for k, _ in items]
            vals = [v for _, v in items]
            ax.barh(range(len(vals)), vals, color="#1f77b4")
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(bucket)
            ax.grid(alpha=0.25, axis="x")
        fig.suptitle(
            f"Sobol S1 sensitivity (via surrogate) of continuous "
            f"hyperparameters vs {outcome}",
            fontweight="bold",
        )
        fig.tight_layout()
        path = FIG_DIR / f"figure_sensitivity_{outcome}.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


def _top_findings(summary: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    for bucket in BUCKETS:
        facs = summary["buckets"][bucket]["factors"]
        ranking = []
        for factor, out in facs.items():
            if "rank_mae" not in out:
                continue
            entry = out["rank_mae"]
            score = (
                entry["sobol_s1"]
                if entry["method"] == "sobol_via_surrogate"
                else abs(entry["spearman_rho"])
            )
            ranking.append((factor, score, entry["method"]))
        ranking.sort(key=operator.itemgetter(1), reverse=True)
        top = ranking[:3]
        if top:
            findings.append(
                f"{bucket}: top rank_mae factors = "
                + ", ".join(f"{f} ({m}={s:.2f})" for f, s, m in top)
            )
    return findings


def main() -> None:
    """CLI entry point for the sensitivity report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmt", default="png")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    summary = summarize(seed=args.seed)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"saved {OUT_JSON}")
    _plot_sobol(summary, args.fmt)
    print("\nKey findings:")
    for line in _top_findings(summary):
        print("-", line)


if __name__ == "__main__":
    main()
