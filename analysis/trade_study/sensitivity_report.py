#!/usr/bin/env python
"""Sensitivity report for VBPCA hyperparameters under the Option A design.

Uses the existing surrogate-training table (``optionA/surrogate_train.json``)
to quantify how each tuned VBPCA hyperparameter affects:

- ``rank_mae``          : primary objective
- ``rank_under``        : missed signal (under-selection)
- ``rank_over``         : hallucinated noise (over-selection)
- ``holdout_rmse``      : reconstruction guard
- ``total_iters``       : efficiency

Outputs:
- ``analysis/results/optionA/sensitivity_summary.json``
- ``analysis/results/figures/optionA/figure_sensitivity_*.png``

The intent is to decide whether to revisit the tuned settings and which knobs
should enter a second-stage retuning (Hyperband / Optuna / Pareto) rather than
blindly re-optimizing all factors.
"""

from __future__ import annotations

import argparse
import json
import operator
import pathlib
from collections import defaultdict
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.stats import spearmanr

RESULTS = pathlib.Path("analysis/results/optionA/surrogate_train.json")
OUT_JSON = pathlib.Path("analysis/results/optionA/sensitivity_summary.json")
FIG_DIR = pathlib.Path("analysis/results/figures/optionA")

FACTORS = [
    "hp_va",
    "hp_vb",
    "hp_v",
    "va_init",
    "niter_broadprior",
    "maxiters",
    "minangle",
    "patience",
    "cfstop_rel",
    "rmsstop_window",
    "xprobe_fraction",
]
OUTCOMES = ["rank_mae", "rank_under", "rank_over", "holdout_rmse", "total_iters"]
BUCKETS = ("smallp", "trans", "large")


def _bucket(p: int) -> str:
    if p <= 30:
        return "smallp"
    if p <= 70:
        return "trans"
    return "large"


def _load_rows() -> list[dict[str, Any]]:
    if not RESULTS.exists():
        msg = f"Missing Option A results: {RESULTS}"
        raise SystemExit(msg)
    return json.loads(RESULTS.read_text())["rows"]


def _finite_pairs(
    rows: list[dict[str, Any]], factor: str, outcome: str
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        xv = r["config"].get(factor)
        yv = r["scores"].get(outcome)
        if xv is None or yv is None:
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


def summarize() -> dict[str, Any]:
    rows = _load_rows()
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bucket[_bucket(int(r["regime"]["p"]))].append(r)

    summary: dict[str, Any] = {"buckets": {}}
    for bucket in BUCKETS:
        bucket_rows = by_bucket[bucket]
        bucket_out: dict[str, Any] = {"n_rows": len(bucket_rows), "factors": {}}
        for factor in FACTORS:
            factor_out: dict[str, Any] = {}
            for outcome in OUTCOMES:
                x, y = _finite_pairs(bucket_rows, factor, outcome)
                if x.size < 8:
                    continue
                rho, pval = spearmanr(x, y)
                if np.isnan(rho):
                    continue
                bins = _quantile_bins(x)
                tercile_means = []
                for lo, hi in pairwise(bins):
                    if hi == bins[-1]:
                        mask = (x >= lo) & (x <= hi)
                    else:
                        mask = (x >= lo) & (x < hi)
                    tercile_means.append(
                        float(np.mean(y[mask])) if mask.any() else float("nan")
                    )
                factor_out[outcome] = {
                    "spearman_rho": float(rho),
                    "spearman_p": float(pval),
                    "q0": float(bins[0]),
                    "q33": float(bins[1]),
                    "q66": float(bins[2]),
                    "q100": float(bins[3]),
                    "tercile_means": tercile_means,
                }
            if factor_out:
                bucket_out["factors"][factor] = factor_out
        summary["buckets"][bucket] = bucket_out
    return summary


def _plot_spearman(summary: dict[str, Any], fmt: str) -> None:
    import matplotlib.pyplot as plt

    for outcome in ["rank_mae", "rank_under", "rank_over", "total_iters"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, bucket in zip(axes, BUCKETS, strict=True):
            facs = summary["buckets"][bucket]["factors"]
            items = [
                (factor, facs[factor][outcome]["spearman_rho"])
                for factor in facs
                if outcome in facs[factor]
            ]
            items.sort(key=lambda kv: abs(kv[1]), reverse=True)
            labels = [k for k, _ in items]
            vals = [v for _, v in items]
            colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
            ax.barh(range(len(vals)), vals, color=colors)
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(bucket)
            ax.grid(alpha=0.25, axis="x")
        fig.suptitle(
            f"Spearman sensitivity of hyperparameters vs {outcome}",
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
            ranking.append((
                factor,
                abs(out["rank_mae"]["spearman_rho"]),
                out["rank_mae"]["spearman_rho"],
            ))
        ranking.sort(key=operator.itemgetter(1), reverse=True)
        top = ranking[:3]
        if top:
            findings.append(
                f"{bucket}: top rank_mae factors = "
                + ", ".join(f"{f} (rho={rho:+.2f})" for f, _a, rho in top)
            )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmt", default="png")
    args = parser.parse_args()

    summary = summarize()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"saved {OUT_JSON}")
    _plot_spearman(summary, args.fmt)
    print("\nKey findings:")
    for line in _top_findings(summary):
        print("-", line)


if __name__ == "__main__":
    main()
