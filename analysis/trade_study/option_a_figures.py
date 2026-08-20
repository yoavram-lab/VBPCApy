#!/usr/bin/env python
"""Option A figure suite (2026-07 reframe).

Figures
-------
F1  Convergence knee: holdout RMSE/MAE/coverage vs iterations per p-bucket.
F2  Broadprior waste: final quality and iterations-to-knee vs niter_broadprior.
F3  Coverage fix: latent-only vs predictive (+sigma^2) coverage per bucket.
F5  rank_mae vs total_iters Pareto from the surrogate-training data.
F7  Regime map: surrogate-predicted rank_mae over the (n, p) plane.

F1/F2 read ``results/convergence_trace/sweep.json``; F5/F7 read
``results/optionA/``.  F3 fits a small fresh sweep.

Usage
-----
    python -m analysis.trade_study.option_a_figures --fmt png
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np

FIG_DIR = pathlib.Path("analysis/results/figures/optionA")
TRACE = pathlib.Path("analysis/results/convergence_trace/sweep.json")
OPTIONA = pathlib.Path("analysis/results/optionA")

_BUCKET_ORDER = ["smallp", "trans", "large"]
_BUCKET_COLOR = {"smallp": "#1f77b4", "trans": "#ff7f0e", "large": "#2ca02c"}


def _load_trace() -> list[dict[str, Any]] | None:
    """Load convergence-trace rows, or None if that data isn't present.

    The convergence-characterization paper (and the script that produced
    this data) moved to its own repo, jcm-sci/vbpca-convergence -- F1/F2
    skip rather than crash the whole figure suite when it's absent from
    this one.
    """
    if not TRACE.exists():
        print(
            f"  skipping -- convergence-trace data lives in "
            f"jcm-sci/vbpca-convergence now, not found at {TRACE}"
        )
        return None
    return [
        {k: v for k, v in row.items() if k != "lc"}
        for row in json.loads(TRACE.read_text())
    ]


def _median_curve(
    rows: list[dict[str, Any]], bucket: str, metric: str, bp: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Median metric vs maxiters for a bucket at a broadprior level."""
    by_iter: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["p_bucket"] == bucket and r["niter_broadprior"] == bp:
            by_iter[r["maxiters"]].append(r[metric])
    iters = np.array(sorted(by_iter))
    vals = np.array([float(np.median(by_iter[i])) for i in iters])
    return iters, vals


def fig_f1_knee(fmt: str) -> None:
    """F1 — quality vs iterations per p-bucket."""
    import matplotlib.pyplot as plt

    rows = _load_trace()
    if rows is None:
        return
    metrics = [
        ("holdout_rmse", "Holdout RMSE", False),
        ("holdout_mae", "Holdout MAE", False),
        ("coverage_95", "Coverage @ 95%", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (metric, label, is_cov) in zip(axes, metrics, strict=True):
        for bucket in _BUCKET_ORDER:
            iters, vals = _median_curve(rows, bucket, metric, bp=0)
            if iters.size:
                ax.plot(
                    iters,
                    vals,
                    "o-",
                    color=_BUCKET_COLOR[bucket],
                    label=bucket,
                    markersize=4,
                )
        ax.set_xscale("log")
        ax.set_xlabel("maxiters (log)")
        ax.set_ylabel(label)
        ax.axvspan(1, 20, alpha=0.08, color="gray")
        if is_cov:
            ax.axhline(0.95, ls="--", color="k", lw=0.8, alpha=0.6)
        ax.grid(alpha=0.3)
    axes[0].legend(title="p-bucket", fontsize=8)
    fig.suptitle(
        "F1 — Quality converges by ~15-20 iterations (shaded) in every regime",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "figure_f1_knee", fmt)


def fig_f2_broadprior(fmt: str) -> None:
    """F2 — broadprior adds iterations without improving quality."""
    import matplotlib.pyplot as plt

    rows = _load_trace()
    if rows is None:
        return
    bps = sorted({r["niter_broadprior"] for r in rows})
    fig, (ax_q, ax_i) = plt.subplots(1, 2, figsize=(11, 4.2))

    for bucket in _BUCKET_ORDER:
        finals, iters_used = [], []
        for bp in bps:
            sub = [
                r
                for r in rows
                if r["p_bucket"] == bucket and r["niter_broadprior"] == bp
            ]
            if not sub:
                continue
            max_it = max(r["maxiters"] for r in sub)
            fin = [r["holdout_rmse"] for r in sub if r["maxiters"] == max_it]
            finals.append(float(np.median(fin)))
            iters_used.append(float(np.median([r["n_iter"] for r in sub])))
        ax_q.plot(
            bps[: len(finals)], finals, "o-", color=_BUCKET_COLOR[bucket], label=bucket
        )
        ax_i.plot(
            bps[: len(iters_used)],
            iters_used,
            "o-",
            color=_BUCKET_COLOR[bucket],
            label=bucket,
        )

    ax_q.set_xlabel("niter_broadprior")
    ax_q.set_ylabel("Final holdout RMSE")
    ax_q.set_title("Quality is flat in broadprior")
    ax_q.grid(alpha=0.3)
    ax_q.legend(title="p-bucket", fontsize=8)
    ax_i.set_xlabel("niter_broadprior")
    ax_i.set_ylabel("Median iterations run")
    ax_i.set_title("Iterations rise with broadprior (pure cost)")
    ax_i.grid(alpha=0.3)
    fig.suptitle(
        "F2 — broadprior only adds compute; default 0 recommended",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "figure_f2_broadprior", fmt)


def fig_f3_coverage(fmt: str) -> None:
    """F3 — coverage: latent-only vs predictive (+sigma^2)."""
    import matplotlib.pyplot as plt

    from vbpca_py import VBPCA

    from ._common import (
        HOLDOUT_FRACTION,
        apply_missingness,
        generate_low_rank,
        holdout_split,
    )

    regimes = [
        (
            "smallp",
            {
                "n": 50,
                "p": 10,
                "true_rank": 2,
                "missingness": "complete",
                "noise_std": 0.3,
            },
        ),
        (
            "trans",
            {
                "n": 100,
                "p": 50,
                "true_rank": 5,
                "missingness": "mcar",
                "noise_std": 0.5,
            },
        ),
        (
            "large",
            {
                "n": 100,
                "p": 100,
                "true_rank": 5,
                "missingness": "complete",
                "noise_std": 0.5,
            },
        ),
        (
            "large",
            {
                "n": 200,
                "p": 200,
                "true_rank": 10,
                "missingness": "block",
                "noise_std": 1.0,
            },
        ),
    ]

    def _cov(x, xr, var, ho, extra=0.0):
        std = np.sqrt(np.maximum(var + extra, 0.0))
        inside = (x >= xr - 1.96 * std) & (x <= xr + 1.96 * std)
        return float(inside[ho].mean())

    labels, lat, pred = [], [], []
    for bucket, reg in regimes:
        cl, cp = [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            x = generate_low_rank(
                reg["n"], reg["p"], reg["true_rank"], reg["noise_std"], rng
            )
            mask = apply_missingness(x, reg["missingness"], rng)
            tr, ho = holdout_split(mask, HOLDOUT_FRACTION, rng)
            m = VBPCA(
                n_components=reg["true_rank"],
                maxiters=100,
                niter_broadprior=0,
                verbose=0,
            )
            m.fit(x, mask=tr)
            cl.append(_cov(x, m.reconstruction_, m.variance_, ho))
            cp.append(_cov(x, m.reconstruction_, m.variance_, ho, m.noise_variance_))
        labels.append(f"{reg['n']}x{reg['p']}\n{bucket}")
        lat.append(np.mean(cl) * 100)
        pred.append(np.mean(cp) * 100)

    xpos = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(xpos - w / 2, lat, w, label="latent only (variance_)", color="#d62728")
    ax.bar(xpos + w / 2, pred, w, label="predictive (+sigma^2)", color="#2ca02c")
    ax.axhline(95, ls="--", color="k", lw=0.9, label="nominal 95%")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Coverage @ 95% (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    fig.suptitle(
        "F3 — predictive_variance_ restores calibration (48-65% -> ~95%)",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "figure_f3_coverage", fmt)


def fig_f5_pareto(fmt: str) -> None:
    """F5 — rank_mae vs total_iters from the surrogate-training data."""
    import matplotlib.pyplot as plt

    train = OPTIONA / "surrogate_train.json"
    if not train.exists():
        print(f"  [skip F5] {train} missing (run option_a_pipeline collect)")
        return
    data = json.loads(train.read_text())
    rows = data["rows"]

    def _bucket(p: int) -> str:
        return "smallp" if p <= 30 else "trans" if p <= 70 else "large"

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for bucket in _BUCKET_ORDER:
        xs = [
            r["scores"]["total_iters"]
            for r in rows
            if _bucket(r["regime"]["p"]) == bucket
        ]
        ys = [
            r["scores"]["rank_mae"] for r in rows if _bucket(r["regime"]["p"]) == bucket
        ]
        ax.scatter(xs, ys, s=12, alpha=0.4, color=_BUCKET_COLOR[bucket], label=bucket)
    ax.set_xlabel("total_iters (efficiency)")
    ax.set_ylabel("rank_mae (primary objective)")
    ax.grid(alpha=0.3)
    ax.legend(title="p-bucket", fontsize=8)
    fig.suptitle(
        "F5 — rank_mae vs iterations (primary vs efficiency)", fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "figure_f5_pareto", fmt)


def fig_f7_regime_map(fmt: str) -> None:
    """F7 — surrogate-predicted rank_mae over the (n, p) plane."""
    import matplotlib.pyplot as plt
    from trade_study import fit_regime_surrogate

    from ._common import REGIME_FEATURES
    from ._surrogate import _design_factors, load_results_table, recommend_ranked

    train = OPTIONA / "surrogate_train.json"
    if not train.exists():
        print(f"  [skip F7] {train} missing (run option_a_pipeline collect)")
        return
    table = load_results_table(train)
    surrogate = fit_regime_surrogate(
        table,
        regime_factors=REGIME_FEATURES,
        factors=_design_factors(),
        method="rf",
        seed=42,
    )

    n_vals = [30, 50, 70, 100, 150, 200]
    p_vals = [10, 20, 30, 50, 70, 100, 200]
    grid = np.full((len(p_vals), len(n_vals)), np.nan)
    for i, p in enumerate(p_vals):
        for j, n in enumerate(n_vals):
            if p >= n:
                continue
            reg = {
                "n": n,
                "p": p,
                "true_rank": 5,
                "missingness": "mcar",
                "noise_std": 0.5,
            }
            _, pred = recommend_ranked(
                surrogate, reg, primary="rank_mae", n_candidates=512
            )
            grid[i, j] = pred.get("rank_mae", np.nan)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(n_vals)))
    ax.set_xticklabels(n_vals)
    ax.set_yticks(range(len(p_vals)))
    ax.set_yticklabels(p_vals)
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("p (features)")
    fig.colorbar(im, ax=ax, label="predicted rank_mae (recommended config)")
    fig.suptitle(
        "F7 — recommended-config rank_mae across the (n, p) plane", fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "figure_f7_regime_map", fmt)


def _save(fig: Any, name: str, fmt: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")
    import matplotlib.pyplot as plt

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmt", default="png")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Subset of figures to generate (f1 f2 f3 f5 f7).",
    )
    args = parser.parse_args()

    figs = {
        "f1": fig_f1_knee,
        "f2": fig_f2_broadprior,
        "f3": fig_f3_coverage,
        "f5": fig_f5_pareto,
        "f7": fig_f7_regime_map,
    }
    selected = args.only or list(figs)
    for key in selected:
        if key in figs:
            print(f"Generating {key.upper()} ...")
            figs[key](args.fmt)


if __name__ == "__main__":
    main()
