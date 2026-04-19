"""Model-selection stability analysis across stopping metrics and missingness.

Generates synthetic low-rank data and runs ``select_n_components`` with each
metric (cost, prms) over a grid of (n, p, true_rank) settings under four
missingness patterns (complete, mcar, mnar_censored, block).

Produces nine figures for the JOSS paper:

1. **Figure 1 -- Model Selection Accuracy**: 2x4 exact-rate heatmaps
   comparing VBPCApy (cost) vs sklearn PCA (EVR95) across missingness.
2. **Figure 2 -- Error Structure**: over/under rates, MAE, selected vs true.
3. **Figure 3 -- Detection Power**: power_rate vs true_rank + overall bars.
4. **Figure 4 -- Posterior Predictive Coverage**: calibration curves + heatmap
   + interval width.
5. **Figure 5 -- Holdout RMSE**: VBPCApy vs impute+PCA reconstruction error.
6. **Figure 6 -- RMSE Improvement Heatmap**: % improvement by (n, p).
7. **Figure 7 -- Accuracy-Coverage Pareto Front**: tradeoff by (n, p).
8. **Figure 8 -- MAE Heatmap**: VBPCApy cost MAE by (n, p).
9. **Figure 9 -- ∆MAE Heatmap**: MAE advantage of VBPCApy over EVR95.

Results are also saved as JSON (and Parquet when pandas is available).

Usage::

    python analysis/stability_analysis.py                # full grid
    python analysis/stability_analysis.py --smoke        # fast CI check
    python analysis/stability_analysis.py --fmt pdf      # publication quality
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import pathlib
from collections import defaultdict
from dataclasses import asdict, dataclass

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from vbpca_py import VBPCA, SelectionConfig, select_n_components

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid definitions — n and p chosen to match pp-eigentest regime
# ---------------------------------------------------------------------------

FULL_GRID: dict[str, list[int]] = {
    "n": [20, 30, 50, 70, 100, 150, 200],
    "p": [10, 20, 30, 50, 70, 100, 200],
    "true_rank": [2, 5, 10],
}
SMOKE_GRID: dict[str, list[int]] = {
    "n": [20],
    "p": [10],
    "true_rank": [2],
}
METRICS: list[str] = ["cost", "prms"]
MISSINGNESS: list[str] = ["complete", "mcar", "mnar_censored", "block"]
MISS_FRACTION = 0.15
NOISE_STD = 0.5
REPS_FULL = 10
REPS_SMOKE = 1
MAXITERS = 200

# Pretty labels for figures
_MISS_LABEL: dict[str, str] = {
    "complete": "Complete",
    "mcar": "MCAR",
    "mnar_censored": "MNAR",
    "block": "Block",
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


@dataclass
class _Trial:
    n: int
    p: int
    true_rank: int
    metric: str
    missingness: str
    rep: int
    selected_k: int
    method: str  # "vbpca" or "sklearn_pca"
    # Per-run selection indicators
    over: float
    under: float
    exact: float
    power: float
    abs_error: float


def summarize_selection(true_rank: int, selected: int) -> dict[str, float]:
    """Return per-run selection indicators (ported from pp-eigentest)."""
    over = selected > true_rank
    under = selected < true_rank
    exact = selected == true_rank
    power = true_rank > 0 and selected >= true_rank
    abs_error = abs(selected - true_rank)
    return {
        "over": float(over),
        "under": float(under),
        "exact": float(exact),
        "power": float(power),
        "abs_error": float(abs_error),
    }


def _generate_low_rank(
    n: int, p: int, rank: int, rng: np.random.Generator
) -> np.ndarray:
    """Return a (n, p) low-rank-plus-noise matrix."""
    w = rng.standard_normal((n, rank))
    s = rng.standard_normal((rank, p))
    return w @ s + NOISE_STD * rng.standard_normal((n, p))


def _apply_missingness(
    x: np.ndarray,
    pattern: str,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Apply a missingness pattern to *x*.

    Returns:
        Boolean mask (True = observed) or None for complete data.
        Modifies *x* in-place by setting missing entries to NaN.

    Raises:
        ValueError: If *pattern* is not recognised.
    """
    if pattern == "complete":
        return None

    n, p = x.shape
    mask = np.ones((n, p), dtype=bool)

    if pattern == "mcar":
        miss = rng.random((n, p)) < MISS_FRACTION
        mask[miss] = False

    elif pattern == "mnar_censored":
        # Entries below per-column percentile are missing (detection limit).
        for j in range(p):
            threshold = np.percentile(x[:, j], MISS_FRACTION * 100)
            below = x[:, j] < threshold
            mask[below, j] = False

    elif pattern == "block":
        # Random contiguous block covering ~MISS_FRACTION of entries.
        total_miss = max(1, int(n * p * MISS_FRACTION))
        block_h = max(1, int(np.sqrt(total_miss * n / p)))
        block_w = max(1, int(total_miss / block_h))
        block_h = min(block_h, n)
        block_w = min(block_w, p)
        r0 = rng.integers(0, max(1, n - block_h + 1))
        c0 = rng.integers(0, max(1, p - block_w + 1))
        mask[r0 : r0 + block_h, c0 : c0 + block_w] = False

    else:
        msg = f"Unknown missingness pattern: {pattern!r}"
        raise ValueError(msg)

    # Ensure at least one observed entry per row and column.
    for i in range(n):
        if not mask[i].any():
            mask[i, rng.integers(0, p)] = True
    for j in range(p):
        if not mask[:, j].any():
            mask[rng.integers(0, n), j] = True

    x[~mask] = np.nan
    return mask


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def _run_grid(grid: dict[str, list[int]], reps: int, seed: int = 42) -> list[_Trial]:
    """Run select_n_components across the full parameter grid.

    Returns:
        List of trial results.
    """
    rng = np.random.default_rng(seed)
    trials: list[_Trial] = []
    settings = list(itertools.product(grid["n"], grid["p"], grid["true_rank"]))
    total = len(settings) * len(METRICS) * len(MISSINGNESS) * reps
    LOGGER.info(
        "Running %d trials (%d settings x %d metrics x %d missingness x %d reps)",
        total,
        len(settings),
        len(METRICS),
        len(MISSINGNESS),
        reps,
    )

    for idx, (n, p, true_rank) in enumerate(settings):
        if true_rank >= min(n, p):
            continue
        for rep in range(reps):
            x_clean = _generate_low_rank(n, p, true_rank, rng)
            max_k = min(true_rank + 6, min(n, p) - 1)
            candidates = list(range(1, max_k + 1))

            for miss_pattern in MISSINGNESS:
                x = x_clean.copy()
                mask = _apply_missingness(x, miss_pattern, rng)

                for metric in METRICS:
                    cfg = SelectionConfig(
                        metric=metric,  # type: ignore[arg-type]
                        patience=2,
                        max_trials=len(candidates),
                        compute_explained_variance=False,
                    )
                    opts: dict[str, object] = {
                        "maxiters": MAXITERS,
                        "verbose": 0,
                    }
                    best_k, _metrics, _trace, _ = select_n_components(
                        x, mask=mask, components=candidates, config=cfg, **opts
                    )
                    sel = summarize_selection(true_rank, best_k)
                    trials.append(
                        _Trial(
                            n=n,
                            p=p,
                            true_rank=true_rank,
                            metric=metric,
                            missingness=miss_pattern,
                            rep=rep,
                            selected_k=best_k,
                            method="vbpca",
                            **sel,
                        )
                    )

                # --- sklearn PCA baseline (once per rep x missingness) ---
                x_imp = x.copy()
                if mask is not None:
                    imp = SimpleImputer(strategy="mean")
                    x_imp = imp.fit_transform(x_imp)
                pca = PCA(n_components=max(candidates))
                pca.fit(x_imp)
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                sklearn_k = int(np.searchsorted(cumvar, 0.95) + 1)
                sel_sk = summarize_selection(true_rank, sklearn_k)
                trials.append(
                    _Trial(
                        n=n,
                        p=p,
                        true_rank=true_rank,
                        metric="evr95",
                        missingness=miss_pattern,
                        rep=rep,
                        selected_k=sklearn_k,
                        method="sklearn_pca",
                        **sel_sk,
                    )
                )

        done = idx + 1
        LOGGER.info(
            "  [%d/%d] (n=%d, p=%d, rank=%d) complete",
            done,
            len(settings),
            n,
            p,
            true_rank,
        )
    return trials


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _rate_matrix(
    trials: list[_Trial],
    field: str,
    metric: str,
    missingness: str,
    method: str = "vbpca",
) -> tuple[np.ndarray, list[int], list[int]]:
    """Compute per-(n, p) rate for *field*, filtered by metric and missingness.

    Returns:
        Tuple of (matrix, ns, ps).
    """
    vals: dict[tuple[int, int], list[float]] = defaultdict(list)
    for t in trials:
        if t.metric != metric or t.missingness != missingness:
            continue
        if t.method != method:
            continue
        vals[t.n, t.p].append(getattr(t, field))

    ns = sorted({k[0] for k in vals})
    ps = sorted({k[1] for k in vals})
    mat = np.full((len(ns), len(ps)), np.nan)
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            v = vals.get((n, p), [])
            if v:
                mat[i, j] = float(np.mean(v))
    return mat, ns, ps


def _aggregate_by_group(
    trials: list[_Trial],
    field: str,
    group_key: str,
    method: str = "vbpca",
) -> dict[str, float]:
    """Mean of *field* grouped by *group_key* (metric or missingness).

    Returns:
        Dict mapping group key to mean value.
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for t in trials:
        if t.method != method:
            continue
        key = getattr(t, group_key)
        buckets[key].append(getattr(t, field))
    return {k: float(np.mean(v)) for k, v in sorted(buckets.items())}


def _aggregate_by_rank(
    trials: list[_Trial],
    field: str,
    metric: str | None = None,
    missingness: str | None = None,
    method: str = "vbpca",
) -> tuple[list[int], list[float], list[float]]:
    """Mean +/- std of *field* by true_rank, optionally filtered.

    Returns:
        Tuple of (ranks, means, stds).
    """
    buckets: dict[int, list[float]] = defaultdict(list)
    for t in trials:
        if t.method != method:
            continue
        if metric is not None and t.metric != metric:
            continue
        if missingness is not None and t.missingness != missingness:
            continue
        buckets[t.true_rank].append(getattr(t, field))
    ranks = sorted(buckets)
    means = [float(np.mean(buckets[r])) for r in ranks]
    stds = [float(np.std(buckets[r])) for r in ranks]
    return ranks, means, stds


# ---------------------------------------------------------------------------
# Coverage sweep — posterior predictive calibration
# ---------------------------------------------------------------------------

COVERAGE_NOMINALS = [0.50, 0.80, 0.90, 0.95, 0.99]
HOLDOUT_FRACTION = 0.10  # fraction of observed entries held out


@dataclass
class _CoverageTrial:
    n: int
    p: int
    true_rank: int
    missingness: str
    rep: int
    nominal: float
    coverage: float
    holdout_rmse: float
    baseline_rmse: float = 0.0  # impute+PCA holdout RMSE
    mean_interval_width: float = 0.0  # mean 2*z*sqrt(var) at this nominal


def _run_single_coverage_trial(
    n: int,
    p: int,
    true_rank: int,
    miss_pattern: str,
    rep: int,
    x_clean: np.ndarray,
    rng: np.random.Generator,
    stats: object,
) -> list[_CoverageTrial]:
    """Fit one VBPCA model and compute coverage at all nominals.

    Returns:
        List of coverage trial results, one per nominal level.
    """
    from typing import Any

    x = x_clean.copy()
    mask = _apply_missingness(x, miss_pattern, rng)

    # Create hold-out set from observed entries
    obs_mask = np.ones((n, p), dtype=bool) if mask is None else mask.copy()
    obs_indices = np.argwhere(obs_mask)
    n_holdout = max(1, int(len(obs_indices) * HOLDOUT_FRACTION))
    holdout_idx = rng.choice(len(obs_indices), size=n_holdout, replace=False)
    holdout_coords = obs_indices[holdout_idx]
    holdout_vals = x_clean[holdout_coords[:, 0], holdout_coords[:, 1]]

    # Build training mask with holdout removed
    train_mask = obs_mask.copy()
    train_mask[holdout_coords[:, 0], holdout_coords[:, 1]] = False
    x_train = x_clean.copy()
    x_train[~train_mask] = np.nan

    # Ensure every row/col has at least one observed entry
    for i in range(n):
        if not train_mask[i].any():
            j = rng.integers(0, p)
            train_mask[i, j] = True
            x_train[i, j] = x_clean[i, j]
    for j in range(p):
        if not train_mask[:, j].any():
            i = rng.integers(0, n)
            train_mask[i, j] = True
            x_train[i, j] = x_clean[i, j]

    # Fit VBPCA at true rank
    model = VBPCA(n_components=true_rank, maxiters=MAXITERS, verbose=0)
    model.fit(x_train, mask=train_mask)

    # Get predictions and variances at held-out entries
    xrec = model.reconstruction_
    vr = model.variance_
    if xrec is None or vr is None:
        LOGGER.warning(
            "Missing reconstruction/variance for n=%d p=%d rank=%d %s rep=%d; skipping",
            n,
            p,
            true_rank,
            miss_pattern,
            rep,
        )
        return []

    pred = xrec[holdout_coords[:, 0], holdout_coords[:, 1]]
    var = vr[holdout_coords[:, 0], holdout_coords[:, 1]]
    residuals = np.abs(holdout_vals - pred)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # sklearn impute+PCA baseline RMSE on same holdout set
    x_bl = x_clean.copy()
    x_bl[~train_mask] = np.nan
    imp = SimpleImputer(strategy="mean")
    x_imp = imp.fit_transform(x_bl)
    pca = PCA(n_components=true_rank)
    scores = pca.fit_transform(x_imp)
    x_pca = scores @ pca.components_ + pca.mean_
    bl_pred = x_pca[holdout_coords[:, 0], holdout_coords[:, 1]]
    bl_rmse = float(np.sqrt(np.mean((holdout_vals - bl_pred) ** 2)))

    std = np.sqrt(np.maximum(var, 1e-12))
    stats_mod: Any = stats
    results: list[_CoverageTrial] = []
    for nominal in COVERAGE_NOMINALS:
        z = stats_mod.norm.ppf(0.5 + nominal / 2.0)
        half_width = z * std
        covered = residuals <= half_width
        cov_rate = float(np.mean(covered))
        width = float(np.mean(2.0 * half_width))
        results.append(
            _CoverageTrial(
                n=n,
                p=p,
                true_rank=true_rank,
                missingness=miss_pattern,
                rep=rep,
                nominal=nominal,
                coverage=cov_rate,
                holdout_rmse=rmse,
                baseline_rmse=bl_rmse,
                mean_interval_width=width,
            )
        )
    return results


def _run_coverage_grid(
    grid: dict[str, list[int]], reps: int, seed: int = 42
) -> list[_CoverageTrial]:
    """Fit VBPCA at true_rank, compute posterior coverage on held-out entries.

    Returns:
        List of coverage trial results.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    results: list[_CoverageTrial] = []
    settings = list(itertools.product(grid["n"], grid["p"], grid["true_rank"]))
    valid_settings = [(n, p, r) for n, p, r in settings if r < min(n, p)]
    total_fits = len(valid_settings) * len(MISSINGNESS) * reps
    LOGGER.info(
        "Coverage sweep: %d settings x %d miss x %d reps = %d fits",
        len(valid_settings),
        len(MISSINGNESS),
        reps,
        total_fits,
    )

    fit_count = 0
    for n, p, true_rank in valid_settings:
        for rep in range(reps):
            x_clean = _generate_low_rank(n, p, true_rank, rng)

            for miss_pattern in MISSINGNESS:
                fit_count += 1
                LOGGER.info(
                    "  [%d/%d] n=%d p=%d rank=%d %s rep=%d",
                    fit_count,
                    total_fits,
                    n,
                    p,
                    true_rank,
                    miss_pattern,
                    rep,
                )
                trials = _run_single_coverage_trial(
                    n,
                    p,
                    true_rank,
                    miss_pattern,
                    rep,
                    x_clean,
                    rng,
                    stats,
                )
                results.extend(trials)

    LOGGER.info("Coverage sweep done: %d results", len(results))
    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _annotate_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    *,
    fmt: str = "{:.0%}",
    threshold: float = 0.5,
    fontsize: int = 8,
) -> None:
    """Annotate each cell of a heatmap with its value."""
    nrows, ncols = mat.shape
    for i in range(nrows):
        for j in range(ncols):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    fmt.format(val),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="white" if val < threshold else "black",
                )


# ---------------------------------------------------------------------------
# Figure 1 — Model Selection Accuracy (exact_rate heatmaps)
# ---------------------------------------------------------------------------


def plot_figure1(
    trials: list[_Trial], output_dir: pathlib.Path, fmt: str = "png"
) -> None:
    """2x4 exact-rate heatmaps.

    Rows = method (VBPCApy cost vs sklearn EVR95), columns = missingness.
    """
    n_miss = len(MISSINGNESS)
    row_defs = [
        ("cost", "vbpca", "VBPCApy (cost)"),
        ("evr95", "sklearn_pca", "sklearn PCA (EVR95)"),
    ]
    n_rows = len(row_defs)
    fig, axes = plt.subplots(
        n_rows, n_miss, figsize=(4.0 * n_miss + 1.2, 3.2 * n_rows), squeeze=False
    )

    im = None
    for row, (metric, method, row_label) in enumerate(row_defs):
        for col, miss in enumerate(MISSINGNESS):
            ax = axes[row, col]
            mat, ns, ps = _rate_matrix(trials, "exact", metric, miss, method=method)
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(range(len(ps)))
            ax.set_xticklabels(ps)
            ax.set_yticks(range(len(ns)))
            ax.set_yticklabels(ns)
            if row == n_rows - 1:
                ax.set_xlabel("p (features)")
            if col == 0:
                ax.set_ylabel(f"{row_label}\nn (samples)")
            ax.set_title(_MISS_LABEL.get(miss, miss) if row == 0 else "")
            for i in range(len(ns)):
                for j in range(len(ps)):
                    val = mat[i, j]
                    if not np.isnan(val):
                        ax.text(
                            j,
                            i,
                            f"{val:.0%}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if val < 0.5 else "black",
                        )

    fig.subplots_adjust(right=0.88, wspace=0.30, hspace=0.35)
    if im is not None:
        cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cax, label="Exact rate")
    fig.suptitle("Model selection accuracy: VBPCApy vs sklearn PCA", fontsize=12)
    out = output_dir / f"figure_accuracy.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Figure 2 — Error Structure (over/under, MAE, selected vs true_rank)
# ---------------------------------------------------------------------------


def plot_figure2(
    trials: list[_Trial], output_dir: pathlib.Path, fmt: str = "png"
) -> None:
    """3-panel error decomposition: over/under bars, MAE bars, selected line."""
    fig, (ax_ou, ax_mae, ax_sel) = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel A: over_rate + under_rate by missingness ---
    over = _aggregate_by_group(trials, "over", "missingness")
    under = _aggregate_by_group(trials, "under", "missingness")
    labels = list(over.keys())
    pretty = [_MISS_LABEL.get(la, la) for la in labels]
    x_pos = np.arange(len(labels))
    w = 0.35
    ax_ou.bar(
        x_pos - w / 2, [over[la] for la in labels], w, label="Over", color="#d62728"
    )
    ax_ou.bar(
        x_pos + w / 2, [under[la] for la in labels], w, label="Under", color="#1f77b4"
    )
    ax_ou.set_xticks(x_pos)
    ax_ou.set_xticklabels(pretty, rotation=0, ha="center")
    ax_ou.set_ylabel("Rate")
    ax_ou.set_title("A) Over/Under-selection")
    ax_ou.legend(fontsize=8)
    ax_ou.set_ylim(0, 1.05)

    # --- Panel B: MAE by missingness x metric (VBPCA + sklearn) ---
    bar_labels: list[str] = []
    bar_vals: list[float] = []
    bar_colors: list[str] = []
    metric_colors = {"cost": "#ff7f0e", "prms": "#2ca02c", "evr95": "#9467bd"}
    methods_metrics = [("vbpca", "cost"), ("vbpca", "prms"), ("sklearn_pca", "evr95")]
    for miss in MISSINGNESS:
        for method, metric in methods_metrics:
            vals_b = [
                t.abs_error
                for t in trials
                if t.missingness == miss and t.metric == metric and t.method == method
            ]
            if vals_b:
                label_m = "EVR95" if method == "sklearn_pca" else metric
                bar_labels.append(f"{_MISS_LABEL.get(miss, miss)}\n{label_m}")
                bar_vals.append(float(np.mean(vals_b)))
                bar_colors.append(metric_colors[metric])

    ax_mae.bar(range(len(bar_labels)), bar_vals, color=bar_colors)
    ax_mae.set_xticks(range(len(bar_labels)))
    ax_mae.set_xticklabels(bar_labels, fontsize=6, rotation=30, ha="right")
    ax_mae.set_ylabel("Mean absolute error")
    ax_mae.set_title("B) MAE by missingness x method")

    # --- Panel C: selected_mean vs true_rank (by metric + sklearn) ---
    for metric in METRICS:
        ranks, means, stds = _aggregate_by_rank(trials, "selected_k", metric=metric)
        ax_sel.errorbar(ranks, means, yerr=stds, marker="o", label=metric, capsize=3)
    # sklearn baseline
    ranks_sk, means_sk, stds_sk = _aggregate_by_rank(
        trials, "selected_k", metric="evr95", method="sklearn_pca"
    )
    if ranks_sk:
        ax_sel.errorbar(
            ranks_sk,
            means_sk,
            yerr=stds_sk,
            marker="^",
            linestyle=":",
            color="#9467bd",
            label="EVR95",
            capsize=3,
        )
    # Reference line
    all_ranks = sorted({t.true_rank for t in trials})
    ax_sel.plot(all_ranks, all_ranks, "k--", alpha=0.4, label="ideal")
    ax_sel.set_xlabel("True rank")
    ax_sel.set_ylabel("Selected k (mean +/- std)")
    ax_sel.set_title("C) Selected vs true rank")
    ax_sel.legend(fontsize=8)

    fig.tight_layout()
    out = output_dir / f"figure_errors.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Figure 3 — Detection Power
# ---------------------------------------------------------------------------


def plot_figure3(
    trials: list[_Trial], output_dir: pathlib.Path, fmt: str = "png"
) -> None:
    """2-panel detection: power vs true_rank + overall power bars."""
    fig, (ax_line, ax_bar) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel A: power_rate vs true_rank, one line per metric ---
    styles = {"cost": ("o", "-"), "prms": ("s", "--")}
    for metric in METRICS:
        marker, ls = styles[metric]
        ranks, means, stds = _aggregate_by_rank(trials, "power", metric=metric)
        lo = [max(0, m - s) for m, s in zip(means, stds, strict=True)]
        hi = [min(1, m + s) for m, s in zip(means, stds, strict=True)]
        ax_line.plot(ranks, means, marker=marker, linestyle=ls, label=metric)
        ax_line.fill_between(ranks, lo, hi, alpha=0.15)

    ax_line.set_xlabel("True rank")
    ax_line.set_ylabel("Power rate")
    ax_line.set_ylim(-0.05, 1.05)
    ax_line.set_title("A) Power vs true rank")
    ax_line.legend(fontsize=8)

    # --- Panel B: overall power by missingness x method ---
    bar_labels_b: list[str] = []
    bar_vals_b: list[float] = []
    bar_colors_b: list[str] = []
    metric_colors = {"cost": "#ff7f0e", "prms": "#2ca02c", "evr95": "#9467bd"}
    methods_metrics = [("vbpca", "cost"), ("vbpca", "prms"), ("sklearn_pca", "evr95")]
    for miss in MISSINGNESS:
        for method, metric in methods_metrics:
            vals_p = [
                t.power
                for t in trials
                if t.missingness == miss and t.metric == metric and t.method == method
            ]
            if vals_p:
                label_m = "EVR95" if method == "sklearn_pca" else metric
                bar_labels_b.append(f"{_MISS_LABEL.get(miss, miss)}\n{label_m}")
                bar_vals_b.append(float(np.mean(vals_p)))
                bar_colors_b.append(metric_colors[metric])

    ax_bar.bar(range(len(bar_labels_b)), bar_vals_b, color=bar_colors_b)
    ax_bar.set_xticks(range(len(bar_labels_b)))
    ax_bar.set_xticklabels(bar_labels_b, fontsize=6, rotation=30, ha="right")
    ax_bar.set_ylabel("Power rate")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("B) Power by missingness x method")

    fig.tight_layout()
    out = output_dir / f"figure_power.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Figure 4 — Posterior Predictive Coverage
# ---------------------------------------------------------------------------


def _plot_coverage_calibration(
    ax: plt.Axes,
    cov_results: list[_CoverageTrial],
) -> None:
    """Panel A: coverage vs nominal, one line per missingness."""
    miss_colors = {
        "complete": "#1f77b4",
        "mcar": "#ff7f0e",
        "mnar_censored": "#2ca02c",
        "block": "#d62728",
    }
    for miss in MISSINGNESS:
        nominals: list[float] = []
        coverages: list[float] = []
        for nom in COVERAGE_NOMINALS:
            vals = [
                r.coverage
                for r in cov_results
                if r.missingness == miss and r.nominal == nom
            ]
            if vals:
                nominals.append(nom)
                coverages.append(float(np.mean(vals)))
        if nominals:
            ax.plot(
                nominals,
                coverages,
                "o-",
                color=miss_colors.get(miss, "grey"),
                label=_MISS_LABEL.get(miss, miss),
            )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.45, 1.01)
    ax.set_ylim(0.45, 1.01)
    ax.set_title("A) Coverage calibration")
    ax.legend(fontsize=7, loc="lower right")


def _plot_coverage_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    cov_results: list[_CoverageTrial],
) -> None:
    """Panel B: heatmap of coverage at 95% nominal."""
    nominal_95 = 0.95
    vals_95: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in cov_results:
        if abs(r.nominal - nominal_95) < 1e-9:
            vals_95[r.n, r.p].append(r.coverage)

    ns = sorted({k[0] for k in vals_95})
    ps = sorted({k[1] for k in vals_95})
    mat = np.full((len(ns), len(ps)), np.nan)
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            v = vals_95.get((n, p), [])
            if v:
                mat[i, j] = float(np.mean(v))

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels(ps)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(ns)
    ax.set_xlabel("p (features)")
    ax.set_ylabel("n (samples)")
    ax.set_title("B) 95% coverage by (n, p)")
    _annotate_heatmap(ax, mat, fmt="{:.0%}", threshold=0.75)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Coverage rate")


def _plot_interval_width(
    ax: plt.Axes,
    cov_results: list[_CoverageTrial],
) -> None:
    """Panel C: mean interval width vs nominal, one line per missingness."""
    miss_colors = {
        "complete": "#1f77b4",
        "mcar": "#ff7f0e",
        "mnar_censored": "#2ca02c",
        "block": "#d62728",
    }
    for miss in MISSINGNESS:
        nominals_w: list[float] = []
        widths: list[float] = []
        for nom in COVERAGE_NOMINALS:
            vals_w = [
                r.mean_interval_width
                for r in cov_results
                if r.missingness == miss and r.nominal == nom
            ]
            if vals_w:
                nominals_w.append(nom)
                widths.append(float(np.mean(vals_w)))
        if nominals_w:
            ax.plot(
                nominals_w,
                widths,
                "o-",
                color=miss_colors.get(miss, "grey"),
                label=_MISS_LABEL.get(miss, miss),
            )
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Mean interval width")
    ax.set_title("C) Interval width")
    ax.legend(fontsize=7, loc="upper left")


def plot_figure4(
    cov_results: list[_CoverageTrial], output_dir: pathlib.Path, fmt: str = "png"
) -> None:
    """3-panel coverage: calibration curves, per-(n,p) heatmap, interval width."""
    fig, (ax_cal, ax_heat, ax_width) = plt.subplots(1, 3, figsize=(16, 4.5))

    _plot_coverage_calibration(ax_cal, cov_results)
    _plot_coverage_heatmap(fig, ax_heat, cov_results)
    _plot_interval_width(ax_width, cov_results)

    fig.suptitle("Posterior predictive coverage on held-out entries", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = output_dir / f"figure_coverage.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Figure 5 — Holdout RMSE comparison
# ---------------------------------------------------------------------------


def plot_figure5(
    cov_results: list[_CoverageTrial], output_dir: pathlib.Path, fmt: str = "png"
) -> None:
    """Bar chart: VBPCApy vs impute+PCA holdout RMSE by missingness pattern."""
    # Deduplicate: one RMSE per (n, p, rank, miss, rep) — take first nominal
    seen: set[tuple[int, int, int, str, int]] = set()
    vb_vals: dict[str, list[float]] = defaultdict(list)
    bl_vals: dict[str, list[float]] = defaultdict(list)
    for r in cov_results:
        key = (r.n, r.p, r.true_rank, r.missingness, r.rep)
        if key in seen:
            continue
        seen.add(key)
        vb_vals[r.missingness].append(r.holdout_rmse)
        bl_vals[r.missingness].append(r.baseline_rmse)

    miss_order = [m for m in MISSINGNESS if m in vb_vals]
    pretty = [_MISS_LABEL.get(m, m) for m in miss_order]
    x_pos = np.arange(len(miss_order))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        x_pos - w / 2,
        [float(np.mean(vb_vals[m])) for m in miss_order],
        w,
        label="VBPCApy",
        color="#1f77b4",
    )
    ax.bar(
        x_pos + w / 2,
        [float(np.mean(bl_vals[m])) for m in miss_order],
        w,
        label="Impute + PCA",
        color="#9467bd",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pretty)
    ax.set_ylabel("Holdout RMSE")
    ax.set_title("Reconstruction accuracy on held-out entries")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = output_dir / f"figure_rmse.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Figure 6 — RMSE improvement heatmap by (n, p)
# ---------------------------------------------------------------------------


def plot_figure6(
    cov_results: list[_CoverageTrial], output_dir: pathlib.Path, fmt: str = "png"
) -> None:
    """Heatmap of % RMSE improvement (VBPCApy vs impute+PCA) by (n, p)."""
    # Deduplicate: one RMSE pair per (n, p, rank, miss, rep)
    seen: set[tuple[int, int, int, str, int]] = set()
    rmse_by: dict[tuple[int, int], list[float]] = defaultdict(list)
    base_by: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in cov_results:
        key = (r.n, r.p, r.true_rank, r.missingness, r.rep)
        if key in seen:
            continue
        seen.add(key)
        rmse_by[r.n, r.p].append(r.holdout_rmse)
        base_by[r.n, r.p].append(r.baseline_rmse)

    ns = sorted({k[0] for k in rmse_by})
    ps = sorted({k[1] for k in rmse_by})
    mat = np.full((len(ns), len(ps)), np.nan)
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            vb = rmse_by.get((n, p), [])
            bl = base_by.get((n, p), [])
            if vb and bl:
                mv = float(np.mean(vb))
                mb = float(np.mean(bl))
                mat[i, j] = (mb - mv) / mb * 100 if mb > 0 else 0

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(mat, aspect="auto", cmap="YlGn", vmin=0, vmax=60)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels(ps)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(ns)
    ax.set_xlabel("p (features)")
    ax.set_ylabel("n (samples)")
    ax.set_title("RMSE improvement over impute+PCA (%)")
    for i in range(len(ns)):
        for j in range(len(ps)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if val < 30 else "black",
                )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Improvement (%)")
    fig.tight_layout()
    out = output_dir / f"figure_rmse_heatmap.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Figure 7 — Accuracy vs Coverage Pareto front
# ---------------------------------------------------------------------------


def _compute_pareto_front(
    points: list[tuple[float, float, int, int]],
) -> list[tuple[float, float]]:
    """Return non-dominated points in (accuracy-up, coverage-up) space."""
    pareto: list[tuple[float, float]] = []
    for a, c, _n, _p in points:
        dominated = False
        for a2, c2, _, _ in points:
            if a2 >= a and c2 >= c and (a2 > a or c2 > c):
                dominated = True
                break
        if not dominated:
            pareto.append((a, c))
    pareto.sort()
    return pareto


def plot_figure7(
    trials: list[_Trial],
    cov_results: list[_CoverageTrial],
    output_dir: pathlib.Path,
    fmt: str = "png",
) -> None:
    """Scatter of model selection accuracy vs coverage at 95% nominal by (n, p)."""
    # Model selection accuracy: vbpca cost, averaged over missingness and ranks
    acc_by: dict[tuple[int, int], list[float]] = defaultdict(list)
    for t in trials:
        if t.method == "vbpca" and t.metric == "cost":
            acc_by[t.n, t.p].append(t.exact)

    # Coverage at 95%: averaged over missingness and ranks
    cov_by: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in cov_results:
        if abs(r.nominal - 0.95) < 1e-9:
            cov_by[r.n, r.p].append(r.coverage)

    # Build scatter data
    points: list[tuple[float, float, int, int]] = []
    for key in sorted(acc_by.keys()):
        if key in cov_by:
            a = float(np.mean(acc_by[key]))
            c = float(np.mean(cov_by[key]))
            points.append((a, c, key[0], key[1]))

    if not points:
        LOGGER.warning("No overlapping (n, p) keys for Pareto plot; skipping.")
        return

    accs = [pt[0] for pt in points]
    covs = [pt[1] for pt in points]
    ps_vals = [pt[3] for pt in points]

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        accs,
        covs,
        c=ps_vals,
        cmap="viridis",
        s=100,
        edgecolors="black",
        linewidths=0.5,
        norm=plt.matplotlib.colors.LogNorm(vmin=min(ps_vals), vmax=max(ps_vals)),
    )

    for a, c, n_val, p_val in points:
        ax.annotate(
            f"({n_val},{p_val})",
            (a, c),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=6,
            alpha=0.8,
        )

    pareto_pts = _compute_pareto_front(points)
    if pareto_pts:
        pa, pc = zip(*pareto_pts, strict=True)
        ax.plot(pa, pc, "r--", alpha=0.7, linewidth=1.5, label="Pareto front")

    ax.set_xlabel("Model selection accuracy (exact match rate)")
    ax.set_ylabel("Posterior coverage at 95% nominal")
    ax.set_title("Accuracy-coverage tradeoff by (n, p)")
    ax.legend(fontsize=8, loc="lower left")
    fig.colorbar(sc, ax=ax, label="p (features)", shrink=0.8)
    fig.tight_layout()
    out = output_dir / f"figure_pareto.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Figure 8 — MAE heatmap by (n, p)
# ---------------------------------------------------------------------------


def plot_figure8(
    trials: list[_Trial], output_dir: pathlib.Path, fmt: str = "png"
) -> None:
    """Heatmap of VBPCApy cost MAE by (n, p), averaged over missingness and ranks."""
    mat, ns, ps = _rate_matrix(trials, "abs_error", "cost", missingness="", method="")
    # _rate_matrix filters; build manually for grand average
    vals: dict[tuple[int, int], list[float]] = defaultdict(list)
    for t in trials:
        if t.method == "vbpca" and t.metric == "cost":
            vals[t.n, t.p].append(t.abs_error)

    ns = sorted({k[0] for k in vals})
    ps = sorted({k[1] for k in vals})
    mat = np.full((len(ns), len(ps)), np.nan)
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            v = vals.get((n, p), [])
            if v:
                mat[i, j] = float(np.mean(v))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    vmax = max(3.0, float(np.nanmax(mat)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels(ps)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(ns)
    ax.set_xlabel("p (features)")
    ax.set_ylabel("n (samples)")
    ax.set_title("VBPCApy rank-selection MAE by (n, p)")
    for i in range(len(ns)):
        for j in range(len(ps)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if val > vmax * 0.6 else "black",
                )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean absolute error")
    fig.tight_layout()
    out = output_dir / f"figure_mae_heatmap.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Figure 9 — ΔMAE heatmap: VBPCApy cost vs EVR95
# ---------------------------------------------------------------------------


def _annotate_signed_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    abs_max: float,
    fontsize: int = 8,
) -> None:
    """Annotate a diverging heatmap with signed values."""
    nrows, ncols = mat.shape
    for i in range(nrows):
        for j in range(ncols):
            val = mat[i, j]
            if not np.isnan(val):
                sign = "+" if val > 0 else ""
                ax.text(
                    j,
                    i,
                    f"{sign}{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="white" if abs(val) > abs_max * 0.6 else "black",
                )


def plot_figure9(
    trials: list[_Trial], output_dir: pathlib.Path, fmt: str = "png"
) -> None:
    """Heatmap of MAE difference (EVR95 - VBPCApy cost) by (n, p).

    Positive values (green) mean VBPCApy has lower MAE (better).
    """
    cost_vals: dict[tuple[int, int], list[float]] = defaultdict(list)
    evr_vals: dict[tuple[int, int], list[float]] = defaultdict(list)
    for t in trials:
        if t.method == "vbpca" and t.metric == "cost":
            cost_vals[t.n, t.p].append(t.abs_error)
        elif t.method == "sklearn_pca" and t.metric == "evr95":
            evr_vals[t.n, t.p].append(t.abs_error)

    all_keys = sorted(set(cost_vals.keys()) & set(evr_vals.keys()))
    if not all_keys:
        LOGGER.warning("No overlapping (n, p) keys for dMAE plot; skipping.")
        return

    ns = sorted({k[0] for k in all_keys})
    ps = sorted({k[1] for k in all_keys})
    mat = np.full((len(ns), len(ps)), np.nan)
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            c = cost_vals.get((n, p), [])
            e = evr_vals.get((n, p), [])
            if c and e:
                mat[i, j] = float(np.mean(e)) - float(np.mean(c))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    abs_max = max(1.0, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-abs_max, vmax=abs_max)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels(ps)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(ns)
    ax.set_xlabel("p (features)")
    ax.set_ylabel("n (samples)")
    ax.set_title("MAE advantage: VBPCApy cost vs EVR95 (EVR95 - cost)")
    _annotate_signed_heatmap(ax, mat, abs_max)
    fig.colorbar(im, ax=ax, shrink=0.8, label="dMAE (positive = VBPCApy better)")
    fig.tight_layout()
    out = output_dir / f"figure_delta_mae.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_results(trials: list[_Trial], output_dir: pathlib.Path) -> None:
    """Save raw trial data as JSON (always) and Parquet (if pandas available)."""
    records = [asdict(t) for t in trials]

    json_path = output_dir / "stability_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    LOGGER.info("Wrote %s (%d trials)", json_path, len(records))

    try:
        import pandas as pd

        df = pd.DataFrame(records)
        parquet_path = output_dir / "stability_results.parquet"
        df.to_parquet(parquet_path, index=False)
        LOGGER.info("Wrote %s", parquet_path)
    except ImportError:
        LOGGER.info("pandas not available; skipping Parquet output")


def _save_coverage_results(
    results: list[_CoverageTrial], output_dir: pathlib.Path
) -> None:
    """Save coverage trial data as JSON."""
    records = [asdict(t) for t in results]
    json_path = output_dir / "coverage_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    LOGGER.info("Wrote %s (%d records)", json_path, len(records))


def _load_trials(path: pathlib.Path) -> list[_Trial]:
    with path.open(encoding="utf-8") as f:
        return [_Trial(**r) for r in json.load(f)]


def _load_coverage(path: pathlib.Path) -> list[_CoverageTrial]:
    with path.open(encoding="utf-8") as f:
        return [_CoverageTrial(**r) for r in json.load(f)]


def _plot_stability_figs(
    trials: list[_Trial],
    output_dir: pathlib.Path,
    fmt: str,
) -> None:
    plot_figure1(trials, output_dir, fmt)
    plot_figure2(trials, output_dir, fmt)
    plot_figure3(trials, output_dir, fmt)
    plot_figure8(trials, output_dir, fmt)
    plot_figure9(trials, output_dir, fmt)


def _plot_coverage_figs(
    cov_results: list[_CoverageTrial],
    output_dir: pathlib.Path,
    fmt: str,
    trials: list[_Trial] | None = None,
) -> None:
    plot_figure4(cov_results, output_dir, fmt)
    plot_figure5(cov_results, output_dir, fmt)
    plot_figure6(cov_results, output_dir, fmt)
    if trials is not None:
        plot_figure7(trials, cov_results, output_dir, fmt)


def _run_plot_only(output_dir: pathlib.Path, fmt: str) -> None:
    stab_path = output_dir / "stability_results.json"
    if not stab_path.exists():
        LOGGER.error("No %s found; run simulation first.", stab_path)
        return
    trials = _load_trials(stab_path)
    LOGGER.info("Loaded %d trials from %s", len(trials), stab_path)
    _plot_stability_figs(trials, output_dir, fmt)

    cov_path = output_dir / "coverage_results.json"
    if cov_path.exists():
        cov_results = _load_coverage(cov_path)
        LOGGER.info("Loaded %d coverage records from %s", len(cov_results), cov_path)
        _plot_coverage_figs(cov_results, output_dir, fmt, trials=trials)
    else:
        LOGGER.warning("No %s found; skipping coverage figures.", cov_path)


def _run_coverage_only(
    grid: dict[str, list[int]],
    reps: int,
    seed: int,
    output_dir: pathlib.Path,
    fmt: str,
) -> None:
    stab_path = output_dir / "stability_results.json"
    trials: list[_Trial] | None = None
    if stab_path.exists():
        trials = _load_trials(stab_path)
        LOGGER.info("Loaded %d trials from %s", len(trials), stab_path)
        _plot_stability_figs(trials, output_dir, fmt)
    else:
        LOGGER.warning("No %s found; skipping stability figures.", stab_path)

    cov_results = _run_coverage_grid(grid, reps=reps, seed=seed)
    if cov_results:
        _save_coverage_results(cov_results, output_dir)
        _plot_coverage_figs(cov_results, output_dir, fmt, trials=trials)


def _run_full(
    grid: dict[str, list[int]],
    reps: int,
    seed: int,
    output_dir: pathlib.Path,
    fmt: str,
) -> None:
    trials = _run_grid(grid, reps=reps, seed=seed)
    if not trials:
        LOGGER.warning("No valid trials; nothing to plot.")
        return

    _save_results(trials, output_dir)
    _plot_stability_figs(trials, output_dir, fmt)

    cov_results = _run_coverage_grid(grid, reps=reps, seed=seed)
    if cov_results:
        _save_coverage_results(cov_results, output_dir)
        _plot_coverage_figs(cov_results, output_dir, fmt, trials=trials)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for model-selection stability analysis."""
    parser = argparse.ArgumentParser(
        description="VBPCApy model-selection stability across metrics and missingness"
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("paper"),
    )
    parser.add_argument("--fmt", default="pdf", choices=["png", "pdf", "svg"])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a minimal grid for fast CI validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from existing JSON results (no simulation)",
    )
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Rerun only the coverage sweep; replot stability figs from existing JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        _run_plot_only(args.output_dir, args.fmt)
        LOGGER.info("Done. Figures written to %s/", args.output_dir)
        return

    grid = SMOKE_GRID if args.smoke else FULL_GRID
    reps = REPS_SMOKE if args.smoke else REPS_FULL

    if args.coverage_only:
        _run_coverage_only(grid, reps, args.seed, args.output_dir, args.fmt)
    else:
        _run_full(grid, reps, args.seed, args.output_dir, args.fmt)

    LOGGER.info("Done. Figures written to %s/", args.output_dir)


if __name__ == "__main__":
    main()
