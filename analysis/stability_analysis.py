"""Model-selection stability analysis across stopping metrics and missingness.

Generates synthetic low-rank data and runs ``select_n_components`` with each
metric (cost, prms) over a grid of (n, p, true_rank) settings under four
missingness patterns (complete, mcar, mnar_censored, block).

Produces three figures for the JOSS paper:

1. **Figure 1 -- Model Selection Accuracy**: 2x4 exact-rate heatmaps
   (metric x missingness), each cell an (n x p) heatmap.
2. **Figure 2 -- Error Structure**: over/under rates, MAE, selected vs true.
3. **Figure 3 -- Detection Power**: power_rate vs true_rank + overall bars.

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
    "n": [20, 50, 100, 200],
    "p": [10, 20, 50, 100, 200],
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
REPS_FULL = 5
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


def _run_grid(grid: dict[str, list[int]], reps: int, seed: int = 42) -> list[_Trial]:  # noqa: PLR0914
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


def _run_coverage_grid(  # noqa: PLR0914, C901
    grid: dict[str, list[int]], reps: int, seed: int = 42
) -> list[_CoverageTrial]:
    """Fit VBPCA at true_rank, compute posterior coverage on held-out entries.

    Returns:
        List of coverage trial results.
    """
    from scipy import stats  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    results: list[_CoverageTrial] = []
    settings = list(itertools.product(grid["n"], grid["p"], grid["true_rank"]))
    LOGGER.info(
        "Coverage sweep: %d settings x %d miss x %d reps",
        len(settings),
        len(MISSINGNESS),
        reps,
    )

    for n, p, true_rank in settings:
        if true_rank >= min(n, p):
            continue
        for rep in range(reps):
            x_clean = _generate_low_rank(n, p, true_rank, rng)

            for miss_pattern in MISSINGNESS:
                x = x_clean.copy()
                mask = _apply_missingness(x, miss_pattern, rng)

                # Create hold-out set from observed entries
                obs_mask = np.ones((n, p), dtype=bool) if mask is None else mask.copy()
                obs_indices = np.argwhere(obs_mask)
                n_holdout = max(1, int(len(obs_indices) * HOLDOUT_FRACTION))
                holdout_idx = rng.choice(
                    len(obs_indices), size=n_holdout, replace=False
                )
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
                        "Missing reconstruction/variance for"
                        " n=%d p=%d rank=%d %s rep=%d; skipping",
                        n,
                        p,
                        true_rank,
                        miss_pattern,
                        rep,
                    )
                    continue

                pred = xrec[holdout_coords[:, 0], holdout_coords[:, 1]]
                var = vr[holdout_coords[:, 0], holdout_coords[:, 1]]
                residuals = np.abs(holdout_vals - pred)
                rmse = float(np.sqrt(np.mean(residuals**2)))

                for nominal in COVERAGE_NOMINALS:
                    z = stats.norm.ppf(0.5 + nominal / 2.0)
                    covered = residuals <= z * np.sqrt(np.maximum(var, 1e-12))
                    cov_rate = float(np.mean(covered))
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
                        )
                    )

    LOGGER.info("Coverage sweep done: %d results", len(results))
    return results


# ---------------------------------------------------------------------------
# Figure 1 — Model Selection Accuracy (exact_rate heatmaps)
# ---------------------------------------------------------------------------


def plot_figure1(
    trials: list[_Trial], output_dir: pathlib.Path, fmt: str = "png"
) -> None:
    """2x4 exact-rate heatmaps: rows=metric, cols=missingness."""
    n_miss = len(MISSINGNESS)
    n_met = len(METRICS)
    fig, axes = plt.subplots(
        n_met, n_miss, figsize=(4.0 * n_miss + 1.2, 3.2 * n_met), squeeze=False
    )

    im = None
    for row, metric in enumerate(METRICS):
        for col, miss in enumerate(MISSINGNESS):
            ax = axes[row, col]
            mat, ns, ps = _rate_matrix(trials, "exact", metric, miss)
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(range(len(ps)))
            ax.set_xticklabels(ps)
            ax.set_yticks(range(len(ns)))
            ax.set_yticklabels(ns)
            if row == n_met - 1:
                ax.set_xlabel("p (features)")
            if col == 0:
                ax.set_ylabel(f"{metric}\nn (samples)")
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
    fig.suptitle(
        "Model selection accuracy by metric and missingness pattern", fontsize=12
    )
    out = output_dir / f"figure_accuracy.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Figure 2 — Error Structure (over/under, MAE, selected vs true_rank)
# ---------------------------------------------------------------------------


def plot_figure2(  # noqa: PLR0914
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


def plot_figure3(  # noqa: PLR0914
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


def plot_figure4(  # noqa: C901, PLR0914
    cov_results: list[_CoverageTrial], output_dir: pathlib.Path, fmt: str = "png"
) -> None:
    """2-panel coverage: calibration curves + per-(n,p) heatmap."""
    fig, (ax_cal, ax_heat) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel A: coverage vs nominal, one line per missingness ---
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
            ax_cal.plot(
                nominals,
                coverages,
                "o-",
                color=miss_colors.get(miss, "grey"),
                label=_MISS_LABEL.get(miss, miss),
            )

    ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal")
    ax_cal.set_xlabel("Nominal coverage")
    ax_cal.set_ylabel("Empirical coverage")
    ax_cal.set_xlim(0.45, 1.01)
    ax_cal.set_ylim(0.45, 1.01)
    ax_cal.set_title("A) Coverage calibration")
    ax_cal.legend(fontsize=7, loc="lower right")

    # --- Panel B: heatmap of coverage at 95% nominal ---
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

    im = ax_heat.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax_heat.set_xticks(range(len(ps)))
    ax_heat.set_xticklabels(ps)
    ax_heat.set_yticks(range(len(ns)))
    ax_heat.set_yticklabels(ns)
    ax_heat.set_xlabel("p (features)")
    ax_heat.set_ylabel("n (samples)")
    ax_heat.set_title("B) 95% coverage by (n, p)")
    for i in range(len(ns)):
        for j in range(len(ps)):
            val = mat[i, j]
            if not np.isnan(val):
                ax_heat.text(
                    j,
                    i,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if val < 0.75 else "black",
                )
    fig.colorbar(im, ax=ax_heat, shrink=0.8, label="Coverage rate")

    fig.suptitle("Posterior predictive coverage on held-out entries", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = output_dir / f"figure_coverage.{fmt}"
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
        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame(records)
        parquet_path = output_dir / "stability_results.parquet"
        df.to_parquet(parquet_path, index=False)
        LOGGER.info("Wrote %s", parquet_path)
    except ImportError:
        LOGGER.info("pandas not available; skipping Parquet output")


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        # Load existing results and regenerate figures only
        stab_path = args.output_dir / "stability_results.json"
        cov_path = args.output_dir / "coverage_results.json"
        if not stab_path.exists():
            LOGGER.error("No %s found; run simulation first.", stab_path)
            return
        with stab_path.open(encoding="utf-8") as f:
            trials = [_Trial(**r) for r in json.load(f)]
        LOGGER.info("Loaded %d trials from %s", len(trials), stab_path)
        plot_figure1(trials, args.output_dir, args.fmt)
        plot_figure2(trials, args.output_dir, args.fmt)
        plot_figure3(trials, args.output_dir, args.fmt)
        if cov_path.exists():
            with cov_path.open(encoding="utf-8") as f:
                cov_results = [_CoverageTrial(**r) for r in json.load(f)]
            LOGGER.info(
                "Loaded %d coverage results from %s", len(cov_results), cov_path
            )
            plot_figure4(cov_results, args.output_dir, args.fmt)
        LOGGER.info("Done. Figures written to %s/", args.output_dir)
        return

    grid = SMOKE_GRID if args.smoke else FULL_GRID
    reps = REPS_SMOKE if args.smoke else REPS_FULL

    trials = _run_grid(grid, reps=reps, seed=args.seed)

    if not trials:
        LOGGER.warning("No valid trials; nothing to plot.")
        return

    _save_results(trials, args.output_dir)
    plot_figure1(trials, args.output_dir, args.fmt)
    plot_figure2(trials, args.output_dir, args.fmt)
    plot_figure3(trials, args.output_dir, args.fmt)

    # Coverage sweep
    cov_results = _run_coverage_grid(grid, reps=reps, seed=args.seed + 1)
    if cov_results:
        cov_records = [asdict(r) for r in cov_results]
        cov_path = args.output_dir / "coverage_results.json"
        with cov_path.open("w", encoding="utf-8") as f:
            json.dump(cov_records, f, indent=2)
        LOGGER.info("Wrote %s (%d results)", cov_path, len(cov_records))
        plot_figure4(cov_results, args.output_dir, args.fmt)

    LOGGER.info("Done. Figures written to %s/", args.output_dir)


if __name__ == "__main__":
    main()
