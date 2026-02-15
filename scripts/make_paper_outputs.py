#!/usr/bin/env python3
"""Generate paper-facing tables and figures from benchmark summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VBPCA_METHOD = "vbpca_vb_modern"


def _format_ci(low: float, high: float) -> str:
    return f"[{low:.4f}, {high:.4f}]"


def _build_table1(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "mechanism",
        "pattern",
        "missing_rate",
        "n_components_requested",
        "method",
        "n_reps",
        "rmse_mean",
        "rmse_median",
        "rmse_std",
        "rmse_ci_low",
        "rmse_ci_high",
        "mae_mean",
        "mae_median",
        "mae_std",
        "mae_ci_low",
        "mae_ci_high",
        "wall_time_sec_mean",
        "wall_time_sec_median",
        "wall_time_sec_std",
        "wall_time_sec_ci_low",
        "wall_time_sec_ci_high",
        "vbpca_mean_variance_mean",
    ]
    out = summary.loc[:, cols].copy()
    out["rmse_ci"] = [
        _format_ci(lo, hi)
        for lo, hi in zip(out["rmse_ci_low"], out["rmse_ci_high"], strict=True)
    ]
    out["mae_ci"] = [
        _format_ci(lo, hi)
        for lo, hi in zip(out["mae_ci_low"], out["mae_ci_high"], strict=True)
    ]
    out["time_ci"] = [
        _format_ci(lo, hi)
        for lo, hi in zip(
            out["wall_time_sec_ci_low"],
            out["wall_time_sec_ci_high"],
            strict=True,
        )
    ]
    return out


def _build_table2(pairwise: pd.DataFrame) -> pd.DataFrame:
    out = pairwise.copy()
    out["delta_ci"] = [
        _format_ci(lo, hi)
        for lo, hi in zip(out["delta_ci_low"], out["delta_ci_high"], strict=True)
    ]
    return out


def _make_figure_runtime_scaling(replicates: pd.DataFrame, out_path: Path) -> None:
    plot_df = replicates.copy()
    plot_df["size"] = plot_df["n_samples"] * plot_df["n_features"]

    grouped = (
        plot_df.groupby(["method", "size"], as_index=False)["wall_time_sec"]
        .agg(
            mean="mean",
            ci_low=lambda s: np.percentile(np.asarray(s, dtype=float), 2.5),
            ci_high=lambda s: np.percentile(np.asarray(s, dtype=float), 97.5),
        )
        .sort_values(["method", "size"])
    )

    plt.figure(figsize=(7.2, 4.8))
    for method, subset in grouped.groupby("method"):
        x = subset["size"].to_numpy()
        y = subset["mean"].to_numpy()
        err_low = y - subset["ci_low"].to_numpy()
        err_high = subset["ci_high"].to_numpy() - y
        plt.errorbar(x, y, yerr=[err_low, err_high], marker="o", label=method)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dataset size (n_samples × n_features)")
    plt.ylabel("Wall time (seconds)")
    plt.title("Figure 1: Runtime scaling by method")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _make_figure_uncertainty(replicates: pd.DataFrame, out_path: Path) -> None:
    vbpca = replicates[replicates["method"] == VBPCA_METHOD].copy()
    if vbpca.empty:
        return

    labels = (
        vbpca["dataset"]
        + "|"
        + vbpca["mechanism"]
        + "|"
        + vbpca["pattern"]
        + "|r="
        + vbpca["missing_rate"].map(lambda x: f"{x:.1f}")
    )
    vbpca["setting_label"] = labels

    grouped = [
        np.asarray(sub["vbpca_mean_variance"], dtype=float)
        for _, sub in vbpca.groupby("setting_label")
    ]
    tick_labels = [label for label, _ in vbpca.groupby("setting_label")]

    if not grouped:
        return

    plt.figure(figsize=(10, 4.8))
    plt.boxplot(grouped, showfliers=False)
    plt.xticks(range(1, len(tick_labels) + 1), tick_labels, rotation=60, ha="right")
    plt.ylabel("Mean marginal variance")
    plt.title("Figure 2: VBPCA predictive uncertainty across settings")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicates",
        type=Path,
        default=Path("results/replicates.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/summary.csv"),
    )
    parser.add_argument(
        "--pairwise",
        type=Path,
        default=Path("results/pairwise_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paper"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    replicates = pd.read_csv(args.replicates)
    summary = pd.read_csv(args.summary)
    pairwise = pd.read_csv(args.pairwise)

    table1 = _build_table1(summary)
    table2 = _build_table2(pairwise)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table1.to_csv(args.out_dir / "table1_method_summary.csv", index=False)
    table2.to_csv(args.out_dir / "table2_pairwise_deltas.csv", index=False)

    _make_figure_runtime_scaling(replicates, args.out_dir / "figure1_runtime_scaling.png")
    _make_figure_uncertainty(replicates, args.out_dir / "figure2_vbpca_uncertainty.png")

    print(
        "Generated paper outputs in",
        args.out_dir,
    )


if __name__ == "__main__":
    main()
