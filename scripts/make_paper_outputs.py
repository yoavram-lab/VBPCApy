#!/usr/bin/env python3
"""Generate paper-facing tables and figures from benchmark summaries."""

from __future__ import annotations

import argparse
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VBPCA_METHOD = "vbpca_vb_modern"
VBPCA_REP_UNCERTAINTY_COL = "vbpca_median_variance"
VBPCA_REP_UNCERTAINTY_HOLDOUT_COL = "vbpca_median_variance_holdout"
VBPCA_REP_UNCERTAINTY_OBSERVED_COL = "vbpca_median_variance_observed"


def _format_ci(low: float, high: float) -> str:
    return f"[{low:.4f}, {high:.4f}]"


def _index_to_setting_id(index: int) -> str:
    """Convert 0-based index to alphabetic setting ID: A..Z, AA..AZ, ..."""
    if index < 0:
        msg = "index must be non-negative"
        raise ValueError(msg)

    letters = string.ascii_uppercase
    base = len(letters)
    n = index
    out: list[str] = []
    while True:
        n, rem = divmod(n, base)
        out.append(letters[rem])
        if n == 0:
            break
        n -= 1
    return "".join(reversed(out))


def _build_setting_key(replicates: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "mechanism",
        "pattern",
        "missing_rate",
        "n_components_requested",
        "synthetic_shape",
    ]
    key = (
        replicates.loc[:, cols]
        .drop_duplicates()
        .sort_values(["dataset", "mechanism", "pattern", "missing_rate"])
        .reset_index(drop=True)
    )
    key.insert(0, "setting_id", [_index_to_setting_id(i) for i in range(len(key))])
    key["setting_id_dataset"] = (
        key.groupby("dataset").cumcount().map(_index_to_setting_id)
    )
    key["missing_setting"] = (
        key["mechanism"]
        + "|"
        + key["pattern"]
        + "|r="
        + key["missing_rate"].map(lambda x: f"{x:.1f}")
    )
    return key


def _build_vbpca_uncertainty_long(
    replicates: pd.DataFrame,
    setting_key: pd.DataFrame,
) -> pd.DataFrame:
    vbpca = replicates[replicates["method"] == VBPCA_METHOD].copy()
    if vbpca.empty:
        return pd.DataFrame(
            columns=[
                "setting_id",
                "setting_id_dataset",
                "dataset",
                "mechanism",
                "pattern",
                "missing_rate",
                "n_components_requested",
                "synthetic_shape",
                "missing_setting",
                "replicate_id",
                "scope",
                "median_marginal_variance",
            ]
        )

    required_cols = {
        VBPCA_REP_UNCERTAINTY_HOLDOUT_COL,
        VBPCA_REP_UNCERTAINTY_OBSERVED_COL,
    }
    missing = sorted(required_cols.difference(vbpca.columns))
    if missing:
        msg = (
            "Replicates are missing required uncertainty split columns: "
            f"{', '.join(missing)}. Rerun benchmark_missing_pca.py."
        )
        raise ValueError(msg)

    merge_cols = [
        "dataset",
        "mechanism",
        "pattern",
        "missing_rate",
        "n_components_requested",
        "synthetic_shape",
    ]
    vbpca = vbpca.merge(setting_key, how="left", on=merge_cols, validate="many_to_one")

    long = vbpca.melt(
        id_vars=[
            "setting_id",
            "setting_id_dataset",
            "dataset",
            "mechanism",
            "pattern",
            "missing_rate",
            "n_components_requested",
            "synthetic_shape",
            "missing_setting",
            "replicate_id",
        ],
        value_vars=[
            VBPCA_REP_UNCERTAINTY_OBSERVED_COL,
            VBPCA_REP_UNCERTAINTY_HOLDOUT_COL,
        ],
        var_name="scope_raw",
        value_name="median_marginal_variance",
    )
    scope_map = {
        VBPCA_REP_UNCERTAINTY_OBSERVED_COL: "Observed",
        VBPCA_REP_UNCERTAINTY_HOLDOUT_COL: "Held-out",
    }
    long["scope"] = long["scope_raw"].map(scope_map)
    long = long.drop(columns=["scope_raw"])
    long = long.dropna(subset=["median_marginal_variance"])
    setting_order = {
        sid: i for i, sid in enumerate(setting_key["setting_id_dataset"].drop_duplicates().tolist())
    }
    long["setting_order"] = long["setting_id_dataset"].map(setting_order)
    long = long.sort_values(["dataset", "setting_order", "replicate_id", "scope"]).drop(
        columns=["setting_order"]
    )
    return long.reset_index(drop=True)


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

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.2, 4.8))
    ax = sns.lineplot(
        data=grouped,
        x="size",
        y="mean",
        hue="method",
        marker="o",
        linewidth=1.2,
        estimator=None,
    )
    palette = sns.color_palette(n_colors=grouped["method"].nunique())
    color_by_method = {
        method: palette[idx]
        for idx, method in enumerate(grouped["method"].drop_duplicates().tolist())
    }
    for method, subset in grouped.groupby("method", sort=False):
        x = subset["size"].to_numpy(dtype=float)
        ci_low = subset["ci_low"].to_numpy(dtype=float)
        ci_high = subset["ci_high"].to_numpy(dtype=float)
        ax.fill_between(x, ci_low, ci_high, alpha=0.15, color=color_by_method[method])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dataset size (n_samples × n_features)")
    plt.ylabel("Wall time (seconds)")
    plt.title("Figure 1: Runtime scaling by method")
    plt.legend(frameon=False, title="method")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _make_figure_uncertainty(replicates: pd.DataFrame, out_path: Path) -> None:
    setting_key = _build_setting_key(replicates)
    long = _build_vbpca_uncertainty_long(replicates, setting_key)
    if long.empty:
        return

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=long,
        x="setting_id_dataset",
        y="median_marginal_variance",
        hue="scope",
        col="dataset",
        col_wrap=2,
        kind="box",
        palette="Paired",
        showfliers=False,
        linewidth=0.8,
        height=4.2,
        aspect=1.35,
        sharey=True,
    )
    g.set_axis_labels("Setting", "Median marginal variance")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=0)
    if g._legend is not None:
        g._legend.set_title("Entry subset")
        g._legend.set_frame_on(False)
    g.figure.suptitle(
        "Figure 2: VBPCA uncertainty by dataset (paired observed vs held-out)",
        y=1.02,
    )
    g.figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(out_path, dpi=180)
    plt.close(g.figure)


def _build_robust_uncertainty_tables(replicates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    setting_key = _build_setting_key(replicates)
    long = _build_vbpca_uncertainty_long(replicates, setting_key)
    if long.empty:
        cols_setting = [
            "setting_id",
            "dataset",
            "mechanism",
            "pattern",
            "missing_rate",
            "scope",
            "median",
            "q25",
            "q75",
            "p95",
            "max",
        ]
        cols_dataset = ["dataset", "scope", "median", "q25", "q75", "p95", "max"]
        return pd.DataFrame(columns=cols_setting), pd.DataFrame(columns=cols_dataset)

    by_setting = (
        long.groupby(
            [
                "setting_id",
                "setting_id_dataset",
                "dataset",
                "mechanism",
                "pattern",
                "missing_rate",
                "missing_setting",
                "scope",
            ],
            as_index=False,
        )["median_marginal_variance"]
        .agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            p95=lambda s: s.quantile(0.95),
            max="max",
        )
        .sort_values(["dataset", "setting_id_dataset", "scope"])
    )
    by_dataset = (
        long.groupby(["dataset", "scope"], as_index=False)["median_marginal_variance"]
        .agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            p95=lambda s: s.quantile(0.95),
            max="max",
        )
        .sort_values(["dataset", "scope"])
    )
    return by_setting, by_dataset


def _build_setting_legend_table(replicates: pd.DataFrame) -> pd.DataFrame:
    key = _build_setting_key(replicates)
    return key.rename(columns={"missing_rate": "missing_rate_r"})


def _build_uncertainty_replicate_long_table(replicates: pd.DataFrame) -> pd.DataFrame:
    setting_key = _build_setting_key(replicates)
    return _build_vbpca_uncertainty_long(replicates, setting_key)


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
    table_s0 = _build_setting_legend_table(replicates)
    table_s5, table_s6 = _build_robust_uncertainty_tables(replicates)
    table_s7 = _build_uncertainty_replicate_long_table(replicates)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table1.to_csv(args.out_dir / "table1_method_summary.csv", index=False)
    table2.to_csv(args.out_dir / "table2_pairwise_deltas.csv", index=False)
    table_s0.to_csv(args.out_dir / "tableS0_setting_key.csv", index=False)
    table_s5.to_csv(args.out_dir / "tableS5_vbpca_uncertainty_robust_by_setting.csv", index=False)
    table_s6.to_csv(args.out_dir / "tableS6_vbpca_uncertainty_robust_by_dataset.csv", index=False)
    table_s7.to_csv(args.out_dir / "tableS7_vbpca_uncertainty_replicate_long.csv", index=False)

    _make_figure_runtime_scaling(replicates, args.out_dir / "figure1_runtime_scaling.png")
    _make_figure_uncertainty(replicates, args.out_dir / "figure2_vbpca_uncertainty.png")

    print(
        "Generated paper outputs in",
        args.out_dir,
    )


if __name__ == "__main__":
    main()
