#!/usr/bin/env python3
"""Aggregate benchmark CSVs into reproducible figures and tables.

Inputs are the long-form CSVs produced by benchmark scripts:
- benchmark_missing_pca.py (replicates + selection trace)
- benchmark_dense_runtime.py
- benchmark_sparse_mask_explicit.py

Outputs are saved under a target directory (figures/*.png, tables/*.csv).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warnings.warn(f"Missing input CSV, skipping: {path}")
        return None
    return pd.read_csv(path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _summarize_imputation(replicates: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "mechanism", "pattern", "missing_rate", "method"]
    agg = replicates.groupby(group_cols, as_index=False).agg(
        n_reps=("rmse", "size"),
        rmse_mean=("rmse", "mean"),
        rmse_median=("rmse", "median"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_median=("mae", "median"),
        mae_std=("mae", "std"),
        wall_time_sec_mean=("wall_time_sec", "mean"),
        wall_time_sec_median=("wall_time_sec", "median"),
        wall_time_sec_std=("wall_time_sec", "std"),
        vbpca_selected_k_median=("vbpca_selected_k", "median"),
    )
    return agg


def _best_method_table(imputation_summary: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["dataset", "mechanism", "pattern", "missing_rate"]
    best_rows: list[dict[str, object]] = []
    for _, group in imputation_summary.groupby(key_cols):
        group_sorted = group.sort_values("rmse_median")
        best = group_sorted.iloc[0]
        delta_col = "rmse_delta_to_best"
        group_sorted[delta_col] = group_sorted["rmse_median"] - float(
            best["rmse_median"]
        )
        best_rows.append(
            {
                **{col: best[col] for col in key_cols},
                "best_method": best["method"],
                "rmse_median": float(best["rmse_median"]),
                "rmse_mean": float(best["rmse_mean"]),
                "runner_up_delta": float(group_sorted[delta_col].iloc[1])
                if len(group_sorted) > 1
                else np.nan,
            }
        )
    return pd.DataFrame(best_rows)


def _runtime_summary(runtime_df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    group_cols = ["shape", "n_features", "n_samples", "n_components", "missing_rate"]
    agg = runtime_df.copy()
    agg["method"] = label
    summary = agg.groupby(group_cols, as_index=False).agg(
        n_reps=("time_sec", "size"),
        time_median=("time_sec", "median"),
        time_mean=("time_sec", "mean"),
        time_std=("time_sec", "std"),
        observed_rate_median=("observed_rate", "median"),
    )
    return summary


def _selection_stability(replicates: pd.DataFrame) -> pd.DataFrame:
    if "vbpca_selected_k" not in replicates.columns:
        return pd.DataFrame()
    group_cols = ["dataset", "mechanism", "pattern", "missing_rate"]
    summary = replicates.groupby(group_cols, as_index=False).agg(
        n_reps=("vbpca_selected_k", "size"),
        k_median=("vbpca_selected_k", "median"),
        k_min=("vbpca_selected_k", "min"),
        k_max=("vbpca_selected_k", "max"),
        k_std=("vbpca_selected_k", "std"),
    )
    return summary


def _selection_stability_dataset(replicates: pd.DataFrame) -> pd.DataFrame:
    if "vbpca_selected_k" not in replicates.columns:
        return pd.DataFrame()
    group_cols = ["dataset"]
    return (
        replicates.groupby(group_cols, as_index=False)
        .agg(
            n_reps=("vbpca_selected_k", "size"),
            k_median=("vbpca_selected_k", "median"),
            k_min=("vbpca_selected_k", "min"),
            k_max=("vbpca_selected_k", "max"),
            k_std=("vbpca_selected_k", "std"),
        )
        .sort_values(group_cols)
    )


def _seed_sensitivity(replicates: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "method"]
    if "seed_method" not in replicates.columns:
        return pd.DataFrame()
    agg = (
        replicates.groupby(group_cols, as_index=False)
        .agg(
            rmse_std_over_seeds=("rmse", "std"),
            mae_std_over_seeds=("mae", "std"),
            rmse_median=("rmse", "median"),
            n_reps=("rmse", "size"),
        )
        .sort_values(group_cols)
    )
    return agg


def _plot_imputation(replicates: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.relplot(
        data=replicates,
        x="missing_rate",
        y="rmse",
        hue="method",
        style="pattern",
        col="dataset",
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
    )
    ax.set_titles("{col_name}")
    ax.set_axis_labels("Missing rate", "RMSE (holdout)")
    if ax._legend is not None:
        ax._legend.set_title("method")
        ax._legend.set_frame_on(False)
    out_path = out_dir / "figure_imputation_quality.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ax.savefig(out_path, dpi=180)
    plt.close(ax.fig)


def _plot_runtime(runtime_df: pd.DataFrame, out_dir: Path, *, label: str) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.5, 4.5))
    plot_df = runtime_df.copy()
    plot_df["size"] = plot_df["n_samples"] * plot_df["n_features"]
    ax = sns.lineplot(
        data=plot_df,
        x="size",
        y="time_sec",
        hue="n_components",
        style="missing_rate",
        marker="o",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size (n_samples × n_features)")
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title(f"Runtime scaling ({label})")
    ax.legend(title="n_components", frameon=False)
    out_path = out_dir / f"figure_runtime_{label}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_stability(replicates: pd.DataFrame, out_dir: Path) -> None:
    if "vbpca_selected_k" not in replicates.columns:
        return
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.5, 4.5))
    ax = sns.boxplot(
        data=replicates,
        x="dataset",
        y="vbpca_selected_k",
        hue="pattern",
        showfliers=False,
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Selected k (VBPCA)")
    ax.set_title("Model-selection stability across seeds")
    ax.legend(title="pattern", frameon=False)
    out_path = out_dir / "figure_model_selection_stability.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_seed_sensitivity(replicates: pd.DataFrame, out_dir: Path) -> None:
    if "seed_method" not in replicates.columns:
        return
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.stripplot(
        data=replicates,
        x="method",
        y="rmse",
        hue="dataset",
        dodge=True,
        alpha=0.65,
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("RMSE by seed")
    ax.set_title("Seed sensitivity (per replicate)")
    ax.legend(title="dataset", frameon=False)
    out_path = out_dir / "figure_seed_sensitivity.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicates", type=Path, default=Path("results/replicates.csv")
    )
    parser.add_argument(
        "--selection-trace",
        type=Path,
        default=Path("results/vbpca_selection_trace.csv"),
    )
    parser.add_argument(
        "--dense-runtime",
        type=Path,
        default=Path("results/perf_baseline/dense_runtime.csv"),
    )
    parser.add_argument(
        "--sparse-runtime",
        type=Path,
        default=Path("results/perf_baseline/sparse_mask_explicit.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/figures_tables"),
        help="Root directory for figures/ and tables/ outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    replicates = _read_csv_optional(args.replicates)
    selection_trace = _read_csv_optional(args.selection_trace)
    dense_runtime = _read_csv_optional(args.dense_runtime)
    sparse_runtime = _read_csv_optional(args.sparse_runtime)

    tables_dir = args.out_dir / "tables"
    figures_dir = args.out_dir / "figures"

    if replicates is not None:
        imputation_summary = _summarize_imputation(replicates)
        _write_csv(imputation_summary, tables_dir / "imputation_summary.csv")
        best_table = _best_method_table(imputation_summary)
        _write_csv(best_table, tables_dir / "best_method_by_setting.csv")
        stability = _selection_stability(replicates)
        if not stability.empty:
            _write_csv(stability, tables_dir / "vbpca_selection_stability.csv")
        stability_ds = _selection_stability_dataset(replicates)
        if not stability_ds.empty:
            _write_csv(
                stability_ds, tables_dir / "vbpca_selection_stability_by_dataset.csv"
            )
        seed_variance = _seed_sensitivity(replicates)
        if not seed_variance.empty:
            _write_csv(seed_variance, tables_dir / "seed_sensitivity.csv")

        _plot_imputation(replicates, figures_dir)
        _plot_stability(replicates, figures_dir)
        _plot_seed_sensitivity(replicates, figures_dir)

    if dense_runtime is not None:
        dense_summary = _runtime_summary(dense_runtime, label="dense")
        _write_csv(dense_summary, tables_dir / "dense_runtime_summary.csv")
        _plot_runtime(dense_runtime, figures_dir, label="dense")

    if sparse_runtime is not None:
        sparse_summary = _runtime_summary(sparse_runtime, label="sparse")
        _write_csv(sparse_summary, tables_dir / "sparse_runtime_summary.csv")
        _plot_runtime(sparse_runtime, figures_dir, label="sparse")

    if selection_trace is not None:
        # Store raw selection trace for reproducibility and future plotting.
        _write_csv(selection_trace, tables_dir / "vbpca_selection_trace.csv")

    print("Wrote figures to", figures_dir)
    print("Wrote tables to", tables_dir)


if __name__ == "__main__":
    main()
