#!/usr/bin/env python3
"""Aggregate per-replicate benchmark results into publication-ready summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_COLS = [
    "dataset",
    "mechanism",
    "pattern",
    "missing_rate",
    "n_components_requested",
    "synthetic_shape",
    "method",
]
REPLICATE_KEY_COLS = [
    "dataset",
    "mechanism",
    "pattern",
    "missing_rate",
    "n_components_requested",
    "synthetic_shape",
    "replicate_id",
]
METRICS = ["rmse", "mae", "wall_time_sec"]
VBPCA_METHOD = "vbpca_vb_modern"


def _summary_stats(values: pd.Series) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }


def _aggregate_method_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, group in frame.groupby(GROUP_COLS, dropna=False):
        key_map = dict(zip(GROUP_COLS, group_key, strict=True))
        row: dict[str, object] = {**key_map, "n_reps": int(group.shape[0])}
        for metric in METRICS:
            stats = _summary_stats(group[metric])
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_median"] = stats["median"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_ci_low"] = stats["ci_low"]
            row[f"{metric}_ci_high"] = stats["ci_high"]

        if key_map["method"] == VBPCA_METHOD:
            var_stats = _summary_stats(group["vbpca_mean_variance"])
            row["vbpca_mean_variance_mean"] = var_stats["mean"]
            row["vbpca_mean_variance_median"] = var_stats["median"]
            row["vbpca_mean_variance_std"] = var_stats["std"]
            row["vbpca_mean_variance_ci_low"] = var_stats["ci_low"]
            row["vbpca_mean_variance_ci_high"] = var_stats["ci_high"]
            if "vbpca_median_variance" in group.columns:
                med_var_stats = _summary_stats(group["vbpca_median_variance"])
                row["vbpca_median_variance_mean"] = med_var_stats["mean"]
                row["vbpca_median_variance_median"] = med_var_stats["median"]
                row["vbpca_median_variance_std"] = med_var_stats["std"]
                row["vbpca_median_variance_ci_low"] = med_var_stats["ci_low"]
                row["vbpca_median_variance_ci_high"] = med_var_stats["ci_high"]
            else:
                row["vbpca_median_variance_mean"] = np.nan
                row["vbpca_median_variance_median"] = np.nan
                row["vbpca_median_variance_std"] = np.nan
                row["vbpca_median_variance_ci_low"] = np.nan
                row["vbpca_median_variance_ci_high"] = np.nan
        else:
            row["vbpca_mean_variance_mean"] = np.nan
            row["vbpca_mean_variance_median"] = np.nan
            row["vbpca_mean_variance_std"] = np.nan
            row["vbpca_mean_variance_ci_low"] = np.nan
            row["vbpca_mean_variance_ci_high"] = np.nan
            row["vbpca_median_variance_mean"] = np.nan
            row["vbpca_median_variance_median"] = np.nan
            row["vbpca_median_variance_std"] = np.nan
            row["vbpca_median_variance_ci_low"] = np.nan
            row["vbpca_median_variance_ci_high"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values(GROUP_COLS).reset_index(drop=True)


def _aggregate_pairwise(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index=REPLICATE_KEY_COLS,
        columns="method",
        values=METRICS,
        aggfunc="first",
    )

    if VBPCA_METHOD not in pivot.columns.get_level_values(1):
        msg = f"Missing required method {VBPCA_METHOD!r} for pairwise comparisons"
        raise ValueError(msg)

    methods = sorted(set(frame["method"]) - {VBPCA_METHOD})
    rows: list[dict[str, object]] = []

    for comparator in methods:
        if comparator not in pivot.columns.get_level_values(1):
            continue

        for metric in METRICS:
            delta = pivot[(metric, comparator)] - pivot[(metric, VBPCA_METHOD)]
            delta = delta.dropna()
            if delta.empty:
                continue

            base_cols = REPLICATE_KEY_COLS[:-1]
            group_index_frame = delta.index.to_frame(index=False)

            for group_vals, idx in group_index_frame.groupby(base_cols, dropna=False).groups.items():
                group_delta = np.asarray(delta.iloc[list(idx)], dtype=float)
                key_map = dict(zip(base_cols, group_vals, strict=True))
                rows.append(
                    {
                        **key_map,
                        "comparator": comparator,
                        "reference": VBPCA_METHOD,
                        "metric": metric,
                        "n_reps": int(group_delta.size),
                        "delta_mean": float(np.mean(group_delta)),
                        "delta_median": float(np.median(group_delta)),
                        "delta_std": float(np.std(group_delta, ddof=1)) if group_delta.size > 1 else 0.0,
                        "delta_ci_low": float(np.percentile(group_delta, 2.5)),
                        "delta_ci_high": float(np.percentile(group_delta, 97.5)),
                        "comparator_win_rate": float(np.mean(group_delta < 0.0)),
                        "vbpca_win_rate": float(np.mean(group_delta > 0.0)),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        [
            "dataset",
            "mechanism",
            "pattern",
            "missing_rate",
            "n_components_requested",
            "synthetic_shape",
            "metric",
            "comparator",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/replicates.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/summary.csv"),
    )
    parser.add_argument(
        "--pairwise-output",
        type=Path,
        default=Path("results/pairwise_summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frame = pd.read_csv(args.input)

    summary = _aggregate_method_summary(frame)
    pairwise = _aggregate_pairwise(frame)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    pairwise.to_csv(args.pairwise_output, index=False)

    print(
        "Wrote summaries:",
        f"summary={args.summary_output}",
        f"pairwise={args.pairwise_output}",
    )


if __name__ == "__main__":
    main()
