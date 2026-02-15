#!/usr/bin/env python3
"""Generate supplementary model-selection artifacts for VBPCA."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SETTING_COLS = [
    "dataset",
    "mechanism",
    "pattern",
    "missing_rate",
    "n_components_requested",
    "synthetic_shape",
]


def _setting_label(row: pd.Series) -> str:
    return (
        f"{row['dataset']} | {row['mechanism']} | {row['pattern']} | "
        f"r={float(row['missing_rate']):.1f}"
    )


def _mode_selected_k(series: pd.Series) -> int:
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return -1
    return int(counts.index[0])


def _extract_reversal_points(trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = [*SETTING_COLS, "replicate_id"]

    for group_key, group in trace.groupby(group_cols, dropna=False):
        group_sorted = group.sort_values("trace_k")
        k_vals = group_sorted["trace_k"].to_numpy(dtype=int)
        rms_vals = group_sorted["trace_rms"].to_numpy(dtype=float)

        reversal_k: int | None = None
        q_k: int
        for idx in range(1, len(rms_vals)):
            if rms_vals[idx] > rms_vals[idx - 1]:
                reversal_k = int(k_vals[idx])
                q_k = int(k_vals[idx - 1])
                break
        else:
            q_k = int(k_vals[-1])

        key_map = dict(zip(group_cols, group_key, strict=True))
        rows.append(
            {
                **key_map,
                "q": q_k,
                "q_plus_1": reversal_k,
                "has_reversal": reversal_k is not None,
                "trace_k_max": int(k_vals.max()),
            }
        )

    return pd.DataFrame(rows)


def _summarize_q_variation(reversal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for group_key, group in reversal.groupby(SETTING_COLS, dropna=False):
        key_map = dict(zip(SETTING_COLS, group_key, strict=True))
        q_vals = np.asarray(group["q"], dtype=float)
        q_plus_1_vals = np.asarray(
            group.loc[group["q_plus_1"].notna(), "q_plus_1"],
            dtype=float,
        )
        reversal_rate = float(np.mean(np.asarray(group["has_reversal"], dtype=bool)))

        row: dict[str, object] = {
            **key_map,
            "n_reps": int(len(group)),
            "n_with_reversal": int(np.count_nonzero(group["has_reversal"])),
            "reversal_rate": reversal_rate,
            "q_min": int(np.min(q_vals)),
            "q_p25": float(np.percentile(q_vals, 25)),
            "q_median": float(np.percentile(q_vals, 50)),
            "q_p75": float(np.percentile(q_vals, 75)),
            "q_max": int(np.max(q_vals)),
            "q_mode": _mode_selected_k(group["q"]),
        }

        if q_plus_1_vals.size > 0:
            row["q_plus_1_min"] = int(np.min(q_plus_1_vals))
            row["q_plus_1_median"] = float(np.percentile(q_plus_1_vals, 50))
            row["q_plus_1_max"] = int(np.max(q_plus_1_vals))
        else:
            row["q_plus_1_min"] = np.nan
            row["q_plus_1_median"] = np.nan
            row["q_plus_1_max"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values(SETTING_COLS).reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicates",
        type=Path,
        default=Path("results/replicates.csv"),
    )
    parser.add_argument(
        "--selection-trace",
        type=Path,
        default=Path("results/vbpca_selection_trace.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paper"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reps = pd.read_csv(args.replicates)
    trace = pd.read_csv(args.selection_trace)

    vbpca_reps = reps[reps["method"] == "vbpca_vb_modern"].copy()
    if vbpca_reps.empty or trace.empty:
        print("No VBPCA selection data found; skipping supplement generation.")
        return

    vbpca_reps["selected_k"] = vbpca_reps["vbpca_selected_k"].astype(int)

    freq = (
        vbpca_reps.groupby(SETTING_COLS + ["selected_k"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = freq.groupby(SETTING_COLS, as_index=False)["count"].sum().rename(
        columns={"count": "total"}
    )
    freq = freq.merge(totals, on=SETTING_COLS, how="left")
    freq["fraction"] = freq["count"] / freq["total"]
    freq.to_csv(args.out_dir / "tableS1_vbpca_selected_k_frequency.csv", index=False)

    trace_stats = (
        trace.groupby(SETTING_COLS + ["trace_k"], as_index=False)["trace_rms"]
        .agg(
            rms_mean="mean",
            rms_ci_low=lambda s: np.percentile(np.asarray(s, dtype=float), 2.5),
            rms_ci_high=lambda s: np.percentile(np.asarray(s, dtype=float), 97.5),
        )
        .sort_values(SETTING_COLS + ["trace_k"])
        .reset_index(drop=True)
    )
    trace_stats.to_csv(args.out_dir / "tableS2_vbpca_trace_rms_by_k.csv", index=False)

    reversal = _extract_reversal_points(trace)
    reversal.to_csv(args.out_dir / "tableS3_vbpca_reversal_points_by_replicate.csv", index=False)

    q_summary = _summarize_q_variation(reversal)
    q_summary.to_csv(args.out_dir / "tableS4_vbpca_reversal_summary_by_setting.csv", index=False)

    selected_mode = (
        vbpca_reps.groupby(SETTING_COLS, as_index=False)["selected_k"]
        .agg(_mode_selected_k)
        .rename(columns={"selected_k": "selected_k_mode"})
    )

    plot_data = trace_stats.merge(selected_mode, on=SETTING_COLS, how="left")
    plot_data = plot_data.merge(
        q_summary[
            [
                *SETTING_COLS,
                "q_min",
                "q_median",
                "q_max",
                "q_plus_1_max",
                "reversal_rate",
            ]
        ],
        on=SETTING_COLS,
        how="left",
    )
    settings = (
        plot_data[SETTING_COLS]
        .drop_duplicates()
        .sort_values(SETTING_COLS)
        .reset_index(drop=True)
    )

    n_settings = len(settings)
    n_cols = min(3, n_settings)
    n_rows = int(np.ceil(n_settings / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )

    for idx, (_, setting) in enumerate(settings.iterrows()):
        ax = axes[idx // n_cols][idx % n_cols]
        mask = np.ones(len(plot_data), dtype=bool)
        for col in SETTING_COLS:
            mask &= plot_data[col] == setting[col]
        subset = plot_data.loc[mask].sort_values("trace_k")

        x = subset["trace_k"].to_numpy(dtype=int)
        y = subset["rms_mean"].to_numpy(dtype=float)
        low = subset["rms_ci_low"].to_numpy(dtype=float)
        high = subset["rms_ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - low, high - y])

        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3)

        q_min = int(subset["q_min"].iloc[0])
        q_max = int(subset["q_max"].iloc[0])
        q_median = float(subset["q_median"].iloc[0])
        q_plus_1_max_val = subset["q_plus_1_max"].iloc[0]
        reversal_rate = float(subset["reversal_rate"].iloc[0])

        ax.axvspan(q_min - 0.1, q_max + 0.1, color="tab:orange", alpha=0.12)
        ax.axvline(q_median, color="tab:red", linestyle="--", linewidth=1.2)

        if np.isfinite(q_plus_1_max_val):
            axis_upper_k = max(int(q_plus_1_max_val), q_max + 1)
        else:
            axis_upper_k = q_max + 1
        axis_upper_k = max(axis_upper_k, int(np.max(x)))
        ax.set_xlim(0.8, axis_upper_k + 0.2)

        selected_k_mode = int(subset["selected_k_mode"].iloc[0])
        if selected_k_mode > 0:
            ax.axvline(
                selected_k_mode,
                color="tab:purple",
                linestyle=":",
                linewidth=1.0,
            )

        ax.set_title(
            f"{_setting_label(setting)}\n"
            f"q range [{q_min}, {q_max}], reversal rate={reversal_rate:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("Candidate k")
        ax.set_ylabel("Held-out RMS")
        ax.grid(alpha=0.25)

    for idx in range(n_settings, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("Figure S1: VBPCA model-selection trace (RMS vs k)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out_dir / "figureS1_vbpca_model_selection.png", dpi=180)
    plt.close(fig)

    print("Generated selection supplement in", args.out_dir)


if __name__ == "__main__":
    main()
