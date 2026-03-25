#!/usr/bin/env python3
"""Validate deterministic reproducibility for the benchmark pipeline."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        msg = f"Command failed ({completed.returncode}): {' '.join(command)}"
        raise RuntimeError(msg)


def _run_once(out_prefix: Path, n_jobs: int) -> tuple[Path, Path]:
    replicates = out_prefix.with_suffix(".replicates.csv")
    summary = out_prefix.with_suffix(".summary.csv")
    pairwise = out_prefix.with_suffix(".pairwise.csv")

    _run_command([
        "python",
        "scripts/benchmark_missing_pca.py",
        "--datasets",
        "synthetic",
        "--mechanisms",
        "MCAR",
        "--patterns",
        "random",
        "--missing-rates",
        "0.3",
        "--n-reps",
        "6",
        "--n-components",
        "5",
        "--n-jobs",
        str(n_jobs),
        "--random-seed",
        "424242",
        "--synthetic-shape",
        "300x25",
        "--vbpca-maxiters",
        "20",
        "--mice-max-iter",
        "5",
        "--output",
        str(replicates),
    ])

    _run_command([
        "python",
        "scripts/summarize_missing_pca.py",
        "--input",
        str(replicates),
        "--summary-output",
        str(summary),
        "--pairwise-output",
        str(pairwise),
    ])

    return summary, pairwise


def _assert_frame_close(left: pd.DataFrame, right: pd.DataFrame, atol: float) -> None:
    if list(left.columns) != list(right.columns):
        msg = "Column mismatch between reproducibility outputs"
        raise AssertionError(msg)

    for column in left.columns:
        lcol = left[column]
        rcol = right[column]
        if pd.api.types.is_numeric_dtype(lcol):
            left_vals = np.asarray(lcol, dtype=float)
            right_vals = np.asarray(rcol, dtype=float)

            is_time_col = "wall_time_sec" in column
            is_delta_col = column.startswith("delta_")
            metric_is_time = (
                "metric" in left.columns
                and left["metric"].astype(str).to_numpy() == "wall_time_sec"
            )

            base_mask = np.ones(left_vals.shape, dtype=bool)
            if is_delta_col and "metric" in left.columns:
                strict_mask = base_mask & ~metric_is_time
                loose_mask = base_mask & metric_is_time
            elif is_time_col:
                strict_mask = np.zeros(left_vals.shape, dtype=bool)
                loose_mask = base_mask
            else:
                strict_mask = base_mask
                loose_mask = np.zeros(left_vals.shape, dtype=bool)

            if np.any(strict_mask):
                if not np.allclose(
                    left_vals[strict_mask],
                    right_vals[strict_mask],
                    atol=atol,
                    rtol=0.0,
                    equal_nan=True,
                ):
                    msg = f"Numeric mismatch in column {column!r}"
                    raise AssertionError(msg)

            if np.any(loose_mask):
                if not np.allclose(
                    left_vals[loose_mask],
                    right_vals[loose_mask],
                    atol=max(atol, 1e-2),
                    rtol=0.25,
                    equal_nan=True,
                ):
                    msg = f"Time-like numeric mismatch in column {column!r}"
                    raise AssertionError(msg)
        else:
            if not lcol.astype(str).equals(rcol.astype(str)):
                msg = f"Categorical mismatch in column {column!r}"
                raise AssertionError(msg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("results/repro"),
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=None,
        help="Optional summary CSV to compare against the first run.",
    )
    parser.add_argument(
        "--baseline-pairwise",
        type=Path,
        default=None,
        help="Optional pairwise CSV to compare against the first run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    first_summary, first_pairwise = _run_once(
        args.work_dir / "run1", n_jobs=args.n_jobs
    )
    second_summary, second_pairwise = _run_once(
        args.work_dir / "run2", n_jobs=args.n_jobs
    )

    summary_a = pd.read_csv(first_summary)
    summary_b = pd.read_csv(second_summary)
    pairwise_a = pd.read_csv(first_pairwise)
    pairwise_b = pd.read_csv(second_pairwise)

    summary_keys = [
        "dataset",
        "mechanism",
        "pattern",
        "missing_rate",
        "n_components",
        "synthetic_shape",
        "method",
        "n_reps",
    ]
    pairwise_keys = [
        "dataset",
        "mechanism",
        "pattern",
        "missing_rate",
        "n_components",
        "synthetic_shape",
        "comparator",
        "reference",
        "metric",
        "n_reps",
    ]

    summary_a = summary_a.sort_values(summary_keys).reset_index(drop=True)
    summary_b = summary_b.sort_values(summary_keys).reset_index(drop=True)
    pairwise_a = pairwise_a.sort_values(pairwise_keys).reset_index(drop=True)
    pairwise_b = pairwise_b.sort_values(pairwise_keys).reset_index(drop=True)

    _assert_frame_close(summary_a, summary_b, atol=args.atol)
    _assert_frame_close(pairwise_a, pairwise_b, atol=args.atol)

    if args.baseline_summary is not None:
        baseline_summary = pd.read_csv(args.baseline_summary)
        baseline_summary = baseline_summary.sort_values(summary_keys).reset_index(
            drop=True
        )
        _assert_frame_close(summary_a, baseline_summary, atol=args.atol)

    if args.baseline_pairwise is not None:
        baseline_pairwise = pd.read_csv(args.baseline_pairwise)
        baseline_pairwise = baseline_pairwise.sort_values(pairwise_keys).reset_index(
            drop=True
        )
        _assert_frame_close(pairwise_a, baseline_pairwise, atol=args.atol)

    print("Reproducibility check passed:", f"work_dir={args.work_dir}")


if __name__ == "__main__":
    main()
