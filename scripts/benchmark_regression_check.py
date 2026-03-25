#!/usr/bin/env python3
"""Lightweight regression check for benchmark harnesses.

Runs minimal dense and sparse benchmarks with fixed seeds and optionally
compares outputs against provided baseline CSVs (excluding timing columns
which are hardware-dependent).
"""

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


def _run_dense(out_path: Path) -> Path:
    _run_command([
        "python",
        "scripts/benchmark_dense_runtime.py",
        "--shapes",
        "500x500,200x800",
        "--missing-rates",
        "0.0,0.1",
        "--n-components",
        "6",
        "--maxiters",
        "40",
        "--reps",
        "2",
        "--seed",
        "31415",
        "--runtime-tuning",
        "safe",
        "--compat-mode",
        "modern",
        "--out",
        str(out_path),
    ])
    return out_path


def _run_sparse(out_path: Path) -> Path:
    _run_command([
        "python",
        "scripts/benchmark_sparse_mask_explicit.py",
        "--shape",
        "200x400",
        "--n-components",
        "6",
        "--maxiters",
        "30",
        "--reps",
        "3",
        "--seed",
        "27182",
        "--observed-rate",
        "0.1",
        "--zero-rate",
        "0.5",
        "--runtime-tuning",
        "safe",
        "--compat-mode",
        "modern",
        "--out",
        str(out_path),
    ])
    return out_path


def _filter_comparable_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in frame.columns if "time" not in c]
    return frame.loc[:, cols].copy()


def _assert_frame_close(
    left: pd.DataFrame, right: pd.DataFrame, *, atol: float
) -> None:
    if list(left.columns) != list(right.columns):
        msg = "Column mismatch between benchmark outputs"
        raise AssertionError(msg)

    for column in left.columns:
        lcol = left[column]
        rcol = right[column]
        if pd.api.types.is_numeric_dtype(lcol):
            lvals = np.asarray(lcol, dtype=float)
            rvals = np.asarray(rcol, dtype=float)
            if not np.allclose(lvals, rvals, atol=atol, rtol=0.0, equal_nan=True):
                msg = f"Numeric mismatch in column {column!r}"
                raise AssertionError(msg)
        else:
            if not lcol.astype(str).equals(rcol.astype(str)):
                msg = f"Categorical mismatch in column {column!r}"
                raise AssertionError(msg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("results/bench_repro"))
    parser.add_argument("--dense-baseline", type=Path, default=None)
    parser.add_argument("--sparse-baseline", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    dense_out = _run_dense(args.work_dir / "dense_runtime.csv")
    sparse_out = _run_sparse(args.work_dir / "sparse_explicit.csv")

    dense_frame = _filter_comparable_columns(pd.read_csv(dense_out))
    sparse_frame = _filter_comparable_columns(pd.read_csv(sparse_out))

    if args.dense_baseline is not None:
        dense_base = _filter_comparable_columns(pd.read_csv(args.dense_baseline))
        dense_base = dense_base.sort_values(list(dense_base.columns)).reset_index(
            drop=True
        )
        dense_cur = dense_frame.sort_values(list(dense_frame.columns)).reset_index(
            drop=True
        )
        _assert_frame_close(dense_cur, dense_base, atol=args.atol)

    if args.sparse_baseline is not None:
        sparse_base = _filter_comparable_columns(pd.read_csv(args.sparse_baseline))
        sparse_base = sparse_base.sort_values(list(sparse_base.columns)).reset_index(
            drop=True
        )
        sparse_cur = sparse_frame.sort_values(list(sparse_frame.columns)).reset_index(
            drop=True
        )
        _assert_frame_close(sparse_cur, sparse_base, atol=args.atol)

    print("Benchmark regression check passed:", f"work_dir={args.work_dir}")


if __name__ == "__main__":
    main()
