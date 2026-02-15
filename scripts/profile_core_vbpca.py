#!/usr/bin/env python3
"""Profile core VBPCA runtime on fixed dense/sparse workloads.

This script is intended for iterative optimization phases and records
repeatable wall-time baselines for the core `pca_full` implementation.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from vbpca_py._pca_full import pca_full


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    n_features: int
    n_samples: int
    n_components: int
    missing_rate: float


def _build_core_cases() -> list[Case]:
    return [
        Case("dense_small", "dense", 60, 200, 8, 0.20),
        Case("dense_medium", "dense", 120, 400, 12, 0.30),
        Case("sparse_small", "sparse", 80, 250, 10, 0.85),
        Case("sparse_medium", "sparse", 160, 500, 16, 0.92),
    ]


def _build_genetics_like_cases() -> list[Case]:
    return [
        Case("sparse_genetics_2k", "sparse", 2_000, 1_000, 24, 0.97),
        Case("sparse_genetics_5k", "sparse", 5_000, 1_000, 32, 0.985),
    ]


def _build_cases(case_set: str) -> list[Case]:
    if case_set == "core":
        return _build_core_cases()
    if case_set == "genetics":
        return _build_genetics_like_cases()
    if case_set == "all":
        return _build_core_cases() + _build_genetics_like_cases()
    msg = f"Unsupported case_set: {case_set!r}"
    raise ValueError(msg)


def _make_dense(
    rng: np.random.Generator,
    n_features: int,
    n_samples: int,
    missing_rate: float,
) -> np.ndarray:
    x = rng.standard_normal((n_features, n_samples))
    holdout = rng.random((n_features, n_samples)) < missing_rate
    x[holdout] = np.nan
    return x


def _make_sparse(
    rng: np.random.Generator,
    n_features: int,
    n_samples: int,
    missing_rate: float,
) -> sp.csr_matrix:
    observed_prob = max(0.01, 1.0 - missing_rate)
    dense = rng.standard_normal((n_features, n_samples))
    keep = rng.random((n_features, n_samples)) < observed_prob
    dense[~keep] = 0.0
    return sp.csr_matrix(dense)


def _run_case(
    case: Case,
    seed: int,
    maxiters: int,
    reps: int,
    rotate2pca: int,
) -> dict[str, float | int | str]:
    times: list[float] = []

    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        if case.kind == "dense":
            x = _make_dense(
                rng,
                n_features=case.n_features,
                n_samples=case.n_samples,
                missing_rate=case.missing_rate,
            )
        else:
            x = _make_sparse(
                rng,
                n_features=case.n_features,
                n_samples=case.n_samples,
                missing_rate=case.missing_rate,
            )

        start = time.perf_counter()
        _ = pca_full(
            x,
            case.n_components,
            algorithm="vb",
            compat_mode="modern",
            rotate2pca=int(bool(rotate2pca)),
            verbose=0,
            maxiters=maxiters,
        )
        times.append(time.perf_counter() - start)

    arr = np.asarray(times, dtype=float)
    return {
        "case": case.name,
        "kind": case.kind,
        "n_features": case.n_features,
        "n_samples": case.n_samples,
        "n_components": case.n_components,
        "missing_rate": case.missing_rate,
        "rotate2pca": int(bool(rotate2pca)),
        "repetitions": reps,
        "time_mean_sec": float(np.mean(arr)),
        "time_median_sec": float(np.median(arr)),
        "time_std_sec": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "time_min_sec": float(np.min(arr)),
        "time_max_sec": float(np.max(arr)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--maxiters", type=int, default=40)
    parser.add_argument(
        "--rotate2pca",
        type=int,
        choices=(0, 1),
        default=1,
        help="Enable rotate-to-PCA during training (1=on, 0=off).",
    )
    parser.add_argument(
        "--case-set",
        type=str,
        choices=("core", "genetics", "all"),
        default="core",
        help="Case set to profile: core defaults, genetics-like large sparse, or all.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/perf_baseline/core_vbpca_baseline.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = [
        _run_case(
            case,
            seed=args.seed,
            maxiters=args.maxiters,
            reps=args.reps,
            rotate2pca=args.rotate2pca,
        )
        for case in _build_cases(args.case_set)
    ]

    frame = pd.DataFrame(rows).sort_values(["kind", "case"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print("Wrote core baseline:", args.out)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
