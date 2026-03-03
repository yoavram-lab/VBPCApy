#!/usr/bin/env python3
"""Benchmark explicit-mask VBPCA runtime on sparse synthetic data."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from vbpca_py import VBPCA


def _host_info() -> dict[str, object]:
    cpu_count = os.cpu_count() or 1
    mem_gb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        mem_kb = float(parts[1])
                        mem_gb = mem_kb / (1024**2)
                    break
    except OSError:
        mem_gb = None

    return {"host_cpu_count": int(cpu_count), "host_mem_gb": mem_gb}


def _parse_shape(raw: str) -> tuple[int, int]:
    parts = raw.lower().split("x")
    if len(parts) != 2:
        msg = f"Shape must be like '200x400', got {raw!r}"
        raise ValueError(msg)
    return int(parts[0]), int(parts[1])


def _stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _make_sparse_problem(
    n_features: int,
    n_samples: int,
    observed_rate: float,
    zero_rate: float,
    seed: int,
) -> tuple[sp.csr_matrix, np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_features, n_samples))

    mask = rng.random(data.shape) < observed_rate
    data = data * mask

    if zero_rate > 0.0:
        zero_mask = (rng.random(data.shape) < zero_rate) & mask
        data[zero_mask] = 0.0

    csr = sp.csr_matrix(data)
    observed_frac = float(np.mean(mask))
    nnz_frac = float(csr.nnz / float(n_features * n_samples))
    return csr, mask, observed_frac, nnz_frac


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        type=str,
        default="200x400",
        help="Matrix shape as '<features>x<samples>'.",
    )
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--maxiters", type=int, default=40)
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Number of timing repetitions.",
    )
    parser.add_argument("--seed", type=int, default=202601, help="Base RNG seed.")
    parser.add_argument(
        "--observed-rate",
        type=float,
        default=0.1,
        help="Fraction of entries marked observed in the explicit mask.",
    )
    parser.add_argument(
        "--zero-rate",
        type=float,
        default=0.5,
        help="Fraction of observed entries forced to zero (controls sparsity).",
    )
    parser.add_argument(
        "--compat-mode",
        type=str,
        default="modern",
        choices=["modern", "strict_legacy"],
    )
    parser.add_argument(
        "--runtime-tuning",
        type=str,
        default="safe",
        choices=["off", "safe", "aggressive"],
    )
    parser.add_argument("--num-cpu", type=int, default=None)
    parser.add_argument("--rotate2pca", type=int, default=1)
    parser.add_argument(
        "--accessor-mode",
        type=str,
        default="auto",
        choices=["auto", "legacy", "buffered"],
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default="sparse_explicit_mask",
        help="Label for the case column in the output CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/perf_baseline/sparse_mask_explicit.csv"),
        help="Output CSV path for aggregate timings.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity passed to VBPCA (0 is silent).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (0.0 < args.observed_rate <= 1.0):
        msg = "--observed-rate must be in (0, 1]"
        raise ValueError(msg)
    if not (0.0 <= args.zero_rate <= 1.0):
        msg = "--zero-rate must be in [0, 1]"
        raise ValueError(msg)

    n_features, n_samples = _parse_shape(args.shape)

    timings: list[float] = []
    observed_fracs: list[float] = []
    nnz_fracs: list[float] = []

    for rep in range(int(args.reps)):
        seed = int(args.seed + rep)
        x, mask, obs_frac, nnz_frac = _make_sparse_problem(
            n_features=n_features,
            n_samples=n_samples,
            observed_rate=float(args.observed_rate),
            zero_rate=float(args.zero_rate),
            seed=seed,
        )

        model_kwargs: dict[str, object] = {
            "n_components": int(args.n_components),
            "maxiters": int(args.maxiters),
            "compat_mode": str(args.compat_mode),
            "runtime_tuning": str(args.runtime_tuning),
            "rotate2pca": int(args.rotate2pca),
            "verbose": int(args.verbose),
            "accessor_mode": str(args.accessor_mode),
        }
        if args.num_cpu is not None:
            model_kwargs["num_cpu"] = int(args.num_cpu)

        model = VBPCA(**model_kwargs)

        start = time.perf_counter()
        model.fit(x, mask=mask)
        elapsed = time.perf_counter() - start

        timings.append(float(elapsed))
        observed_fracs.append(obs_frac)
        nnz_fracs.append(nnz_frac)

    timing_stats = _stats(timings)
    observed_mean = float(np.mean(observed_fracs))
    nnz_mean = float(np.mean(nnz_fracs))
    host = _host_info()

    row = {
        "case": args.case_name,
        "kind": "sparse_explicit_mask",
        "n_features": int(n_features),
        "n_samples": int(n_samples),
        "n_components": int(args.n_components),
        "observed_rate": observed_mean,
        "missing_rate": 1.0 - observed_mean,
        "zero_rate": float(args.zero_rate),
        "nnz_fraction": nnz_mean,
        "rotate2pca": int(args.rotate2pca),
        "runtime_tuning": str(args.runtime_tuning),
        "compat_mode": str(args.compat_mode),
        "num_cpu": None if args.num_cpu is None else int(args.num_cpu),
        "accessor_mode": str(args.accessor_mode),
        "repetitions": int(args.reps),
        "seed_base": int(args.seed),
        "host_cpu_count": host["host_cpu_count"],
        "host_mem_gb": host["host_mem_gb"],
        "time_mean_sec": timing_stats["mean"],
        "time_median_sec": timing_stats["median"],
        "time_std_sec": timing_stats["std"],
        "time_min_sec": timing_stats["min"],
        "time_max_sec": timing_stats["max"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and args.out.exists() else "w"
    header = not (args.append and args.out.exists())
    pd.DataFrame([row]).to_csv(args.out, mode=mode, header=header, index=False)

    print("Recorded timing row:")
    for key, value in row.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
