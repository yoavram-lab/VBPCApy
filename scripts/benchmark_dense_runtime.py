#!/usr/bin/env python3
"""Benchmark dense VBPCA runtime on selected shapes with small missingness.

Records wall-clock timing for a handful of shapes and missingness rates while
capturing host info to contextualize autotuning decisions.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from vbpca_py import VBPCA


def _parse_shape(raw: str) -> tuple[int, int]:
    parts = raw.lower().split("x")
    if len(parts) != 2:
        msg = f"Shape must be like '500x500', got {raw!r}"
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


def _make_dense_problem(
    n_features: int,
    n_samples: int,
    missing_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_features, n_samples))
    if missing_rate <= 0.0:
        mask = np.ones_like(x, dtype=bool)
        return x, mask

    mask = rng.random(x.shape) > missing_rate
    x = x.copy()
    x[~mask] = np.nan
    return x, mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=str,
        default="500x500,1000x200,200x2000",
        help="Comma-separated shapes as '<features>x<samples>'.",
    )
    parser.add_argument(
        "--missing-rates",
        type=str,
        default="0.0,0.1",
        help="Comma-separated missingness rates in [0,1).",
    )
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument(
        "--n-components-list",
        type=str,
        default="8,64,128,256",
        help="Comma-separated component counts to sweep; overrides --n-components when set.",
    )
    parser.add_argument("--maxiters", type=int, default=60)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=202602)
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
    parser.add_argument(
        "--accessor-mode",
        type=str,
        default="auto",
        choices=["auto", "legacy", "buffered"],
    )
    parser.add_argument(
        "--cov-writeback-modes",
        type=str,
        default="auto,kernel,python",
        help="Comma-separated covariance writeback modes to sweep (auto, python, bulk, kernel).",
    )
    parser.add_argument("--rotate2pca", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/perf_baseline/dense_runtime.csv"),
        help="Output CSV path for aggregate timings.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting.",
    )
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def _parse_csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    args = _parse_args()
    shapes = [_parse_shape(item) for item in _parse_csv_values(args.shapes)]
    missing_rates = [float(item) for item in _parse_csv_values(args.missing_rates)]
    cov_modes = _parse_csv_values(args.cov_writeback_modes) or ["auto"]
    component_values = [int(val) for val in _parse_csv_values(args.n_components_list)]
    if not component_values:
        component_values = [int(args.n_components)]

    if not shapes:
        msg = "At least one shape must be provided"
        raise ValueError(msg)
    for rate in missing_rates:
        if not (0.0 <= rate < 1.0):
            msg = f"Invalid missing rate {rate}; expected 0 <= rate < 1"
            raise ValueError(msg)

    rng_base = int(args.seed)
    timings: list[float] = []
    rows: list[dict[str, object]] = []
    host = _host_info()

    for shape_idx, (n_features, n_samples) in enumerate(shapes):
        for rate_idx, missing_rate in enumerate(missing_rates):
            for rep in range(int(args.reps)):
                seed = rng_base + 1000 * shape_idx + 100 * rate_idx + rep
                x, mask = _make_dense_problem(
                    n_features=n_features,
                    n_samples=n_samples,
                    missing_rate=missing_rate,
                    seed=seed,
                )

                for cov_mode in cov_modes:
                    for n_comp in component_values:
                        model_kwargs: dict[str, object] = {
                            "n_components": int(n_comp),
                            "maxiters": int(args.maxiters),
                            "compat_mode": str(args.compat_mode),
                            "runtime_tuning": str(args.runtime_tuning),
                            "rotate2pca": int(args.rotate2pca),
                            "verbose": int(args.verbose),
                            "accessor_mode": str(args.accessor_mode),
                            "cov_writeback_mode": str(cov_mode),
                        }

                        if args.num_cpu is not None:
                            model_kwargs["num_cpu"] = int(args.num_cpu)

                        model = VBPCA(**model_kwargs)

                        start = time.perf_counter()
                        model.fit(x, mask=mask)
                        elapsed = time.perf_counter() - start
                        timings.append(float(elapsed))

                        rows.append(
                            {
                                "shape": f"{n_features}x{n_samples}",
                                "n_features": int(n_features),
                                "n_samples": int(n_samples),
                                "missing_rate": float(missing_rate),
                                "observed_rate": float(np.mean(mask)),
                                "n_components": int(n_comp),
                                "runtime_tuning": str(args.runtime_tuning),
                                "compat_mode": str(args.compat_mode),
                                "cov_writeback_mode": str(cov_mode),
                                "num_cpu": None
                                if args.num_cpu is None
                                else int(args.num_cpu),
                                "repetition": int(rep),
                                "seed": int(seed),
                                "rotate2pca": int(args.rotate2pca),
                                "maxiters": int(args.maxiters),
                                "accessor_mode": str(args.accessor_mode),
                                "host_cpu_count": host["host_cpu_count"],
                                "host_mem_gb": host["host_mem_gb"],
                                "time_sec": float(elapsed),
                            }
                        )

    timing_stats = _stats(timings) if timings else {}

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and args.out.exists() else "w"
    header = not (args.append and args.out.exists())
    frame.to_csv(args.out, mode=mode, header=header, index=False)

    print("Recorded rows:", len(rows))
    if timing_stats:
        print(
            "Timing summary (s):",
            {
                "mean": timing_stats.get("mean"),
                "median": timing_stats.get("median"),
                "std": timing_stats.get("std"),
                "min": timing_stats.get("min"),
                "max": timing_stats.get("max"),
            },
        )


if __name__ == "__main__":
    main()
