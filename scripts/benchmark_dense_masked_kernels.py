#!/usr/bin/env python3
"""Benchmark masked dense kernel updates (scores/loadings) across shapes and num_cpu.

This focuses on the raw dense masked kernels without full model overhead.
Results are written to a CSV for later comparison.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from vbpca_py import dense_update_kernels as duk


def _parse_shape(raw: str) -> tuple[int, int]:
    parts = raw.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Shape must be '<features>x<samples>', got {raw!r}")
    return int(parts[0]), int(parts[1])


def _parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_num_cpu(raw: str) -> list[int]:
    out: list[int] = []
    for item in _parse_list(raw):
        try:
            out.append(int(item))
        except ValueError as exc:  # pragma: no cover - CLI validation
            raise ValueError(f"num_cpu entries must be ints, got {item!r}") from exc
    return out


def _stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _host_info() -> dict[str, object]:
    cpu_count = os.cpu_count() or 1
    mem_gb = None
    try:
        # macOS fallback via sysctl; Linux /proc/meminfo would also be fine.
        import subprocess

        res = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=False,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            mem_bytes = float(res.stdout.strip())
            mem_gb = mem_bytes / (1024**3)
    except Exception:
        mem_gb = None

    return {"host_cpu_count": int(cpu_count), "host_mem_gb": mem_gb}


def _make_problem(
    n_features: int,
    n_samples: int,
    n_components: int,
    missing_rate: float,
    seed: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(seed)
    x_data = rng.standard_normal((n_features, n_samples))
    mask = (rng.random((n_features, n_samples)) > missing_rate).astype(float)

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))

    av = np.stack([np.eye(n_components) * 0.05 for _ in range(n_features)])
    sv = np.stack([np.eye(n_components) * 0.02 for _ in range(n_samples)])
    prior_prec = np.eye(n_components) * 0.1
    return x_data, mask, loadings, scores, av, sv, prior_prec


def _timeit(fn, reps: int) -> dict[str, float]:
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    stats = _stats(times)
    stats["ops_per_sec"] = 1.0 / stats["mean"] if stats["mean"] > 0 else float("nan")
    return stats


def _run_kernel_benchmarks(
    shapes: Sequence[tuple[int, int]],
    missing_rate: float,
    n_components: int,
    num_cpu_values: Sequence[int],
    reps: int,
    seed: int,
    return_covariances: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    host = _host_info()

    for shape_idx, (n_features, n_samples) in enumerate(shapes):
        base_seed = seed + 1000 * shape_idx
        x_data, mask, loadings, scores, av, sv, prior_prec = _make_problem(
            n_features=n_features,
            n_samples=n_samples,
            n_components=n_components,
            missing_rate=missing_rate,
            seed=base_seed,
        )
        observed_rate = float(np.mean(mask))

        for num_cpu in num_cpu_values:
            score_stats = _timeit(
                lambda: duk.score_update_dense_masked_nopattern(
                    x_data=x_data,
                    mask=mask,
                    loadings=loadings,
                    loading_covariances=av,
                    noise_var=0.5,
                    return_covariances=return_covariances,
                    num_cpu=num_cpu,
                ),
                reps=reps,
            )
            rows.append(
                {
                    "kernel": "score",
                    "num_cpu": int(num_cpu),
                    "n_features": int(n_features),
                    "n_samples": int(n_samples),
                    "n_components": int(n_components),
                    "missing_rate": float(missing_rate),
                    "observed_rate": observed_rate,
                    "reps": int(reps),
                    "time_mean_sec": score_stats["mean"],
                    "time_median_sec": score_stats["median"],
                    "time_min_sec": score_stats["min"],
                    "time_max_sec": score_stats["max"],
                    "time_std_sec": score_stats["std"],
                    "ops_per_sec": score_stats["ops_per_sec"],
                    "seed": int(base_seed),
                    **host,
                }
            )

            loading_stats = _timeit(
                lambda: duk.loadings_update_dense_masked_nopattern(
                    x_data=x_data,
                    mask=mask,
                    scores=scores,
                    score_covariances=sv,
                    prior_prec=prior_prec,
                    noise_var=0.5,
                    return_covariances=return_covariances,
                    num_cpu=num_cpu,
                ),
                reps=reps,
            )
            rows.append(
                {
                    "kernel": "loadings",
                    "num_cpu": int(num_cpu),
                    "n_features": int(n_features),
                    "n_samples": int(n_samples),
                    "n_components": int(n_components),
                    "missing_rate": float(missing_rate),
                    "observed_rate": observed_rate,
                    "reps": int(reps),
                    "time_mean_sec": loading_stats["mean"],
                    "time_median_sec": loading_stats["median"],
                    "time_min_sec": loading_stats["min"],
                    "time_max_sec": loading_stats["max"],
                    "time_std_sec": loading_stats["std"],
                    "ops_per_sec": loading_stats["ops_per_sec"],
                    "seed": int(base_seed),
                    **host,
                }
            )

    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=str,
        default="200x200,800x800,2000x2000",
        help="Comma-separated shapes as '<features>x<samples>'.",
    )
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--num-cpu", type=str, default="0,4,8")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--return-covariances",
        action="store_true",
        help="Include covariance outputs during benchmarking (slower).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/perf_baseline/dense_masked_kernels.csv"),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shapes = [_parse_shape(item) for item in _parse_list(args.shapes)]
    num_cpu_values = _parse_num_cpu(args.num_cpu)
    if not shapes:
        raise ValueError("At least one shape is required")
    if not num_cpu_values:
        raise ValueError("At least one num_cpu value is required")
    if not (0.0 <= args.missing_rate < 1.0):
        raise ValueError("missing_rate must be in [0, 1)")

    rows = _run_kernel_benchmarks(
        shapes=shapes,
        missing_rate=float(args.missing_rate),
        n_components=int(args.n_components),
        num_cpu_values=num_cpu_values,
        reps=int(args.reps),
        seed=int(args.seed),
        return_covariances=bool(args.return_covariances),
    )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and args.out.exists() else "w"
    header = not (args.append and args.out.exists())
    frame.to_csv(args.out, index=False, mode=mode, header=header)

    print("Recorded rows:", len(rows))
    print("Output:", args.out)
    if not frame.empty:
        grouped = frame.groupby(["kernel", "num_cpu"]).agg(
            {"time_mean_sec": "mean", "ops_per_sec": "mean"}
        )
        print("Mean time (s) / ops_per_sec by kernel,num_cpu:\n", grouped)


if __name__ == "__main__":
    main()
