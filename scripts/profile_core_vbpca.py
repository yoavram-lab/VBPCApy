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

_TUNING_STAGE_MAXITERS = {
    "quick": 20,
    "confirm": 40,
    "final": 80,
}


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
    runtime_tuning: str,
    runtime_profile: str | None,
    warmup_reps: int,
    tuning_stage: str,
    collect_phase_timings: bool,
) -> dict[str, float | int | str]:
    times: list[float] = []
    phase_totals: dict[str, list[float]] = {
        "phase_scores_sec": [],
        "phase_loadings_sec": [],
        "phase_rms_sec": [],
        "phase_noise_sec": [],
        "phase_converge_sec": [],
        "phase_total_sec": [],
    }

    for warmup in range(max(0, int(warmup_reps))):
        rng = np.random.default_rng(seed + warmup)
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

        _ = pca_full(
            x,
            case.n_components,
            algorithm="vb",
            compat_mode="modern",
            rotate2pca=int(bool(rotate2pca)),
            runtime_tuning=runtime_tuning,
            runtime_profile=runtime_profile,
            verbose=0,
            maxiters=maxiters,
        )

    for rep in range(reps):
        rng = np.random.default_rng(seed + warmup_reps + rep)
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
        result = pca_full(
            x,
            case.n_components,
            algorithm="vb",
            compat_mode="modern",
            rotate2pca=int(bool(rotate2pca)),
            runtime_tuning=runtime_tuning,
            runtime_profile=runtime_profile,
            verbose=0,
            maxiters=maxiters,
        )
        times.append(time.perf_counter() - start)

        if collect_phase_timings:
            lc = result.get("lc", {})
            if isinstance(lc, dict):
                for key in phase_totals:
                    vals = lc.get(key, [])
                    if isinstance(vals, list):
                        phase_totals[key].append(
                            float(sum(float(v) for v in vals[1:]))
                        )

    arr = np.asarray(times, dtype=float)
    out: dict[str, float | int | str] = {
        "case": case.name,
        "kind": case.kind,
        "n_features": case.n_features,
        "n_samples": case.n_samples,
        "n_components": case.n_components,
        "missing_rate": case.missing_rate,
        "rotate2pca": int(bool(rotate2pca)),
        "runtime_tuning": runtime_tuning,
        "runtime_profile": runtime_profile or "",
        "tuning_stage": tuning_stage,
        "maxiters_used": int(maxiters),
        "repetitions": reps,
        "time_mean_sec": float(np.mean(arr)),
        "time_median_sec": float(np.median(arr)),
        "time_std_sec": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "time_min_sec": float(np.min(arr)),
        "time_max_sec": float(np.max(arr)),
    }

    if collect_phase_timings:
        for key, values in phase_totals.items():
            key_prefix = key.replace("_sec", "")
            out[f"{key_prefix}_mean_sec"] = float(np.mean(values)) if values else 0.0
            out[f"{key_prefix}_median_sec"] = (
                float(np.median(values)) if values else 0.0
            )

    return out


def _resolve_maxiters(maxiters: int | None, tuning_stage: str) -> int:
    if maxiters is not None:
        return int(maxiters)
    return int(_TUNING_STAGE_MAXITERS[tuning_stage])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--maxiters",
        type=int,
        default=None,
        help="Explicit max iterations override. If unset, --tuning-stage is used.",
    )
    parser.add_argument(
        "--tuning-stage",
        type=str,
        choices=("quick", "confirm", "final"),
        default="quick",
        help="Preset stage mapping to maxiters (quick=20, confirm=40, final=80).",
    )
    parser.add_argument(
        "--rotate2pca",
        type=int,
        choices=(0, 1),
        default=1,
        help="Enable rotate-to-PCA during training (1=on, 0=off).",
    )
    parser.add_argument(
        "--runtime-tuning",
        type=str,
        choices=("off", "safe", "aggressive"),
        default="off",
        help="Runtime tuning mode passed to pca_full.",
    )
    parser.add_argument(
        "--runtime-profile",
        type=str,
        default="",
        help="Path to runtime profile JSON (empty disables profile).",
    )
    parser.add_argument(
        "--warmup-reps",
        type=int,
        default=0,
        help="Number of warmup runs per case excluded from timing statistics.",
    )
    parser.add_argument(
        "--collect-phase-timings",
        action="store_true",
        help="Include aggregated per-phase timing fields from lc diagnostics.",
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
    maxiters = _resolve_maxiters(args.maxiters, args.tuning_stage)
    rows = [
        _run_case(
            case,
            seed=args.seed,
            maxiters=maxiters,
            reps=args.reps,
            rotate2pca=args.rotate2pca,
            runtime_tuning=args.runtime_tuning,
            runtime_profile=(args.runtime_profile or None),
            warmup_reps=args.warmup_reps,
            tuning_stage=args.tuning_stage,
            collect_phase_timings=bool(args.collect_phase_timings),
        )
        for case in _build_cases(args.case_set)
    ]

    frame = pd.DataFrame(rows).sort_values(["kind", "case"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print("Wrote core baseline:", args.out)
    print(
        "Resolved run config:",
        {
            "tuning_stage": args.tuning_stage,
            "maxiters_used": maxiters,
            "warmup_reps": args.warmup_reps,
            "collect_phase_timings": bool(args.collect_phase_timings),
        },
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
