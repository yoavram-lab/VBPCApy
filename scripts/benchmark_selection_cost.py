"""Interleaved fixed-policy selection sweeps with trace or endpoint cost."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_info

import vbpca_py._converge as convergence
from vbpca_py.model_selection import SelectionConfig, select_n_components


def _probe(rank, repeats):
    rng = np.random.default_rng(295)
    x = rng.normal(size=(512, 256))
    x[rng.random(x.shape) < 0.18] = np.nan
    times = {"trace": [], "endpoint": []}
    results, counts = {}, {}
    real = convergence.compute_full_cost
    cost_calls = 0

    def counted(*args, **kwargs):
        nonlocal cost_calls
        cost_calls += 1
        return real(*args, **kwargs)

    convergence.compute_full_cost = counted
    try:
        for repeat in range(repeats + 1):
            for mode in list(times) if repeat % 2 == 0 else list(reversed(times)):
                cost_calls = 0
                started = perf_counter()
                results[mode] = select_n_components(
                    x,
                    components=[0, rank // 2, rank],
                    config=SelectionConfig(
                        cost_diagnostics=mode, compute_explained_variance=False
                    ),
                    record_cost=False,
                    maxiters=4,
                    niter_broadprior=0,
                    random_state=295,
                    convergence_criteria=dict.fromkeys(
                        (
                            "angle",
                            "earlystop",
                            "rms_plateau",
                            "cost",
                            "composite",
                            "slowing_down",
                        ),
                        False,
                    ),
                    runtime_tuning="off",
                    num_cpu=1,
                    verbose=0,
                    return_diagnostics=False,
                )
                if repeat:
                    times[mode].append(perf_counter() - started)
                counts[mode] = cost_calls
    finally:
        convergence.compute_full_cost = real
    if results["trace"][0] != results["endpoint"][0]:
        raise AssertionError("Selected capacity differs")
    for a, b in zip(results["trace"][2], results["endpoint"][2], strict=True):
        for key in ("rms", "prms", "cost", "n_iter", "converged"):
            np.testing.assert_allclose(a[key], b[key], rtol=0, atol=0)
    return {
        "shape": list(x.shape),
        "components": [0, rank // 2, rank],
        "fixed_iterations": 4,
        "seconds": times,
        "cost_calls": counts,
        "selected_capacity": results["endpoint"][0],
        "median_seconds": {key: median(values) for key, values in times.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("repeats must be at least two")
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ("numpy", "scipy", "vbpca_py")},
        "blas_pools": threadpool_info(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "cases": [_probe(rank, args.repeats) for rank in (8, 32)],
        "scope": "Shared-host, fixed-iteration sweep timing; no default or publication-policy qualification.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["cases"]))


if __name__ == "__main__":
    main()
