"""Compare full-covariance contractions, packing, and NumPy allocation peaks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tracemalloc
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_info

from vbpca_py._pca_full import _marginal_variance
from vbpca_py._posterior_variance import marginal_variance


def _einsum(state, *, optimize=False):
    # Include the same covariance packing paid by the historical diagnostic.
    av, sv = np.stack(state.av), np.stack(state.sv)
    return (
        np.einsum("ik,jkl,il->ij", state.a, sv, state.a, optimize=optimize)
        + np.einsum("ikl,kj,lj->ij", av, state.s, state.s, optimize=optimize)
        + np.einsum("ikl,jkl->ij", av, sv, optimize=optimize)
        + state.muv
    )


def _probe(features, samples, rank, repeats):
    rng = np.random.default_rng(290)
    ra = rng.normal(size=(features, rank, rank)) / rank
    rs = rng.normal(size=(samples, rank, rank)) / rank
    state = SimpleNamespace(
        a=rng.normal(size=(features, rank)),
        s=rng.normal(size=(rank, samples)),
        av=list(ra @ ra.transpose(0, 2, 1)),
        sv=list(rs @ rs.transpose(0, 2, 1)),
        muv=rng.uniform(size=(features, 1)),
        pattern_index=None,
    )
    calls = {
        "historical_einsum": lambda: _einsum(state),
        "optimized_einsum": lambda: _einsum(state, optimize=True),
        "whole_gemm": lambda: marginal_variance(
            state, block_size=max(features, samples)
        ),
        "blocked_gemm": lambda: _marginal_variance(state),
    }
    reference = calls["historical_einsum"]()
    timings = {key: [] for key in calls}
    errors, peaks = {}, {}
    for key, call in calls.items():
        actual = call()  # Warm every implementation before alternating order.
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
        errors[key] = float(np.max(abs(actual - reference)))
        tracemalloc.start()
        try:
            call()
            peaks[key] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    for repeat in range(repeats):
        order = list(calls) if repeat % 2 == 0 else list(reversed(calls))
        for key in order:
            start = perf_counter()
            calls[key]()
            timings[key].append(perf_counter() - start)
    return {
        "shape": [features, samples],
        "rank": rank,
        "layout": "C",
        "covariances": "full",
        "packing_included": True,
        "seconds": timings,
        "median_seconds": {key: median(x) for key, x in timings.items()},
        "numpy_peak_bytes": peaks,
        "max_absolute_error": errors,
        "output_bytes": reference.nbytes,
    }


def _fit_probe(rank, repeats):
    import vbpca_py._pca_full as fitting

    rng = np.random.default_rng(290)
    x = rng.normal(size=(512, 256))
    x[rng.random(x.shape) < 0.18] = np.nan
    original = fitting._marginal_variance
    times = {"historical_einsum": [], "blocked_gemm": []}
    results = {}
    try:
        for repeat in range(repeats + 1):
            names = list(times) if repeat % 2 == 0 else list(reversed(times))
            for name in names:
                fitting._marginal_variance = (
                    _einsum if name == "historical_einsum" else original
                )
                started = perf_counter()
                results[name] = fitting.pca_full(
                    x,
                    rank,
                    maxiters=4,
                    niter_broadprior=0,
                    num_cpu=1,
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
                    random_state=290,
                    verbose=0,
                    compat_mode="modern",
                    return_diagnostics=True,
                )
                if repeat:
                    times[name].append(perf_counter() - started)
    finally:
        fitting._marginal_variance = original
    for key in ("A", "S", "Mu", "V", "Xrec", "Vr", "Vpred"):
        np.testing.assert_allclose(
            results["blocked_gemm"][key],
            results["historical_einsum"][key],
            rtol=1e-12,
            atol=1e-12,
        )
    return {
        "shape": [512, 256],
        "rank": rank,
        "missing_fraction": 0.18,
        "fixed_iterations": 4,
        "seconds": times,
        "median_seconds": {key: median(x) for key, x in times.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-fits", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("repeats must be at least two")
    import vbpca_py._posterior_variance as implementation

    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "versions": {name: version(name) for name in ("numpy", "scipy", "vbpca_py")},
        "blas_pools": threadpool_info(),
        "source_sha256": hashlib.sha256(
            Path(implementation.__file__).read_bytes()
        ).hexdigest(),
        "scope": "Synthetic diagnostic phase only; NumPy peaks exclude native BLAS workspace. No fit/default/pin changes.",
        "cases": [
            _probe(p, n, q, args.repeats)
            for p, n, q in (
                (512, 256, 8),
                (512, 256, 32),
                (128, 1024, 8),
                (1024, 128, 32),
            )
        ],
        "whole_fits": [_fit_probe(q, args.repeats) for q in (8, 32)]
        if args.include_fits
        else [],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["cases"]:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
