"""Profile VBPCA phases without making performance checks part of CI.

Run from the repository root, for example::

    python analysis/benchmark_backend_phases.py --features 300 --samples 500

For process-level peak memory, wrap the command with ``/usr/bin/time -v``.
The JSON output is intended for comparisons between git revisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_info

import vbpca_py
import vbpca_py._pca_full as pca_module  # noqa: PLC2701

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

_CONVERGENCE_OFF = {
    "angle": False,
    "earlystop": False,
    "rms_plateau": False,
    "cost": False,
    "composite": False,
    "slowing_down": False,
}
_PHASE_TARGETS = {
    "prepare": "_prepare_problem",
    "initialize": "_initialize_model",
    "runtime_policy": "_resolve_runtime_threads_for_training",
    "training": "_run_training_loop",
    "rotation": "_maybe_finalize_rotation",
    "restore_shape": "_restore_original_shape",
    "pack_result": "_pack_result",
    "explained_variance": "_explained_variance",
    "sparse_views": "_coerce_sparse_views",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=300)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--true-rank", type=int, default=6)
    parser.add_argument("--missing-rate", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=207)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument(
        "--runtime-tuning",
        choices=("off", "safe", "aggressive"),
        default="off",
    )
    parser.add_argument(
        "--dense-sufficient-statistics",
        choices=("auto", "observed", "complement"),
        default="auto",
    )
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--skip-bias", action="store_true")
    parser.add_argument("--skip-rotation", action="store_true")
    parser.add_argument(
        "--noise-sufficient-statistics",
        choices=("auto", "observed", "complement"),
        default="observed",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    return parser


def _make_problem(
    args: argparse.Namespace,
) -> tuple[np.ndarray | sp.csr_matrix, np.ndarray | sp.csr_matrix]:
    rng = np.random.default_rng(args.seed)
    rank = min(args.true_rank, args.features, args.samples)
    loadings = rng.normal(size=(args.features, rank))
    scores = rng.normal(size=(rank, args.samples))
    x = loadings @ scores + 0.3 * rng.normal(
        size=(args.features, args.samples),
    )
    mask = rng.random(x.shape) >= args.missing_rate

    # Avoid empty rows and columns so the requested shape is the fitted shape.
    mask[:, 0] = True
    mask[0, :] = True
    if args.sparse:
        rows, columns = np.nonzero(mask)
        # Include an explicit observed zero in the prepared input.
        x[0, 0] = 0.0
        return sp.csr_matrix(
            (x[rows, columns], (rows, columns)), shape=x.shape
        ), sp.csr_matrix(mask)
    return np.array(x, order=args.order), np.array(mask, order=args.order)


@contextmanager
def _profile_orchestration() -> Generator[dict[str, dict[str, float | int]]]:
    measurements = {name: {"calls": 0, "seconds": 0.0} for name in _PHASE_TARGETS}
    originals: dict[str, Callable[..., Any]] = {}

    for phase, attribute in _PHASE_TARGETS.items():
        original = getattr(pca_module, attribute)
        originals[attribute] = original

        def timed(
            *args: object,
            _phase: str = phase,
            _fn: Callable[..., Any] = original,
            **kwargs: object,
        ) -> Any:
            started = time.perf_counter()
            try:
                return _fn(*args, **kwargs)
            finally:
                measurements[_phase]["calls"] += 1
                measurements[_phase]["seconds"] += time.perf_counter() - started

        setattr(pca_module, attribute, timed)

    try:
        yield measurements
    finally:
        for attribute, original in originals.items():
            setattr(pca_module, attribute, original)


def _sum_learning_curve_phase(result: dict[str, object], key: str) -> float:
    learning_curve = result["lc"]
    values = np.asarray(learning_curve.get(key, []), dtype=float)
    return float(np.nansum(values))


def _autotune_seconds(runtime_report: dict[str, object]) -> float:
    elapsed = 0.0
    for key, value in runtime_report.items():
        if not key.startswith("autotune") and key != "cov_writeback_autotune":
            continue
        if isinstance(value, dict):
            elapsed += float(value.get("elapsed_sec", 0.0))
    return elapsed


def _run_once(
    args: argparse.Namespace,
    x: np.ndarray | sp.csr_matrix,
    mask: np.ndarray | sp.csr_matrix,
    repeat: int,
) -> dict[str, object]:
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    with _profile_orchestration() as orchestration:
        result = pca_module.pca_full(
            x,
            n_components=args.components,
            mask=mask,
            algorithm="vb",
            maxiters=args.iterations,
            niter_broadprior=0,
            convergence_criteria=_CONVERGENCE_OFF,
            rotate2pca=not args.skip_rotation,
            bias=not args.skip_bias,
            record_cost=False,
            return_diagnostics=not args.skip_diagnostics,
            runtime_tuning=args.runtime_tuning,
            dense_sufficient_statistics=args.dense_sufficient_statistics,
            noise_sufficient_statistics=args.noise_sufficient_statistics,
            runtime_report=True,
            num_cpu=args.threads,
            compat_mode="modern",
            random_state=args.seed + repeat,
            display=0,
            verbose=0,
        )
    wall_seconds = time.perf_counter() - started
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    runtime_report = result["RuntimeReport"] or {}

    if args.snapshot_dir is not None:
        arrays = {
            key: np.asarray(result[key])
            for key in (
                "A",
                "S",
                "Mu",
                "Muv",
                "Av",
                "Sv",
                "V",
                "Va",
                "Vmu",
                "Xrec",
                "Vr",
                "Vpred",
            )
            if result[key] is not None
        }
        arrays.update({
            f"lc_{key}": np.asarray(result["lc"][key])
            for key in ("rms", "prms", "cost")
        })
        arrays["returned_iteration"] = np.asarray(result["ReturnedIteration"])
        arrays["lc_stop"] = np.asarray(result["lc"].get("_stop", []))
        np.savez(args.snapshot_dir / f"fit-{repeat}.npz", **arrays)

    return {
        "repeat": repeat,
        "shape": [args.features, args.samples],
        "components": args.components,
        "missing_rate": args.missing_rate,
        "observed_fraction": float(mask.sum()) / (args.features * args.samples),
        "storage": "csr" if args.sparse else "dense",
        "bias": not args.skip_bias,
        "rotation": not args.skip_rotation,
        "order": args.order,
        "threads": args.threads,
        "iterations": args.iterations,
        "runtime_tuning": args.runtime_tuning,
        "dense_sufficient_statistics": args.dense_sufficient_statistics,
        "noise_sufficient_statistics": args.noise_sufficient_statistics,
        "diagnostics": not args.skip_diagnostics,
        "wall_seconds": wall_seconds,
        "max_rss_before_kib": int(rss_before),
        "max_rss_after_kib": int(rss_after),
        "orchestration": orchestration,
        "iteration_phases": {
            name: _sum_learning_curve_phase(result, f"phase_{name}_sec")
            for name in ("scores", "loadings", "rms", "noise", "converge", "total")
        },
        "autotune_seconds_reported": _autotune_seconds(runtime_report),
        "runtime_report": runtime_report,
        "terminal_rms": float(result["RMS"]),
        "noise_variance": float(result["V"]),
    }


def main() -> None:
    args = _parser().parse_args()
    if not 0.0 <= args.missing_rate < 1.0:
        raise ValueError("missing-rate must be in [0, 1)")
    if min(args.features, args.samples, args.components, args.iterations) <= 0:
        raise ValueError("shape, components, and iterations must be positive")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.snapshot_dir is not None:
        args.snapshot_dir.mkdir(parents=True, exist_ok=False)

    x, mask = _make_problem(args)
    records = [_run_once(args, x, mask, repeat) for repeat in range(args.repeats)]
    package = Path(vbpca_py.__file__).parent
    identity = {
        "source_path": str(package),
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "versions": {name: version(name) for name in ("numpy", "scipy", "vbpca_py")},
        "blas_pools": threadpool_info(),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.glob("*.py"))
        },
        "native_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for suffix in ("*.so", "*.pyd")
            for path in sorted(package.glob(suffix))
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VBPCA_NUM_THREADS",
            )
        },
    }
    for record in records:
        record["identity"] = identity
    payload = json.dumps(records, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
