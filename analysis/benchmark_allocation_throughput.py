"""Measure fixed-policy outer-worker throughput within an explicit allocation.

Linux advisory experiment. Every worker fits the same registered synthetic
problem. Each allocation gets the same number of completed fits. A pilot's
whole-process peak RSS screens configurations before launch; summed worker
high-water marks conservatively bound, rather than measure simultaneous RSS.
No performance or statistical default is selected by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess  # noqa: S404 - explicit local executable/argument lists
import sys
from pathlib import Path
from time import perf_counter

from compare_backend_revisions import _child_environment  # noqa: PLC2701


def _batch(args: argparse.Namespace, workers: int, destination: Path) -> dict:
    destination.mkdir()
    driver = Path(__file__).with_name("benchmark_backend_phases.py").resolve()
    root = Path(__file__).resolve().parents[1]
    per_worker = len(args.cpus) // workers
    processes = []
    handles = []
    started = perf_counter()
    try:
        for index in range(workers):
            directory = destination / str(index)
            directory.mkdir()
            snapshots = directory / "snapshots"
            cpu_ids = args.cpus[index * per_worker : (index + 1) * per_worker]
            command = [
                "taskset",
                "-c",
                ",".join(map(str, cpu_ids)),
                sys.executable,
                str(driver),
                "--features",
                str(args.features),
                "--samples",
                str(args.samples),
                "--components",
                str(args.components),
                "--iterations",
                str(args.iterations),
                "--repeats",
                str(args.fits // workers),
                "--threads",
                str(per_worker),
                "--initialization-offset",
                str(index * (args.fits // workers)),
                "--runtime-tuning",
                "off",
                "--output",
                str(directory / "profile.json"),
                "--snapshot-dir",
                str(snapshots),
            ]
            handle = (directory / "process.log").open("w")
            handles.append(handle)
            processes.append(
                subprocess.Popen(  # noqa: S603 - trusted source and explicit arguments
                    command,
                    cwd=root,
                    env=_child_environment(root, per_worker, 1),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"Worker failures: {codes}; inspect {destination}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for handle in handles:
            handle.close()
    elapsed = perf_counter() - started
    records = [
        json.loads((destination / str(i) / "profile.json").read_text())
        for i in range(workers)
    ]
    peak = sum(max(row["max_rss_after_kib"] for row in group) for group in records)
    return {
        "workers": workers,
        "native_per_worker": per_worker,
        "blas_per_worker": 1,
        "fits": args.fits,
        "wall_seconds_including_startup": elapsed,
        "fits_per_second": args.fits / elapsed,
        "fit_cpu_seconds": sum(
            row["cpu_seconds"] for group in records for row in group
        ),
        "summed_process_peak_rss_kib": peak,
        "within_memory_budget": peak <= args.memory_mib * 1024,
        "profiles": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--fits", type=int, default=4)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--memory-mib", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not shutil.which("taskset") or not hasattr(os, "sched_getaffinity"):
        parser.error("Linux taskset and CPU affinity are required")
    positive_limits = all(
        value > 0 for value in [args.fits, args.repeats, args.memory_mib]
    )
    valid_workers = all(
        w > 0 and len(args.cpus) % w == 0 and args.fits % w == 0 for w in args.workers
    )
    if (
        not set(args.cpus).issubset(os.sched_getaffinity(0))
        or len(set(args.cpus)) != len(args.cpus)
        or not positive_limits
        or not valid_workers
    ):
        parser.error(
            "Use distinct allocated CPUs, positive limits and workers dividing CPUs and fits"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    pilot = _batch(args, 1, args.output / "pilot")
    rows = []
    for repeat in range(args.repeats):
        for workers in args.workers[:: 1 if repeat % 2 == 0 else -1]:
            if workers * pilot["summed_process_peak_rss_kib"] > args.memory_mib * 1024:
                rows.append({
                    "workers": workers,
                    "repeat": repeat,
                    "status": "skipped_pilot_memory_bound",
                })
                continue
            directory = args.output / f"repeat-{repeat}-workers-{workers}"
            row = _batch(args, workers, directory)
            row["repeat"] = repeat
            # Worker j's repeat r has exactly pilot seed+r. Full arrays and
            # covariance stores are checked, not merely scalar fit summaries.
            receipts = [
                _compare_worker_prefix(
                    args.output / "pilot/0/snapshots",
                    directory / str(worker) / "snapshots",
                    offset=worker * (args.fits // workers),
                )
                for worker in range(workers)
            ]
            row["numerical_receipts"] = receipts
            rows.append(row)
    payload = {
        "allocation_cpus": args.cpus,
        "memory_budget_mib": args.memory_mib,
        "pilot": pilot,
        "measurements": rows,
        "scope": "Advisory shared-host throughput. Fixed priors/stopping/rank; startup included. Summed per-process RSS high-water marks are conservative. No wheel ISA changes.",
    }
    (args.output / "throughput.json").write_text(json.dumps(payload, indent=2) + "\n")
    if any(not row.get("within_memory_budget", True) for row in rows):
        raise RuntimeError("A measured configuration exceeded its RSS budget")


def _compare_worker_prefix(baseline: Path, candidate: Path, *, offset: int) -> dict:
    import numpy as np

    receipts = []
    for path in sorted(candidate.glob("*.npz")):
        reference = baseline / f"fit-{offset + int(path.stem.split('-')[1])}.npz"
        with np.load(reference) as expected, np.load(path) as actual:
            if set(expected.files) != set(actual.files):
                raise ValueError("Snapshot keys differ")
            for key in expected.files:
                if expected[key].dtype != actual[key].dtype:
                    raise ValueError(f"Snapshot dtype differs: {key}")
                if expected[key].dtype.kind in "fc":
                    np.testing.assert_allclose(
                        actual[key],
                        expected[key],
                        rtol=1e-8,
                        atol=1e-10,
                        equal_nan=True,
                    )
                else:
                    np.testing.assert_array_equal(actual[key], expected[key])
            receipts.append(path.name)
    return {
        "snapshots": receipts,
        "comparison": "rtol=1e-8, atol=1e-10; metadata/dtypes exact",
    }


if __name__ == "__main__":
    main()
