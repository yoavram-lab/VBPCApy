"""Compare two source trees in fresh processes with fixed inputs and CPU limits.

Requires Linux affinity control and installed native extensions in both trees.
Generated reports/snapshots belong outside tracked source. The fit measurements
exclude imports and input generation; peak RSS includes the whole child process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess  # noqa: S404  # Fixed local interpreter, argument lists, no shell.
import sys
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rms_child(argv: list[str]) -> None:
    import resource

    import numpy as np
    from threadpoolctl import threadpool_info

    import vbpca_py
    from vbpca_py._rms import RmsConfig, compute_rms  # noqa: PLC2701

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(292)
    x = rng.normal(size=(3000, 1500))
    a = rng.normal(size=(3000, 8))
    s = rng.normal(size=(8, 1500))
    mask = rng.random(x.shape) > 0.2
    config = RmsConfig()
    compute_rms(x, a, s, mask, config)
    timings = []
    for _ in range(7):
        started = perf_counter()
        compute_rms(x, a, s, mask, config)
        timings.append(perf_counter() - started)
    gc.collect()
    tracemalloc.start()
    value, error = compute_rms(x, a, s, mask, config)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    package = Path(vbpca_py.__file__).parent
    record = {
        "operation": "rms",
        "shape": list(x.shape),
        "wall_seconds": median(timings),
        "timings_seconds": timings,
        "traced_peak_bytes": peak,
        "max_rss_after_kib": rss,
        "identity": {
            "blas_pools": threadpool_info(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "source_path": str(package),
            "source_sha256": {
                name: _digest(package / name) for name in ("_rms.py", "_sparsity.py")
            },
        },
    }
    args.snapshot_dir.mkdir(parents=True, exist_ok=False)
    np.savez(args.snapshot_dir / "fit-0.npz", rms=value, error=error)
    args.output.write_text(json.dumps([record], indent=2) + "\n")


def _compare_snapshots(reference: Path, candidate: Path) -> dict[str, object]:
    import numpy as np

    receipts = []
    for name in sorted(path.name for path in reference.glob("*.npz")):
        expected_path, actual_path = reference / name, candidate / name
        with (
            np.load(expected_path, allow_pickle=False) as expected,
            np.load(actual_path, allow_pickle=False) as actual,
        ):
            if set(actual.files) != set(expected.files):
                raise ValueError("Snapshot fields differ")
            for key in expected.files:
                if actual[key].dtype != expected[key].dtype:
                    raise ValueError(f"Snapshot dtype differs: {key}")
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
            receipts.append({
                "name": name,
                "arrays": sorted(expected.files),
                "baseline_sha256": _digest(expected_path),
                "candidate_sha256": _digest(actual_path),
            })
    if not receipts or sorted(path.name for path in candidate.glob("*.npz")) != [
        receipt["name"] for receipt in receipts
    ]:
        raise ValueError("Snapshot roster differs or is empty")
    return {"comparison": "array equality with equal NaNs", "snapshots": receipts}


def _child_environment(root: Path, native: int, blas: int) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not (key.startswith("VBPCA_") and key.endswith("THREADS"))
    }
    environment.update({
        "PYTHONPATH": str(root / "src"),
        "OMP_NUM_THREADS": str(blas),
        "OPENBLAS_NUM_THREADS": str(blas),
        "MKL_NUM_THREADS": str(blas),
        "BLIS_NUM_THREADS": str(blas),
        "NUMEXPR_NUM_THREADS": str(blas),
        "MKL_DYNAMIC": "FALSE",
        "OMP_DYNAMIC": "FALSE",
        "VBPCA_NUM_THREADS": str(native),
    })
    return environment


@dataclass(frozen=True, slots=True)
class _Profile:
    cpus: tuple[int, ...]
    native_threads: int
    blas_threads: int


def _run_child(
    command: list[str], root: Path, directory: Path, profile: _Profile
) -> dict:
    with (directory / "process.log").open("x") as stream:
        subprocess.run(  # noqa: S603  # Trusted source trees and explicit executable.
            command,
            cwd=root,
            env=_child_environment(root, profile.native_threads, profile.blas_threads),
            preexec_fn=lambda: os.sched_setaffinity(0, set(profile.cpus)),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    record = json.loads((directory / "profile.json").read_text())[0]
    if record["identity"]["affinity"] != sorted(profile.cpus):
        raise ValueError("Child CPU affinity differs from the profile")
    if Path(record["identity"]["source_path"]).resolve() != root / "src/vbpca_py":
        raise ValueError("Child imported the wrong source tree")
    pools = [
        pool for pool in record["identity"]["blas_pools"] if pool["user_api"] == "blas"
    ]
    if not pools or any(pool["num_threads"] != profile.blas_threads for pool in pools):
        raise ValueError("Loaded BLAS thread limits differ from the profile")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cpus",
        type=int,
        nargs=4,
        required=True,
        help="Four distinct physical-core CPU IDs from the current allocation",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not hasattr(os, "sched_setaffinity"):
        parser.error("Linux CPU affinity control is required")
    if (
        args.repeats < 1
        or len(set(args.cpus)) != 4
        or not set(args.cpus).issubset(os.sched_getaffinity(0))
    ):
        parser.error("Use positive repeats and four distinct allocated CPU IDs")
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    for root in roots.values():
        if not (root / "src/vbpca_py/_pca_full.py").is_file():
            parser.error(f"Not a source tree: {root}")
    args.output.mkdir(parents=True, exist_ok=False)
    driver = Path(__file__).resolve()
    benchmark = driver.with_name("benchmark_backend_phases.py")
    profiles = [(1, 1, 1), (2, 2, 1), (4, 4, 1), (4, 1, 4), (4, 4, 4)]
    workloads = {
        "dense-q8": [
            "--features",
            "1536",
            "--samples",
            "1024",
            "--components",
            "8",
            "--missing-rate",
            "0.01",
            "--skip-diagnostics",
        ],
        "dense-q32": [
            "--features",
            "512",
            "--samples",
            "256",
            "--components",
            "32",
            "--missing-rate",
            "0.2",
        ],
        "rms": [],
    }
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpus": args.cpus,
        "driver_sha256": _digest(driver),
        "benchmark_sha256": _digest(benchmark),
        "roots": {key: str(root) for key, root in roots.items()},
        "pairs": [],
    }
    report_path = args.output / "comparison.json"
    for workload, options in workloads.items():
        for cpu_count, native, blas in profiles:
            for repeat in range(args.repeats):
                label = f"{workload}-c{cpu_count}-n{native}-b{blas}-r{repeat}"
                pair = {}
                snapshots = {}
                for revision in (
                    ("baseline", "candidate")
                    if repeat % 2 == 0
                    else ("candidate", "baseline")
                ):
                    directory = args.output / f"{label}-{revision}"
                    directory.mkdir()
                    output, snapshot = (
                        directory / "profile.json",
                        directory / "snapshots",
                    )
                    snapshots[revision] = snapshot
                    if workload == "rms":
                        command = [
                            sys.executable,
                            str(driver),
                            "rms-child",
                            "--output",
                            str(output),
                            "--snapshot-dir",
                            str(snapshot),
                        ]
                    else:
                        command = [
                            sys.executable,
                            str(benchmark),
                            *options,
                            "--iterations",
                            "4",
                            "--threads",
                            str(native),
                            "--dense-sufficient-statistics",
                            "complement",
                            "--snapshot-dir",
                            str(snapshot),
                            "--output",
                            str(output),
                        ]
                    profile = _Profile(tuple(args.cpus[:cpu_count]), native, blas)
                    pair[revision] = _run_child(
                        command, roots[revision], directory, profile
                    )
                fidelity = _compare_snapshots(
                    snapshots["baseline"], snapshots["candidate"]
                )
                record = {
                    "case": label,
                    "workload": workload,
                    "cpu_count": cpu_count,
                    "native_threads": native,
                    "blas_threads": blas,
                    "repeat": repeat,
                    "fidelity": fidelity,
                    "profiles": pair,
                }
                report["pairs"].append(record)
                report_path.write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
                # Retain identities and comparison receipts, not every dense
                # residual matrix. Leave all snapshots in place on failure.
                for directory in snapshots.values():
                    for path in directory.glob("*.npz"):
                        path.unlink()
                    directory.rmdir()
                print(label, "equal", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["rms-child"]:
        _rms_child(sys.argv[2:])
    else:
        main()
