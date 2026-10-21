"""Build two immutable revisions and produce advisory interleaved CI timings.

Deterministic allocation and numerical checks remain hard failures. Timing
signals never fail this command. Use compatible backend revisions (16df5d7 or
later) and a Linux allocation. Generated builds, snapshots and reports are
written only beneath --output; an existing output directory is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import sysconfig
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import median


def _run(command, *, cwd, log, env=None):
    with log.open("w") as stream:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _revision(root, ref):
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
        cwd=root,
        text=True,
    ).strip()


def _physical_cpus():
    selected, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            identity = (
                (topology / "physical_package_id").read_text().strip(),
                (topology / "core_id").read_text().strip(),
            )
        except OSError:
            identity = ("unknown", str(cpu))
        if identity not in seen:
            seen.add(identity)
            selected.append(cpu)
    try:
        from vbpca_py._cpu_limits import available_cpu_count

        cap = available_cpu_count()
    except ImportError:
        cap = len(selected)
    return selected[: min(4, cap)]


def _cpu_model():
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return platform.processor()
    return next(
        (
            line.split(":", 1)[1].strip()
            for line in lines
            if line.startswith("model name")
        ),
        platform.processor(),
    )


def _summarize(report):
    groups = defaultdict(list)
    for pair in report["pairs"]:
        base, candidate = pair["profiles"]["baseline"], pair["profiles"]["candidate"]
        groups[
            pair["workload"],
            pair["cpu_count"],
            pair["native_threads"],
            pair["blas_threads"],
        ].append(base["wall_seconds"] / candidate["wall_seconds"])
    rows = []
    for key, ratios in groups.items():
        center = median(ratios)
        mad = median(abs(value - center) for value in ratios)
        signal = "within_practical_margin_or_variability"
        if abs(center - 1) > max(0.05, 2 * mad):
            signal = "candidate_faster" if center > 1 else "candidate_slower"
        rows.append({
            "workload": key[0],
            "cpus": key[1],
            "native": key[2],
            "blas": key[3],
            "baseline_over_candidate_ratios": ratios,
            "median_ratio": center,
            "median_absolute_deviation": mad,
            "minimum_ratio": min(ratios),
            "maximum_ratio": max(ratios),
            "advisory_signal": signal,
        })
    return {
        "timing_is_advisory": True,
        "hard_timing_gate": False,
        "practical_margin_fraction": 0.05,
        "signal_requires_effect_above_twice_ratio_MAD": True,
        "groups": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3 or not hasattr(os, "sched_getaffinity"):
        parser.error(
            "Linux affinity and at least three interleaved repeats are required"
        )
    root = Path(__file__).resolve().parents[1]
    revisions = {
        "baseline": _revision(root, args.baseline),
        "candidate": _revision(root, args.candidate),
    }
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cpus = _physical_cpus()
    identity = {
        "revisions": revisions,
        "python": sys.version,
        "platform": platform.platform(),
        "cpus": cpus,
        "compiler_cflags": sysconfig.get_config_var("CFLAGS"),
        "build_environment": {
            k: os.environ.get(k)
            for k in [
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "CPPFLAGS",
                "LDFLAGS",
                "EIGEN_INCLUDE_DIR",
            ]
        },
        "runner_environment": {
            k: os.environ.get(k)
            for k in ["RUNNER_NAME", "RUNNER_OS", "RUNNER_ARCH", "GITHUB_RUN_ID"]
        },
        "runner_label_is_not_a_hardware_guarantee": True,
        "cpu_model": _cpu_model(),
        "full_affinity": sorted(os.sched_getaffinity(0)),
    }
    (args.output / "registration.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    sources = {}
    for name, revision in revisions.items():
        source = args.output / name
        _run(
            ["git", "clone", "--shared", "--no-checkout", str(root), str(source)],
            cwd=root,
            log=args.output / f"{name}-clone.log",
        )
        _run(
            ["git", "checkout", "--detach", revision],
            cwd=source,
            log=args.output / f"{name}-checkout.log",
        )
        wheels = args.output / f"{name}-wheel"
        _run(
            [
                "uv",
                "build",
                "--wheel",
                "--python",
                sys.executable,
                "--out-dir",
                str(wheels),
            ],
            cwd=source,
            log=args.output / f"{name}-build.log",
        )
        (wheel,) = wheels.glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            for item in archive.namelist():
                if item.startswith("vbpca_py/") and item.endswith((".so", ".pyd")):
                    destination = source / "src" / item
                    destination.write_bytes(archive.read(item))
        env = dict(
            os.environ,
            PYTHONPATH=str(source / "src"),
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        _run(
            [sys.executable, str(source / "scripts/check_sparse_allocations.py")],
            cwd=source,
            env=env,
            log=args.output / f"{name}-allocations.json",
        )
        sources[name] = source
    _run(
        [
            sys.executable,
            str(root / "analysis/compare_backend_revisions.py"),
            "--baseline",
            str(sources["baseline"]),
            "--candidate",
            str(sources["candidate"]),
            "--suite",
            "ci",
            "--cpus",
            *map(str, cpus),
            "--repeats",
            str(args.repeats),
            "--output",
            str(args.output / "comparison"),
        ],
        cwd=root,
        log=args.output / "comparison.log",
    )
    report = json.loads((args.output / "comparison/comparison.json").read_text())
    summary = _summarize(report)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
