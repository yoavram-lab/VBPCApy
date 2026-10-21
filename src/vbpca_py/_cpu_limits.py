"""CPU allocation limits, including discoverable Linux cgroup quotas."""

from __future__ import annotations

import os
from pathlib import Path


def _read_quota(directory: Path) -> int | None:
    try:
        quota_text, period_text = (
            (directory / "cpu.max").read_text(encoding="utf-8").split()
        )
        quota, period = int(quota_text), int(period_text)
    except (OSError, ValueError):
        try:
            quota = int((directory / "cpu.cfs_quota_us").read_text(encoding="utf-8"))
            period = int((directory / "cpu.cfs_period_us").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return max(1, quota // period) if quota > 0 and period > 0 else None


def cgroup_cpu_limit(
    *,
    root: Path | None = None,
    membership: Path | None = None,
) -> int | None:
    """Return a conservative whole-worker quota, when cgroup files are readable.

    Both v2 ``cpu.max`` and v1 CFS quota/period pairs are supported. Ancestor
    quotas are included. Missing/unlimited/malformed files impose no new cap.

    Returns:
        At least one worker, or ``None`` when no quota is discoverable.
    """
    root = root or Path("/sys/fs/cgroup")
    membership = membership or Path("/proc/self/cgroup")
    directories = [root, root / "cpu", root / "cpu,cpuacct", root / "cpuacct,cpu"]
    try:
        lines = membership.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        controllers, group = fields[1:]
        mounts = (
            [root]
            if not controllers
            else [root / "cpu", root / "cpu,cpuacct", root / "cpuacct,cpu"]
            if "cpu" in controllers.split(",")
            else []
        )
        for mount in mounts:
            try:
                base = mount.resolve()
                directory = (mount / group.lstrip("/")).resolve()
            except OSError:
                continue
            if not directory.is_relative_to(base):
                continue
            directories.extend((
                directory,
                *directory.parents[: len(directory.parts) - len(base.parts)],
            ))
    limits = []
    for directory in dict.fromkeys(directories):
        quota = _read_quota(directory)
        if quota is not None:
            limits.append(quota)
    return min(limits) if limits else None


def available_cpu_count() -> int:
    """Return the minimum positive hardware, affinity, scheduler and quota cap.

    Returns:
        A positive worker count, conservatively rounding fractional quotas down.
    """
    candidates = [int(os.cpu_count() or 1)]
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if callable(process_cpu_count):
        count = process_cpu_count()
        if count is not None and count > 0:
            candidates.append(int(count))
    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            count = len(affinity(0))
        except OSError:
            count = 0
        if count > 0:
            candidates.append(count)
    for key in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        try:
            count = int(os.environ.get(key, "0"))
        except ValueError:
            continue
        if count > 0:
            candidates.append(count)
    quota = cgroup_cpu_limit()
    if quota is not None:
        candidates.append(quota)
    return max(1, min(candidates))
