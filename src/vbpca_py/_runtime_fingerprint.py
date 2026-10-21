"""Reproducible identities for optional hardware-specific execution profiles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import sysconfig
import tomllib
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import cast

from threadpoolctl import threadpool_info  # type: ignore[import-untyped]

from ._cpu_limits import available_cpu_count, cgroup_cpu_limit


@lru_cache(maxsize=128)
def _cached_digest(name: str, modified_ns: int, size: int) -> str:
    del modified_ns, size  # They invalidate the cache when a build changes.
    return hashlib.sha256(Path(name).read_bytes()).hexdigest()


def _digest(path: Path) -> str:
    stat = path.stat()
    return _cached_digest(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=1)
def _cpu_model() -> str:
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


def blas_pools() -> list[dict[str, object]]:
    """Return actual loaded BLAS/OpenMP pool identities and independent limits.

    Returns:
        Deterministically ordered threadpoolctl metadata.
    """
    return sorted(
        cast("list[dict[str, object]]", threadpool_info()),
        key=lambda row: str(row.get("filepath", "")),
    )


def runtime_environment() -> dict[str, object]:
    """Capture the machine/allocation, numerical environment and native build.

    File digests are cached by path, size and modification time. Dynamic pool
    counts, affinity, scheduler limits and quotas are read for every identity.

    Returns:
        JSON-compatible profile context. No statistical options are changed.
    """
    package = Path(__file__).parent
    project = package.parent.parent / "pyproject.toml"
    source_version: str | None = None
    if project.is_file():
        try:
            with project.open("rb") as stream:
                source_version = str(tomllib.load(stream)["project"]["version"])
        except (OSError, tomllib.TOMLDecodeError, KeyError):
            source_version = None
    affinity = getattr(os, "sched_getaffinity", None)
    try:
        cpu_ids = sorted(affinity(0)) if callable(affinity) else None
    except OSError:
        cpu_ids = None
    keys = (
        "VBPCA_NUM_THREADS",
        "VBPCA_SCORE_THREADS",
        "VBPCA_LOADINGS_THREADS",
        "VBPCA_NOISE_THREADS",
        "VBPCA_RMS_THREADS",
        "VBPCA_SUBTRACT_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "SLURM_CPUS_PER_TASK",
        "PBS_NP",
        "NSLOTS",
    )
    return {
        "identity_schema": 1,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "affinity": cpu_ids,
        "available_cpu_count": available_cpu_count(),
        "cgroup_cpu_limit": cgroup_cpu_limit(),
        "python": sys.version,
        "python_compiler": platform.python_compiler(),
        "python_build_cflags": sysconfig.get_config_var("CFLAGS"),
        "versions": {key: version(key) for key in ("numpy", "scipy", "vbpca_py")},
        "source_project_version": source_version,
        "thread_environment": {key: os.environ.get(key) for key in keys},
        "blas_pools": blas_pools(),
        "source_sha256": {
            path.name: _digest(path) for path in sorted(package.glob("*.py"))
        },
        "native_sha256": {
            path.name: _digest(path)
            for suffix in ("*.so", "*.pyd", "*.dylib")
            for path in sorted(package.glob(suffix))
        },
    }


def environment_digest(environment: dict[str, object]) -> str:
    """Return a stable hash of the complete environment descriptor.

    Returns:
        A hexadecimal SHA256 digest.
    """
    encoded = json.dumps(
        environment, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
