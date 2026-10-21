"""Validate staged source snapshots and record a cluster's numerical environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Identity of one source tree, independent of the installed native build.

    Attributes:
        revision: Git source revision used to create the archive.
        files: Relative file names and their SHA256 digests.

    Examples:
        >>> files = {"example.py": "b" * 64}
        >>> snapshot = SourceSnapshot("a" * 40, files)
        >>> files.clear()
        >>> list(snapshot.files)
        ['example.py']
    """

    revision: str
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        """Copy the manifest and validate its identities.

        Raises:
            ValueError: If a revision or content digest is not a full SHA.
        """
        hexadecimal = frozenset("0123456789abcdef")
        if len(self.revision) != 40 or not set(self.revision) <= hexadecimal:
            raise ValueError("A full Git SHA is required")
        files = dict(self.files)
        if any(
            len(value) != 64 or not set(value) <= hexadecimal
            for value in files.values()
        ):
            raise ValueError("Full SHA256 content digests are required")
        object.__setattr__(self, "files", MappingProxyType(files))

    def validate(self, root: Path) -> None:
        """Require the expected contents before creating extension links.

        Raises:
            ValueError: If a path escapes the snapshot or its content differs.
        """
        for relative, expected in self.files.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Snapshot file escapes its source root")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Source content differs: {relative}")


def physical_cpu_ids(allowed: set[int], count: int = 4) -> tuple[int, ...]:
    """Select distinct physical cores from the process's allocated CPUs.

    Returns:
        One logical CPU per selected physical core.

    Raises:
        ValueError: If the count is nonpositive or too few cores are available.
    """
    if count < 1:
        raise ValueError("The physical core count must be positive")
    selected: dict[tuple[str, ...], int] = {}
    for cpu in sorted(allowed):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        # Package/core IDs identify siblings even on kernels without die_id.
        key = tuple(
            (topology / name).read_text().strip()
            for name in ("physical_package_id", "core_id")
        )
        selected.setdefault(key, cpu)
    if len(selected) < count:
        raise ValueError(f"Need {count} distinct allocated physical cores")
    return tuple(selected.values())[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    registration = root / "registration.json"
    if (
        hashlib.sha256(registration.read_bytes()).hexdigest()
        != args.registration_sha256
    ):
        raise ValueError("Registration checksum differs")
    config = json.loads(registration.read_text())
    if version("vbpca_py") != "0.4.10":
        raise ValueError("Use the registered isolated VBPCApy 0.4.10 environment")

    import numpy as np
    import scipy.linalg
    from threadpoolctl import threadpool_info

    import vbpca_py

    package = Path(vbpca_py.__file__).parent
    extensions = sorted(package.glob("*.so"))
    expected_modules = {
        "dense_update_kernels",
        "sparse_update_kernels",
        "noise_update_kernels",
        "rotate_update_kernels",
        "errpca_pt",
        "subtract_mu_from_sparse",
    }
    if {path.name.split(".")[0] for path in extensions} != expected_modules:
        raise ValueError("Native extension roster differs")
    cpus = physical_cpu_ids(os.sched_getaffinity(0))
    for name in ("baseline", "candidate"):
        spec = config["sources"][name]
        SourceSnapshot(spec["revision"], spec["files"]).validate(root / name)
    for name in ("baseline", "candidate"):
        for extension in extensions:
            target = root / name / "src/vbpca_py" / extension.name
            target.symlink_to(extension)

    info = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("numpy", "scipy", "vbpca_py", "threadpoolctl")
        },
        "cpus": cpus,
        "cpuinfo": Path("/proc/cpuinfo").read_text(),
        "slurm": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_CPUS_PER_NODE",
                "SLURM_MEM_PER_NODE",
            )
        },
        "native_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in extensions
        },
        "initial_blas_pools": threadpool_info(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scientific_claims_eligible": False,
    }
    (root / "hardware.json").write_text(json.dumps(info, indent=2) + "\n")
    (root / "cpu-ids.txt").write_text(" ".join(map(str, cpus)) + "\n")


if __name__ == "__main__":
    main()
