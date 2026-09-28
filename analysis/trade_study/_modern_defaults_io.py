"""Manifest, shard, and checkpoint I/O for the modern-defaults study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy

if TYPE_CHECKING:
    from pathlib import Path

from ._modern_defaults_design import (
    build_manifest,
    registered_screen_manifest,
    validate_manifest,
)


def manifest_sha256(path: Path) -> str:
    """Return the byte-level checksum of a manifest file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Read and validate an immutable study manifest."""
    manifest = json.loads(path.read_text())
    validate_manifest(manifest)
    return manifest


def write_manifest(
    path: Path,
    *,
    profile: str,
    n_reps: int,
    seed: int,
    candidate_ids: tuple[str, ...] | None = None,
    registered_screen: bool = False,
) -> dict[str, Any]:
    """Build and atomically write a study manifest."""
    manifest = (
        registered_screen_manifest()
        if registered_screen
        else build_manifest(
            profile,
            n_reps=n_reps,
            seed=seed,
            candidate_ids=candidate_ids,
        )
    )
    atomic_json(path, manifest)
    return manifest


def shards(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    """Return candidate/regime shards in stable array-index order."""
    return [
        (str(candidate["id"]), str(regime["name"]))
        for candidate in manifest["candidates"]
        for regime in manifest["regimes"]
    ]


def shard_at_index(manifest: dict[str, Any], index: int) -> tuple[str, str]:
    """Resolve one zero-based array index."""
    available = shards(manifest)
    if not 0 <= index < len(available):
        msg = f"shard-index must be in [0, {len(available)})"
        raise ValueError(msg)
    return available[index]


def candidate_by_id(manifest: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    """Return the unique candidate with the requested identifier."""
    matches = [item for item in manifest["candidates"] if item["id"] == candidate_id]
    if len(matches) != 1:
        msg = f"manifest does not contain exactly one candidate {candidate_id!r}"
        raise ValueError(msg)
    return matches[0]


def regime_by_name(manifest: dict[str, Any], regime_name: str) -> dict[str, Any]:
    """Return the unique regime with the requested name."""
    matches = [item for item in manifest["regimes"] if item["name"] == regime_name]
    if len(matches) != 1:
        msg = f"manifest does not contain exactly one regime {regime_name!r}"
        raise ValueError(msg)
    return matches[0]


def checkpoint_path(
    output_dir: Path,
    manifest: dict[str, Any],
    candidate_id: str,
    regime_name: str,
) -> Path:
    """Return the controlled destination for one shard."""
    return output_dir / str(manifest["profile"]) / candidate_id / f"{regime_name}.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON beside its destination and atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def run_provenance(*, num_cpu: int) -> dict[str, Any]:
    """Capture the software and scheduler identity of a shard process."""
    return {
        "vbpca_version": version("vbpca_py"),
        "vbpca_revision": os.environ.get("VBPCA_REVISION"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "num_cpu": num_cpu,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def new_checkpoint(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    candidate_id: str,
    regime_name: str,
    num_cpu: int,
) -> dict[str, Any]:
    """Create an empty resumable shard checkpoint."""
    return {
        "manifest_sha256": manifest_sha256(manifest_path),
        "design_version": manifest["design_version"],
        "profile": manifest["profile"],
        "candidate_id": candidate_id,
        "regime": regime_name,
        "n_reps": int(manifest["n_reps"]),
        "complete": False,
        "provenance": run_provenance(num_cpu=num_cpu),
        "records": [],
    }


def validate_checkpoint(
    checkpoint: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    candidate_id: str,
    regime_name: str,
) -> None:
    """Reject a checkpoint that cannot belong to the requested shard."""
    expected = {
        "manifest_sha256": manifest_sha256(manifest_path),
        "design_version": manifest["design_version"],
        "profile": manifest["profile"],
        "candidate_id": candidate_id,
        "regime": regime_name,
        "n_reps": int(manifest["n_reps"]),
    }
    differing = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if differing:
        msg = f"checkpoint identity differs in fields: {differing}"
        raise ValueError(msg)
    reps = [int(record["rep"]) for record in checkpoint.get("records", [])]
    if len(reps) != len(set(reps)) or any(
        not 0 <= rep < expected["n_reps"] for rep in reps
    ):
        msg = "checkpoint contains duplicate or out-of-range replicates"
        raise ValueError(msg)
    if bool(checkpoint.get("complete")) != (
        set(reps) == set(range(expected["n_reps"]))
    ):
        msg = "checkpoint complete flag disagrees with recorded replicates"
        raise ValueError(msg)
