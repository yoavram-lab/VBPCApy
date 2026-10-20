"""Replay selected policies from complete retained version 2 screen artifacts.

This is exploratory analysis. Stored fidelity labels and original study inputs
remain unchanged; a replay cannot qualify a replacement production policy.
"""

# ruff: noqa: SLF001 - reuse the existing scientific replay and shard contract

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess  # noqa: S404 - fixed, read-only Git argv
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from vbpca_py import __version__ as vbpca_version

from . import convergence_detector as detector
from . import convergence_stability_study as study
from ._convergence_detector_design import MANIFEST_VERSION, policy_from_json
from .convergence_detector import DetectorPolicy
from .convergence_detector_summary import summaries_by, summarize_records


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path) -> tuple[Any, str]:
    payload = path.read_bytes()
    return json.loads(payload), _digest(payload)


def _integer(value: object, minimum: int) -> bool:
    return type(value) is int and value >= minimum


def _threshold(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def _validate_policy(payload: dict[str, Any]) -> DetectorPolicy:
    """Reject coercible flags/counts and nonfinite settings before replay."""
    defaults = asdict(DetectorPolicy(name="placeholder", enabled=()))
    if (
        not isinstance(payload, dict)
        or set(payload) - set(defaults)
        or not {"name", "enabled"}.issubset(payload)
    ):
        msg = "policy must be an object containing only DetectorPolicy fields"
        raise ValueError(msg)
    values = defaults | payload
    valid_name = isinstance(values["name"], str) and bool(values["name"].strip())
    valid_enabled = isinstance(values["enabled"], (list, tuple)) and all(
        isinstance(name, str) for name in values["enabled"]
    )
    valid_controls = (
        type(values["earlystop"]) is bool
        and _integer(values["patience"], 1)
        and _integer(values["warmup"], 0)
    )
    valid_order = (
        isinstance(values["criterion_order"], (list, tuple))
        and len(values["criterion_order"]) == len(detector.VALID_CRITERIA)
        and all(isinstance(name, str) for name in values["criterion_order"])
    )
    if not all((valid_name, valid_enabled, valid_controls, valid_order)):
        msg = "policy names, flags, counts or criterion order are invalid"
        raise ValueError(msg)
    for key in ("minangle", "cfstop_rel", "cfstop_curv"):
        if values[key] is not None and not _threshold(values[key]):
            msg = f"policy {key} must be finite and nonnegative"
            raise ValueError(msg)
    for key in ("rmsstop", "cfstop"):
        triple = values[key]
        if triple is not None and (
            not isinstance(triple, (list, tuple))
            or len(triple) != 3
            or not _integer(triple[0], 1)
            or not all(_threshold(value) for value in triple[1:])
        ):
            msg = f"policy {key} needs an integer window and finite thresholds"
            raise ValueError(msg)
    composite = values["composite_stop"]
    if not isinstance(composite, (list, tuple)) or any(
        not isinstance(pair, (list, tuple))
        or len(pair) != 2
        or pair[0] not in {"angle", "rms", "elbo_rel", "elbo_curv"}
        or not _threshold(pair[1])
        for pair in composite
    ):
        msg = "invalid composite policy thresholds"
        raise ValueError(msg)
    if len({pair[0] for pair in composite}) != len(composite):
        msg = "duplicate composite policy thresholds"
        raise ValueError(msg)
    return policy_from_json(values)


def _source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = [
        *sorted((root / "analysis/trade_study").glob("*convergence*.py")),
        *[
            root / "src/vbpca_py" / name
            for name in (
                "_converge.py",
                "_pca_full.py",
                "defaults.py",
            )
        ],
    ]
    files = {str(path.relative_to(root)): _digest(path.read_bytes()) for path in paths}
    git = shutil.which("git")
    if git is None:
        msg = "Git is required to record replay source identity"
        raise FileNotFoundError(msg)
    revision = subprocess.run(  # noqa: S603 - read-only fixed Git argv
        [git, "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(  # noqa: S603 - read-only fixed Git argv
        [git, "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "revision": revision,
        "tracked_changes": bool(dirty),
        "files": files,
        "vbpca_release": vbpca_version,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _validate_record(
    record: dict[str, Any],
    manifest: dict[str, Any],
    manifest_sha: str,
    revision: str,
    index: int,
) -> None:
    cell, rep = study._shard_at_index(manifest, index)
    expected = {
        "manifest_sha256": manifest_sha,
        "vbpca_revision": revision,
        "status": "ok",
        "result_schema_version": 2,
        "shard_index": index,
        "rep": rep,
        "cell": cell,
    }
    if (
        any(record.get(key) != value for key, value in expected.items())
        or type(record.get("shard_index")) is not int
        or type(record.get("rep")) is not int
    ):
        msg = f"shard {index} identity, revision, schema or completion mismatch"
        raise ValueError(msg)
    curve = record.get("learning_curve", {})
    endpoint = manifest["checkpoints"][-1]
    if curve.get("schema_version") != 1:
        msg = f"shard {index} has no supported retained curve"
        raise ValueError(msg)
    for name in ("rms", "prms", "cost", "angle"):
        series = curve.get(name)
        # The initialization may lack cost/angle/probe metrics. Subsequent
        # missing values cannot support a policy replay and are refused.
        if (
            not isinstance(series, list)
            or len(series) != endpoint + 1
            or not all(
                value is None
                or _threshold(value)
                or (type(value) in {int, float} and math.isfinite(value))
                for value in series
            )
            or any(value is None for value in series[1:])
        ):
            msg = f"shard {index} has a missing, nonfinite or misaligned {name} curve"
            raise ValueError(msg)
    fidelity = record.get("fidelity", {})
    rows = fidelity.get("rows", [])
    checkpoints = manifest["checkpoints"]
    if (
        fidelity.get("endpoint_checkpoint") != endpoint
        or [row.get("checkpoint") for row in rows] != checkpoints
        or any(type(row.get("acceptable")) is not bool for row in rows)
    ):
        msg = f"shard {index} has invalid frozen fidelity rows"
        raise ValueError(msg)
    tail = int(manifest["tail_points"])
    stable = len(rows[-tail - 1 : -1]) == tail and all(
        row["acceptable"] for row in rows[-tail - 1 : -1]
    )
    earliest = next(
        (
            row["checkpoint"]
            for i, row in enumerate(rows)
            if stable and all(later["acceptable"] for later in rows[i:])
        ),
        None,
    )
    if (
        type(fidelity.get("endpoint_stable")) is not bool
        or fidelity["endpoint_stable"] != stable
        or fidelity.get("earliest_fidelity_checkpoint") != earliest
    ):
        msg = f"shard {index} frozen fidelity labels are inconsistent"
        raise ValueError(msg)


def _baseline_policies(
    record: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, DetectorPolicy]:
    policies = [_validate_policy(payload) for payload in manifest["policies"]]
    if manifest.get("include_production_policies") is True:
        options = dict(record["base_options"])
        options["convergence_criteria"] = dict(
            options.get("convergence_criteria", {}), slowing_down=False
        )
        policies.extend(study._production_policies(options))
    return {policy.name: policy for policy in policies}


def _replay_rows(
    record: dict[str, Any],
    manifest: dict[str, Any],
    selected: tuple[str, ...],
    additional: list[DetectorPolicy],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    policies = _baseline_policies(record, manifest)
    saved = record["policies"]
    available = [name for name in selected if name in saved["aliases"]]
    if set(available) - set(policies) or set(policies) & {p.name for p in additional}:
        msg = (
            "unknown selected baseline or additional policy name collides with baseline"
        )
        raise ValueError(msg)
    roster = [policies[name] for name in available] + additional
    representatives, aliases = detector.collapse_equivalent_policies(roster)
    evaluated = study._evaluate_policies(
        record["learning_curve"],
        manifest
        | {
            "include_production_policies": False,
            "policies": [asdict(policy) for policy in representatives],
        },
        record["fidelity"],
        base_options=record["base_options"],
    )
    by_name = {row["policy"]: row for row in evaluated["rows"]}
    expanded = [
        by_name[aliases[policy.name]] | {"policy": policy.name} for policy in roster
    ]
    stored = {row["policy"]: row for row in saved["rows"]}
    for row in expanded[: len(available)]:
        original = stored[saved["aliases"][row["policy"]]] | {"policy": row["policy"]}
        # Ordering changes attribution for simultaneous rules. Stop-time and
        # safety classification must match; retain both reasons separately.
        if any(value != original[key] for key, value in row.items() if key != "reason"):
            msg = f"baseline replay differs: shard {record['shard_index']}, {row['policy']}"
            raise ValueError(msg)
        row["stored_reason"] = original["reason"]
    return expanded, aliases


def replay(
    manifest_path: Path,
    input_dir: Path,
    *,
    manifest_sha256: str,
    generating_revision: str,
    selected: tuple[str, ...],
    policies_path: Path | None = None,
) -> dict[str, Any]:
    """Validate every shard and require baseline agreement for exploratory reports."""
    if (
        not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)
        or not re.fullmatch(
            r"[0-9a-f]{40}",
            generating_revision,
        )
        or not selected
        or len(set(selected)) != len(selected)
    ):
        msg = "provide trusted full input identities and distinct baseline policy names"
        raise ValueError(msg)
    manifest, actual_sha = _load(manifest_path)
    if actual_sha != manifest_sha256:
        msg = "manifest checksum differs from expected identity"
        raise ValueError(msg)
    study.validate_manifest(manifest)
    if manifest["manifest_version"] != MANIFEST_VERSION:
        msg = "offline replay requires a retained version 2 manifest"
        raise ValueError(msg)
    additional: list[DetectorPolicy] = []
    inputs = {str(manifest_path.resolve()): actual_sha}
    if policies_path is not None:
        payload, identity = _load(policies_path)
        if not isinstance(payload, list):
            msg = "additional policies must be a JSON list"
            raise ValueError(msg)
        additional = [_validate_policy(item) for item in payload]
        if any(
            not policy.enabled or "slowing_down" in policy.enabled
            for policy in additional
        ):
            msg = "additional policies need supported live criteria; slowing_down has no stream"
            raise ValueError(msg)
        detector.collapse_equivalent_policies(additional)
        inputs[str(policies_path.resolve())] = identity
    count = len(study._shards(manifest))
    expected = {f"shard-{index:05d}.json" for index in range(count)}
    if {path.name for path in input_dir.glob("shard-*.json")} != expected:
        msg = "replay requires the exact complete shard set"
        raise ValueError(msg)
    source = _source_identity()
    records: list[dict[str, Any]] = []
    generating_runtime: dict[str, str] | None = None
    availability = dict.fromkeys(selected, 0)
    for index in range(count):
        path = input_dir / f"shard-{index:05d}.json"
        record, identity = _load(path)
        _validate_record(record, manifest, actual_sha, generating_revision, index)
        runtime = {
            key: record[key] for key in ("python", "numpy", "scipy", "vbpca_release")
        }
        if any(
            not isinstance(value, str) or not value for value in runtime.values()
        ) or (generating_runtime is not None and runtime != generating_runtime):
            msg = "mixed or missing generating runtime identities"
            raise ValueError(msg)
        generating_runtime = runtime
        rows, aliases = _replay_rows(record, manifest, selected, additional)
        for name in selected:
            availability[name] += int(name in aliases)
        records.append({
            "shard_index": index,
            "cell": record["cell"],
            "rep": record["rep"],
            "fidelity": record["fidelity"],
            "policies": {"rows": rows, "aliases": aliases},
            "unavailable_baselines": [name for name in selected if name not in aliases],
        })
        inputs[str(path.resolve())] = identity
    if not all(availability.values()):
        msg = "selected baseline does not occur in any stored shard"
        raise ValueError(msg)
    if _source_identity() != source or any(
        _digest(Path(path).read_bytes()) != identity
        for path, identity in inputs.items()
    ):
        msg = "source or input changed during replay"
        raise ValueError(msg)
    return {
        "schema_version": "vbpca.convergence-replay.v1",
        "exploratory": True,
        "qualification": "No policy promotion or independent confirmation is implied.",
        "manifest_sha256": actual_sha,
        "generating_revision": generating_revision,
        "generating_runtime": generating_runtime,
        "replay_source": source,
        "input_sha256": inputs,
        "baseline_policies": list(selected),
        "baseline_availability": availability,
        "additional_policies": [asdict(policy) for policy in additional],
        "n_unstable_endpoints": sum(
            not r["fidelity"]["endpoint_stable"] for r in records
        ),
        "summary": summarize_records(records),
        "by_shape": summaries_by(records, "shape"),
        "by_scenario": summaries_by(records, "scenario"),
        "by_missingness": summaries_by(records, "missingness"),
        "records": records,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Publish a complete JSON report atomically, refusing any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(report, handle, allow_nan=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    """Run offline replay without fitting or overwriting original study files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--generating-revision", required=True)
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--policies", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(
        args.input_dir.resolve()
    ):
        parser.error("use a new output file outside the retained input directory")
    report = replay(
        args.manifest,
        args.input_dir,
        manifest_sha256=args.manifest_sha256,
        generating_revision=args.generating_revision,
        selected=tuple(args.policy),
        policies_path=args.policies,
    )
    write_report(args.output, report)
    print(
        f"Wrote exploratory replay for {report['summary']['n_records']} shards -> {args.output}"
    )


if __name__ == "__main__":
    main()
