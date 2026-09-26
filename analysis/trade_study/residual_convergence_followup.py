#!/usr/bin/env python
"""Run the released-default residual convergence follow-up (#174).

The registered study compares VBPCA 0.4's recommended configuration with a
doubled iteration cap and a forced-long endpoint at the same doubled cap.
Each condition/regime pair is an independent, resumable Rockfish shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from trade_study import load_results, run_grid, save_results

from vbpca_py import __version__ as vbpca_version

from ._residual_convergence_design import (
    CONDITIONS,
    REGIME_PROFILES,
    build_manifest,
    condition_config,
    registered_manifest,
    validate_manifest,
)
from ._world import VBPCAScorer, VBPCASimulator
from .validate_convergence_margins import (
    OBSERVABLES,
    _means,
    _paired_summary,
    _require_complete_table,
    _score_rows,
)


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    validate_manifest(manifest)
    return manifest


def _shards(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    """Return condition/regime shards in stable array-index order.

    Returns:
        ``(condition, regime_name)`` pairs.
    """
    return [
        (str(condition), str(regime["name"]))
        for condition in manifest["conditions"]
        for regime in manifest["regimes"]
    ]


def _shard_at_index(manifest: dict[str, Any], index: int) -> tuple[str, str]:
    shards = _shards(manifest)
    if not 0 <= index < len(shards):
        msg = f"shard-index must be in [0, {len(shards)}) for this manifest"
        raise ValueError(msg)
    return shards[index]


def _regime_by_name(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [regime for regime in manifest["regimes"] if regime["name"] == name]
    if len(matches) != 1:
        msg = f"manifest does not contain exactly one regime named {name!r}"
        raise ValueError(msg)
    return matches[0]


def _shard_grid(
    manifest: dict[str, Any], condition: str, regime_name: str
) -> list[dict[str, Any]]:
    if condition not in manifest["conditions"]:
        msg = f"condition {condition!r} is absent from the manifest"
        raise ValueError(msg)
    regime = _regime_by_name(manifest, regime_name)
    n, p = int(regime["n"]), int(regime["p"])
    config = {
        key: value for key, value in regime.items() if key not in {"name", "bucket"}
    }
    config.update(condition_config(n, p, condition))
    config["_regime"] = regime_name
    config["_bucket"] = regime["bucket"]
    config["_condition"] = condition
    return [config]


def write_manifest(
    output: Path,
    *,
    registered: bool,
    profile: str,
    n_reps: int,
    seed: int,
) -> None:
    """Write a manifest atomically for the registered run or a smoke run."""
    manifest = (
        registered_manifest()
        if registered
        else build_manifest(profile, n_reps=n_reps, seed=seed)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output)
    print(
        f"Wrote {output}: {len(_shards(manifest))} shards; "
        f"sha256={_manifest_sha256(output)}"
    )


def _run_provenance(manifest_path: Path, *, n_jobs: int) -> dict[str, Any]:
    return {
        "manifest_sha256": _manifest_sha256(manifest_path),
        "vbpca_release": vbpca_version,
        "vbpca_revision": os.environ.get("VBPCA_REVISION"),
        "trade_study_revision": os.environ.get("VBPCA_TRADE_STUDY_REVISION"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "n_jobs": n_jobs,
    }


def _save_shard_checkpoint(
    table: Any,
    grid: list[dict[str, Any]],
    *,
    n_reps: int,
    temporary: Path,
    manifest_path: Path,
    condition: str,
    regime_name: str,
    n_jobs: int,
) -> None:
    """Validate and stage one shard with its complete provenance."""
    _require_complete_table(table, grid, n_reps)
    save_results(table, temporary)
    run_metadata = {
        **_run_provenance(manifest_path, n_jobs=n_jobs),
        "condition": condition,
        "regime": regime_name,
        "n_rows": len(table.configs),
    }
    (temporary / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")


def run_shard(
    manifest_path: Path,
    output_dir: Path,
    *,
    condition: str,
    regime_name: str,
    n_jobs: int,
) -> Path:
    """Run or validate one condition/regime checkpoint.

    Returns:
        The completed shard directory.
    """
    manifest = _load_manifest(manifest_path)
    if vbpca_version != str(manifest["vbpca_release"]):
        msg = (
            f"installed VBPCA release {vbpca_version!r} differs from manifest "
            f"release {manifest['vbpca_release']!r}"
        )
        raise ValueError(msg)
    n_reps = int(manifest["n_reps"])
    grid = _shard_grid(manifest, condition, regime_name)
    destination = output_dir / condition / regime_name
    if destination.exists():
        table = load_results(destination)
        _require_complete_table(table, grid, n_reps)
        run_meta = json.loads((destination / "run.json").read_text())
        if run_meta["manifest_sha256"] != _manifest_sha256(manifest_path):
            msg = f"checkpoint manifest differs from {manifest_path}"
            raise ValueError(msg)
        print(f"Validated existing complete shard -> {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{regime_name}-", dir=destination.parent)
    )
    try:
        table = run_grid(
            VBPCASimulator(),
            VBPCAScorer(),
            grid,
            OBSERVABLES,
            n_jobs=n_jobs,
            n_reps=n_reps,
        )
        _save_shard_checkpoint(
            table,
            grid,
            n_reps=n_reps,
            temporary=temporary,
            manifest_path=manifest_path,
            condition=condition,
            regime_name=regime_name,
            n_jobs=n_jobs,
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Saved complete shard -> {destination}")
    return destination


def _load_rows(
    manifest: dict[str, Any], manifest_path: Path, output_dir: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_reps = int(manifest["n_reps"])
    expected_sha = _manifest_sha256(manifest_path)
    for condition, regime_name in _shards(manifest):
        grid = _shard_grid(manifest, condition, regime_name)
        destination = output_dir / condition / regime_name
        table = load_results(destination)
        _require_complete_table(table, grid, n_reps)
        run_meta = json.loads((destination / "run.json").read_text())
        if run_meta["manifest_sha256"] != expected_sha:
            msg = f"checkpoint manifest differs in {destination}"
            raise ValueError(msg)
        rows.extend(_score_rows(table))
    return rows


def summarize(
    manifest_path: Path,
    output_dir: Path,
    *,
    output: Path,
    n_resamples: int,
) -> dict[str, Any]:
    """Validate all shards and write the preregistered paired summary.

    Returns:
        JSON-compatible summary mapping.
    """
    manifest = _load_manifest(manifest_path)
    rows = _load_rows(manifest, manifest_path, output_dir)
    conditions = tuple(str(condition) for condition in manifest["conditions"])
    regime_names = tuple(str(regime["name"]) for regime in manifest["regimes"])

    by_condition: dict[str, Any] = {}
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "overall": _means(condition_rows),
            "by_regime": {
                regime: _means([
                    row for row in condition_rows if row["regime"] == regime
                ])
                for regime in regime_names
            },
        }

    reference_name = str(manifest["reference_condition"])
    reference_rows = [row for row in rows if row["condition"] == reference_name]
    paired: dict[str, Any] = {}
    for condition_index, condition in enumerate(conditions):
        if condition == reference_name:
            continue
        condition_rows = [row for row in rows if row["condition"] == condition]
        paired[condition] = {
            "overall": _paired_summary(
                condition_rows,
                reference_rows,
                n_resamples=n_resamples,
                seed=int(manifest["seed"]) + condition_index * 10_003,
            ),
            "by_regime": {
                regime: _paired_summary(
                    [row for row in condition_rows if row["regime"] == regime],
                    [row for row in reference_rows if row["regime"] == regime],
                    n_resamples=n_resamples,
                    seed=(
                        int(manifest["seed"])
                        + condition_index * 10_003
                        + regime_index * 101
                    ),
                )
                for regime_index, regime in enumerate(regime_names)
            },
        }

    result = {
        "manifest_sha256": _manifest_sha256(manifest_path),
        "design_version": manifest["design_version"],
        "vbpca_release": manifest["vbpca_release"],
        "profile": manifest["profile"],
        "n_reps": manifest["n_reps"],
        "reference_condition": reference_name,
        "by_condition": by_condition,
        "paired_vs_reference": paired,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(f"Saved paired summary -> {output}")
    return result


def main() -> None:
    """Run the residual-convergence command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--registered", action="store_true")
    manifest_parser.add_argument(
        "--profile", choices=tuple(REGIME_PROFILES), default="smoke"
    )
    manifest_parser.add_argument("--n-reps", type=int, default=1)
    manifest_parser.add_argument("--seed", type=int, default=20260926)
    manifest_parser.add_argument("--output", type=Path, required=True)

    run_parser = subparsers.add_parser("run-shard")
    run_parser.add_argument("--manifest", type=Path, required=True)
    shard_group = run_parser.add_mutually_exclusive_group(required=True)
    shard_group.add_argument("--shard-index", type=int)
    shard_group.add_argument("--condition", choices=CONDITIONS)
    run_parser.add_argument("--regime")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--n-jobs", type=int, default=1)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--manifest", type=Path, required=True)
    summary_parser.add_argument("--output-dir", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    summary_parser.add_argument("--n-resamples", type=int, default=10_000)

    args = parser.parse_args()
    if args.command == "manifest":
        write_manifest(
            args.output,
            registered=args.registered,
            profile=args.profile,
            n_reps=args.n_reps,
            seed=args.seed,
        )
        return
    if args.command == "run-shard":
        manifest = _load_manifest(args.manifest)
        if args.shard_index is not None:
            condition, regime = _shard_at_index(manifest, args.shard_index)
        else:
            if args.regime is None:
                parser.error("--regime is required with --condition")
            condition, regime = args.condition, args.regime
        run_shard(
            args.manifest,
            args.output_dir,
            condition=condition,
            regime_name=regime,
            n_jobs=args.n_jobs,
        )
        return
    summarize(
        args.manifest,
        args.output_dir,
        output=args.output,
        n_resamples=args.n_resamples,
    )


if __name__ == "__main__":
    main()
