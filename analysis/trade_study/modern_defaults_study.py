#!/usr/bin/env python
"""Run the registered post-optimization VBPCA defaults study (#214/#226)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ._modern_defaults_io import (
    atomic_json,
    candidate_by_id,
    checkpoint_path,
    load_manifest,
    manifest_sha256,
    new_checkpoint,
    regime_by_name,
    shard_at_index,
    shards,
    validate_checkpoint,
    write_manifest,
)
from ._modern_defaults_reduction import (
    summarize_confirmation,
    summarize_scale,
    summarize_screen,
)
from ._modern_defaults_spec import (
    PROFILE_REGIMES,
    REGISTERED_CONFIRM_REPS,
    REGISTERED_CONFIRM_SEED,
    REGISTERED_SCALE_REPS,
    REGISTERED_SCALE_SEED,
)
from ._modern_defaults_trial import run_trial


def run_shard(
    manifest_path: Path,
    output_dir: Path,
    *,
    candidate_id: str,
    regime_name: str,
    num_cpu: int,
) -> Path:
    """Run or resume every replicate in one candidate/regime shard."""
    if num_cpu < 1:
        msg = f"num_cpu must be positive, got {num_cpu}"
        raise ValueError(msg)
    manifest = load_manifest(manifest_path)
    candidate = candidate_by_id(manifest, candidate_id)
    regime = regime_by_name(manifest, regime_name)
    destination = checkpoint_path(
        output_dir,
        manifest,
        candidate_id,
        regime_name,
    )

    if destination.exists():
        checkpoint: dict[str, Any] = json.loads(destination.read_text())
        validate_checkpoint(
            checkpoint,
            manifest_path,
            manifest,
            candidate_id=candidate_id,
            regime_name=regime_name,
        )
        if checkpoint["complete"]:
            print(f"Validated complete shard -> {destination}")
            return destination
    else:
        checkpoint = new_checkpoint(
            manifest_path,
            manifest,
            candidate_id=candidate_id,
            regime_name=regime_name,
            num_cpu=num_cpu,
        )
        atomic_json(destination, checkpoint)

    completed = {int(record["rep"]) for record in checkpoint["records"]}
    for rep in range(int(manifest["n_reps"])):
        if rep in completed:
            continue
        scores = run_trial(
            manifest,
            candidate,
            regime,
            rep=rep,
            num_cpu=num_cpu,
        )
        scores.update({
            "candidate_id": candidate_id,
            "regime": regime_name,
        })
        checkpoint["records"].append(scores)
        checkpoint["records"].sort(key=lambda item: int(item["rep"]))
        completed.add(rep)
        checkpoint["complete"] = len(completed) == int(manifest["n_reps"])
        atomic_json(destination, checkpoint)
        print(
            f"Saved {candidate_id}/{regime_name} replicate "
            f"{rep + 1}/{manifest['n_reps']}"
        )

    validate_checkpoint(
        checkpoint,
        manifest_path,
        manifest,
        candidate_id=candidate_id,
        regime_name=regime_name,
    )
    print(f"Completed shard -> {destination}")
    return destination


def _manifest_command(args: argparse.Namespace) -> None:
    candidate_ids = tuple(args.candidate) if args.candidate else None
    manifest = write_manifest(
        args.output,
        profile=args.profile,
        n_reps=args.n_reps,
        seed=args.seed,
        candidate_ids=candidate_ids,
        registered_screen=args.registered_screen,
    )
    print(
        f"Wrote {args.output}: {len(shards(manifest))} shards; "
        f"sha256={manifest_sha256(args.output)}"
    )


def _run_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    if args.shard_index is not None:
        candidate_id, regime_name = shard_at_index(manifest, args.shard_index)
    else:
        if args.regime is None:
            parser.error("--regime is required with --candidate-id")
        candidate_id, regime_name = args.candidate_id, args.regime
    run_shard(
        args.manifest,
        args.output_dir,
        candidate_id=candidate_id,
        regime_name=regime_name,
        num_cpu=args.num_cpu,
    )


def _list_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    for index, (candidate_id, regime_name) in enumerate(shards(manifest)):
        print(f"{index}\t{candidate_id}\t{regime_name}")


def _summary_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    summary = summarize_screen(args.manifest, manifest, args.output_dir)
    atomic_json(args.output, summary)
    print(f"Saved screen summary -> {args.output}")


def _scale_summary_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    summary = summarize_scale(args.manifest, manifest, args.output_dir)
    atomic_json(args.output, summary)
    print(f"Saved scale summary -> {args.output}")


def _promote_command(args: argparse.Namespace) -> None:
    screen_manifest = load_manifest(args.screen_manifest)
    summary = summarize_screen(args.screen_manifest, screen_manifest, args.output_dir)
    if args.summary_output is not None:
        atomic_json(args.summary_output, summary)
    candidate_ids = tuple(str(item) for item in summary["confirmation_candidate_ids"])
    manifest = write_manifest(
        args.output,
        profile="confirm",
        n_reps=REGISTERED_CONFIRM_REPS,
        seed=REGISTERED_CONFIRM_SEED,
        candidate_ids=candidate_ids,
    )
    print(
        f"Wrote {args.output}: {len(shards(manifest))} shards; "
        f"sha256={manifest_sha256(args.output)}"
    )


def _promote_scale_command(args: argparse.Namespace) -> None:
    confirm_manifest = load_manifest(args.confirm_manifest)
    summary = summarize_confirmation(
        args.confirm_manifest,
        confirm_manifest,
        args.output_dir,
    )
    if args.summary_output is not None:
        atomic_json(args.summary_output, summary)
    manifest = write_manifest(
        args.output,
        profile="scale",
        n_reps=REGISTERED_SCALE_REPS,
        seed=REGISTERED_SCALE_SEED,
        candidate_ids=tuple(summary["scale_candidate_ids"]),
    )
    print(
        f"Wrote {args.output}: {len(shards(manifest))} shards; "
        f"sha256={manifest_sha256(args.output)}"
    )


def main() -> None:
    """Run the modern-defaults study command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_REGIMES),
        default="smoke",
    )
    manifest_parser.add_argument("--n-reps", type=int, default=1)
    manifest_parser.add_argument("--seed", type=int, default=20260928)
    manifest_parser.add_argument("--candidate", action="append")
    manifest_parser.add_argument("--registered-screen", action="store_true")
    manifest_parser.add_argument("--output", type=Path, required=True)

    run_parser = subparsers.add_parser("run-shard")
    run_parser.add_argument("--manifest", type=Path, required=True)
    shard_group = run_parser.add_mutually_exclusive_group(required=True)
    shard_group.add_argument("--shard-index", type=int)
    shard_group.add_argument("--candidate-id")
    run_parser.add_argument("--regime")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--num-cpu", type=int, default=1)

    list_parser = subparsers.add_parser("list-shards")
    list_parser.add_argument("--manifest", type=Path, required=True)

    summary_parser = subparsers.add_parser("summarize-screen")
    summary_parser.add_argument("--manifest", type=Path, required=True)
    summary_parser.add_argument("--output-dir", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)

    scale_summary_parser = subparsers.add_parser("summarize-scale")
    scale_summary_parser.add_argument("--manifest", type=Path, required=True)
    scale_summary_parser.add_argument("--output-dir", type=Path, required=True)
    scale_summary_parser.add_argument("--output", type=Path, required=True)

    promote_parser = subparsers.add_parser("promote-confirm")
    promote_parser.add_argument("--screen-manifest", type=Path, required=True)
    promote_parser.add_argument("--output-dir", type=Path, required=True)
    promote_parser.add_argument("--summary-output", type=Path)
    promote_parser.add_argument("--output", type=Path, required=True)

    scale_parser = subparsers.add_parser("promote-scale")
    scale_parser.add_argument("--confirm-manifest", type=Path, required=True)
    scale_parser.add_argument("--output-dir", type=Path, required=True)
    scale_parser.add_argument("--summary-output", type=Path)
    scale_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "manifest":
        _manifest_command(args)
    elif args.command == "run-shard":
        _run_command(parser, args)
    elif args.command == "list-shards":
        _list_command(args)
    elif args.command == "summarize-screen":
        _summary_command(args)
    elif args.command == "summarize-scale":
        _scale_summary_command(args)
    elif args.command == "promote-confirm":
        _promote_command(args)
    else:
        _promote_scale_command(args)


if __name__ == "__main__":
    main()
