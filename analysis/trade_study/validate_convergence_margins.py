#!/usr/bin/env python
"""Run and reduce the post-warmup convergence-margin study (#133).

The study is paired by regime and replicate. It compares the exact shipped
configuration with larger iteration caps, shorter warmups, and a forced
800-iteration diagnostic. Results are checkpointed once per condition so six
Rockfish shared-array tasks can run independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from trade_study import (
    Direction,
    Observable,
    ResultsTable,
    load_results,
    run_grid,
    save_results,
)

from ._convergence_margin_design import (
    CONDITIONS,
    build_manifest,
    condition_config,
    validate_manifest,
)
from ._world import VBPCAScorer, VBPCASimulator

OBSERVABLES = [
    Observable("rank_mae", Direction.MINIMIZE),
    Observable("rank_under", Direction.MINIMIZE),
    Observable("rank_over", Direction.MINIMIZE),
    Observable("holdout_rmse", Direction.MINIMIZE),
    Observable("holdout_mae", Direction.MINIMIZE),
    Observable("coverage_95", Direction.MAXIMIZE),
    Observable("interval_score", Direction.MINIMIZE),
    Observable("selected_k", Direction.MINIMIZE),
    Observable("best_k_iters", Direction.MINIMIZE),
    Observable("total_iters", Direction.MINIMIZE),
    Observable("best_k_converged", Direction.MAXIMIZE),
    Observable("best_k_budget_hit", Direction.MINIMIZE),
    Observable("candidate_converged_rate", Direction.MAXIMIZE),
    Observable("candidate_budget_hit_rate", Direction.MINIMIZE),
]
OBSERVABLE_NAMES = [observable.name for observable in OBSERVABLES]


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    validate_manifest(manifest)
    return manifest


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition_grid(manifest: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for regime in manifest["regimes"]:
        n, p = int(regime["n"]), int(regime["p"])
        config = {
            key: value for key, value in regime.items() if key not in {"name", "bucket"}
        }
        config.update(condition_config(n, p, condition))
        config["_regime"] = regime["name"]
        config["_bucket"] = regime["bucket"]
        config["_condition"] = condition
        grid.append(config)
    return grid


def _expected_configs(grid: list[dict[str, Any]], n_reps: int) -> list[dict[str, Any]]:
    return [config for config in grid for _rep in range(n_reps)]


def _require_complete_table(
    table: ResultsTable,
    grid: list[dict[str, Any]],
    n_reps: int,
) -> None:
    expected = _expected_configs(grid, n_reps)
    if table.observable_names != OBSERVABLE_NAMES:
        msg = "saved observable schema differs from the current design"
        raise ValueError(msg)
    if table.configs != expected:
        msg = "saved configs differ from the manifest-derived design"
        raise ValueError(msg)
    if table.scores.shape != (len(expected), len(OBSERVABLE_NAMES)):
        msg = f"saved score shape is invalid: {table.scores.shape}"
        raise ValueError(msg)
    if not np.isfinite(table.scores).all():
        msg = "saved scores contain non-finite values"
        raise ValueError(msg)
    expected_metadata = [
        (design_point, rep)
        for design_point in range(len(grid))
        for rep in range(n_reps)
    ]
    observed_metadata = [
        (int(row["design_point"]), int(row["rep"])) for row in table.metadata
    ]
    if observed_metadata != expected_metadata:
        msg = "saved replicate metadata are incomplete or out of order"
        raise ValueError(msg)


def write_manifest(
    output: Path,
    *,
    profile: str,
    n_reps: int,
    seed: int,
) -> None:
    """Write one immutable manifest atomically."""
    manifest = build_manifest(profile, n_reps=n_reps, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output)
    print(
        f"Wrote {output}: {len(manifest['conditions'])} conditions x "
        f"{len(manifest['regimes'])} regimes x {n_reps} replicates; "
        f"sha256={_manifest_sha256(output)}"
    )


def run_condition(
    manifest_path: Path,
    output_dir: Path,
    *,
    condition: str,
    n_jobs: int,
) -> Path:
    """Run or validate one condition checkpoint.

    Returns:
        Condition result directory.
    """
    manifest = _load_manifest(manifest_path)
    if condition not in manifest["conditions"]:
        msg = f"condition {condition!r} is absent from the manifest"
        raise ValueError(msg)
    n_reps = int(manifest["n_reps"])
    grid = _condition_grid(manifest, condition)
    destination = output_dir / condition
    if destination.exists():
        table = load_results(destination)
        _require_complete_table(table, grid, n_reps)
        print(f"Validated existing complete checkpoint -> {destination}")
        return destination

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{condition}-", dir=output_dir))
    try:
        table = run_grid(
            VBPCASimulator(),
            VBPCAScorer(),
            grid,
            OBSERVABLES,
            n_jobs=n_jobs,
            n_reps=n_reps,
        )
        _save_condition_checkpoint(
            table,
            grid,
            n_reps=n_reps,
            temporary=temporary,
            destination=destination,
            manifest_path=manifest_path,
            condition=condition,
            n_jobs=n_jobs,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Saved complete condition -> {destination}")
    return destination


def _save_condition_checkpoint(
    table: ResultsTable,
    grid: list[dict[str, Any]],
    *,
    n_reps: int,
    temporary: Path,
    destination: Path,
    manifest_path: Path,
    condition: str,
    n_jobs: int,
) -> None:
    """Validate and atomically install one completed condition."""
    _require_complete_table(table, grid, n_reps)
    save_results(table, temporary)
    run_metadata = {
        "manifest_sha256": _manifest_sha256(manifest_path),
        "condition": condition,
        "n_jobs": n_jobs,
        "n_rows": len(table.configs),
    }
    (temporary / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    temporary.replace(destination)


def _score_rows(table: ResultsTable) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(table.configs):
        score = dict(
            zip(table.observable_names, table.scores[index].tolist(), strict=True)
        )
        rows.append({
            "condition": config["_condition"],
            "regime": config["_regime"],
            "bucket": config["_bucket"],
            "rep": int(table.metadata[index]["rep"]),
            "true_rank": int(config["true_rank"]),
            **score,
        })
    return rows


def _means(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    output: dict[str, float | int] = {"n": len(rows)}
    for name in OBSERVABLE_NAMES:
        output[name] = float(np.mean([float(row[name]) for row in rows]))
    output["exact_rank_recovery"] = float(
        np.mean([
            int(round(float(row["selected_k"]))) == int(row["true_rank"])
            for row in rows
        ])
    )
    return output


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_resamples: int,
) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _paired_summary(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    reference_by_key = {(row["regime"], row["rep"]): row for row in reference}
    pairs = [(row, reference_by_key[row["regime"], row["rep"]]) for row in candidate]
    if len(pairs) != len(reference) or len(reference_by_key) != len(reference):
        msg = "candidate/reference replicate keys are not one-to-one"
        raise ValueError(msg)

    candidate_exact = np.asarray([
        int(round(float(row["selected_k"]))) == int(row["true_rank"])
        for row, _reference in pairs
    ])
    reference_exact = np.asarray([
        int(round(float(row["selected_k"]))) == int(row["true_rank"])
        for _row, row in pairs
    ])
    differences = {
        "exact_rank_recovery_difference": candidate_exact.astype(float)
        - reference_exact.astype(float),
        "mae_gain": np.asarray([
            float(reference_row["rank_mae"]) - float(candidate_row["rank_mae"])
            for candidate_row, reference_row in pairs
        ]),
        "holdout_rmse_difference": np.asarray([
            float(candidate_row["holdout_rmse"]) - float(reference_row["holdout_rmse"])
            for candidate_row, reference_row in pairs
        ]),
        "coverage_95_difference": np.asarray([
            float(candidate_row["coverage_95"]) - float(reference_row["coverage_95"])
            for candidate_row, reference_row in pairs
        ]),
        "best_k_iters_difference": np.asarray([
            float(candidate_row["best_k_iters"]) - float(reference_row["best_k_iters"])
            for candidate_row, reference_row in pairs
        ]),
        "best_k_budget_hit_difference": np.asarray([
            float(candidate_row["best_k_budget_hit"])
            - float(reference_row["best_k_budget_hit"])
            for candidate_row, reference_row in pairs
        ]),
    }
    output: dict[str, Any] = {
        "n_pairs": len(pairs),
        "selected_rank_disagreement_rate": float(
            np.mean([
                int(round(float(candidate_row["selected_k"])))
                != int(round(float(reference_row["selected_k"])))
                for candidate_row, reference_row in pairs
            ])
        ),
        "rescues": int(np.sum(candidate_exact & ~reference_exact)),
        "spoils": int(np.sum(~candidate_exact & reference_exact)),
    }
    rng = np.random.default_rng(seed)
    for name, values in differences.items():
        output[name] = float(np.mean(values))
        output[f"{name}_ci95"] = _bootstrap_mean_interval(
            values,
            rng=rng,
            n_resamples=n_resamples,
        )
    return output


def summarize(
    manifest_path: Path,
    output_dir: Path,
    *,
    output: Path,
    n_resamples: int,
) -> dict[str, Any]:
    """Validate every condition and write a paired summary.

    Returns:
        JSON-compatible summary mapping.
    """
    manifest = _load_manifest(manifest_path)
    n_reps = int(manifest["n_reps"])
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        grid = _condition_grid(manifest, condition)
        table = load_results(output_dir / condition)
        _require_complete_table(table, grid, n_reps)
        rows.extend(_score_rows(table))

    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "overall": _means(condition_rows),
            "by_regime": {
                regime["name"]: _means([
                    row for row in condition_rows if row["regime"] == regime["name"]
                ])
                for regime in manifest["regimes"]
            },
        }

    reference_name = str(manifest["reference_condition"])
    reference_rows = [row for row in rows if row["condition"] == reference_name]
    paired = {
        condition: _paired_summary(
            [row for row in rows if row["condition"] == condition],
            reference_rows,
            n_resamples=n_resamples,
            seed=int(manifest["seed"]) + index * 10_003,
        )
        for index, condition in enumerate(CONDITIONS)
        if condition != reference_name
    }
    result = {
        "manifest_sha256": _manifest_sha256(manifest_path),
        "design_version": manifest["design_version"],
        "profile": manifest["profile"],
        "n_reps": n_reps,
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
    """Run the convergence-margin command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--profile", choices=("smoke", "screen", "confirm"))
    manifest_parser.add_argument("--n-reps", type=int, required=True)
    manifest_parser.add_argument("--seed", type=int, default=20260922)
    manifest_parser.add_argument("--output", type=Path, required=True)

    run_parser = subparsers.add_parser("run-condition")
    run_parser.add_argument("--manifest", type=Path, required=True)
    condition_group = run_parser.add_mutually_exclusive_group(required=True)
    condition_group.add_argument("--condition", choices=CONDITIONS)
    condition_group.add_argument("--condition-index", type=int)
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
            profile=args.profile,
            n_reps=args.n_reps,
            seed=args.seed,
        )
        return
    if args.command == "run-condition":
        condition = args.condition
        if args.condition_index is not None:
            if not 0 <= args.condition_index < len(CONDITIONS):
                msg = f"condition-index must be in [0, {len(CONDITIONS)})"
                raise ValueError(msg)
            condition = CONDITIONS[args.condition_index]
        run_condition(
            args.manifest,
            args.output_dir,
            condition=condition,
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
