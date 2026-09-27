#!/usr/bin/env python
"""Run and reduce posterior-stability convergence validation (#186)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.linalg import subspace_angles

from vbpca_py import VBPCA, recommend_config
from vbpca_py import __version__ as vbpca_version
from vbpca_py._pca_full import (
    _build_options,
)

from ._convergence_detector_design import (
    build_manifest,
    policy_from_json,
    validate_manifest,
)
from .convergence_detector import (
    DetectorPolicy,
    collapse_equivalent_policies,
    policy_from_options,
    replay_policies,
)
from .convergence_detector_summary import summaries_by

ALL_CRITERIA_FALSE = {
    "angle": False,
    "earlystop": False,
    "rms_plateau": False,
    "cost": False,
    "composite": False,
    "slowing_down": False,
}


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    validate_manifest(manifest)
    return manifest


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_manifest(output: Path, *, profile: str, n_reps: int, seed: int) -> None:
    """Write one immutable manifest atomically."""
    manifest = build_manifest(profile, n_reps=n_reps, seed=seed)
    _atomic_json(output, manifest)
    n_shards = len(manifest["cells"]) * int(manifest["n_reps"])
    print(
        f"Wrote {output}: {len(manifest['cells'])} cells, "
        f"{n_shards} shards, {len(manifest['policies'])} policies; "
        f"sha256={_manifest_sha256(output)}"
    )


def _shards(manifest: dict[str, Any]) -> list[tuple[dict[str, Any], int]]:
    return [
        (cell, rep)
        for cell in manifest["cells"]
        for rep in range(int(manifest["n_reps"]))
    ]


def _shard_at_index(manifest: dict[str, Any], index: int) -> tuple[dict[str, Any], int]:
    shards = _shards(manifest)
    if not 0 <= index < len(shards):
        msg = f"shard-index must be in [0, {len(shards)})"
        raise ValueError(msg)
    return shards[index]


def _seed(base: int, cell_id: str, rep: int, stream: str) -> int:
    payload = f"{base}|{cell_id}|{rep}|{stream}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _standardize(values: np.ndarray) -> np.ndarray:
    centered = values - float(np.mean(values))
    scale = float(np.std(centered))
    return centered / max(scale, np.finfo(float).eps)


def _missing_probabilities(scores: np.ndarray, fraction: float) -> np.ndarray:
    standardized = _standardize(np.asarray(scores, dtype=float))
    lower, upper = -20.0, 20.0
    for _ in range(80):
        intercept = 0.5 * (lower + upper)
        probability = 1.0 / (1.0 + np.exp(-(intercept + 1.5 * standardized)))
        if float(np.mean(probability)) < fraction:
            lower = intercept
        else:
            upper = intercept
    intercept = 0.5 * (lower + upper)
    return 1.0 / (1.0 + np.exp(-(intercept + 1.5 * standardized)))


def _generate_matrix(
    cell: dict[str, Any], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    p, n = int(cell["p"]), int(cell["n"])
    rank = int(cell["true_rank"])
    signal = np.zeros((p, n), dtype=float)
    if rank:
        signal = rng.standard_normal((p, rank)) @ rng.standard_normal((rank, n))
        signal *= float(cell["signal_std"]) / max(
            float(np.std(signal)), np.finfo(float).eps
        )
    noise_model = str(cell["noise_model"])
    noise_std = float(cell["noise_std"])
    if noise_model == "gaussian":
        noise = noise_std * rng.standard_normal((p, n))
    elif noise_model == "heteroskedastic_gaussian":
        feature_scale = noise_std * np.exp(np.linspace(-0.8, 0.8, p))
        noise = feature_scale[:, None] * rng.standard_normal((p, n))
    elif noise_model == "student_t3":
        noise = noise_std * rng.standard_t(3.0, size=(p, n)) / math.sqrt(3.0)
    else:
        msg = f"unknown noise model {noise_model!r}"
        raise ValueError(msg)
    covariate = rng.standard_normal(n)
    return signal + noise, covariate


def _repair_mask_support(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    repaired = np.array(mask, dtype=bool, copy=True)
    p, n = repaired.shape
    for row in range(p):
        if not repaired[row].any():
            repaired[row, int(rng.integers(n))] = True
    for column in range(n):
        if not repaired[:, column].any():
            repaired[int(rng.integers(p)), column] = True
    return repaired


def _apply_missingness(
    matrix: np.ndarray,
    covariate: np.ndarray,
    mechanism: str,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    p, n = matrix.shape
    if mechanism == "complete":
        return np.ones_like(matrix, dtype=bool)
    if mechanism == "mcar":
        observed = rng.random(matrix.shape) >= fraction
    elif mechanism == "mar":
        probability = _missing_probabilities(covariate, fraction)
        observed = rng.random(matrix.shape) >= probability[None, :]
    elif mechanism == "mnar":
        probability = _missing_probabilities(matrix.ravel(), fraction).reshape(
            matrix.shape
        )
        observed = rng.random(matrix.shape) >= probability
    elif mechanism == "block":
        observed = np.ones_like(matrix, dtype=bool)
        side = math.sqrt(fraction)
        n_rows = min(p - 1, max(1, int(round(side * p))))
        n_columns = min(n - 1, max(1, int(round(side * n))))
        row_start = int(rng.integers(0, p - n_rows + 1))
        column_start = int(rng.integers(0, n - n_columns + 1))
        observed[
            row_start : row_start + n_rows,
            column_start : column_start + n_columns,
        ] = False
    else:
        msg = f"unknown missingness mechanism {mechanism!r}"
        raise ValueError(msg)
    return _repair_mask_support(observed, rng)


def _holdout_split(
    observed: np.ndarray, fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    training = np.array(observed, copy=True)
    holdout = np.zeros_like(observed, dtype=bool)
    candidates = rng.permutation(np.argwhere(observed))
    target = max(1, int(round(fraction * len(candidates))))
    selected = 0
    for row, column in candidates:
        if selected >= target:
            break
        if int(np.sum(training[row])) <= 1 or int(np.sum(training[:, column])) <= 1:
            continue
        training[row, column] = False
        holdout[row, column] = True
        selected += 1
    if selected == 0:
        msg = "could not create a support-preserving validation holdout"
        raise ValueError(msg)
    return training, holdout


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _fit_checkpoint(
    matrix: np.ndarray,
    training_mask: np.ndarray,
    holdout_mask: np.ndarray,
    *,
    rank_cap: int,
    checkpoint: int,
    random_state: int,
    base_options: dict[str, Any],
) -> dict[str, Any]:
    xprobe = np.full(matrix.shape, np.nan, dtype=float)
    xprobe[holdout_mask] = matrix[holdout_mask]
    options = dict(base_options)
    options.update(
        {
            "maxiters": checkpoint,
            "random_state": random_state,
            "convergence_criteria": dict(ALL_CRITERIA_FALSE),
            "earlystop": False,
            "record_cost": True,
            "runtime_tuning": "off",
            "num_cpu": 1,
            "verbose": 0,
        }
    )
    started = time.perf_counter()
    model = VBPCA(rank_cap, **options).fit(
        matrix,
        mask=training_mask,
        xprobe=xprobe,
    )
    elapsed = time.perf_counter() - started
    reconstruction = np.asarray(model.reconstruction_, dtype=float)
    variance = model.predictive_variance_
    if variance is None:
        if model.variance_ is None or model.noise_variance_ is None:
            msg = "VBPCA fit did not return predictive variance"
            raise RuntimeError(msg)
        variance = np.asarray(model.variance_, dtype=float) + float(
            model.noise_variance_
        )
    residual = matrix[holdout_mask] - reconstruction[holdout_mask]
    return {
        "checkpoint": checkpoint,
        "n_iter": int(model.n_iter_ or 0),
        "active_components": int(model.components_.shape[1]),
        "noise_variance": float(model.noise_variance_),
        "holdout_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "wall_seconds": elapsed,
        "reconstruction": reconstruction,
        "predictive_variance": np.asarray(variance, dtype=float),
        "loadings": np.asarray(model.components_, dtype=float),
        "learning_curve": model.learning_curve_,
    }


def _relative_frobenius(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).eps)
    return float(np.linalg.norm(value - reference) / denominator)


def _subspace_angle(value: np.ndarray, reference: np.ndarray) -> float:
    if value.shape[1] != reference.shape[1]:
        return float(np.pi / 2.0)
    if value.shape[1] == 0:
        return 0.0
    angles = subspace_angles(value, reference)
    return float(np.max(angles)) if angles.size else 0.0


def _fidelity_row(
    fit: dict[str, Any], reference: dict[str, Any], margins: dict[str, Any]
) -> dict[str, Any]:
    reference_rmse = max(float(reference["holdout_rmse"]), np.finfo(float).eps)
    metrics = {
        "reconstruction_relative_frobenius": _relative_frobenius(
            fit["reconstruction"], reference["reconstruction"]
        ),
        "predictive_variance_relative_frobenius": _relative_frobenius(
            fit["predictive_variance"], reference["predictive_variance"]
        ),
        "noise_variance_relative_change": abs(
            float(fit["noise_variance"]) - float(reference["noise_variance"])
        )
        / max(abs(float(reference["noise_variance"])), np.finfo(float).eps),
        "loading_subspace_max_angle_radians": _subspace_angle(
            fit["loadings"], reference["loadings"]
        ),
        "holdout_rmse_relative_change": abs(
            float(fit["holdout_rmse"]) - float(reference["holdout_rmse"])
        )
        / reference_rmse,
        "active_components_match": int(fit["active_components"])
        == int(reference["active_components"]),
    }
    acceptable = all(
        float(metrics[name]) <= float(limit)
        for name, limit in margins.items()
        if name != "active_components_must_match"
    ) and (
        bool(metrics["active_components_match"])
        or not bool(margins["active_components_must_match"])
    )
    return {
        "checkpoint": int(fit["checkpoint"]),
        **metrics,
        "acceptable": acceptable,
        "n_iter": int(fit["n_iter"]),
        "active_components": int(fit["active_components"]),
        "noise_variance": float(fit["noise_variance"]),
        "holdout_rmse": float(fit["holdout_rmse"]),
        "wall_seconds": float(fit["wall_seconds"]),
    }


def _assess_fidelity(
    fits: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    reference = fits[-1]
    rows = [_fidelity_row(fit, reference, manifest["fidelity_margins"]) for fit in fits]
    tail_points = int(manifest["tail_points"])
    tail_rows = rows[-(tail_points + 1) : -1]
    endpoint_stable = len(tail_rows) == tail_points and all(
        bool(row["acceptable"]) for row in tail_rows
    )
    earliest: int | None = None
    if endpoint_stable:
        for index, row in enumerate(rows):
            if all(bool(later["acceptable"]) for later in rows[index:]):
                earliest = int(row["checkpoint"])
                break
    return {
        "endpoint_checkpoint": int(reference["checkpoint"]),
        "endpoint_stable": endpoint_stable,
        "earliest_fidelity_checkpoint": earliest,
        "rows": rows,
    }


def _production_policies(base_options: dict[str, Any]) -> list[DetectorPolicy]:
    """Return the exact effective production policy and criterion ablations."""
    resolved_input = dict(base_options)
    resolved_input.pop("xprobe_fraction", None)
    resolved = _build_options(resolved_input)
    production = policy_from_options("production", resolved)
    ablations = [
        policy_from_options(
            f"production_without_{criterion}",
            resolved,
            disabled=(criterion,),
        )
        for criterion in production.enabled
    ]
    return [production, *ablations]


def _compact_learning_curve(learning_curve: dict[str, Any]) -> dict[str, Any]:
    """Retain only versioned scalar series required for offline replay."""
    names = ("rms", "prms", "cost", "angle")
    arrays = {
        name: np.asarray(learning_curve.get(name, []), dtype=float) for name in names
    }
    lengths = {len(array) for array in arrays.values() if array.ndim == 1}
    if any(array.ndim != 1 for array in arrays.values()) or len(lengths) != 1:
        msg = "replay learning-curve series must be one-dimensional and aligned"
        raise ValueError(msg)
    length = lengths.pop()
    if length < 2:
        msg = "replay learning curve must include initialization and one iteration"
        raise ValueError(msg)
    return {
        "schema_version": 1,
        **{
            name: [float(value) if np.isfinite(value) else None for value in array]
            for name, array in arrays.items()
        },
    }


def _evaluate_policies(
    learning_curve: dict[str, Any],
    manifest: dict[str, Any],
    fidelity: dict[str, Any],
    *,
    base_options: dict[str, Any],
) -> dict[str, Any]:
    policies = [policy_from_json(payload) for payload in manifest["policies"]]
    if manifest.get("include_production_policies") is True:
        policies.extend(_production_policies(base_options))
    representatives, aliases = collapse_equivalent_policies(policies)
    results = replay_policies(learning_curve, representatives)
    earliest = fidelity["earliest_fidelity_checkpoint"]
    evaluable = bool(fidelity["endpoint_stable"] and earliest is not None)
    late_margin = int(manifest["late_margin"])
    rows: list[dict[str, Any]] = []
    for result in results:
        stop = result.stop_iteration
        rows.append(
            {
                "policy": result.policy,
                "stop_iteration": stop,
                "reason": result.reason,
                "evaluable": evaluable,
                "premature": bool(evaluable and stop is not None and stop < earliest),
                "late_or_no_stop": bool(
                    evaluable and (stop is None or stop > earliest + late_margin)
                ),
                "excess_iterations": (
                    max(0, int(stop) - int(earliest))
                    if evaluable and stop is not None
                    else None
                ),
            }
        )
    return {"aliases": aliases, "rows": rows}


def _run_one(
    manifest: dict[str, Any], cell: dict[str, Any], rep: int
) -> dict[str, Any]:
    base_seed = int(manifest["seed"])
    cell_id = str(cell["cell_id"])
    data_rng = np.random.default_rng(_seed(base_seed, cell_id, rep, "data"))
    matrix, covariate = _generate_matrix(cell, data_rng)
    mask_rng = np.random.default_rng(_seed(base_seed, cell_id, rep, "mask"))
    observed = _apply_missingness(
        matrix,
        covariate,
        str(cell["missingness"]),
        float(manifest["missing_fraction"]),
        mask_rng,
    )
    split_rng = np.random.default_rng(_seed(base_seed, cell_id, rep, "holdout"))
    training, holdout = _holdout_split(
        observed, float(manifest["holdout_fraction"]), split_rng
    )
    rank_cap = min(
        max(8, int(cell["true_rank"]) + 5),
        min(matrix.shape) - 1,
    )
    base_options = recommend_config(n=int(cell["n"]), p=int(cell["p"]))
    fit_seed = _seed(base_seed, cell_id, rep, "fit")
    fits = [
        _fit_checkpoint(
            matrix,
            training,
            holdout,
            rank_cap=rank_cap,
            checkpoint=int(checkpoint),
            random_state=fit_seed,
            base_options=base_options,
        )
        for checkpoint in manifest["checkpoints"]
    ]
    fidelity = _assess_fidelity(fits, manifest)
    endpoint_curve = fits[-1]["learning_curve"]
    if not isinstance(endpoint_curve, dict):
        msg = "forced-long fit did not return a learning curve"
        raise TypeError(msg)
    policies = _evaluate_policies(
        endpoint_curve,
        manifest,
        fidelity,
        base_options=base_options,
    )
    payload = {
        "status": "ok",
        "cell": cell,
        "rep": rep,
        "rank_cap": rank_cap,
        "observed_fraction": float(np.mean(observed)),
        "training_fraction": float(np.mean(training)),
        "base_options": _jsonable(base_options),
        "fidelity": fidelity,
        "policies": policies,
    }
    if manifest.get("retain_learning_curve") is True:
        payload["learning_curve"] = _compact_learning_curve(endpoint_curve)
    return payload


def run_shard(
    manifest_path: Path,
    output_dir: Path,
    *,
    shard_index: int,
    retry_errors: bool,
) -> Path:
    """Run or validate one cell/replicate shard.

    Returns:
        Completed shard path.
    """
    manifest = _load_manifest(manifest_path)
    cell, rep = _shard_at_index(manifest, shard_index)
    destination = output_dir / f"shard-{shard_index:05d}.json"
    manifest_sha = _manifest_sha256(manifest_path)
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing.get("manifest_sha256") != manifest_sha:
            msg = f"existing shard manifest differs: {destination}"
            raise ValueError(msg)
        if existing.get("status") == "ok" or not retry_errors:
            print(f"Validated existing shard -> {destination}")
            return destination
    envelope = {
        "manifest_sha256": manifest_sha,
        "shard_index": shard_index,
        "vbpca_release": vbpca_version,
        "vbpca_revision": os.environ.get("VBPCA_REVISION"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    result_schema = manifest.get("result_schema_version")
    if result_schema is not None:
        envelope["result_schema_version"] = int(result_schema)

    try:
        payload = {**envelope, **_run_one(manifest, cell, rep)}
    except Exception as error:
        payload = {
            **envelope,
            "status": "error",
            "cell": cell,
            "rep": rep,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    _atomic_json(destination, payload)
    print(f"Wrote {payload['status']} shard -> {destination}")
    return destination


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(manifest_path: Path, output_dir: Path, output: Path) -> None:
    """Validate complete shards and write aggregate safety outcomes."""
    manifest = _load_manifest(manifest_path)
    expected_sha = _manifest_sha256(manifest_path)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, _shard in enumerate(_shards(manifest)):
        path = output_dir / f"shard-{index:05d}.json"
        record = json.loads(path.read_text())
        if record.get("manifest_sha256") != expected_sha:
            msg = f"shard manifest differs: {path}"
            raise ValueError(msg)
        if record.get("status") == "ok":
            records.append(record)
        else:
            errors.append(record)
    if errors:
        msg = f"cannot summarize with {len(errors)} error shards"
        raise RuntimeError(msg)

    endpoint_stable = [bool(row["fidelity"]["endpoint_stable"]) for row in records]
    earliest = [
        int(row["fidelity"]["earliest_fidelity_checkpoint"])
        for row in records
        if row["fidelity"]["earliest_fidelity_checkpoint"] is not None
    ]
    policy_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for row in record["policies"]["rows"]:
            policy_rows[str(row["policy"])].append(row)
    policy_summary: dict[str, Any] = {}
    for name, rows in policy_rows.items():
        evaluable = [row for row in rows if row["evaluable"]]
        excess = [
            float(row["excess_iterations"])
            for row in evaluable
            if row["excess_iterations"] is not None
        ]
        policy_summary[name] = {
            "n": len(rows),
            "n_evaluable": len(evaluable),
            "premature_rate": _mean([float(row["premature"]) for row in evaluable]),
            "late_or_no_stop_rate": _mean(
                [float(row["late_or_no_stop"]) for row in evaluable]
            ),
            "median_excess_iterations": (float(np.median(excess)) if excess else None),
        }
    payload = {
        "manifest_sha256": expected_sha,
        "n_records": len(records),
        "endpoint_stable_rate": float(np.mean(endpoint_stable)),
        "earliest_fidelity_checkpoint_counts": {
            str(value): earliest.count(value) for value in sorted(set(earliest))
        },
        "policies": policy_summary,
        "by_shape": summaries_by(records, "shape"),
        "by_scenario": summaries_by(records, "scenario"),
        "by_missingness": summaries_by(records, "missingness"),
    }
    _atomic_json(output, payload)
    print(f"Wrote summary -> {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument(
        "--profile", choices=("smoke", "screen"), required=True
    )
    manifest_parser.add_argument("--n-reps", type=int, required=True)
    manifest_parser.add_argument("--seed", type=int, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run-shard")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--shard-index", type=int, required=True)
    run_parser.add_argument("--retry-errors", action="store_true")
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--manifest", type=Path, required=True)
    summary_parser.add_argument("--output-dir", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Run the command-line interface."""
    args = _parser().parse_args()
    if args.command == "manifest":
        write_manifest(
            args.output, profile=args.profile, n_reps=args.n_reps, seed=args.seed
        )
    elif args.command == "run-shard":
        run_shard(
            args.manifest,
            args.output_dir,
            shard_index=args.shard_index,
            retry_errors=args.retry_errors,
        )
    else:
        summarize(args.manifest, args.output_dir, args.output)


if __name__ == "__main__":
    main()
