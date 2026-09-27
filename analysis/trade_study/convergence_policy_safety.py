#!/usr/bin/env python
"""Run held-out production-versus-no-RMS convergence validation (#195)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
from scipy.linalg import subspace_angles

from vbpca_py import SelectionConfig, recommend_config, select_n_components
from vbpca_py import __version__ as vbpca_version

from ._convergence_policy_safety_design import (
    build_manifest,
    condition_options,
    validate_manifest,
)
from .convergence_stability_study import (
    _apply_missingness,
    _generate_matrix,
    _holdout_split,
    _seed,
)

if TYPE_CHECKING:
    from vbpca_py import VBPCA

_Z_975 = 1.959963984540054
_INTERVAL_ALPHA = 0.05


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
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_manifest(output: Path, *, profile: str, n_reps: int, seed: int) -> None:
    """Write one immutable safety-study manifest atomically."""
    manifest = build_manifest(profile, n_reps=n_reps, seed=seed)
    _atomic_json(output, manifest)
    print(
        f"Wrote {output}: {len(manifest['cells'])} cells x {n_reps} replicates; "
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


def _predictive_variance(model: VBPCA) -> np.ndarray:
    variance = model.predictive_variance_
    if variance is not None:
        return np.asarray(variance, dtype=float)
    if model.variance_ is None or model.noise_variance_ is None:
        msg = "selected VBPCA model did not return predictive variance"
        raise RuntimeError(msg)
    return np.asarray(model.variance_, dtype=float) + float(model.noise_variance_)


def _predictive_metrics(
    matrix: np.ndarray,
    holdout: np.ndarray,
    reconstruction: np.ndarray,
    predictive_variance: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(matrix[holdout], dtype=float)
    mean = np.asarray(reconstruction[holdout], dtype=float)
    variance = np.maximum(
        np.asarray(predictive_variance[holdout], dtype=float),
        np.finfo(float).tiny,
    )
    residual = truth - mean
    scale = np.sqrt(variance)
    lower = mean - _Z_975 * scale
    upper = mean + _Z_975 * scale
    interval_score = upper - lower
    interval_score += (2.0 / _INTERVAL_ALPHA) * np.maximum(lower - truth, 0.0)
    interval_score += (2.0 / _INTERVAL_ALPHA) * np.maximum(truth - upper, 0.0)
    return {
        "holdout_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "holdout_mae": float(np.mean(np.abs(residual))),
        "coverage_95": float(np.mean((truth >= lower) & (truth <= upper))),
        "interval_score": float(np.mean(interval_score)),
    }


def _downstream_ranks(
    reconstruction: np.ndarray,
    predictive_variance: np.ndarray,
    *,
    manifest: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    try:
        from pp_eigentest import (
            RankSelectionOptions,
            ReplicateSeedPlan,
            select_rank,
        )
        from pp_eigentest import __version__ as pp_eigentest_version
    except ImportError as error:
        msg = (
            "pp-eigentest is required for convergence-policy safety analysis; "
            "install or expose the pinned pp-eigentest checkout"
        )
        raise RuntimeError(msg) from error

    k_max = min(int(manifest["max_components"]), min(reconstruction.shape) - 1)
    started = time.perf_counter()
    result = select_rank(
        reconstruction,
        predictive_variance,
        options=RankSelectionOptions(
            seed_plan=ReplicateSeedPlan(base_seed=seed),
            backend="numpy",
            k_max=k_max,
            reps=int(manifest["pp_reps"]),
            layer1_method="pa_upper_quantile",
            quantile=float(manifest["pa_quantile"]),
            layer3_method="fixed_sequence",
            alpha=float(manifest["seq_alpha"]),
            store_null=False,
            progress=False,
            inner_threads=1,
        ),
    )
    return {
        "pa_rank": int(result.layer1.rank),
        "seq_rank": int(result.layer3.rank),
        "pp_wall_seconds": float(time.perf_counter() - started),
        "pp_eigentest_version": pp_eigentest_version,
        "backend": str(result.config["backend"]),
        "seed_plan": result.config["seed_plan"],
    }


def _fit_condition(
    matrix: np.ndarray,
    training: np.ndarray,
    xprobe: np.ndarray,
    *,
    manifest: dict[str, Any],
    cell: dict[str, Any],
    condition: str,
    fit_seed: int,
    pp_seed: int,
) -> dict[str, Any]:
    max_components = min(
        int(manifest["max_components"]),
        min(matrix.shape) - 1,
    )
    components = list(range(1, max_components + 1))
    base_options = recommend_config(n=int(cell["n"]), p=int(cell["p"]))
    options = condition_options(base_options, condition)
    started = time.perf_counter()
    selected_k, best_metrics, trace, model = select_n_components(
        matrix,
        mask=training,
        components=components,
        config=SelectionConfig(
            metric="prms",
            compute_explained_variance=True,
            return_best_model=True,
        ),
        xprobe=xprobe,
        random_state=fit_seed,
        **options,
    )
    selection_wall_seconds = time.perf_counter() - started
    if model is None or model.reconstruction_ is None:
        msg = "component selection did not retain the selected VBPCA model"
        raise RuntimeError(msg)
    reconstruction = np.asarray(model.reconstruction_, dtype=float)
    predictive_variance = _predictive_variance(model)
    holdout = np.isfinite(xprobe)
    predictive = _predictive_metrics(
        matrix,
        holdout,
        reconstruction,
        predictive_variance,
    )
    downstream = _downstream_ranks(
        reconstruction,
        predictive_variance,
        manifest=manifest,
        seed=pp_seed,
    )
    iterations = [int(entry["n_iter"]) for entry in trace]
    converged = [bool(entry["converged"]) for entry in trace]
    reasons = [str(entry["convergence_reason"]) for entry in trace]
    return {
        "record": {
            "condition": condition,
            "selected_capacity": int(selected_k),
            "best_prms": float(best_metrics["prms"]),
            **predictive,
            **downstream,
            "selection_wall_seconds": float(selection_wall_seconds),
            "total_candidate_iterations": int(sum(iterations)),
            "max_candidate_iterations": int(max(iterations, default=0)),
            "candidate_convergence_rate": float(np.mean(converged)),
            "candidate_budget_hit_rate": float(
                np.mean([not item for item in converged])
            ),
            "selected_n_iter": int(model.n_iter_ or 0),
            "selected_converged": bool(model.converged_),
            "selected_convergence_reason": model.convergence_reason_ or "maxiters",
            "candidate_convergence_reasons": {
                reason: reasons.count(reason) for reason in sorted(set(reasons))
            },
        },
        "posterior": {
            "reconstruction": reconstruction,
            "predictive_variance": predictive_variance,
            "loadings": np.asarray(model.components_, dtype=float),
            "noise_variance": float(model.noise_variance_),
        },
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


def _posterior_drift(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    metrics = {
        "reconstruction_relative_frobenius": _relative_frobenius(
            candidate["reconstruction"], reference["reconstruction"]
        ),
        "predictive_variance_relative_frobenius": _relative_frobenius(
            candidate["predictive_variance"], reference["predictive_variance"]
        ),
        "noise_variance_relative_change": abs(
            float(candidate["noise_variance"]) - float(reference["noise_variance"])
        )
        / max(abs(float(reference["noise_variance"])), np.finfo(float).eps),
        "loading_subspace_max_angle_radians": _subspace_angle(
            candidate["loadings"], reference["loadings"]
        ),
    }
    margins = manifest["posterior_drift_reference_margins"]
    return {
        **metrics,
        "within_reference_margins": all(
            float(metrics[name]) <= float(margin) for name, margin in margins.items()
        ),
    }


def _run_one(
    manifest: dict[str, Any], cell: dict[str, Any], rep: int
) -> dict[str, Any]:
    base_seed = int(manifest["seed"])
    cell_id = str(cell["cell_id"])
    matrix, covariate = _generate_matrix(
        cell,
        np.random.default_rng(_seed(base_seed, cell_id, rep, "data")),
    )
    observed = _apply_missingness(
        matrix,
        covariate,
        str(cell["missingness"]),
        float(manifest["missing_fraction"]),
        np.random.default_rng(_seed(base_seed, cell_id, rep, "mask")),
    )
    training, holdout = _holdout_split(
        observed,
        float(manifest["holdout_fraction"]),
        np.random.default_rng(_seed(base_seed, cell_id, rep, "holdout")),
    )
    xprobe = np.full(matrix.shape, np.nan, dtype=float)
    xprobe[holdout] = matrix[holdout]
    fit_seed = _seed(base_seed, cell_id, rep, "fit")
    pp_seed = _seed(base_seed, cell_id, rep, "pp")
    fits = {
        condition: _fit_condition(
            matrix,
            training,
            xprobe,
            manifest=manifest,
            cell=cell,
            condition=str(condition),
            fit_seed=fit_seed,
            pp_seed=pp_seed,
        )
        for condition in manifest["conditions"]
    }
    reference_name = str(manifest["reference_condition"])
    candidate_name = str(manifest["candidate_condition"])
    return {
        "status": "ok",
        "cell": cell,
        "rep": rep,
        "observed_fraction": float(np.mean(observed)),
        "training_fraction": float(np.mean(training)),
        "holdout_count": int(np.sum(holdout)),
        "conditions": {name: fit["record"] for name, fit in fits.items()},
        "posterior_drift": _posterior_drift(
            fits[candidate_name]["posterior"],
            fits[reference_name]["posterior"],
            manifest,
        ),
    }


def _provenance() -> dict[str, Any]:
    return {
        "vbpca_release": vbpca_version,
        "vbpca_revision": os.environ.get("VBPCA_REVISION"),
        "pp_eigentest_revision": os.environ.get("PP_EIGENTEST_REVISION"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def run_shard(
    manifest_path: Path,
    output_dir: Path,
    *,
    shard_index: int,
    retry_errors: bool,
) -> Path:
    """Run or validate one paired cell/replicate checkpoint."""
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
        "result_schema_version": int(manifest["result_schema_version"]),
        "shard_index": shard_index,
        **_provenance(),
    }
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


def _condition_means(
    records: list[dict[str, Any]], condition: str
) -> dict[str, float | int]:
    rows = [record["conditions"][condition] for record in records]
    true_rank = np.asarray([int(record["cell"]["true_rank"]) for record in records])
    output: dict[str, float | int] = {"n": len(rows)}
    scalar_names = (
        "selected_capacity",
        "best_prms",
        "holdout_rmse",
        "holdout_mae",
        "coverage_95",
        "interval_score",
        "pa_rank",
        "seq_rank",
        "selection_wall_seconds",
        "pp_wall_seconds",
        "total_candidate_iterations",
        "candidate_budget_hit_rate",
        "selected_n_iter",
        "selected_converged",
    )
    for name in scalar_names:
        output[name] = float(np.mean([float(row[name]) for row in rows]))
    for method in ("pa_rank", "seq_rank"):
        values = np.asarray([int(row[method]) for row in rows])
        output[f"{method}_exact_recovery"] = float(np.mean(values == true_rank))
        output[f"{method}_mae"] = float(np.mean(np.abs(values - true_rank)))
    null = true_rank == 0
    output["pa_null_positive_rate"] = float(
        np.mean(np.asarray([int(row["pa_rank"]) for row in rows])[null] > 0)
    )
    output["seq_null_positive_rate"] = float(
        np.mean(np.asarray([int(row["seq_rank"]) for row in rows])[null] > 0)
    )
    return output


def _cell_interval(
    values: np.ndarray,
    cell_ids: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
) -> list[float]:
    unique = np.unique(cell_ids)
    cell_means = np.asarray([np.mean(values[cell_ids == cell]) for cell in unique])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(unique), size=(n_resamples, len(unique)))
    means = np.mean(cell_means[indices], axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _contrast(
    records: list[dict[str, Any]],
    *,
    candidate: str,
    reference: str,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    cell_ids = np.asarray([str(record["cell"]["cell_id"]) for record in records])
    true_rank = np.asarray([int(record["cell"]["true_rank"]) for record in records])
    candidate_rows = [record["conditions"][candidate] for record in records]
    reference_rows = [record["conditions"][reference] for record in records]

    def interval(values: np.ndarray, offset: int) -> list[float]:
        return _cell_interval(
            values,
            cell_ids,
            seed=seed + offset * 10_003,
            n_resamples=n_resamples,
        )

    output: dict[str, Any] = {}
    for offset, metric in enumerate(("holdout_rmse", "interval_score"), start=1):
        cand = np.asarray([float(row[metric]) for row in candidate_rows])
        ref = np.asarray([float(row[metric]) for row in reference_rows])
        values = (cand - ref) / np.maximum(np.abs(ref), np.finfo(float).eps)
        output[f"relative_{metric}"] = {
            "estimate": float(np.mean(values)),
            "interval": interval(values, offset),
        }
    coverage = np.asarray([
        float(cand["coverage_95"]) - float(ref["coverage_95"])
        for cand, ref in zip(candidate_rows, reference_rows, strict=True)
    ])
    output["coverage_difference"] = {
        "estimate": float(np.mean(coverage)),
        "interval": interval(coverage, 3),
    }
    for method_index, method in enumerate(
        ("selected_capacity", "pa_rank", "seq_rank"), start=4
    ):
        cand = np.asarray([int(row[method]) for row in candidate_rows])
        ref = np.asarray([int(row[method]) for row in reference_rows])
        output[method] = {
            "agreement": float(np.mean(cand == ref)),
            "agreement_interval": interval(
                (cand == ref).astype(float), method_index * 2 - 1
            ),
            "mean_rank_shift": float(np.mean(cand - ref)),
            "mean_rank_shift_interval": interval(
                (cand - ref).astype(float), method_index * 2 - 3
            ),
            "mean_absolute_rank_difference": {
                "estimate": float(np.mean(np.abs(cand - ref))),
                "interval": interval(
                    np.abs(cand - ref).astype(float), method_index * 2 - 2
                ),
            },
        }
        if method != "selected_capacity":
            exact = (cand == true_rank).astype(float) - (ref == true_rank).astype(float)
            mae_gain = np.abs(ref - true_rank) - np.abs(cand - true_rank)
            output[method].update({
                "exact_recovery_difference": {
                    "estimate": float(np.mean(exact)),
                    "interval": interval(exact, method_index * 2),
                },
                "mae_gain": {
                    "estimate": float(np.mean(mae_gain)),
                    "interval": interval(mae_gain, method_index * 2 + 1),
                },
                "rescues": int(np.sum((cand == true_rank) & (ref != true_rank))),
                "spoils": int(np.sum((cand != true_rank) & (ref == true_rank))),
            })
    null = true_rank == 0
    null_cell_ids = cell_ids[null]
    for index, method in enumerate(("pa_rank", "seq_rank"), start=20):
        cand = np.asarray([int(row[method]) for row in candidate_rows])[null]
        ref = np.asarray([int(row[method]) for row in reference_rows])[null]
        values = (cand > 0).astype(float) - (ref > 0).astype(float)
        output[f"{method}_null_positive_rate_difference"] = {
            "estimate": float(np.mean(values)),
            "interval": _cell_interval(
                values,
                null_cell_ids,
                seed=seed + index * 10_003,
                n_resamples=n_resamples,
            ),
        }
    return output


def _decision_gate(
    contrast: dict[str, Any], margins: dict[str, float]
) -> dict[str, Any]:
    checks = {
        "holdout_rmse_noninferior": contrast["relative_holdout_rmse"]["interval"][1]
        <= margins["relative_holdout_rmse"],
        "interval_score_noninferior": contrast["relative_interval_score"]["interval"][1]
        <= margins["relative_interval_score"],
        "coverage_noninferior": contrast["coverage_difference"]["interval"][0]
        >= margins["coverage_difference"],
    }
    checks["selected_capacity_not_inflated"] = (
        contrast["selected_capacity"]["mean_rank_shift_interval"][1]
        <= margins["capacity_mean_increase"]
    )
    for method in ("pa_rank", "seq_rank"):
        checks[f"{method}_exact_noninferior"] = (
            contrast[method]["exact_recovery_difference"]["interval"][0]
            >= margins["exact_recovery_difference"]
        )
        checks[f"{method}_mae_noninferior"] = (
            contrast[method]["mae_gain"]["interval"][0] >= margins["mae_gain"]
        )
    for method in ("pa_rank", "seq_rank"):
        checks[f"{method}_null_rate_noninferior"] = (
            contrast[f"{method}_null_positive_rate_difference"]["interval"][1]
            <= margins["null_positive_rate_difference"]
        )
    return {"checks": checks, "passes_all": all(checks.values())}


def summarize(
    manifest_path: Path,
    output_dir: Path,
    output: Path,
    *,
    n_resamples: int,
) -> dict[str, Any]:
    """Validate every paired shard and write the preregistered summary."""
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
    reference = str(manifest["reference_condition"])
    candidate = str(manifest["candidate_condition"])
    contrast = _contrast(
        records,
        candidate=candidate,
        reference=reference,
        n_resamples=n_resamples,
        seed=int(manifest["seed"]),
    )
    drift_names = tuple(manifest["posterior_drift_reference_margins"])
    posterior_drift = {
        name: {
            "mean": float(
                np.mean([record["posterior_drift"][name] for record in records])
            ),
            "median": float(
                np.median([record["posterior_drift"][name] for record in records])
            ),
            "q95": float(
                np.quantile(
                    [record["posterior_drift"][name] for record in records], 0.95
                )
            ),
        }
        for name in drift_names
    }
    posterior_drift["within_reference_margins_rate"] = float(
        np.mean([
            bool(record["posterior_drift"]["within_reference_margins"])
            for record in records
        ])
    )
    payload = {
        "manifest_sha256": expected_sha,
        "design_version": manifest["design_version"],
        "profile": manifest["profile"],
        "n_records": len(records),
        "n_resamples": n_resamples,
        "reference_condition": reference,
        "candidate_condition": candidate,
        "by_condition": {
            condition: _condition_means(records, condition)
            for condition in manifest["conditions"]
        },
        "paired_candidate_vs_reference": contrast,
        "posterior_drift_candidate_vs_reference": posterior_drift,
        "decision_gate": _decision_gate(contrast, manifest["noninferiority_margins"]),
    }
    _atomic_json(output, payload)
    print(f"Wrote safety summary -> {output}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument(
        "--profile", choices=("smoke", "confirm"), required=True
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
    summary_parser.add_argument("--n-resamples", type=int, default=5_000)
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
        summarize(
            args.manifest,
            args.output_dir,
            args.output,
            n_resamples=args.n_resamples,
        )


if __name__ == "__main__":
    main()
