"""Paired inference and preregistered promotion for the modern-defaults study."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from ._modern_defaults_io import manifest_sha256
from ._modern_defaults_results import descriptive_summary, load_complete_records
from ._modern_defaults_spec import MANDATORY_CANDIDATES, MAX_FINALISTS

REFERENCE_CANDIDATE = "recommended_post_factor"
_EFFECTS: tuple[tuple[str, str, str, str], ...] = (
    ("holdout_rmse_relative", "holdout_rmse", "relative", "all"),
    ("interval_score_relative", "interval_score", "relative", "all"),
    ("coverage_difference", "coverage_95", "difference", "all"),
    ("rank_mae_difference", "rank_mae", "difference", "all"),
    ("null_selection_rate_difference", "null_selected", "difference", "null"),
    ("iteration_ratio", "selection_total_iters", "ratio", "all"),
)

if TYPE_CHECKING:
    from pathlib import Path


def _paired_rows(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    field: str,
    subset: str,
) -> dict[str, list[tuple[float, float]]]:
    reference = {(str(row["regime"]), int(row["rep"])): row for row in reference_rows}
    paired: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in candidate_rows:
        if subset == "null" and int(row["true_rank"]) != 0:
            continue
        key = (str(row["regime"]), int(row["rep"]))
        if key not in reference:
            msg = f"reference is missing paired record {key}"
            raise ValueError(msg)
        candidate_value = row.get(field)
        reference_value = reference[key].get(field)
        if candidate_value is None or reference_value is None:
            msg = f"paired field {field!r} is missing at {key}"
            raise ValueError(msg)
        paired[key[0]].append((float(candidate_value), float(reference_value)))
    if not paired:
        msg = f"no paired rows available for {field!r} subset {subset!r}"
        raise ValueError(msg)
    return dict(paired)


def _effect(candidate: np.ndarray, reference: np.ndarray, mode: str) -> float:
    candidate_mean = float(np.mean(candidate))
    reference_mean = float(np.mean(reference))
    if mode == "difference":
        return candidate_mean - reference_mean
    if np.isclose(reference_mean, 0.0):
        msg = f"cannot compute {mode} effect relative to a zero reference mean"
        raise ValueError(msg)
    if mode == "relative":
        return candidate_mean / reference_mean - 1.0
    if mode == "ratio":
        return candidate_mean / reference_mean
    msg = f"unknown effect mode {mode!r}"
    raise ValueError(msg)


def paired_bootstrap(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    field: str,
    mode: str,
    subset: str,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap paired replicates independently within each regime."""
    paired = _paired_rows(candidate_rows, reference_rows, field, subset)
    candidate = np.asarray([item[0] for values in paired.values() for item in values])
    reference = np.asarray([item[1] for values in paired.values() for item in values])
    estimate = _effect(candidate, reference, mode)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for draw in range(n_resamples):
        candidate_draw: list[float] = []
        reference_draw: list[float] = []
        for values in paired.values():
            indices = rng.integers(0, len(values), size=len(values))
            candidate_draw.extend(values[index][0] for index in indices)
            reference_draw.extend(values[index][1] for index in indices)
        draws[draw] = _effect(
            np.asarray(candidate_draw),
            np.asarray(reference_draw),
            mode,
        )
    alpha = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "n_pairs": int(candidate.size),
        "n_regimes": len(paired),
    }


def _gate_results(
    effects: dict[str, dict[str, Any]], gates: dict[str, float]
) -> dict[str, Any]:
    checks = {
        "holdout_rmse_relative_upper": effects["holdout_rmse_relative"]["upper"]
        <= gates["holdout_rmse_relative_upper"],
        "interval_score_relative_upper": effects["interval_score_relative"]["upper"]
        <= gates["interval_score_relative_upper"],
        "coverage_difference_lower": effects["coverage_difference"]["lower"]
        >= gates["coverage_difference_lower"],
        "rank_mae_difference_upper": effects["rank_mae_difference"]["upper"]
        <= gates["rank_mae_difference_upper"],
        "null_selection_rate_difference_upper": effects[
            "null_selection_rate_difference"
        ]["upper"]
        <= gates["null_selection_rate_difference_upper"],
    }
    return {"checks": checks, "eligible": all(checks.values())}


def _pareto_front(
    candidates: list[str],
    means: dict[str, dict[str, Any]],
) -> list[str]:
    objectives = ("rank_mae", "holdout_rmse", "selection_total_iters")

    def dominates(left: str, right: str) -> bool:
        left_values = [float(means[left]["overall"][field]) for field in objectives]
        right_values = [float(means[right]["overall"][field]) for field in objectives]
        return all(
            a <= b for a, b in zip(left_values, right_values, strict=True)
        ) and any(a < b for a, b in zip(left_values, right_values, strict=True))

    return [
        candidate
        for candidate in candidates
        if not any(
            dominates(other, candidate) for other in candidates if other != candidate
        )
    ]


def _paired_comparisons(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    gates: dict[str, float],
) -> dict[str, Any]:
    reference_rows = [row for row in rows if row["candidate_id"] == REFERENCE_CANDIDATE]
    bootstrap = manifest["bootstrap"]
    comparisons: dict[str, Any] = {}
    candidate_ids = [str(item["id"]) for item in manifest["candidates"]]
    for candidate_index, candidate_id in enumerate(candidate_ids):
        if candidate_id == REFERENCE_CANDIDATE:
            continue
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        effects = {
            name: paired_bootstrap(
                candidate_rows,
                reference_rows,
                field=field,
                mode=mode,
                subset=subset,
                n_resamples=int(bootstrap["n_resamples"]),
                confidence=float(bootstrap["confidence"]),
                seed=int(bootstrap["seed"]) + candidate_index * 1009 + effect_index,
            )
            for effect_index, (name, field, mode, subset) in enumerate(_EFFECTS)
        }
        comparisons[candidate_id] = {
            "effects": effects,
            **_gate_results(effects, gates),
        }
    return comparisons


def summarize_screen(
    manifest_path: Path,
    manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Apply the frozen paired gates and Pareto promotion rule."""
    if manifest["profile"] != "screen":
        msg = "screen promotion requires a screen-profile manifest"
        raise ValueError(msg)
    rows = load_complete_records(manifest_path, manifest, output_dir)
    means = descriptive_summary(rows)
    comparisons = _paired_comparisons(rows, manifest, manifest["screen_gates"])

    optional = [
        candidate_id
        for candidate_id, comparison in comparisons.items()
        if comparison["eligible"] and candidate_id not in MANDATORY_CANDIDATES
    ]
    pareto = _pareto_front(optional, means)
    ordered = sorted(
        pareto,
        key=lambda candidate_id: (
            comparisons[candidate_id]["effects"]["rank_mae_difference"]["estimate"],
            comparisons[candidate_id]["effects"]["holdout_rmse_relative"]["estimate"],
            comparisons[candidate_id]["effects"]["iteration_ratio"]["estimate"],
            candidate_id,
        ),
    )
    finalists = ordered[:MAX_FINALISTS]
    return {
        "manifest_sha256": manifest_sha256(manifest_path),
        "design_version": manifest["design_version"],
        "profile": manifest["profile"],
        "n_reps": manifest["n_reps"],
        "reference_candidate": REFERENCE_CANDIDATE,
        "descriptive": means,
        "paired_vs_reference": comparisons,
        "eligible_optional_candidates": optional,
        "pareto_optional_candidates": pareto,
        "promoted_finalists": finalists,
        "confirmation_candidate_ids": [*MANDATORY_CANDIDATES, *finalists],
    }


def summarize_confirmation(
    manifest_path: Path,
    manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Apply confirmation safety gates and material-improvement rules."""
    if manifest["profile"] != "confirm":
        msg = "confirmation promotion requires a confirm-profile manifest"
        raise ValueError(msg)
    rows = load_complete_records(manifest_path, manifest, output_dir)
    comparisons = _paired_comparisons(rows, manifest, manifest["confirm_gates"])
    gates = manifest["confirm_gates"]
    for comparison in comparisons.values():
        effects = comparison["effects"]
        material_checks = {
            "rank_mae_reduction": effects["rank_mae_difference"]["estimate"]
            <= -float(gates["material_rank_mae_improvement"]),
            "iteration_reduction": effects["iteration_ratio"]["estimate"]
            <= 1.0 - float(gates["material_iteration_reduction"]),
        }
        comparison["material_checks"] = material_checks
        comparison["material_improvement"] = any(material_checks.values())
        comparison["adopt"] = bool(
            comparison["eligible"] and comparison["material_improvement"]
        )

    adopted = [
        candidate_id
        for candidate_id, comparison in comparisons.items()
        if candidate_id not in MANDATORY_CANDIDATES and comparison["adopt"]
    ]
    return {
        "manifest_sha256": manifest_sha256(manifest_path),
        "design_version": manifest["design_version"],
        "profile": manifest["profile"],
        "n_reps": manifest["n_reps"],
        "reference_candidate": REFERENCE_CANDIDATE,
        "descriptive": descriptive_summary(rows),
        "paired_vs_reference": comparisons,
        "adopted_optional_candidates": adopted,
        "scale_candidate_ids": [*MANDATORY_CANDIDATES, *adopted],
    }
