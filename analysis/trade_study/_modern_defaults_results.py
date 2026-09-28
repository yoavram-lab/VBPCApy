"""Validated result loading and descriptive summaries for defaults studies."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from ._modern_defaults_io import checkpoint_path, validate_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

_MEAN_FIELDS = (
    "rank_mae",
    "exact_rank",
    "null_selected",
    "holdout_rmse",
    "holdout_mae",
    "coverage_95",
    "interval_score",
    "mean_interval_width",
    "subspace_distance",
    "selection_total_iters",
    "candidate_budget_hit_rate",
    "selected_budget_hit",
    "rms_increase_fraction",
    "rms_two_cycle_amplitude",
    "wall_seconds",
)


def load_complete_records(
    manifest_path: Path,
    manifest: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Validate every expected checkpoint and return decorated records."""
    records: list[dict[str, Any]] = []
    regimes = {str(item["name"]): item for item in manifest["regimes"]}
    for candidate in manifest["candidates"]:
        candidate_id = str(candidate["id"])
        for regime_name, regime in regimes.items():
            path = checkpoint_path(output_dir, manifest, candidate_id, regime_name)
            if not path.exists():
                msg = f"missing checkpoint {path}"
                raise FileNotFoundError(msg)
            checkpoint = json.loads(path.read_text())
            validate_checkpoint(
                checkpoint,
                manifest_path,
                manifest,
                candidate_id=candidate_id,
                regime_name=regime_name,
            )
            if not checkpoint["complete"]:
                msg = f"incomplete checkpoint {path}"
                raise ValueError(msg)
            for raw in checkpoint["records"]:
                row = dict(raw)
                row.update({
                    "candidate_id": candidate_id,
                    "regime": regime_name,
                    "missingness": regime["missingness"],
                    "noise_model": regime["noise_model"],
                    "shape": (
                        "wide"
                        if int(regime["p"]) > int(regime["n"])
                        else "tall"
                        if int(regime["n"]) > int(regime["p"])
                        else "square"
                    ),
                })
                records.append(row)
    return records


def _finite_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def descriptive_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return overall and field-standard stratified descriptive summaries."""
    candidates = sorted({str(row["candidate_id"]) for row in rows})
    factors = ("regime", "shape", "missingness", "noise_model")

    def means(selected: list[dict[str, Any]]) -> dict[str, float | None]:
        return {field: _finite_mean(selected, field) for field in _MEAN_FIELDS}

    result: dict[str, Any] = {}
    for candidate_id in candidates:
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        result[candidate_id] = {"overall": means(candidate_rows)}
        for factor in factors:
            levels = sorted({str(row[factor]) for row in candidate_rows})
            result[candidate_id][f"by_{factor}"] = {
                level: means([
                    row for row in candidate_rows if str(row[factor]) == level
                ])
                for level in levels
            }
    return result
