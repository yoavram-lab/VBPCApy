"""Tests for paired inference and defaults-study promotion."""

from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

import pytest

from analysis.trade_study import modern_defaults_study as study_module
from analysis.trade_study._modern_defaults_io import (
    atomic_json,
    checkpoint_path,
    new_checkpoint,
)
from analysis.trade_study._modern_defaults_reduction import (
    paired_bootstrap,
    summarize_screen,
)
from analysis.trade_study._modern_defaults_results import (
    descriptive_summary,
    load_complete_records,
)
from analysis.trade_study.modern_defaults_study import _promote_command


def _row(
    rep: int,
    *,
    true_rank: int,
    rank_mae: float,
    rmse: float,
    coverage: float,
    interval_score: float,
    iterations: int,
    null_selected: bool = False,
) -> dict[str, Any]:
    return {
        "rep": rep,
        "true_rank": true_rank,
        "rank_mae": rank_mae,
        "exact_rank": not bool(rank_mae),
        "null_selected": null_selected,
        "holdout_rmse": rmse,
        "holdout_mae": rmse * 0.8,
        "coverage_95": coverage,
        "interval_score": interval_score,
        "mean_interval_width": 2.0,
        "subspace_distance": rank_mae / 10.0,
        "selection_total_iters": iterations,
        "candidate_budget_hit_rate": 0.0,
        "selected_budget_hit": False,
        "rms_increase_fraction": 0.0,
        "rms_two_cycle_amplitude": 0.0,
        "wall_seconds": 1.0,
    }


def _manifest() -> dict[str, Any]:
    return {
        "design_version": "test",
        "profile": "screen",
        "n_reps": 2,
        "candidates": [
            {"id": "recommended_post_factor"},
            {"id": "recommended_legacy"},
            {"id": "screen_000"},
            {"id": "screen_001"},
        ],
        "regimes": [
            {
                "name": "wide_signal",
                "n": 10,
                "p": 20,
                "true_rank": 2,
                "missingness": "mar",
                "noise_model": "gaussian",
            },
            {
                "name": "square_null",
                "n": 10,
                "p": 10,
                "true_rank": 0,
                "missingness": "mcar",
                "noise_model": "student_t3",
            },
        ],
        "bootstrap": {"n_resamples": 200, "confidence": 0.95, "seed": 91},
        "screen_gates": {
            "holdout_rmse_relative_upper": 0.01,
            "interval_score_relative_upper": 0.02,
            "coverage_difference_lower": -0.02,
            "rank_mae_difference_upper": 0.10,
            "null_selection_rate_difference_upper": 0.05,
        },
    }


def _records(candidate_id: str, regime: dict[str, Any]) -> list[dict[str, Any]]:
    is_null = int(regime["true_rank"]) == 0
    if candidate_id == "screen_000":
        values = {
            "rank_mae": 0.0,
            "rmse": 0.99,
            "coverage": 0.95,
            "interval_score": 0.99,
            "iterations": 80,
        }
    elif candidate_id == "screen_001":
        values = {
            "rank_mae": 1.0,
            "rmse": 1.2,
            "coverage": 0.85,
            "interval_score": 1.3,
            "iterations": 70,
        }
    else:
        values = {
            "rank_mae": 0.2,
            "rmse": 1.0,
            "coverage": 0.95,
            "interval_score": 1.0,
            "iterations": 100,
        }
    return [
        _row(
            rep,
            true_rank=int(regime["true_rank"]),
            null_selected=is_null and candidate_id == "screen_001",
            **values,
        )
        for rep in range(2)
    ]


def _write_complete_study(tmp_path):
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "results"
    atomic_json(manifest_path, manifest)
    for candidate in manifest["candidates"]:
        candidate_id = str(candidate["id"])
        for regime in manifest["regimes"]:
            regime_name = str(regime["name"])
            checkpoint = new_checkpoint(
                manifest_path,
                manifest,
                candidate_id=candidate_id,
                regime_name=regime_name,
                num_cpu=1,
            )
            checkpoint["records"] = _records(candidate_id, regime)
            checkpoint["complete"] = True
            atomic_json(
                checkpoint_path(output_dir, manifest, candidate_id, regime_name),
                checkpoint,
            )
    return manifest_path, manifest, output_dir


def test_paired_bootstrap_uses_pairing_and_is_reproducible() -> None:
    reference = [
        {"regime": "a", "rep": 0, "value": 2.0},
        {"regime": "a", "rep": 1, "value": 4.0},
    ]
    candidate = [
        {"regime": "a", "rep": 0, "value": 1.0},
        {"regime": "a", "rep": 1, "value": 2.0},
    ]
    first = paired_bootstrap(
        candidate,
        reference,
        field="value",
        mode="ratio",
        subset="all",
        n_resamples=100,
        confidence=0.95,
        seed=4,
    )
    second = paired_bootstrap(
        candidate,
        reference,
        field="value",
        mode="ratio",
        subset="all",
        n_resamples=100,
        confidence=0.95,
        seed=4,
    )

    assert first == second
    assert first["estimate"] == pytest.approx(0.5)
    assert first["n_pairs"] == 2
    assert first["n_regimes"] == 1


def test_descriptive_summary_uses_standard_strata(tmp_path) -> None:
    manifest_path, manifest, output_dir = _write_complete_study(tmp_path)
    rows = load_complete_records(manifest_path, manifest, output_dir)
    summary = descriptive_summary(rows)

    good = summary["screen_000"]
    assert good["overall"]["holdout_rmse"] == pytest.approx(0.99)
    assert set(good["by_shape"]) == {"square", "wide"}
    assert set(good["by_missingness"]) == {"mar", "mcar"}
    assert set(good["by_noise_model"]) == {"gaussian", "student_t3"}


def test_screen_summary_applies_gates_and_promotes_pareto_candidate(tmp_path) -> None:
    manifest_path, manifest, output_dir = _write_complete_study(tmp_path)

    summary = summarize_screen(manifest_path, manifest, output_dir)

    assert summary["paired_vs_reference"]["screen_000"]["eligible"] is True
    assert summary["paired_vs_reference"]["screen_001"]["eligible"] is False
    assert summary["promoted_finalists"] == ["screen_000"]
    assert summary["confirmation_candidate_ids"] == [
        "recommended_post_factor",
        "recommended_legacy",
        "screen_000",
    ]
    good_effects = summary["paired_vs_reference"]["screen_000"]["effects"]
    assert good_effects["holdout_rmse_relative"]["estimate"] == pytest.approx(-0.01)
    assert good_effects["iteration_ratio"]["estimate"] == pytest.approx(0.8)


def test_result_loading_rejects_missing_checkpoint(tmp_path) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, manifest)

    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        load_complete_records(manifest_path, manifest, tmp_path / "results")


def test_screen_summary_rejects_non_screen_profile(tmp_path) -> None:
    manifest_path, manifest, output_dir = _write_complete_study(tmp_path)
    manifest["profile"] = "confirm"

    with pytest.raises(ValueError, match="screen-profile"):
        summarize_screen(manifest_path, manifest, output_dir)


def test_promote_command_recomputes_rules_and_writes_confirm_manifest(
    tmp_path, monkeypatch
) -> None:
    manifest_path, manifest, output_dir = _write_complete_study(tmp_path)
    confirm_path = tmp_path / "confirm.json"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(study_module, "load_manifest", lambda path: manifest)

    _promote_command(
        Namespace(
            screen_manifest=manifest_path,
            output_dir=output_dir,
            summary_output=summary_path,
            output=confirm_path,
        )
    )

    confirm = json.loads(confirm_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert confirm["profile"] == "confirm"
    assert [item["id"] for item in confirm["candidates"]] == [
        "recommended_post_factor",
        "recommended_legacy",
        "screen_000",
    ]
    assert summary["promoted_finalists"] == ["screen_000"]
