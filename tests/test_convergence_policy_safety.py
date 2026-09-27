"""Tests for held-out convergence-policy safety validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis.trade_study import convergence_policy_safety as study
from analysis.trade_study._convergence_policy_safety_design import (
    CANDIDATE_CONDITION,
    CONDITIONS,
    REFERENCE_CONDITION,
    build_manifest,
    condition_options,
    validate_manifest,
)


def test_manifest_freezes_paired_screen_and_heldout_seed() -> None:
    smoke = build_manifest("smoke", n_reps=2, seed=11)
    confirm = build_manifest("confirm", n_reps=5, seed=20261015)

    assert len(smoke["cells"]) == 8
    assert len(confirm["cells"]) == 75
    assert confirm["conditions"] == list(CONDITIONS)
    assert confirm["reference_condition"] == REFERENCE_CONDITION
    assert confirm["candidate_condition"] == CANDIDATE_CONDITION
    assert confirm["seed"] == 20261015
    assert confirm["n_reps"] == 5
    validate_manifest(confirm)


def test_manifest_rejects_changed_claim_gate() -> None:
    manifest = build_manifest("smoke", n_reps=1, seed=11)
    manifest["noninferiority_margins"]["relative_holdout_rmse"] = 0.5

    with pytest.raises(ValueError, match="noninferiority margins"):
        validate_manifest(manifest)


def test_condition_options_disable_only_rms_and_do_not_share_state() -> None:
    base = {
        "xprobe_fraction": 0.1,
        "maxiters": 400,
        "convergence_criteria": {
            "angle": True,
            "rms_plateau": True,
            "cost": True,
        },
    }

    production = condition_options(base, REFERENCE_CONDITION)
    candidate = condition_options(base, CANDIDATE_CONDITION)

    assert "xprobe_fraction" not in production
    assert production["convergence_criteria"]["rms_plateau"] is True
    assert candidate["convergence_criteria"]["rms_plateau"] is False
    assert candidate["convergence_criteria"]["angle"] is True
    assert candidate["convergence_criteria"]["cost"] is True
    candidate["convergence_criteria"]["angle"] = False
    assert production["convergence_criteria"]["angle"] is True
    assert base["convergence_criteria"]["rms_plateau"] is True


def test_predictive_metrics_reward_exact_mean_and_valid_variance() -> None:
    matrix = np.arange(12, dtype=float).reshape(3, 4)
    holdout = np.zeros_like(matrix, dtype=bool)
    holdout[0, 0] = True
    holdout[2, 3] = True

    result = study._predictive_metrics(
        matrix,
        holdout,
        matrix.copy(),
        np.ones_like(matrix),
    )

    assert result["holdout_rmse"] == pytest.approx(0.0)
    assert result["holdout_mae"] == pytest.approx(0.0)
    assert result["coverage_95"] == pytest.approx(1.0)
    assert result["interval_score"] > 0.0


def test_posterior_drift_is_zero_for_identical_posteriors() -> None:
    manifest = build_manifest("smoke", n_reps=1, seed=11)
    posterior = {
        "reconstruction": np.ones((3, 4)),
        "predictive_variance": np.ones((3, 4)) * 2.0,
        "loadings": np.eye(3, 2),
        "noise_variance": 1.0,
    }

    result = study._posterior_drift(posterior, posterior, manifest)

    assert result["reconstruction_relative_frobenius"] == pytest.approx(0.0)
    assert result["predictive_variance_relative_frobenius"] == pytest.approx(0.0)
    assert result["noise_variance_relative_change"] == pytest.approx(0.0)
    assert result["loading_subspace_max_angle_radians"] == pytest.approx(0.0)
    assert result["within_reference_margins"] is True


def _fake_condition_record(condition: str, *, shift: int = 0) -> dict[str, object]:
    return {
        "condition": condition,
        "selected_capacity": 3 + shift,
        "best_prms": 1.0,
        "holdout_rmse": 1.0,
        "holdout_mae": 0.8,
        "coverage_95": 0.95,
        "interval_score": 4.0,
        "pa_rank": 3 + shift,
        "seq_rank": 3 + shift,
        "pp_wall_seconds": 0.1,
        "pp_eigentest_version": "test",
        "backend": "numpy",
        "seed_plan": {"base_seed": 1, "scheme": "splitmix64-v1"},
        "selection_wall_seconds": 1.0,
        "total_candidate_iterations": 40,
        "max_candidate_iterations": 10,
        "candidate_convergence_rate": 1.0,
        "candidate_budget_hit_rate": 0.0,
        "selected_n_iter": 10,
        "selected_converged": True,
        "selected_convergence_reason": "angle",
        "candidate_convergence_reasons": {"angle": 8},
    }


def test_run_one_pairs_conditions_on_data_probe_and_random_streams(monkeypatch) -> None:
    manifest = build_manifest("smoke", n_reps=1, seed=11)
    cell = manifest["cells"][0]
    calls: list[dict[str, object]] = []

    def fake_fit_condition(matrix, training, xprobe, **kwargs):
        calls.append({
            "matrix": matrix.copy(),
            "training": training.copy(),
            "xprobe": xprobe.copy(),
            **kwargs,
        })
        posterior = {
            "reconstruction": matrix.copy(),
            "predictive_variance": np.ones_like(matrix),
            "loadings": np.eye(matrix.shape[0], 1),
            "noise_variance": 1.0,
        }
        return {
            "record": _fake_condition_record(kwargs["condition"]),
            "posterior": posterior,
        }

    monkeypatch.setattr(study, "_fit_condition", fake_fit_condition)

    result = study._run_one(manifest, cell, 0)

    assert result["status"] == "ok"
    assert [call["condition"] for call in calls] == list(CONDITIONS)
    assert calls[0]["fit_seed"] == calls[1]["fit_seed"]
    assert calls[0]["pp_seed"] == calls[1]["pp_seed"]
    np.testing.assert_array_equal(calls[0]["matrix"], calls[1]["matrix"])
    np.testing.assert_array_equal(calls[0]["training"], calls[1]["training"])
    np.testing.assert_array_equal(calls[0]["xprobe"], calls[1]["xprobe"])
    assert result["posterior_drift"]["within_reference_margins"] is True


def test_run_shard_reuses_valid_complete_checkpoint(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    study.write_manifest(manifest_path, profile="smoke", n_reps=1, seed=11)
    output_dir = tmp_path / "output"
    calls = 0

    def fake_run_one(manifest, cell, rep):
        nonlocal calls
        calls += 1
        return {"status": "ok", "cell": cell, "rep": rep}

    monkeypatch.setattr(study, "_run_one", fake_run_one)

    first = study.run_shard(
        manifest_path,
        output_dir,
        shard_index=0,
        retry_errors=True,
    )
    second = study.run_shard(
        manifest_path,
        output_dir,
        shard_index=0,
        retry_errors=True,
    )

    assert first == second
    assert calls == 1


def test_summary_reports_paired_gate(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    study.write_manifest(manifest_path, profile="smoke", n_reps=1, seed=11)
    manifest = json.loads(manifest_path.read_text())
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    manifest_sha = study._manifest_sha256(manifest_path)
    for index, (cell, rep) in enumerate(study._shards(manifest)):
        true_rank = int(cell["true_rank"])
        reference = _fake_condition_record(REFERENCE_CONDITION)
        candidate = _fake_condition_record(CANDIDATE_CONDITION)
        reference["selected_capacity"] = true_rank
        reference["pa_rank"] = true_rank
        reference["seq_rank"] = true_rank
        candidate["selected_capacity"] = true_rank
        candidate["pa_rank"] = true_rank
        candidate["seq_rank"] = true_rank
        payload = {
            "manifest_sha256": manifest_sha,
            "status": "ok",
            "cell": cell,
            "rep": rep,
            "conditions": {
                REFERENCE_CONDITION: reference,
                CANDIDATE_CONDITION: candidate,
            },
            "posterior_drift": {
                "reconstruction_relative_frobenius": 0.0,
                "predictive_variance_relative_frobenius": 0.0,
                "noise_variance_relative_change": 0.0,
                "loading_subspace_max_angle_radians": 0.0,
                "within_reference_margins": True,
            },
        }
        (output_dir / f"shard-{index:05d}.json").write_text(json.dumps(payload))

    summary = study.summarize(
        manifest_path,
        output_dir,
        tmp_path / "summary.json",
        n_resamples=100,
    )

    assert summary["n_records"] == 8
    assert summary["decision_gate"]["passes_all"] is True
    assert summary["paired_candidate_vs_reference"]["seq_rank"][
        "agreement"
    ] == pytest.approx(1.0)
    assert summary["posterior_drift_candidate_vs_reference"][
        "within_reference_margins_rate"
    ] == pytest.approx(1.0)
