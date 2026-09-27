"""Tests for the posterior-stability convergence study."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis.trade_study import convergence_stability_study as study
from analysis.trade_study._convergence_detector_design import (
    DESIGN_VERSION,
    FIDELITY_MARGINS,
    LEGACY_DESIGN_VERSION,
    LEGACY_MANIFEST_VERSION,
    MANIFEST_VERSION,
    RESULT_SCHEMA_VERSION,
    build_manifest,
    policy_from_json,
    validate_manifest,
)
from analysis.trade_study.convergence_detector import DetectorPolicy, replay_policy


def test_registered_profiles_freeze_expected_cells_and_policy_grid() -> None:
    smoke = build_manifest("smoke", n_reps=2, seed=4)
    screen = build_manifest("screen", n_reps=3, seed=5)

    assert len(smoke["cells"]) == 8
    assert len(screen["cells"]) == 75
    assert len(screen["policies"]) == 208
    policies = [policy_from_json(payload) for payload in screen["policies"]]
    assert len({policy.name for policy in policies}) == 208
    assert {policy.warmup for policy in policies} == {0, 50, 100, 200}
    assert screen["fidelity_margins"] == FIDELITY_MARGINS
    assert screen["manifest_version"] == MANIFEST_VERSION
    assert screen["design_version"] == DESIGN_VERSION
    assert screen["result_schema_version"] == RESULT_SCHEMA_VERSION
    assert screen["include_production_policies"] is True
    assert screen["retain_learning_curve"] is True
    validate_manifest(screen)


def test_legacy_v1_manifest_remains_readable() -> None:
    legacy = dict(build_manifest("smoke", n_reps=1, seed=4))
    legacy["manifest_version"] = LEGACY_MANIFEST_VERSION
    legacy["design_version"] = LEGACY_DESIGN_VERSION
    legacy.pop("result_schema_version")
    legacy.pop("include_production_policies")
    legacy.pop("retain_learning_curve")

    validate_manifest(legacy)


def test_shards_are_cell_major_and_replicate_minor() -> None:
    manifest = build_manifest("smoke", n_reps=2, seed=4)

    first_cell, first_rep = study._shard_at_index(manifest, 0)
    second_cell, second_rep = study._shard_at_index(manifest, 1)
    third_cell, third_rep = study._shard_at_index(manifest, 2)

    assert first_cell["cell_id"] == second_cell["cell_id"]
    assert (first_rep, second_rep) == (0, 1)
    assert third_cell["cell_id"] != first_cell["cell_id"]
    assert third_rep == 0


def test_missingness_mechanisms_preserve_row_and_column_support() -> None:
    matrix = np.random.default_rng(2).normal(size=(200, 160))
    covariate = np.random.default_rng(3).normal(size=160)
    for index, mechanism in enumerate(("mcar", "mar", "mnar", "block")):
        observed = study._apply_missingness(
            matrix,
            covariate,
            mechanism,
            0.30,
            np.random.default_rng(10 + index),
        )

        assert np.all(np.any(observed, axis=0))
        assert np.all(np.any(observed, axis=1))
        assert abs(float(np.mean(~observed)) - 0.30) < 0.04


def _fake_fit(checkpoint: int, value: float) -> dict[str, object]:
    return {
        "checkpoint": checkpoint,
        "n_iter": checkpoint,
        "active_components": 2,
        "noise_variance": 1.0 + value,
        "holdout_rmse": 1.0 + value,
        "wall_seconds": 1.0,
        "reconstruction": np.ones((4, 5)) * (1.0 + value),
        "predictive_variance": np.ones((4, 5)) * (2.0 + value),
        "loadings": np.eye(4, 2),
        "learning_curve": {},
    }


def test_fidelity_requires_a_stable_tail_and_returns_earliest_suffix() -> None:
    manifest = build_manifest("smoke", n_reps=1, seed=4)
    fits = [_fake_fit(5, 0.2), _fake_fit(10, 0.001), _fake_fit(20, 0.0)]

    result = study._assess_fidelity(fits, manifest)

    assert result["endpoint_stable"] is True
    assert result["earliest_fidelity_checkpoint"] == 10
    assert result["rows"][0]["acceptable"] is False
    assert result["rows"][1]["acceptable"] is True


def test_unstable_tail_is_not_evaluable() -> None:
    manifest = build_manifest("smoke", n_reps=1, seed=4)
    fits = [_fake_fit(5, 0.2), _fake_fit(10, 0.1), _fake_fit(20, 0.0)]

    result = study._assess_fidelity(fits, manifest)

    assert result["endpoint_stable"] is False
    assert result["earliest_fidelity_checkpoint"] is None


def _production_options() -> dict[str, object]:
    return {
        "convergence_criteria": dict.fromkeys(study.ALL_CRITERIA_FALSE, True),
        "minangle": 1e-4,
        "rmsstop": [10, 1e-6, 1e-3],
        "cfstop_rel": 1e-5,
        "patience": 2,
        "niter_broadprior": 0,
    }


def test_compact_learning_curve_round_trips_through_replay() -> None:
    compact = study._compact_learning_curve({
        "rms": [1.0, 0.9],
        "prms": [np.nan, 0.8],
        "cost": [np.nan, 5.0],
        "angle": [np.nan, 0.01],
    })
    policy = DetectorPolicy(
        name="angle",
        enabled=("angle",),
        minangle=0.05,
    )

    result = replay_policy(compact, policy)

    assert compact["schema_version"] == 1
    assert compact["prms"][0] is None
    assert result.stop_iteration == 1


def test_v2_evaluator_includes_exact_production_policy_and_ablations() -> None:
    manifest = build_manifest("smoke", n_reps=1, seed=4)
    base_options = _production_options()
    curve = {
        "rms": [1.0, 0.9, 0.89],
        "prms": [np.nan, 0.95, 0.94],
        "cost": [np.nan, 100.0, 99.999],
        "angle": [np.nan, 0.2, 0.01],
    }
    fidelity = {
        "endpoint_stable": True,
        "earliest_fidelity_checkpoint": 1,
    }

    result = study._evaluate_policies(
        curve,
        manifest,
        fidelity,
        base_options=base_options,
    )
    names = {row["policy"] for row in result["rows"]}

    assert {
        "production",
        "production_without_angle",
        "production_without_rms_plateau",
        "production_without_cost",
        "production_without_slowing_down",
    }.issubset(names)


def test_run_shard_reuses_valid_complete_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    study.write_manifest(manifest_path, profile="smoke", n_reps=1, seed=4)
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
