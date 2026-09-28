"""Tests for modern-defaults manifests and resumable checkpoints."""

from __future__ import annotations

import json

import pytest

from analysis.trade_study import modern_defaults_study as study_module
from analysis.trade_study._modern_defaults_io import (
    atomic_json,
    checkpoint_path,
    load_manifest,
    manifest_sha256,
    new_checkpoint,
    shard_at_index,
    shards,
    validate_checkpoint,
    write_manifest,
)
from analysis.trade_study.modern_defaults_study import run_shard


def _smoke_manifest(tmp_path, *, n_reps: int = 2):
    path = tmp_path / "smoke.json"
    manifest = write_manifest(
        path,
        profile="smoke",
        n_reps=n_reps,
        seed=51,
    )
    return path, manifest


def test_manifest_round_trip_and_stable_shard_order(tmp_path) -> None:
    path, manifest = _smoke_manifest(tmp_path)

    assert load_manifest(path) == manifest
    assert len(shards(manifest)) == 4
    assert shards(manifest) == [
        ("recommended_post_factor", "wide_smoke"),
        ("recommended_post_factor", "null_smoke"),
        ("recommended_legacy", "wide_smoke"),
        ("recommended_legacy", "null_smoke"),
    ]
    assert shard_at_index(manifest, 2) == ("recommended_legacy", "wide_smoke")
    assert len(manifest_sha256(path)) == 64

    with pytest.raises(ValueError, match="shard-index"):
        shard_at_index(manifest, 4)


def test_registered_screen_manifest_has_420_shards(tmp_path) -> None:
    path = tmp_path / "screen.json"
    manifest = write_manifest(
        path,
        profile="screen",
        n_reps=1,
        seed=1,
        registered_screen=True,
    )

    assert manifest["profile"] == "screen"
    assert manifest["n_reps"] == 3
    assert len(shards(manifest)) == 420


def test_atomic_json_rejects_nonstandard_numbers_by_normalizing(tmp_path) -> None:
    output = tmp_path / "strict.json"
    atomic_json(output, {"finite": 2.0, "missing": float("nan")})

    text = output.read_text()
    assert "NaN" not in text
    assert json.loads(text) == {"finite": 2.0, "missing": None}
    assert not list(tmp_path.glob("*.tmp.*"))


def test_checkpoint_validation_rejects_identity_and_completion_errors(tmp_path) -> None:
    manifest_path, manifest = _smoke_manifest(tmp_path)
    checkpoint = new_checkpoint(
        manifest_path,
        manifest,
        candidate_id="recommended_post_factor",
        regime_name="wide_smoke",
        num_cpu=2,
    )
    validate_checkpoint(
        checkpoint,
        manifest_path,
        manifest,
        candidate_id="recommended_post_factor",
        regime_name="wide_smoke",
    )

    checkpoint["candidate_id"] = "recommended_legacy"
    with pytest.raises(ValueError, match="identity differs"):
        validate_checkpoint(
            checkpoint,
            manifest_path,
            manifest,
            candidate_id="recommended_post_factor",
            regime_name="wide_smoke",
        )

    checkpoint["candidate_id"] = "recommended_post_factor"
    checkpoint["complete"] = True
    with pytest.raises(ValueError, match="complete flag"):
        validate_checkpoint(
            checkpoint,
            manifest_path,
            manifest,
            candidate_id="recommended_post_factor",
            regime_name="wide_smoke",
        )


def test_run_shard_checkpoints_each_replicate_and_resumes(
    tmp_path, monkeypatch
) -> None:
    manifest_path, manifest = _smoke_manifest(tmp_path)
    output_dir = tmp_path / "results"
    calls: list[int] = []

    def interrupt_on_second(manifest, candidate, regime, *, rep, num_cpu):
        del manifest, candidate, regime, num_cpu
        calls.append(rep)
        if rep == 1:
            raise RuntimeError("simulated preemption")
        return {"rep": rep, "holdout_rmse": float("nan")}

    monkeypatch.setattr(study_module, "run_trial", interrupt_on_second)
    with pytest.raises(RuntimeError, match="preemption"):
        run_shard(
            manifest_path,
            output_dir,
            candidate_id="recommended_post_factor",
            regime_name="wide_smoke",
            num_cpu=2,
        )

    destination = checkpoint_path(
        output_dir,
        manifest,
        "recommended_post_factor",
        "wide_smoke",
    )
    partial = json.loads(destination.read_text())
    assert partial["complete"] is False
    assert [record["rep"] for record in partial["records"]] == [0]
    assert partial["records"][0]["holdout_rmse"] is None

    resumed_calls: list[int] = []

    def complete_remaining(manifest, candidate, regime, *, rep, num_cpu):
        del manifest, candidate, regime, num_cpu
        resumed_calls.append(rep)
        return {"rep": rep, "holdout_rmse": 1.0}

    monkeypatch.setattr(study_module, "run_trial", complete_remaining)
    returned = run_shard(
        manifest_path,
        output_dir,
        candidate_id="recommended_post_factor",
        regime_name="wide_smoke",
        num_cpu=2,
    )

    complete = json.loads(returned.read_text())
    assert calls == [0, 1]
    assert resumed_calls == [1]
    assert complete["complete"] is True
    assert [record["rep"] for record in complete["records"]] == [0, 1]

    resumed_calls.clear()
    run_shard(
        manifest_path,
        output_dir,
        candidate_id="recommended_post_factor",
        regime_name="wide_smoke",
        num_cpu=2,
    )
    assert resumed_calls == []


def test_checkpoint_rejects_manifest_changed_after_creation(tmp_path) -> None:
    manifest_path, manifest = _smoke_manifest(tmp_path)
    checkpoint = new_checkpoint(
        manifest_path,
        manifest,
        candidate_id="recommended_post_factor",
        regime_name="wide_smoke",
        num_cpu=1,
    )
    manifest_path.write_text(manifest_path.read_text() + " ")

    with pytest.raises(ValueError, match="identity differs"):
        validate_checkpoint(
            checkpoint,
            manifest_path,
            manifest,
            candidate_id="recommended_post_factor",
            regime_name="wide_smoke",
        )
