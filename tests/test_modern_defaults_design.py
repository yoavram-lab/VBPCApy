"""Tests for the frozen modern-defaults study design."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from analysis.trade_study._modern_defaults_design import (
    all_screen_candidates,
    build_manifest,
    registered_screen_manifest,
    validate_manifest,
)
from analysis.trade_study._modern_defaults_spec import (
    MANDATORY_CANDIDATES,
    MAX_FINALISTS,
    REGISTERED_CONFIRM_REPS,
    REGISTERED_CONFIRM_SEED,
    REGISTERED_SCALE_REPS,
    REGISTERED_SCALE_SEED,
)

_REGISTERED_SCREEN_HASH = (
    "95c588eea6c1de91df3f537f667f46178499d14b9b1020e88e9d41e1e08ae6ac"
)


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_registered_screen_is_frozen_and_complete() -> None:
    manifest = registered_screen_manifest()
    validate_manifest(manifest)

    assert len(manifest["candidates"]) == 30
    assert len(manifest["regimes"]) == 14
    assert manifest["n_reps"] == 3
    assert manifest["wall_time_is_quality_objective"] is False
    assert manifest["selection"]["metric"] == "prms"
    assert manifest["selection"]["include_rank_zero"] is True
    assert _canonical_hash(manifest) == _REGISTERED_SCREEN_HASH


def test_screen_covers_registered_regimes_and_misspecification() -> None:
    regimes = registered_screen_manifest()["regimes"]
    missingness = {item["missingness"] for item in regimes}
    noise_models = {item["noise_model"] for item in regimes}
    shapes = {
        (
            "wide"
            if item["p"] > item["n"]
            else "tall"
            if item["n"] > item["p"]
            else "square"
        )
        for item in regimes
    }

    assert missingness == {"complete", "mcar", "mar", "mnar_censored", "block"}
    assert noise_models == {"gaussian", "student_t3"}
    assert shapes == {"wide", "square", "tall"}
    assert any(item["true_rank"] == 0 for item in regimes)
    assert any(item["name"] == "genomics_surrogate_mar" for item in regimes)


def test_candidate_space_contains_required_controls_and_valid_ranges() -> None:
    candidates = all_screen_candidates()
    ids = [item["id"] for item in candidates]
    sobol = [item for item in candidates if str(item["id"]).startswith("screen_")]

    assert len(ids) == len(set(ids)) == 30
    assert set(MANDATORY_CANDIDATES).issubset(ids)
    assert {item["bias_update_order"] for item in sobol} == {
        "legacy",
        "post_factor",
    }
    assert {item["criterion_policy"] for item in sobol} == {
        "production",
        "no_rms",
        "angle_cost",
    }
    assert all(0.03 <= item["xprobe_fraction"] <= 0.20 for item in sobol)
    assert all(0.35 <= item["hp_va_scale"] <= 2.0 for item in sobol)
    assert all(item["compat_mode"] == "modern" for item in candidates)


def test_manifest_returns_independent_candidate_copies() -> None:
    first = registered_screen_manifest()
    first["candidates"][0]["id"] = "changed"
    second = registered_screen_manifest()

    assert second["candidates"][0]["id"] == MANDATORY_CANDIDATES[0]


def test_confirm_and_scale_require_controls_and_limit_finalists() -> None:
    finalists = ("screen_000", "screen_001")
    ids = (*MANDATORY_CANDIDATES, *finalists)
    confirm = build_manifest(
        "confirm",
        n_reps=REGISTERED_CONFIRM_REPS,
        seed=REGISTERED_CONFIRM_SEED,
        candidate_ids=ids,
    )
    scale = build_manifest(
        "scale",
        n_reps=REGISTERED_SCALE_REPS,
        seed=REGISTERED_SCALE_SEED,
        candidate_ids=ids,
    )

    validate_manifest(confirm)
    validate_manifest(scale)
    assert len(confirm["regimes"]) == 9
    assert {(item["n"], item["p"]) for item in scale["regimes"]} == {(2504, 5846)}

    with pytest.raises(ValueError, match="must include"):
        build_manifest(
            "confirm",
            n_reps=1,
            seed=1,
            candidate_ids=("screen_000",),
        )
    with pytest.raises(ValueError, match=f"at most {MAX_FINALISTS}"):
        build_manifest(
            "confirm",
            n_reps=1,
            seed=1,
            candidate_ids=(
                *MANDATORY_CANDIDATES,
                "screen_000",
                "screen_001",
                "screen_002",
                "screen_003",
                "screen_004",
            ),
        )


@pytest.mark.parametrize(
    ("candidate_ids", "message"),
    [
        (("recommended_post_factor", "recommended_post_factor"), "duplicates"),
        ((*MANDATORY_CANDIDATES, "not_registered"), "unknown candidate"),
    ],
)
def test_manifest_rejects_invalid_candidate_ids(
    candidate_ids: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_manifest(
            "scale",
            n_reps=1,
            seed=1,
            candidate_ids=candidate_ids,
        )


def test_manifest_validation_rejects_tampering() -> None:
    manifest = registered_screen_manifest()

    changed_gate = copy.deepcopy(manifest)
    changed_gate["screen_gates"]["rank_mae_difference_upper"] = 99.0
    with pytest.raises(ValueError, match="fields differ"):
        validate_manifest(changed_gate)

    changed_candidate = copy.deepcopy(manifest)
    changed_candidate["candidates"][0]["bias_update_order"] = "legacy"
    with pytest.raises(ValueError, match="fields differ"):
        validate_manifest(changed_candidate)

    bad_version = copy.deepcopy(manifest)
    bad_version["manifest_version"] = "unknown"
    with pytest.raises(ValueError, match="unsupported manifest_version"):
        validate_manifest(bad_version)
