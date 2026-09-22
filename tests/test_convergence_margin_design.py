"""Tests for the immutable convergence-margin study design."""

from __future__ import annotations

from pathlib import Path
from runpy import run_path

import pytest

MODULE = run_path(
    str(
        Path(__file__).parents[1]
        / "analysis"
        / "trade_study"
        / "_convergence_margin_design.py"
    )
)
CONDITIONS = MODULE["CONDITIONS"]
build_manifest = MODULE["build_manifest"]
condition_config = MODULE["condition_config"]
validate_manifest = MODULE["validate_manifest"]


@pytest.mark.parametrize("profile", ["smoke", "screen", "confirm"])
def test_profiles_route_only_to_affected_buckets(profile: str) -> None:
    manifest = build_manifest(profile, n_reps=2, seed=100)

    assert {regime["bucket"] for regime in manifest["regimes"]} == {
        "wide_moderate",
        "tall_moderate",
        "tall_extreme",
    }
    assert [regime["seed"] for regime in manifest["regimes"]] == list(
        range(100, 100 + len(manifest["regimes"]))
    )
    validate_manifest(manifest)


def test_conditions_isolate_cap_and_warmup_changes() -> None:
    with pytest.warns(UserWarning, match="wide_moderate"):
        configs = {name: condition_config(50, 300, name) for name in CONDITIONS}

    assert configs["shipped"]["maxiters"] == 200
    assert configs["shipped"]["niter_broadprior"] == 200
    assert configs["cap400"]["maxiters"] == 400
    assert configs["cap400"]["niter_broadprior"] == 200
    assert configs["cap800"]["maxiters"] == 800
    assert configs["warmup50_cap400"]["niter_broadprior"] == 50
    assert configs["warmup50_cap400"]["maxiters"] == 400
    assert configs["no_warmup_cap400"]["niter_broadprior"] == 0
    assert configs["forced800"]["maxiters"] == 800
    assert not any(configs["forced800"]["convergence_criteria"].values())


def test_condition_configs_do_not_share_mutable_values() -> None:
    with pytest.warns(UserWarning, match="tall_moderate"):
        first = condition_config(1000, 50, "cap400")
    first["rmsstop"].append(999)
    first["convergence_criteria"]["angle"] = False

    with pytest.warns(UserWarning, match="tall_moderate"):
        second = condition_config(1000, 50, "cap400")

    assert 999 not in second["rmsstop"]
    assert second["convergence_criteria"]["angle"] is True


def test_manifest_validation_rejects_design_drift() -> None:
    manifest = build_manifest("screen", n_reps=1, seed=10)
    manifest["conditions"] = ["shipped"]

    with pytest.raises(ValueError, match="conditions differ"):
        validate_manifest(manifest)


def test_invalid_profile_condition_and_replicates_fail() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        build_manifest("unknown", n_reps=1, seed=1)
    with pytest.raises(ValueError, match="positive"):
        build_manifest("screen", n_reps=0, seed=1)
    with pytest.raises(ValueError, match="unknown convergence-margin condition"):
        condition_config(50, 300, "unknown")
