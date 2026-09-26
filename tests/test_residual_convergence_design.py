"""Tests for the immutable VBPCA 0.4 residual-cap follow-up."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from runpy import run_path

import pytest

MODULE = run_path(
    str(
        Path(__file__).parents[1]
        / "analysis"
        / "trade_study"
        / "_residual_convergence_design.py"
    )
)
CONDITIONS = MODULE["CONDITIONS"]
build_manifest = MODULE["build_manifest"]
condition_config = MODULE["condition_config"]
registered_manifest = MODULE["registered_manifest"]
validate_manifest = MODULE["validate_manifest"]


@pytest.mark.parametrize("profile", ["smoke", "confirm"])
def test_profiles_route_only_to_residual_buckets(profile: str) -> None:
    manifest = build_manifest(profile, n_reps=2, seed=100)

    assert {regime["bucket"] for regime in manifest["regimes"]} == {
        "wide_moderate",
        "tall_extreme",
    }
    assert [regime["seed"] for regime in manifest["regimes"]] == list(
        range(100, 100 + len(manifest["regimes"]))
    )
    validate_manifest(manifest)


@pytest.mark.parametrize(
    ("n", "p", "released_cap", "bucket"),
    [
        (40, 200, 1600, "wide_moderate"),
        (1020, 20, 800, "tall_extreme"),
    ],
)
def test_conditions_start_from_current_release(
    n: int, p: int, released_cap: int, bucket: str
) -> None:
    with pytest.warns(UserWarning, match=bucket):
        configs = {name: condition_config(n, p, name) for name in CONDITIONS}

    assert configs["released"]["maxiters"] == released_cap
    assert configs["released"]["niter_broadprior"] == 0
    assert configs["double_cap"]["maxiters"] == 2 * released_cap
    assert (
        configs["double_cap"]["convergence_criteria"]
        == (configs["released"]["convergence_criteria"])
    )
    assert configs["forced_long"]["maxiters"] == 2 * released_cap
    assert not any(configs["forced_long"]["convergence_criteria"].values())


def test_condition_configs_do_not_share_mutable_values() -> None:
    with pytest.warns(UserWarning, match="wide_moderate"):
        first = condition_config(40, 200, "double_cap")
    first["rmsstop"].append(999)
    first["convergence_criteria"]["angle"] = False

    with pytest.warns(UserWarning, match="wide_moderate"):
        second = condition_config(40, 200, "double_cap")

    assert 999 not in second["rmsstop"]
    assert second["convergence_criteria"]["angle"] is True


def test_tracked_manifest_is_exact_registered_design() -> None:
    path = (
        Path(__file__).parents[1]
        / "analysis"
        / "trade_study"
        / "manifests"
        / "residual_convergence_v1.json"
    )
    tracked = json.loads(path.read_text())

    assert tracked == registered_manifest()
    validate_manifest(tracked)


def test_manifest_rejects_any_registered_field_drift() -> None:
    manifest = registered_manifest()
    changed = copy.deepcopy(manifest)
    changed["regimes"][0]["noise_std"] = 0.31

    with pytest.raises(ValueError, match="code-registered design"):
        validate_manifest(changed)


def test_invalid_inputs_fail() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        build_manifest("unknown", n_reps=1, seed=1)
    with pytest.raises(ValueError, match="positive"):
        build_manifest("smoke", n_reps=0, seed=1)
    with pytest.raises(ValueError, match="unknown residual-convergence condition"):
        condition_config(40, 200, "unknown")
    with pytest.raises(ValueError, match="does not cover bucket"):
        condition_config(100, 20, "released")
