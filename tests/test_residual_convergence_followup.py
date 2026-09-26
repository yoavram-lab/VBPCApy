"""Tests for residual-convergence Rockfish sharding."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _analysis_module():
    """Import orchestration without requiring the external runner package."""
    if "trade_study" not in sys.modules:

        class Definition:
            def __init__(self, name, *_args, **_kwargs):
                self.name = name
                for key, value in _kwargs.items():
                    setattr(self, key, value)

        dependency = ModuleType("trade_study")
        dependency.Direction = SimpleNamespace(MINIMIZE="minimize", MAXIMIZE="maximize")
        dependency.FactorType = SimpleNamespace(
            CONTINUOUS="continuous",
            DISCRETE="discrete",
            CATEGORICAL="categorical",
        )
        dependency.Constraint = Definition
        dependency.Factor = Definition
        dependency.FactorConstraint = Definition
        dependency.Observable = Definition
        dependency.ResultsTable = object
        dependency.load_results = None
        dependency.run_grid = None
        dependency.save_results = None
        sys.modules["trade_study"] = dependency
    sys.path.insert(0, str(Path(__file__).parents[1]))
    return import_module("analysis.trade_study.residual_convergence_followup")


followup = _analysis_module()


def test_registered_manifest_has_one_shard_per_condition_and_regime() -> None:
    manifest = followup.registered_manifest()
    shards = followup._shards(manifest)

    assert len(shards) == 21
    assert shards[0] == ("released", "wide_complete_anchor")
    assert shards[6] == ("released", "tall_extreme_complete_holdout")
    assert shards[7] == ("double_cap", "wide_complete_anchor")
    assert shards[-1] == ("forced_long", "tall_extreme_complete_holdout")


def test_shard_index_is_manifest_ordered_and_bounded() -> None:
    manifest = followup.build_manifest("smoke", n_reps=1, seed=100)

    assert followup._shard_at_index(manifest, 0) == (
        "released",
        "wide_complete_smoke",
    )
    assert followup._shard_at_index(manifest, 5) == (
        "forced_long",
        "tall_extreme_mnar_smoke",
    )
    with pytest.raises(ValueError, match=r"\[0, 6\)"):
        followup._shard_at_index(manifest, 6)


def test_shard_grid_resolves_release_configuration() -> None:
    manifest = followup.build_manifest("smoke", n_reps=1, seed=100)

    with pytest.warns(UserWarning, match="wide_moderate"):
        grid = followup._shard_grid(
            manifest,
            "released",
            "wide_complete_smoke",
        )

    assert len(grid) == 1
    assert grid[0]["maxiters"] == 1600
    assert grid[0]["_condition"] == "released"
    assert grid[0]["_bucket"] == "wide_moderate"
    assert grid[0]["seed"] == 100
