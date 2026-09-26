"""Tests for selected-condition convergence-margin orchestration."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _analysis_module():
    """Import the orchestration module without requiring its external runner."""
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
    return import_module("analysis.trade_study.validate_convergence_margins")


margin = _analysis_module()


def _paired_row(regime: str, rep: int, *, selected_k: int = 2) -> dict[str, object]:
    return {
        "regime": regime,
        "rep": rep,
        "true_rank": 2,
        "selected_k": selected_k,
        "rank_mae": abs(selected_k - 2),
        "rank_under": max(0, 2 - selected_k),
        "rank_over": max(0, selected_k - 2),
        "holdout_rmse": 1.0,
        "coverage_95": 0.95,
        "interval_score": 1.0,
        "best_k_iters": 10.0,
        "best_k_budget_hit": 0.0,
    }


def test_condition_index_resolves_against_manifest_selection() -> None:
    manifest = {"conditions": ["shipped", "no_warmup_cap400"]}

    assert margin._condition_at_index(manifest, 0) == "shipped"
    assert margin._condition_at_index(manifest, 1) == "no_warmup_cap400"
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        margin._condition_at_index(manifest, 2)


def test_bootstrap_preserves_fixed_regime_composition() -> None:
    values = np.asarray([0.0, 0.0, 10.0, 10.0])
    strata = np.asarray(["first", "first", "second", "second"])

    interval = margin._bootstrap_stratified_mean_interval(
        values,
        strata,
        rng=np.random.default_rng(100),
        n_resamples=100,
    )

    assert interval == [5.0, 5.0]


def test_paired_summary_rejects_different_replicate_keys() -> None:
    candidate = [_paired_row("first", 0)]
    reference = [_paired_row("second", 0)]

    with pytest.raises(ValueError, match="replicate keys differ"):
        margin._paired_summary(candidate, reference, n_resamples=10, seed=100)


def test_summarize_loads_only_manifest_conditions(tmp_path, monkeypatch) -> None:
    conditions = ("shipped", "no_warmup_cap400")
    manifest = margin.build_manifest(
        "confirm",
        n_reps=1,
        seed=100,
        conditions=conditions,
        reference_condition="shipped",
    )
    manifest["regimes"] = [{"name": "test_regime"}]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    grids = {
        condition: [
            {
                "_condition": condition,
                "_regime": "test_regime",
                "_bucket": "wide_moderate",
                "true_rank": 2,
            }
        ]
        for condition in conditions
    }
    monkeypatch.setattr(
        margin,
        "_condition_grid",
        lambda _manifest, condition: grids[condition],
    )

    loaded: list[str] = []

    def fake_load_results(path):
        condition = path.name
        loaded.append(condition)
        scores = np.zeros((1, len(margin.OBSERVABLE_NAMES)))
        scores[0, margin.OBSERVABLE_NAMES.index("selected_k")] = 2
        return SimpleNamespace(
            observable_names=margin.OBSERVABLE_NAMES,
            configs=grids[condition],
            scores=scores,
            metadata=[{"design_point": 0, "rep": 0}],
        )

    monkeypatch.setattr(margin, "load_results", fake_load_results)
    result = margin.summarize(
        manifest_path,
        tmp_path / "conditions",
        output=tmp_path / "summary.json",
        n_resamples=10,
    )

    assert loaded == list(conditions)
    assert set(result["by_condition"]) == set(conditions)
    assert set(result["paired_vs_reference"]) == {"no_warmup_cap400"}
