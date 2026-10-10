"""The release fitting gate must detect small changes and protocol drift."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_fitting_identity.py"
spec = importlib.util.spec_from_file_location("check_fitting_identity", SCRIPT)
assert spec is not None
assert spec.loader is not None
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def _snapshot(path, array, **changes):
    np.savez(path, posterior=array)
    metadata = {
        "version": "0.4.5",
        "schema": 1,
        "policy_sha256": "fixed-policy",
        "policy": "production_no_cost_post_rotation",
        "cases": [],
        "environment": {"numpy": "fixed"},
        "legacy_cv": {},
        **changes,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))


def test_gate_accepts_identical_arrays_across_package_versions(tmp_path):
    baseline, candidate = tmp_path / "old.npz", tmp_path / "new.npz"
    _snapshot(baseline, np.array([1.0, 2.0]))
    _snapshot(candidate, np.array([1.0, 2.0]), version="0.4.6")
    report = checker.compare_snapshots(baseline, candidate)
    assert report["passed"]
    assert report["byte_identical_arrays"] == 1
    assert report["baseline_version"] == "0.4.5"
    assert report["candidate_version"] == "0.4.6"


@pytest.mark.parametrize(
    "changed", [np.array([1 + 1e-15, 2.0]), np.array([[1.0, 2.0]]), np.array([1, 2])]
)
def test_gate_rejects_small_value_shape_or_dtype_changes(tmp_path, changed):
    baseline, candidate = tmp_path / "old.npz", tmp_path / "new.npz"
    _snapshot(baseline, np.array([1.0, 2.0]))
    _snapshot(candidate, changed)
    report = checker.compare_snapshots(baseline, candidate)
    assert not report["passed"]
    assert report["differences"] == ["posterior"]


@pytest.mark.parametrize("field", ["policy_sha256", "environment", "cases"])
def test_gate_rejects_comparing_different_fit_protocols(tmp_path, field):
    baseline, candidate = tmp_path / "old.npz", tmp_path / "new.npz"
    _snapshot(baseline, np.ones(2))
    _snapshot(candidate, np.ones(2), **{field: "changed"})
    with pytest.raises(ValueError, match=f"protocol differs in {field}"):
        checker.compare_snapshots(baseline, candidate)


def test_gate_rejects_missing_fields_and_empty_snapshots(tmp_path):
    baseline, candidate = tmp_path / "old.npz", tmp_path / "new.npz"
    _snapshot(baseline, np.ones(2))
    _snapshot(candidate, np.ones(2))
    np.savez(candidate, other=np.ones(2))
    with pytest.raises(ValueError, match="field sets differ"):
        checker.compare_snapshots(baseline, candidate)
    np.savez(baseline)
    np.savez(candidate)
    with pytest.raises(ValueError, match="empty"):
        checker.compare_snapshots(baseline, candidate)
