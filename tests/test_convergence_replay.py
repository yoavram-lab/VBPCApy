"""Offline replay checks original results before exploring additional policies."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis.trade_study import convergence_replay as replay
from analysis.trade_study import convergence_stability_study as study
from analysis.trade_study._convergence_detector_design import build_manifest
from analysis.trade_study.convergence_detector import DetectorPolicy

REVISION = "a" * 40


@pytest.fixture
def retained(tmp_path):
    root = tmp_path / "retained"
    root.mkdir()
    manifest = build_manifest("smoke", n_reps=2, seed=4)
    manifest["cells"] = manifest["cells"][:1]
    manifest["checkpoints"] = [1, 2, 4]
    policy = DetectorPolicy(name="angle", enabled=("angle",), minangle=0.05)
    manifest["policies"] = [asdict(policy), asdict(policy) | {"name": "angle_alias"}]
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    for i in range(2):
        fidelity = {
            "endpoint_checkpoint": 4,
            "endpoint_stable": i == 0,
            "earliest_fidelity_checkpoint": 2 if i == 0 else None,
            "rows": [
                {"checkpoint": 1, "acceptable": False},
                {"checkpoint": 2, "acceptable": i == 0},
                {"checkpoint": 4, "acceptable": True},
            ],
        }
        curve = {
            "schema_version": 1,
            "rms": [1.0] * 5,
            "prms": [1.0] * 5,
            "cost": [None, 5.0, 4.0, 3.0, 2.0],
            "angle": [None, 0.5, 0.01, 0.001, 0.001],
        }
        record = {
            "status": "ok",
            "shard_index": i,
            "rep": i,
            "result_schema_version": 2,
            "manifest_sha256": identity,
            "vbpca_revision": REVISION,
            "vbpca_release": "0.4.0",
            "python": "3.11.7",
            "numpy": "2.2.6",
            "scipy": "1.17.1",
            "cell": manifest["cells"][0],
            "base_options": {},
            "learning_curve": curve,
            "fidelity": fidelity,
            "policies": study._evaluate_policies(
                curve, manifest, fidelity, base_options={}
            ),
        }
        (root / f"shard-{i:05d}.json").write_text(json.dumps(record))
    return path, root, identity


def _run(retained, **kwargs):
    path, root, identity = retained
    return replay.replay(
        path,
        root,
        manifest_sha256=identity,
        generating_revision=REVISION,
        selected=("angle", "angle_alias"),
        **kwargs,
    )


def _change(retained, **changes):
    path = retained[1] / "shard-00000.json"
    record = json.loads(path.read_text())
    record.update(changes)
    path.write_text(json.dumps(record))


def test_complete_replay_keeps_named_aliases_and_unstable_endpoints(retained, tmp_path):
    before = {p: p.read_bytes() for p in retained[1].glob("*.json")}
    policies = tmp_path / "policies.json"
    policies.write_text(
        json.dumps([
            {
                "name": "later_angle",
                "enabled": ["angle"],
                "minangle": 0.002,
                "patience": 2,
            }
        ])
    )
    result = _run(retained, policies_path=policies)
    assert result["exploratory"] is True
    assert result["n_unstable_endpoints"] == 1
    assert result["summary"]["n_records"] == 2
    for name in ("angle", "angle_alias", "later_angle"):
        assert result["summary"]["policies"][name]["n"] == 2
        assert result["summary"]["policies"][name]["n_evaluable"] == 1
    rows = result["records"][0]["policies"]["rows"]
    assert [row["stop_iteration"] for row in rows] == [2, 2, 4]
    assert rows[1]["stored_reason"] == "angle"
    assert result["generating_revision"] == REVISION
    assert result["replay_source"]["files"]
    assert all(path.read_bytes() == contents for path, contents in before.items())
    output = tmp_path / "report.json"
    replay.write_report(output, result)
    with pytest.raises(FileExistsError):
        replay.write_report(output, {"replacement": True})
    assert json.loads(output.read_text()) == json.loads(json.dumps(result))
    assert not list(tmp_path.glob("tmp*"))


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "error"},
        {"shard_index": 1},
        {"rep": True},
        {"cell": {}},
        {"vbpca_revision": "b" * 40},
        {"manifest_sha256": "b" * 64},
        {"result_schema_version": 1},
        {"python": "changed"},
        {"learning_curve": {}},
    ],
)
def test_mixed_or_misidentified_shards_fail(retained, changes):
    _change(retained, **changes)
    with pytest.raises(ValueError):
        _run(retained)


@pytest.mark.parametrize(
    "value",
    [
        [1.0],
        [1.0, None, 1.0, 1.0, 1.0],
        [1.0, float("nan"), 1.0, 1.0, 1.0],
        [1.0, True, 1.0, 1.0, 1.0],
    ],
)
def test_incomplete_nonfinite_or_coerced_curve_fails(retained, value):
    record = json.loads((retained[1] / "shard-00000.json").read_text())
    record["learning_curve"]["angle"] = value
    _change(retained, learning_curve=record["learning_curve"])
    with pytest.raises(ValueError, match="curve"):
        _run(retained)


def test_inconsistent_fidelity_and_changed_baseline_fail(retained):
    path = retained[1] / "shard-00000.json"
    record = json.loads(path.read_text())
    record["fidelity"]["earliest_fidelity_checkpoint"] = 4
    _change(retained, fidelity=record["fidelity"])
    with pytest.raises(ValueError, match="fidelity labels"):
        _run(retained)
    record["fidelity"]["earliest_fidelity_checkpoint"] = 2
    record["policies"]["rows"][0]["stop_iteration"] = 1
    _change(retained, fidelity=record["fidelity"], policies=record["policies"])
    with pytest.raises(ValueError, match="baseline replay differs"):
        _run(retained)


def test_exact_complete_set_and_manifest_hash_required(retained):
    path, root, _identity = retained
    with pytest.raises(ValueError, match="checksum"):
        replay.replay(
            path,
            root,
            manifest_sha256="b" * 64,
            generating_revision=REVISION,
            selected=("angle",),
        )
    shard = root / "shard-00001.json"
    content = shard.read_bytes()
    shard.unlink()
    with pytest.raises(ValueError, match="complete"):
        _run(retained)
    shard.write_bytes(content)
    (root / "shard-00002.json").write_bytes(content)
    with pytest.raises(ValueError, match="complete"):
        _run(retained)


@pytest.mark.parametrize(
    "change",
    [
        {"minangle": float("nan")},
        {"minangle": -0.1},
        {"minangle": True},
        {"patience": True},
        {"patience": 1.5},
        {"warmup": -1},
        {"earlystop": "False"},
        {"rmsstop": [1.5, 0.0, 0.1]},
        {"composite_stop": [["unknown", 0.1]]},
        {"criterion_order": ["angle"] * 6},
        {"unknown": 1},
    ],
)
def test_malformed_additional_policy_rejected(retained, tmp_path, change):
    path = tmp_path / "policies.json"
    path.write_text(
        json.dumps([{"name": "extra", "enabled": ["angle"], "minangle": 0.1} | change])
    )
    with pytest.raises(ValueError):
        _run(retained, policies_path=path)


def test_new_policy_cannot_claim_unavailable_event_or_collide(retained, tmp_path):
    path = tmp_path / "policies.json"
    for policy in (
        {"name": "extra", "enabled": ["slowing_down"]},
        {"name": "angle", "enabled": ["angle"], "minangle": 0.1},
    ):
        path.write_text(json.dumps([policy]))
        with pytest.raises(ValueError):
            _run(retained, policies_path=path)


def test_mid_replay_artifact_change_is_refused(retained, monkeypatch):
    original = replay._replay_rows

    def modify(*args, **kwargs):
        result = original(*args, **kwargs)
        with (retained[1] / "shard-00000.json").open("a") as handle:
            handle.write(" ")
        return result

    monkeypatch.setattr(replay, "_replay_rows", modify)
    with pytest.raises(ValueError, match="changed during replay"):
        _run(retained)


def test_policy_availability_is_distinct_from_unstable_endpoint(retained):
    path = retained[1] / "shard-00001.json"
    record = json.loads(path.read_text())
    del record["policies"]["aliases"]["angle_alias"]
    path.write_text(json.dumps(record))
    report = _run(retained)
    assert report["baseline_availability"] == {"angle": 2, "angle_alias": 1}
    assert report["records"][1]["unavailable_baselines"] == ["angle_alias"]
    assert report["summary"]["policies"]["angle_alias"]["n"] == 1
    assert report["n_unstable_endpoints"] == 1


def test_unknown_baseline_and_duplicate_new_names_fail(retained, tmp_path):
    path, root, identity = retained
    with pytest.raises(ValueError, match="does not occur"):
        replay.replay(
            path,
            root,
            manifest_sha256=identity,
            generating_revision=REVISION,
            selected=("unknown",),
        )
    path = tmp_path / "policies.json"
    policy = {"name": "extra", "enabled": ["angle"], "minangle": 0.1}
    path.write_text(json.dumps([policy, policy]))
    with pytest.raises(ValueError, match="duplicate"):
        _run(retained, policies_path=path)


def test_cli_creates_new_report_and_refuses_retained_or_existing_output(
    retained,
    tmp_path,
    monkeypatch,
):
    path, root, identity = retained
    argv = [
        "replay",
        "--manifest",
        str(path),
        "--input-dir",
        str(root),
        "--manifest-sha256",
        identity,
        "--generating-revision",
        REVISION,
        "--policy",
        "angle",
        "--output",
    ]
    output = tmp_path / "new-report.json"
    monkeypatch.setattr(sys, "argv", [*argv, str(output)])
    replay.main()
    assert json.loads(output.read_text())["summary"]["n_records"] == 2
    with pytest.raises(SystemExit):
        replay.main()
    monkeypatch.setattr(sys, "argv", [*argv, str(root / "new.json")])
    with pytest.raises(SystemExit):
        replay.main()
    assert not (root / "new.json").exists()


def test_additional_policy_requires_explicit_name_and_criteria():
    for payload in (
        {"enabled": ["angle"], "minangle": 0.1},
        {"name": "missing_criteria", "minangle": 0.1},
    ):
        with pytest.raises(ValueError):
            replay._validate_policy(payload)
