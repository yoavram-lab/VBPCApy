"""Reject changed source snapshots before starting cluster measurements."""

from __future__ import annotations

import doctest
import hashlib
import runpy
from pathlib import Path

import pytest

_MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "analysis/cluster/prepare_backend_comparison.py"
    )
)
_Snapshot = _MODULE["SourceSnapshot"]


def test_snapshot_copies_and_protects_its_manifest(tmp_path) -> None:
    path = tmp_path / "source.py"
    path.write_text("x = 1\n")
    files = {"source.py": hashlib.sha256(path.read_bytes()).hexdigest()}
    snapshot = _Snapshot("a" * 40, files)
    files.clear()
    snapshot.validate(tmp_path)
    with pytest.raises(TypeError):
        snapshot.files["other.py"] = "b" * 64
    path.write_text("x = 2\n")
    with pytest.raises(ValueError, match="content differs"):
        snapshot.validate(tmp_path)


def test_snapshot_rejects_paths_outside_the_source_root(tmp_path) -> None:
    snapshot = _Snapshot("a" * 40, {"../outside.py": "b" * 64})
    with pytest.raises(ValueError, match="escapes"):
        snapshot.validate(tmp_path)


@pytest.mark.parametrize(
    ("revision", "digest"), [("short", "b" * 64), ("a" * 40, "short")]
)
def test_snapshot_rejects_abbreviated_identities(revision, digest) -> None:
    with pytest.raises(ValueError, match="SHA"):
        _Snapshot(revision, {"source.py": digest})


def test_snapshot_docstring_examples() -> None:
    runner = doctest.DocTestRunner()
    for example in doctest.DocTestFinder().find(_Snapshot, globs=_MODULE):
        runner.run(example)
    result = runner.summarize()
    assert result.failed == 0
    assert result.attempted == 4


def test_physical_cores_exclude_smt_siblings(monkeypatch) -> None:
    def topology(path, *args, **kwargs):
        cpu = int(path.parent.parent.name.removeprefix("cpu"))
        return str(cpu // 4 if path.name == "physical_package_id" else cpu % 2)

    monkeypatch.setattr(Path, "read_text", topology)
    assert _MODULE["physical_cpu_ids"](set(range(8))) == (0, 1, 4, 5)
    with pytest.raises(ValueError, match="distinct"):
        _MODULE["physical_cpu_ids"]({0, 1, 2, 3})
    with pytest.raises(ValueError, match="positive"):
        _MODULE["physical_cpu_ids"]({0, 1}, count=0)
