"""Tests for the release metadata consistency gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def _load_checker() -> Callable[[Path, str | None], None]:
    script = Path(__file__).parents[1] / "scripts" / "check_release_metadata.py"
    spec = importlib.util.spec_from_file_location("check_release_metadata", script)
    if spec is None or spec.loader is None:
        msg = f"Could not load {script}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(
        "Callable[[Path, str | None], None]", cast("Any", module).check_release_metadata
    )


check_release_metadata = _load_checker()


def _write_metadata(
    root: Path,
    *,
    project_version: str = "1.2.3",
    citation_version: str = "1.2.3",
    citation_date: str = "2026-09-22",
    changelog_date: str = "2026-09-22",
) -> None:
    root.joinpath("pyproject.toml").write_text(
        f'[project]\nname = "example"\nversion = "{project_version}"\n',
        encoding="utf-8",
    )
    root.joinpath("CITATION.cff").write_text(
        f'version: "{citation_version}"\ndate-released: "{citation_date}"\n',
        encoding="utf-8",
    )
    root.joinpath("CHANGELOG.md").write_text(
        f"# Changelog\n\n## [{project_version}] - {changelog_date}\n",
        encoding="utf-8",
    )


def test_release_metadata_accepts_consistent_values(tmp_path: Path) -> None:
    _write_metadata(tmp_path)

    check_release_metadata(tmp_path, "v1.2.3")


@pytest.mark.parametrize(
    ("overrides", "tag", "message"),
    [
        ({}, "v9.9.9", "release tag"),
        ({"citation_version": "1.2.2"}, "v1.2.3", "CITATION.cff version"),
        ({"citation_date": "2026-09-21"}, "v1.2.3", "date-released"),
    ],
)
def test_release_metadata_rejects_inconsistency(
    tmp_path: Path,
    overrides: dict[str, str],
    tag: str,
    message: str,
) -> None:
    _write_metadata(tmp_path, **overrides)

    with pytest.raises(ValueError, match=message):
        check_release_metadata(tmp_path, tag)
