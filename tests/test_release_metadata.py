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
    root.joinpath("docs").mkdir(exist_ok=True)
    for relative in ("README.md", "docs/citation.md"):
        root.joinpath(relative).write_text(
            "```bibtex\n@software{example,\n"
            f"  version = {{{citation_version}}},\n}}\n```\n",
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


@pytest.mark.parametrize("relative", ["README.md", "docs/citation.md"])
@pytest.mark.parametrize("failure", ["missing", "stale", "no-version", "duplicate"])
def test_release_rejects_unusable_copyable_citation(tmp_path, relative, failure):
    _write_metadata(tmp_path)
    path = tmp_path / relative
    text = path.read_text(encoding="utf-8")
    if failure == "missing":
        path.unlink()
    else:
        if failure == "stale":
            text = text.replace("{1.2.3}", "{1.2.2}")
        elif failure == "no-version":
            text = text.replace("  version = {1.2.3},\n", "")
        else:
            text += "  version = {1.2.3},\n"
        path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="citation"):
        check_release_metadata(tmp_path, "v1.2.3")


def _write_development_metadata(
    root: Path,
    *,
    project_version: str = "0.5.0.dev0",
    citation_version: str = "0.4.12",
) -> None:
    _write_metadata(
        root, project_version=project_version, citation_version=citation_version
    )
    root.joinpath("CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n\n## [{citation_version}] - 2026-09-22\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("development", "stable"),
    [("0.5.0.dev0", "0.4.12"), ("0.10.0.dev7", "0.9.12")],
)
def test_development_retains_stable_citation(
    tmp_path, monkeypatch, development, stable, capsys
):
    monkeypatch.delenv("RELEASE_TAG", raising=False)
    _write_development_metadata(
        tmp_path, project_version=development, citation_version=stable
    )
    check_release_metadata(tmp_path)
    output = capsys.readouterr().out
    assert f"{development} (unreleased)" in output
    assert f"citation remains {stable}" in output


@pytest.mark.parametrize("relative", ["README.md", "docs/citation.md"])
def test_development_citation_examples_keep_the_last_stable_version(
    tmp_path, monkeypatch, relative
):
    monkeypatch.delenv("RELEASE_TAG", raising=False)
    _write_development_metadata(tmp_path)
    path = tmp_path / relative
    path.write_text(
        path.read_text(encoding="utf-8").replace("{0.4.12}", "{0.5.0.dev0}"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="citation versions"):
        check_release_metadata(tmp_path)


@pytest.mark.parametrize("from_environment", [False, True])
def test_development_cannot_pass_production_release_tag(
    tmp_path, monkeypatch, from_environment
):
    _write_development_metadata(tmp_path)
    monkeypatch.delenv("RELEASE_TAG", raising=False)
    tag = "v0.5.0.dev0"
    if from_environment:
        monkeypatch.setenv("RELEASE_TAG", tag)
    with pytest.raises(ValueError, match="cannot be published"):
        check_release_metadata(tmp_path, None if from_environment else tag)


@pytest.mark.parametrize("citation", ["0.5.0", "0.6.0", "0.4.13.dev0"])
def test_development_rejects_nonstable_or_future_citation(
    tmp_path, monkeypatch, citation
):
    monkeypatch.delenv("RELEASE_TAG", raising=False)
    _write_development_metadata(tmp_path, citation_version=citation)
    with pytest.raises(ValueError, match="stable"):
        check_release_metadata(tmp_path)


@pytest.mark.parametrize(
    ("changelog", "message"),
    [
        ("## [0.4.12] - 2026-09-22\n", "Unreleased"),
        (
            "## [Unreleased]\n## [0.5.0.dev0] - 2026-09-22\n## [0.4.12] - 2026-09-22\n",
            "dated release heading",
        ),
        ("## [Unreleased]\n", "no dated"),
        ("## [Unreleased]\n## [0.4.12] - 2026-09-21\n", "date-released"),
    ],
)
def test_development_requires_consistent_unreleased_and_cited_history(
    tmp_path, monkeypatch, changelog, message
):
    monkeypatch.delenv("RELEASE_TAG", raising=False)
    _write_development_metadata(tmp_path)
    tmp_path.joinpath("CHANGELOG.md").write_text(changelog, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        check_release_metadata(tmp_path)


def test_development_rejects_stale_but_internally_consistent_citation(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("RELEASE_TAG", raising=False)
    _write_development_metadata(tmp_path, citation_version="0.4.11")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text(encoding="utf-8") + "\n## [0.4.12] - 2026-09-23\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="most recent stable"):
        check_release_metadata(tmp_path)
