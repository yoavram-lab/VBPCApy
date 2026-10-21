"""Verify that release metadata has one version and date contract."""

from __future__ import annotations

import argparse
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_VERSION = re.compile(r"(\d+\.\d+\.\d+)\.dev\d+")


def _project_version(pyproject: Path) -> str:
    """Read the declared package version.

    Returns:
        Version string from ``[project]``.

    Raises:
        ValueError: If the project version is missing or is not a string.
    """
    with pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    version = project.get("version")
    if not isinstance(version, str) or not version:
        msg = f"Missing [project].version in {pyproject}"
        raise ValueError(msg)
    return version


def _cff_value(citation: Path, key: str) -> str:
    """Read one quoted top-level scalar from ``CITATION.cff``.

    Returns:
        Unquoted scalar value.

    Raises:
        ValueError: If the requested top-level key is missing.
    """
    pattern = re.compile(rf'^{re.escape(key)}:\s*["\']?([^"\'\n]+)["\']?\s*$')
    for line in citation.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    msg = f"Missing top-level {key!r} in {citation}"
    raise ValueError(msg)


def _changelog_date(changelog: Path, version: str) -> str:
    """Read the release date for *version* from the canonical changelog.

    Returns:
        ISO release date from the matching heading.

    Raises:
        ValueError: If the version has no dated release section.
    """
    pattern = re.compile(
        rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$",
        re.MULTILINE,
    )
    match = pattern.search(changelog.read_text(encoding="utf-8"))
    if not match:
        msg = f"CHANGELOG.md has no dated [{version}] release section"
        raise ValueError(msg)
    return match.group(1)


def check_release_metadata(root: Path, release_tag: str | None = None) -> None:
    """Check released metadata or an explicitly unreleased development build.

    A development version retains the last stable citation and its matching
    changelog date. It cannot be published through a release-tag invocation.

    Raises:
        ValueError: If any release metadata value is inconsistent.
    """
    version = _project_version(root / "pyproject.toml")
    expected_tag = f"v{version}"
    supplied_tag = release_tag or os.environ.get("RELEASE_TAG")
    tag = supplied_tag or expected_tag
    citation_version = _cff_value(root / "CITATION.cff", "version")
    citation_date = _cff_value(root / "CITATION.cff", "date-released")
    development = DEVELOPMENT_VERSION.fullmatch(version)
    if development:
        _check_development_metadata(
            root,
            version=version,
            base_version=development.group(1),
            citation_version=citation_version,
            citation_date=citation_date,
            supplied_tag=supplied_tag,
        )
        return
    changelog_date = _changelog_date(root / "CHANGELOG.md", version)

    errors: list[str] = []
    if tag != expected_tag:
        errors.append(f"release tag {tag!r} != expected {expected_tag!r}")
    if citation_version != version:
        errors.append(
            f"CITATION.cff version {citation_version!r} != package {version!r}"
        )
    if citation_date != changelog_date:
        errors.append(
            "CITATION.cff date-released "
            f"{citation_date!r} != changelog date {changelog_date!r}"
        )
    if errors:
        raise ValueError("; ".join(errors))

    print(f"release metadata consistent: {tag} ({changelog_date})")


def _check_development_metadata(
    root: Path,
    *,
    version: str,
    base_version: str,
    citation_version: str,
    citation_date: str,
    supplied_tag: str | None,
) -> None:
    """Validate an unreleased ``major.minor.patch.devN`` checkout.

    Raises:
        ValueError: If the checkout claims a release or a stale/future citation.
    """
    if supplied_tag is not None:
        msg = f"development version {version!r} cannot be published with a release tag"
        raise ValueError(msg)
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if not re.search(r"^## \[Unreleased\]$", changelog, re.MULTILINE):
        msg = "development version requires an [Unreleased] changelog section"
        raise ValueError(msg)
    if re.search(rf"^## \[{re.escape(version)}\] -", changelog, re.MULTILINE):
        msg = "development version must not have a dated release heading"
        raise ValueError(msg)
    if not re.fullmatch(r"\d+\.\d+\.\d+", citation_version):
        msg = "development citation must refer to a stable major.minor.patch release"
        raise ValueError(msg)
    citation_parts = tuple(map(int, citation_version.split(".")))
    base_parts = tuple(map(int, base_version.split(".")))
    if citation_parts >= base_parts:
        msg = "development citation must refer to a stable release before the target"
        raise ValueError(msg)
    stable_versions = re.findall(
        r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", changelog, re.MULTILINE
    )
    if stable_versions and citation_version != max(
        stable_versions, key=lambda value: tuple(map(int, value.split(".")))
    ):
        msg = "development citation must identify the most recent stable changelog release"
        raise ValueError(msg)
    stable_date = _changelog_date(root / "CHANGELOG.md", citation_version)
    if citation_date != stable_date:
        msg = "CITATION.cff date-released does not match its stable changelog date"
        raise ValueError(msg)
    print(
        f"development metadata consistent: {version} (unreleased); "
        f"citation remains {citation_version} ({stable_date})"
    )


def main() -> None:
    """Run the release metadata check from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        help="Release tag to verify (default: RELEASE_TAG or v<project.version>)",
    )
    args = parser.parse_args()
    check_release_metadata(ROOT, args.tag)


if __name__ == "__main__":
    main()
