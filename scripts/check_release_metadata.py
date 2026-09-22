"""Verify that release metadata has one version and date contract."""

from __future__ import annotations

import argparse
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    """Raise when package, citation, changelog, and release tag disagree.

    Raises:
        ValueError: If any release metadata value is inconsistent.
    """
    version = _project_version(root / "pyproject.toml")
    expected_tag = f"v{version}"
    tag = release_tag or os.environ.get("RELEASE_TAG") or expected_tag
    citation_version = _cff_value(root / "CITATION.cff", "version")
    citation_date = _cff_value(root / "CITATION.cff", "date-released")
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
