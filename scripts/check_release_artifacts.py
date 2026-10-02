"""Verify that a release directory holds every advertised artifact (#177).

The package advertises CPython versions through its trove classifiers and is
built for Linux x86_64, macOS arm64 and Windows amd64. Publishing must not
proceed when any (Python, platform) wheel or the source distribution is
missing, as happened when the wheel builder silently skipped CPython 3.14.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTION = "vbpca_py"
PLATFORMS = {
    "linux x86_64": re.compile(r"manylinux.*_x86_64"),
    "macOS arm64": re.compile(r"macosx_.*_arm64"),
    "Windows amd64": re.compile(r"win_amd64"),
}
CLASSIFIER = re.compile(r"^Programming Language :: Python :: 3\.(\d+)$")


def _project(pyproject: Path) -> tuple[str, list[str]]:
    """Read the version and the advertised CPython tags.

    Returns:
        Package version and tags such as ``cp314``.
    """
    with pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    tags = [
        f"cp3{match.group(1)}"
        for classifier in project.get("classifiers", [])
        if (match := CLASSIFIER.match(classifier))
    ]
    return str(project["version"]), tags


def missing_artifacts(dist: Path, pyproject: Path) -> list[str]:
    """List the advertised artifacts absent from ``dist``.

    Returns:
        Human-readable descriptions of each missing artifact.
    """
    version, tags = _project(pyproject)
    names = [path.name for path in dist.iterdir()]
    missing = []
    if f"{DISTRIBUTION}-{version}.tar.gz" not in names:
        missing.append(f"sdist {DISTRIBUTION}-{version}.tar.gz")
    for tag in tags:
        prefix = f"{DISTRIBUTION}-{version}-{tag}-{tag}-"
        for platform, pattern in PLATFORMS.items():
            if not any(
                name.startswith(prefix)
                and name.endswith(".whl")
                and pattern.search(name)
                for name in names
            ):
                missing.append(f"{tag} wheel for {platform}")
    return missing


def main() -> None:
    """Exit non-zero when an advertised artifact is missing.

    Raises:
        SystemExit: If any artifact is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path, help="Directory of built artifacts")
    args = parser.parse_args()
    missing = missing_artifacts(args.dist, ROOT / "pyproject.toml")
    if missing:
        msg = "missing release artifacts:\n  " + "\n  ".join(missing)
        raise SystemExit(msg)
    print(f"all advertised artifacts present in {args.dist}")


if __name__ == "__main__":
    main()
