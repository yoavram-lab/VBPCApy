"""Require included tests to collect and public examples to run from an sdist."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REQUIRED_REFERENCE_FIXTURES = (
    "legacy_pca_full_dense.mat",
    "legacy_pca_full_missing.mat",
    "legacy_rotate_to_pca.mat",
)


def check_archive(archive: Path, *, qualify: bool = False) -> None:
    """Check shipped tests with the installed library and extracted helpers.

    Qualification also runs the non-performance/non-Octave suite, public API
    examples and strict site against an installed package. It requires the
    dev, plot and docs extras in addition to the installed source archive.

    Raises:
        ValueError: The archive does not contain one test-bearing project root.
    """
    with tempfile.TemporaryDirectory(prefix="vbpca-sdist-tests-") as directory:
        destination = Path(directory)
        with tarfile.open(archive) as source:
            source.extractall(destination, filter="data")
        projects = list(destination.iterdir())
        if len(projects) != 1 or not (projects[0] / "tests").is_dir():
            msg = "Source archive must contain one project with its included tests"
            raise ValueError(msg)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTEST_ADDOPTS", None)
        project = projects[0]
        if qualify:
            for name in REQUIRED_REFERENCE_FIXTURES:
                if not (project / "tests" / "data" / name).is_file():
                    msg = f"Source qualification requires reference fixture {name}"
                    raise ValueError(msg)
            # Refuse an editable checkout before accepting any release checks.
            subprocess.run(
                [
                    sys.executable,
                    "scripts/check_public_examples.py",
                    "--require-wheel",
                ],
                cwd=project,
                env=environment,
                check=True,
            )
        # The working directory is the extracted tree. Neither the checkout's
        # analysis helpers nor its test configuration can satisfy missing files.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-o",
                "addopts=",
                "tests",
            ],
            cwd=project,
            env=environment,
            check=True,
        )
        if qualify:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-o",
                    "addopts=",
                    "-m",
                    "not perf and not octave",
                    "tests",
                ],
                cwd=project,
                env=environment,
                check=True,
            )
            site = destination / "site"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mkdocs",
                    "build",
                    "--strict",
                    "--site-dir",
                    str(site),
                ],
                cwd=project,
                env=environment,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/check_gallery_site.py",
                    "--site-dir",
                    str(site),
                ],
                cwd=project,
                env=environment,
                check=True,
            )
            return
        # Collection cannot detect files opened only while a test runs. Execute
        # the public examples from the extracted tree so their Markdown and
        # notebook inputs must also be present in the archive.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "tests/test_markdown_examples.py",
                "tests/test_notebook_examples.py",
            ],
            cwd=project,
            env=environment,
            check=True,
        )


def main() -> None:
    """Check the unique source archive in a release artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--qualify",
        action="store_true",
        help="Run the shipped suite, API examples and strict site against an installed package",
    )
    args = parser.parse_args()
    archives = list(args.directory.glob("*.tar.gz"))
    if len(archives) != 1:
        parser.error("provide a directory with exactly one source archive")
    check_archive(archives[0], qualify=args.qualify)


if __name__ == "__main__":
    main()
