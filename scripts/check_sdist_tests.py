"""Require included tests to collect and public examples to run from an sdist."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def check_archive(archive: Path) -> None:
    """Check shipped tests with the installed library and extracted helpers.

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
            cwd=projects[0],
            env=environment,
            check=True,
        )
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
            cwd=projects[0],
            env=environment,
            check=True,
        )


def main() -> None:
    """Check the unique source archive in a release artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    archives = list(args.directory.glob("*.tar.gz"))
    if len(archives) != 1:
        parser.error("provide a directory with exactly one source archive")
    check_archive(archives[0])


if __name__ == "__main__":
    main()
