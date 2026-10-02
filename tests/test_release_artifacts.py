"""Tests for the release artifact-matrix gate (#177)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def _load_checker() -> Callable[[Path, Path], list[str]]:
    script = Path(__file__).parents[1] / "scripts" / "check_release_artifacts.py"
    spec = importlib.util.spec_from_file_location("check_release_artifacts", script)
    if spec is None or spec.loader is None:
        msg = f"Could not load {script}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(
        "Callable[[Path, Path], list[str]]", cast("Any", module).missing_artifacts
    )


missing_artifacts = _load_checker()

PLATFORM_TAGS = (
    "manylinux_2_27_x86_64.manylinux_2_28_x86_64",
    "macosx_14_0_arm64",
    "win_amd64",
)


def _pyproject(root: Path, minors: tuple[int, ...]) -> Path:
    classifiers = "\n".join(
        f'  "Programming Language :: Python :: 3.{minor}",' for minor in minors
    )
    path = root / "pyproject.toml"
    path.write_text(
        "[project]\n"
        'name = "vbpca_py"\n'
        'version = "1.2.3"\n'
        "classifiers = [\n"
        '  "Programming Language :: Python :: 3",\n'
        f"{classifiers}\n"
        '  "Programming Language :: Python :: 3 :: Only",\n'
        "]\n",
        encoding="utf-8",
    )
    return path


def _dist(root: Path, tags: tuple[str, ...], *, sdist: bool = True) -> Path:
    dist = root / "dist"
    dist.mkdir()
    if sdist:
        (dist / "vbpca_py-1.2.3.tar.gz").touch()
    for tag in tags:
        for platform in PLATFORM_TAGS:
            (dist / f"vbpca_py-1.2.3-{tag}-{tag}-{platform}.whl").touch()
    return dist


def test_complete_matrix_passes(tmp_path: Path) -> None:
    pyproject = _pyproject(tmp_path, (11, 12, 13, 14))
    dist = _dist(tmp_path, ("cp311", "cp312", "cp313", "cp314"))

    assert missing_artifacts(dist, pyproject) == []


def test_missing_python_version_is_reported_per_platform(tmp_path: Path) -> None:
    pyproject = _pyproject(tmp_path, (11, 12, 13, 14))
    dist = _dist(tmp_path, ("cp311", "cp312", "cp313"))

    assert missing_artifacts(dist, pyproject) == [
        "cp314 wheel for linux x86_64",
        "cp314 wheel for macOS arm64",
        "cp314 wheel for Windows amd64",
    ]


def test_missing_platform_and_sdist_are_reported(tmp_path: Path) -> None:
    pyproject = _pyproject(tmp_path, (11,))
    dist = _dist(tmp_path, ("cp311",), sdist=False)
    (dist / "vbpca_py-1.2.3-cp311-cp311-win_amd64.whl").unlink()

    assert missing_artifacts(dist, pyproject) == [
        "sdist vbpca_py-1.2.3.tar.gz",
        "cp311 wheel for Windows amd64",
    ]


def test_free_threaded_and_other_version_wheels_do_not_count(tmp_path: Path) -> None:
    pyproject = _pyproject(tmp_path, (14,))
    dist = _dist(tmp_path, ())
    for platform in PLATFORM_TAGS:
        (dist / f"vbpca_py-1.2.3-cp314-cp314t-{platform}.whl").touch()
        (dist / f"vbpca_py-1.2.2-cp314-cp314-{platform}.whl").touch()

    assert len(missing_artifacts(dist, pyproject)) == 3
