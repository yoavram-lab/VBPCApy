"""Source qualification must retain the fixed numerical reference inputs."""

import runpy
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FIXTURES = (
    "legacy_pca_full_dense.mat",
    "legacy_pca_full_missing.mat",
    "legacy_rotate_to_pca.mat",
)


@pytest.mark.parametrize("missing", FIXTURES)
def test_archive_qualification_refuses_missing_reference_fixture(tmp_path, missing):
    project = tmp_path / "project"
    data = project / "tests" / "data"
    data.mkdir(parents=True)
    for name in FIXTURES:
        if name != missing:
            (data / name).write_bytes(b"fixture placeholder")
    archive = tmp_path / "project.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(project, arcname="project")
    checker = runpy.run_path(str(ROOT / "scripts/check_sdist_tests.py"))
    with pytest.raises(ValueError, match="requires reference fixture") as error:
        checker["check_archive"](archive, qualify=True)
    assert missing in str(error.value)
