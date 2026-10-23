"""Execute actual documentation sources and test build failure propagation."""

import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import vbpca_py

ROOT = Path(__file__).parents[1]


def test_public_api_and_actual_quickstart_migration_examples():
    checker = runpy.run_path(str(ROOT / "scripts/check_public_examples.py"))
    counts = checker["check_examples"](ROOT)
    assert counts
    assert all(count > 0 for count in counts.values())


@pytest.mark.parametrize("failure", ["editable", "wrong-version"])
def test_installed_examples_refuse_wrong_package_provenance(
    tmp_path, monkeypatch, failure
):
    checker = runpy.run_path(str(ROOT / "scripts/check_public_examples.py"))
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "999.0.0"\n', encoding="utf-8"
    )
    location = (
        tmp_path / "src/vbpca_py/__init__.py"
        if failure == "editable"
        else Path(sys.prefix) / "vbpca_py/__init__.py"
    )
    monkeypatch.setattr(vbpca_py, "__file__", str(location))
    with pytest.raises(RuntimeError, match=r"editable checkout|Installed version"):
        checker["check_examples"](tmp_path, require_wheel=True)


@pytest.mark.docs
def test_executed_gallery_preserves_demonstrated_observation_and_score_targets(
    tmp_path,
):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_gallery.py"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    records = {
        r["slug"]: r["summary"]
        for r in json.loads((tmp_path / "manifest.json").read_text())
    }
    assert records["dense"]["training_shape"] == [24, 5]
    assert records["dense"]["latent_covariance_shape"] == [3, 2, 2]
    sparse = records["sparse"]
    assert (
        sparse["input_observations"]
        == sparse["training_observations"] + sparse["probe_observations"]
    )
    assert sparse["stored_observed_zeros"] == 1
    pipeline = records["pipeline"]
    assert pipeline["training_shape"] == [36, 2]
    assert pipeline["new_rows"] == 4
    assert len(pipeline["fold_training_offsets"]) == 3
    assert np.isfinite(pipeline["decoded_size_input_units"]).all()
    assert records["selection"]["common_probe_cells"] == 5
    assert [row["k"] for row in records["selection"]["records"]] == [1, 2, 3]
    raw = records["raw-cv"]
    assert raw["schema_kinds"] == ["categorical", "continuous", "ordinal"]
    assert len(set(raw["fold_training_offsets"])) == 2
    for row in raw["records"]:
        assert np.isfinite([
            row["mean_brier"],
            row["mean_rmse_variable_1"],
            row["mean_rmse_variable_2"],
        ]).all()
    assert len(records["plots"]["component_zero_loadings"]) == 5


@pytest.mark.parametrize("new_output", ["nothing", "summary_only"])
def test_reused_builder_cannot_accept_old_summary_or_old_figures(
    tmp_path, monkeypatch, new_output
):
    module = runpy.run_path(str(ROOT / "scripts/build_gallery.py"))
    builder = module["build_gallery"]
    monkeypatch.setitem(
        builder.__globals__, "EXAMPLES", (("demo", "Demo", "demo.py", "Demo output"),)
    )
    root = tmp_path / "project"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "demo.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "output = Path(sys.argv[-1])\n"
        "(output / 'summary.json').write_text('{}')\n"
        "(output / 'demo.png').write_bytes(b'example figure')\n"
    )
    destination = tmp_path / "gallery"
    builder(root, destination)
    original_manifest = (destination / "manifest.json").read_bytes()
    original_source = (destination / "demo/demo.py").read_bytes()
    if new_output == "nothing":
        script.write_text("print('successful exit without new artifacts')\n")
    else:
        script.write_text(
            "import sys\nfrom pathlib import Path\n"
            "(Path(sys.argv[-1]) / 'summary.json').write_text('{}')\n"
        )
    with pytest.raises(
        RuntimeError, match=r"produced no result summary|produced no figures"
    ):
        builder(root, destination)
    assert (destination / "manifest.json").read_bytes() == original_manifest
    assert (destination / "demo/demo.py").read_bytes() == original_source


@pytest.fixture(scope="module")
def built_site(tmp_path_factory):
    site = tmp_path_factory.mktemp("executed-docs")
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(site)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return site


@pytest.mark.docs
def test_actual_strict_site_matches_executed_sources_and_figures(built_site):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_gallery_site.py"),
            "--site-dir",
            str(built_site),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.docs
@pytest.mark.parametrize(
    "target", ["download", "html", "summary", "figure", "manifest"]
)
def test_site_verifier_refuses_stale_sources_results_or_figures(
    built_site, tmp_path, target
):
    site = tmp_path / "site"
    shutil.copytree(built_site, site)
    dense = site / "gallery/dense"
    if target == "download":
        (dense / "gallery_dense.py").write_text("# stale source\n")
    elif target == "html":
        (dense / "index.html").write_text("<p>source was not rendered</p>")
    elif target == "summary":
        (dense / "summary.json").write_text("{}")
    elif target == "figure":
        (dense / "dense.png").write_bytes(b"broken PNG")
    else:
        (site / "gallery/manifest.json").write_text("[]")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_gallery_site.py"),
            "--site-dir",
            str(site),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0


@pytest.mark.docs
def test_example_failure_fails_the_actual_strict_documentation_build(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for filename in ["build_gallery.py", "docs_gallery_hook.py"]:
        shutil.copyfile(ROOT / "scripts" / filename, scripts / filename)
    (scripts / "gallery_dense.py").write_text(
        'raise RuntimeError("intentional gallery failure")\n'
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/index.md").write_text("# Failure propagation\n")
    (tmp_path / "mkdocs.yml").write_text(
        "site_name: Failure check\nhooks:\n  - scripts/docs_gallery_hook.py\n"
    )
    completed = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "intentional gallery failure" in completed.stdout + completed.stderr
