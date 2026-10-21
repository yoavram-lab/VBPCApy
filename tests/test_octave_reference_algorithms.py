"""Exercise the vendored reference, including empty loading covariance stores."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.io import loadmat, savemat

pytestmark = [
    pytest.mark.octave,
    pytest.mark.skipif(shutil.which("octave") is None, reason="octave not installed"),
]


@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
@pytest.mark.parametrize("uniquesv", [False, True])
@pytest.mark.parametrize("rotate", [False, True])
def test_reference_algorithms_keep_noise_traces_without_debug_exports(
    tmp_path, algorithm, uniquesv, rotate
):
    rng = np.random.default_rng(296)
    x = rng.normal(size=(7, 9))
    x[2:4, 2:6] = np.nan
    savemat(tmp_path / "input.mat", {"X": x})
    tools = Path(__file__).resolve().parents[1] / "tools"
    # Use an isolated working directory; the only output is caller-requested.
    script = tmp_path / "smoke.m"
    script.write_text(
        f"addpath('{tools.as_posix().replace(chr(39), chr(39) * 2)}');\n"
        "load('input.mat'); rand('seed',296); randn('seed',296);\n"
        f"out=pca_full(X,2,'algorithm','{algorithm}','maxiters',3,"
        f"'uniquesv',{int(uniquesv)},'rotate2pca',{int(rotate)},"
        "'niter_broadprior',0,'minangle',0,'rmsstop',[],'cfstop',[10,0,0],"
        "'autosave',Inf,'verbose',0);\n"
        "save('-mat7-binary','output.mat','out');\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["octave", "--no-gui", "-qf", str(script)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = loadmat(tmp_path / "output.mat", simplify_cells=True)["out"]
    lc = out["lc"]
    assert np.isfinite(out["V"])
    assert out["V"] > 0
    assert np.size(lc["rms"]) == 4
    assert np.all(np.isfinite(np.asarray(lc["cost"])[1:]))
    terms = sum(
        np.asarray(lc[key])[1:]
        for key in ("sXv_sv", "sXv_avcov", "sXv_avtrace", "sXv_muv", "sXv_rms")
    )
    assert_allclose(np.asarray(lc["sXv"])[1:], terms, rtol=1e-13)
    assert_allclose(
        np.asarray(lc["V"])[1:],
        (terms + 0.002) / (np.count_nonzero(~np.isnan(x)) + 0.002),
        rtol=1e-13,
    )
    if algorithm != "vb":
        assert np.size(out["Av"]) == 0
        assert np.all(np.isnan(lc["av_sum_trace"]))
        assert_allclose(np.asarray(lc["sXv_avcov"])[1:], 0)
        assert_allclose(np.asarray(lc["sXv_avtrace"])[1:], 0)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "input.mat",
        "output.mat",
        "smoke.m",
    ]
