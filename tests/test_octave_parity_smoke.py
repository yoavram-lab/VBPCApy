import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py._rms import RmsConfig, compute_rms


def _run_octave_eval(cmd: str) -> float:
    result = subprocess.run(
        ["octave", "-qf", "--eval", cmd],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = f"octave failed: {result.stderr}\n{result.stdout}"
        pytest.skip(msg)
    return float(result.stdout.strip() or "nan")


@pytest.mark.skipif(shutil.which("octave") is None, reason="octave not installed")
def test_compute_rms_matches_octave_dense():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    loadings = np.eye(2, dtype=float)
    scores = np.eye(2, dtype=float)
    mask = np.ones_like(x, dtype=float)

    rms_py, _ = compute_rms(x, loadings, scores, mask, RmsConfig(n_observed=4))

    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    x_expr = "[1,2;3,4]"
    a_expr = "[1,0;0,1]"
    s_expr = "[1,0;0,1]"
    m_expr = "[1,1;1,1]"

    with tempfile.TemporaryDirectory() as tmpdir:
        octave_cmd = (
                "format long g;"
                f"addpath('{tools_dir.as_posix()}');"
            f"X={x_expr};A={a_expr};S={s_expr};M={m_expr};"
            f"[rms,errMx]=compute_rms(X,A,S,M,sum(M(:)));"
            f"disp(rms);"
        )
        rms_oct = _run_octave_eval(octave_cmd)

    assert math.isfinite(rms_oct)
    np.testing.assert_allclose(rms_py, rms_oct, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(shutil.which("octave") is None, reason="octave not installed")
def test_compute_rms_matches_octave_dense_masked():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    loadings = np.eye(2, dtype=float)
    scores = np.eye(2, dtype=float)
    mask = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    rms_py, _ = compute_rms(x, loadings, scores, mask, RmsConfig(n_observed=2))

    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    x_expr = "[1,2;3,4]"
    a_expr = "[1,0;0,1]"
    s_expr = "[1,0;0,1]"
    m_expr = "[1,0;0,1]"

    with tempfile.TemporaryDirectory() as tmpdir:
        octave_cmd = (
                "format long g;"
                f"addpath('{tools_dir.as_posix()}');"
            f"X={x_expr};A={a_expr};S={s_expr};M={m_expr};"
            f"[rms,errMx]=compute_rms(X,A,S,M,sum(M(:)));"
            f"disp(rms);"
        )
        rms_oct = _run_octave_eval(octave_cmd)

    assert math.isfinite(rms_oct)
    np.testing.assert_allclose(rms_py, rms_oct, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(shutil.which("octave") is None, reason="octave not installed")
def test_compute_rms_matches_octave_sparse_masked():
    x = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float))
    loadings = np.eye(2, dtype=float)
    scores = np.eye(2, dtype=float)
    # Mask must match sparsity structure for sparse data; values are ignored.
    mask = x.copy()

    rms_py, _ = compute_rms(x, loadings, scores, mask, RmsConfig(n_observed=x.nnz))

    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    a_expr = "[1,0;0,1]"
    s_expr = "[1,0;0,1]"

    with tempfile.TemporaryDirectory() as tmpdir:
        octave_cmd = (
                "format long g;"
                f"addpath('{tools_dir.as_posix()}');"
            "X=sparse([1,2],[1,2],[1,2],2,2);"
            "M=spones(X);"
            f"A={a_expr};S={s_expr};"
            "[rms,errMx]=compute_rms(X,A,S,M,nnz(X));"
            "disp(rms);"
        )
        rms_oct = _run_octave_eval(octave_cmd)

    assert math.isfinite(rms_oct)
    np.testing.assert_allclose(rms_py, rms_oct, rtol=1e-6, atol=1e-6)
