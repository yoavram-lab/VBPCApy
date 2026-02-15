"""
Octave-backed regression + sharp-edge tests for vbpca_py._mean.subtract_mu.

Key notes:
- tools/SubtractMu.m *assumes* Xprobe has the same number of columns as X because
  it uses n2=size(X,2) when subtracting from Xprobe. So regression tests must
  satisfy: size(Xprobe,2)==size(X,2) and Mprobe is n_rows x n_cols.

- For sparse regression, SubtractMu.m calls subtract_mu(X,Mu) which is a mex.
  We compile tools/subtract_mu.cpp to a mex at test time (like rms cpp tests).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.io import loadmat, savemat

from vbpca_py._mean import ProbeMatrices, subtract_mu

# MATLAB / mex uses EPS=1e-15
EPS = 1e-15


# ======================================================================================
# Octave + build harness
# ======================================================================================


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tools_dir() -> Path:
    return _repo_root() / "tools"


def _octave_available() -> bool:
    return shutil.which("octave") is not None


def _run_octave_eval(cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["octave", "--quiet", "--eval", cmd],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _octave_exist_file(symbol: str) -> int:
    """Return octave exist(symbol,'file') value (0=missing, 3=mex, 2=file, etc.)."""
    tools = str(_tools_dir()).replace("\\", "/")
    cmd = f"addpath('{tools}'); disp(exist('{symbol}','file'));"
    out = _run_octave_eval(cmd).stdout.strip().splitlines()[-1]
    return int(float(out))


def _octave_can_call_subtract_mu_mex() -> bool:
    """True if Octave can resolve and execute subtract_mu from tools path."""
    try:
        if _octave_exist_file("subtract_mu") == 0:
            return False

        tools = str(_tools_dir()).replace("\\", "/")
        smoke = (
            f"addpath('{tools}');"
            "x=sparse(1,1,1,1,1);"
            "mu=1;"
            "y=subtract_mu(x,mu);"
            "if ~issparse(y), error('subtract_mu did not return sparse output'); end;"
        )
        _run_octave_eval(smoke)
        return True
    except Exception:
        return False


def _build_subtract_mu_mex_octave() -> None:
    """
    Build tools/subtract_mu.cpp into a mex file using Octave.

    Uses mkoctfile --mex, which is the canonical way for Octave.
    Output lands in tools/ as subtract_mu.mex* (platform-specific suffix).
    """
    tools = str(_tools_dir()).replace("\\", "/")

    # If already exists, do nothing.
    if _octave_can_call_subtract_mu_mex():
        return

    # Build: cd(tools); mkoctfile --mex subtract_mu.cpp
    cmd = (
        f"cd('{tools}');"
        "try, "
        "mkoctfile --mex subtract_mu.cpp; "
        "catch err, disp(getfield(err,'message')); rethrow(err); "
        "end;"
    )
    _run_octave_eval(cmd)

    # Re-check
    if not _octave_can_call_subtract_mu_mex():
        raise RuntimeError(
            "Octave mex build completed but subtract_mu is still not visible on tools path."
        )


@pytest.fixture(scope="session")
def _ensure_mex_built() -> None:
    """Session-scoped mex build, only if Octave is available."""
    if not _octave_available():
        pytest.skip("Octave not available on PATH.")
    _build_subtract_mu_mex_octave()


# ======================================================================================
# MAT IO helpers
# ======================================================================================


def _save_mat_for_octave(
    mat_path: Path,
    *,
    Mu: np.ndarray,
    X: Any,
    M: Any,
    Xprobe: Any | None,
    Mprobe: Any | None,
    update_bias: bool,
) -> None:
    payload: dict[str, Any] = {
        "Mu": np.asarray(Mu, dtype=float),
        "update_bias": float(1.0 if update_bias else 0.0),
    }

    payload["X"] = X if sp.isspmatrix(X) else np.asarray(X, dtype=float)
    payload["M"] = np.asarray(M, dtype=float)

    if Xprobe is None:
        payload["Xprobe"] = np.array([], dtype=float)  # MATLAB empty
        payload["Mprobe"] = np.array([], dtype=float)
    else:
        payload["Xprobe"] = (
            Xprobe if sp.isspmatrix(Xprobe) else np.asarray(Xprobe, dtype=float)
        )
        payload["Mprobe"] = (
            np.asarray(Mprobe, dtype=float)
            if Mprobe is not None
            else np.array([], dtype=float)
        )

    savemat(mat_path, payload)


def _run_octave_subtractmu(mat_in: Path, mat_out: Path) -> None:
    tools = str(_tools_dir()).replace("\\", "/")
    in_path = str(mat_in).replace("\\", "/")
    out_path = str(mat_out).replace("\\", "/")

    cmd = (
        f"addpath('{tools}');"
        f"load('{in_path}');"
        "[Xout, Xprobe_out] = SubtractMu(Mu, X, M, Xprobe, Mprobe, update_bias);"
        f"save('-mat','{out_path}','Xout','Xprobe_out');"
    )
    _run_octave_eval(cmd)


def _load_octave_out(mat_out: Path) -> tuple[Any, Any]:
    mat = loadmat(mat_out, squeeze_me=True, struct_as_record=False)
    return mat["Xout"], mat["Xprobe_out"]


def _as_dense(obj: Any) -> np.ndarray:
    if sp.isspmatrix(obj):
        return obj.toarray()
    if hasattr(obj, "toarray"):
        return obj.toarray()
    return np.asarray(obj, dtype=float)


# ======================================================================================
# Unit-style tests (keep + sharp edges)
# ======================================================================================


def test_subtract_mu_dense_full_mask() -> None:
    x = np.array([[2.0, 3.0], [20.0, 40.0]])
    mu = np.array([1.0, 10.0])
    mask = np.ones_like(x)

    x_out, x_probe_out = subtract_mu(mu, x, mask, update_bias=True)

    expected = np.array([[1.0, 2.0], [10.0, 30.0]])
    assert np.allclose(x_out, expected)
    assert x_probe_out is None


def test_subtract_mu_dense_with_masking() -> None:
    x = np.array([[2.0, 3.0], [20.0, 40.0]])
    mu = np.array([1.0, 10.0])
    mask = np.array([[1.0, 0.0], [1.0, 0.0]])

    x_out, _ = subtract_mu(mu, x, mask, update_bias=True)

    expected = np.array([[1.0, 3.0], [10.0, 40.0]])
    assert np.allclose(x_out, expected)


def test_subtract_mu_dense_with_probe() -> None:
    x = np.array([[2.0, 4.0], [10.0, 20.0]])
    x_probe = np.array([[1.0, 2.0], [5.0, 8.0]])
    mu = np.array([1.0, 3.0])

    mask = np.ones_like(x)
    mask_probe = np.array([[1.0, 0.0], [0.0, 1.0]])
    probe = ProbeMatrices(x=x_probe, mask=mask_probe)

    x_out, x_probe_out = subtract_mu(mu, x, mask, probe=probe, update_bias=True)

    expected_x = x - mu[:, None] * mask
    expected_probe = x_probe - mu[:, None] * mask_probe

    assert np.allclose(x_out, expected_x)
    assert x_probe_out is not None
    assert np.allclose(x_probe_out, expected_probe)


def test_subtract_mu_update_bias_false_returns_unchanged() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    x_probe = np.array([[5.0, 6.0], [7.0, 8.0]])
    mu = np.array([0.5, 1.0])
    mask = np.ones_like(x)

    probe = ProbeMatrices(x=x_probe, mask=mask)
    x_out, x_probe_out = subtract_mu(mu, x, mask, probe=probe, update_bias=False)

    assert x_out is x
    assert x_probe_out is x_probe


def test_subtract_mu_sparse_zero_becomes_eps_sharp_edge() -> None:
    x = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, 1))
    mu = np.array([1.0])
    mask = np.ones((1, 1))

    x_out, _ = subtract_mu(mu, x, mask, update_bias=True)
    assert sp.isspmatrix_csr(x_out)
    assert x_out.nnz == 1
    val = float(x_out.data[0])
    assert val != 0.0
    assert abs(val - EPS) <= 1e-15


def test_subtract_mu_sparse_probe_empty_by_shape_is_skipped_sharp_edge() -> None:
    x = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, 1))
    mu = np.array([1.0])
    mask = np.ones((1, 1))

    x_probe = sp.csr_matrix((1, 0))  # isempty in MATLAB
    probe = ProbeMatrices(x=x_probe, mask=None)

    _, x_probe_out = subtract_mu(mu, x, mask, probe=probe, update_bias=True)
    assert x_probe_out is None


# ======================================================================================
# Octave-backed regression tests (dense + sparse)
# ======================================================================================


@pytest.mark.skipif(not _octave_available(), reason="Octave not available on PATH.")
def test_subtract_mu_regression_dense_full_mask_octave(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n_rows, n_cols = 6, 9

    X = rng.standard_normal((n_rows, n_cols))
    Mu = rng.standard_normal((n_rows, 1))
    M = np.ones((n_rows, n_cols))
    update_bias = True

    X_py, _ = subtract_mu(Mu, X.copy(), M, update_bias=update_bias)

    mat_in = tmp_path / "in.mat"
    mat_out = tmp_path / "out.mat"
    _save_mat_for_octave(
        mat_in, Mu=Mu, X=X, M=M, Xprobe=None, Mprobe=None, update_bias=update_bias
    )
    _run_octave_subtractmu(mat_in, mat_out)
    X_oc, Xp_oc = _load_octave_out(mat_out)

    assert np.asarray(Xp_oc).size == 0
    assert_allclose(np.asarray(X_py), _as_dense(X_oc), rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not _octave_available(), reason="Octave not available on PATH.")
def test_subtract_mu_regression_dense_partial_mask_octave(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n_rows, n_cols = 5, 8

    X = rng.standard_normal((n_rows, n_cols))
    Mu = rng.standard_normal((n_rows, 1))
    M = (rng.random((n_rows, n_cols)) > 0.35).astype(float)
    update_bias = True

    X_py, _ = subtract_mu(Mu, X.copy(), M, update_bias=update_bias)

    mat_in = tmp_path / "in.mat"
    mat_out = tmp_path / "out.mat"
    _save_mat_for_octave(
        mat_in, Mu=Mu, X=X, M=M, Xprobe=None, Mprobe=None, update_bias=update_bias
    )
    _run_octave_subtractmu(mat_in, mat_out)
    X_oc, _ = _load_octave_out(mat_out)

    assert_allclose(np.asarray(X_py), _as_dense(X_oc), rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not _octave_available(), reason="Octave not available on PATH.")
def test_subtract_mu_regression_dense_with_probe_octave(tmp_path: Path) -> None:
    """
    IMPORTANT: SubtractMu.m assumes Xprobe has the same number of columns as X.
    So we set n_cols_probe == n_cols.
    """
    rng = np.random.default_rng(2)
    n_rows, n_cols = 6, 7

    X = rng.standard_normal((n_rows, n_cols))
    Xprobe = rng.standard_normal((n_rows, n_cols))  # must match n_cols
    Mu = rng.standard_normal((n_rows, 1))

    M = (rng.random((n_rows, n_cols)) > 0.2).astype(float)
    Mprobe = (rng.random((n_rows, n_cols)) > 0.4).astype(float)

    update_bias = True

    probe = ProbeMatrices(x=Xprobe.copy(), mask=Mprobe.copy())
    X_py, Xp_py = subtract_mu(Mu, X.copy(), M, probe=probe, update_bias=update_bias)

    mat_in = tmp_path / "in.mat"
    mat_out = tmp_path / "out.mat"
    _save_mat_for_octave(
        mat_in, Mu=Mu, X=X, M=M, Xprobe=Xprobe, Mprobe=Mprobe, update_bias=update_bias
    )
    _run_octave_subtractmu(mat_in, mat_out)
    X_oc, Xp_oc = _load_octave_out(mat_out)

    assert Xp_py is not None
    assert_allclose(np.asarray(X_py), _as_dense(X_oc), rtol=1e-12, atol=1e-12)
    assert_allclose(np.asarray(Xp_py), _as_dense(Xp_oc), rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not _octave_available(), reason="Octave not available on PATH.")
def test_subtract_mu_regression_sparse_basic_octave(
    tmp_path: Path, _ensure_mex_built
) -> None:
    rng = np.random.default_rng(3)
    n_rows, n_cols = 8, 9
    Mu = rng.standard_normal((n_rows, 1))

    rows = np.array([0, 1, 1, 3, 5, 5, 7], dtype=int)
    cols = np.array([0, 2, 4, 1, 3, 8, 2], dtype=int)
    data = rng.standard_normal(len(rows))
    data[0] = Mu[0, 0]  # force exact zero -> EPS
    data[5] = Mu[5, 0]  # force exact zero -> EPS

    X = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    M = np.ones((n_rows, n_cols))  # ignored for sparse

    X_py, _ = subtract_mu(Mu, X, M, update_bias=True)

    mat_in = tmp_path / "in.mat"
    mat_out = tmp_path / "out.mat"
    _save_mat_for_octave(
        mat_in, Mu=Mu, X=X, M=M, Xprobe=None, Mprobe=None, update_bias=True
    )
    _run_octave_subtractmu(mat_in, mat_out)
    X_oc, _ = _load_octave_out(mat_out)

    assert_allclose(X_py.toarray(), _as_dense(X_oc), rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not _octave_available(), reason="Octave not available on PATH.")
def test_subtract_mu_regression_sparse_with_probe_octave(
    tmp_path: Path, _ensure_mex_built
) -> None:
    rng = np.random.default_rng(4)
    n_rows, n_cols = 6, 7

    Mu = rng.standard_normal((n_rows, 1))

    rows = np.array([0, 2, 2, 4], dtype=int)
    cols = np.array([1, 0, 6, 3], dtype=int)
    data = rng.standard_normal(len(rows))
    data[0] = Mu[0, 0]  # force EPS
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # IMPORTANT: For MATLAB SubtractMu.m, probe must have same n_cols as X
    rows_p = np.array([1, 3, 5], dtype=int)
    cols_p = np.array([0, 2, 4], dtype=int)
    data_p = rng.standard_normal(len(rows_p))
    Xprobe = sp.csr_matrix((data_p, (rows_p, cols_p)), shape=(n_rows, n_cols))

    M = np.ones((n_rows, n_cols))
    Mprobe = np.ones((n_rows, n_cols))  # ignored for sparse but shape OK

    probe = ProbeMatrices(x=Xprobe, mask=Mprobe)
    X_py, Xp_py = subtract_mu(Mu, X, M, probe=probe, update_bias=True)

    mat_in = tmp_path / "in.mat"
    mat_out = tmp_path / "out.mat"
    _save_mat_for_octave(
        mat_in, Mu=Mu, X=X, M=M, Xprobe=Xprobe, Mprobe=Mprobe, update_bias=True
    )
    _run_octave_subtractmu(mat_in, mat_out)
    X_oc, Xp_oc = _load_octave_out(mat_out)

    assert Xp_py is not None
    assert_allclose(X_py.toarray(), _as_dense(X_oc), rtol=1e-12, atol=1e-14)
    assert_allclose(Xp_py.toarray(), _as_dense(Xp_oc), rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not _octave_available(), reason="Octave not available on PATH.")
@pytest.mark.parametrize(
    ("seed", "n_rows", "n_cols", "mask_keep_prob", "update_bias"),
    [
        (10, 24, 36, 0.72, True),
        (11, 24, 36, 0.72, False),
        (12, 40, 64, 0.58, True),
    ],
)
def test_subtract_mu_regression_dense_expanded_matrix_octave(
    tmp_path: Path,
    seed: int,
    n_rows: int,
    n_cols: int,
    mask_keep_prob: float,
    update_bias: bool,
) -> None:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_cols))
    Mu = rng.standard_normal((n_rows, 1))
    M = (rng.random((n_rows, n_cols)) < mask_keep_prob).astype(float)

    probe_x = rng.standard_normal((n_rows, n_cols))
    probe_m = (rng.random((n_rows, n_cols)) < 0.7).astype(float)
    probe = ProbeMatrices(x=probe_x.copy(), mask=probe_m.copy())

    X_py, Xp_py = subtract_mu(
        Mu,
        X.copy(),
        M,
        probe=probe,
        update_bias=update_bias,
    )

    mat_in = tmp_path / f"in_dense_{seed}.mat"
    mat_out = tmp_path / f"out_dense_{seed}.mat"
    _save_mat_for_octave(
        mat_in,
        Mu=Mu,
        X=X,
        M=M,
        Xprobe=probe_x,
        Mprobe=probe_m,
        update_bias=update_bias,
    )
    _run_octave_subtractmu(mat_in, mat_out)
    X_oc, Xp_oc = _load_octave_out(mat_out)

    assert Xp_py is not None
    assert_allclose(np.asarray(X_py), _as_dense(X_oc), rtol=1e-12, atol=1e-12)
    assert_allclose(np.asarray(Xp_py), _as_dense(Xp_oc), rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not _octave_available(), reason="Octave not available on PATH.")
def test_subtract_mu_regression_sparse_expanded_matrix_octave(
    tmp_path: Path, _ensure_mex_built
) -> None:
    rng = np.random.default_rng(13)
    n_rows, n_cols = 60, 90
    Mu = rng.standard_normal((n_rows, 1))

    nnz_main = 420
    rows = rng.integers(0, n_rows, size=nnz_main)
    cols = rng.integers(0, n_cols, size=nnz_main)
    data = rng.standard_normal(nnz_main)
    for idx in range(0, nnz_main, 85):
        data[idx] = Mu[rows[idx], 0]
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    nnz_probe = 250
    rows_p = rng.integers(0, n_rows, size=nnz_probe)
    cols_p = rng.integers(0, n_cols, size=nnz_probe)
    data_p = rng.standard_normal(nnz_probe)
    Xprobe = sp.csr_matrix((data_p, (rows_p, cols_p)), shape=(n_rows, n_cols))

    M = np.ones((n_rows, n_cols), dtype=float)
    Mprobe = np.ones((n_rows, n_cols), dtype=float)

    probe = ProbeMatrices(x=Xprobe, mask=Mprobe)
    X_py, Xp_py = subtract_mu(Mu, X, M, probe=probe, update_bias=True)

    mat_in = tmp_path / "in_sparse_expanded.mat"
    mat_out = tmp_path / "out_sparse_expanded.mat"
    _save_mat_for_octave(
        mat_in,
        Mu=Mu,
        X=X,
        M=M,
        Xprobe=Xprobe,
        Mprobe=Mprobe,
        update_bias=True,
    )
    _run_octave_subtractmu(mat_in, mat_out)
    X_oc, Xp_oc = _load_octave_out(mat_out)

    assert Xp_py is not None
    assert_allclose(X_py.toarray(), _as_dense(X_oc), rtol=1e-12, atol=1e-14)
    assert_allclose(Xp_py.toarray(), _as_dense(Xp_oc), rtol=1e-12, atol=1e-14)
