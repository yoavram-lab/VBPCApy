# tests/test_rms_regression.py
#
# End-to-end unified RMS tests with Octave regression.
#
# Requirements satisfied:
# - Octave available on PATH
# - errpca_pt MEX compiled into tools/ (optional; skips true sparse-branch test if missing)
#
# Notes:
# - Dense regression: always-on; recomputes ndata inside Octave.
# - Sparse regression portable: always-on; forces dense branch in Octave (no MEX needed).
# - Sparse regression true sparse branch: optional; requires compiled errpca_pt MEX.
#
# Adjust imports: `from vbpca_py.rms import RmsConfig, compute_rms` to your actual module.

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.io import loadmat, savemat

# ---- Adjust this import to match your package structure ----
from vbpca_py._rms import RmsConfig, compute_rms
from vbpca_py._sparse_error import sparse_reconstruction_error

# ----------------------------
# Repo helpers
# ----------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tools_dir() -> Path:
    return _repo_root() / "tools"


def _octave_available() -> bool:
    return shutil.which("octave") is not None


def _mkoctfile_available() -> bool:
    return shutil.which("mkoctfile") is not None


pytestmark = pytest.mark.skipif(
    not _octave_available(),
    reason="Octave not available on PATH; skipping Octave regression tests.",
)


# ----------------------------
# Octave runner
# ----------------------------


def _run_octave_eval(script: str) -> None:
    proc = subprocess.run(
        ["octave", "--quiet", "--eval", script],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Octave failed.\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
            f"--- script ---\n{script}\n"
        )


def _run_octave_compute_rms_dense(mat_in: Path, mat_out: Path, num_cpu: int) -> None:
    tools = _tools_dir()
    script = (
        "addpath('{tools}');"
        "load('{infile}');"
        "ndata = sum(M(:));"
        "[rms, errMx] = compute_rms(X, A, S, M, ndata, {numCPU});"
        "save('-mat', '{outfile}', 'rms', 'errMx');"
    ).format(
        tools=str(tools).replace("'", "''"),
        infile=str(mat_in).replace("'", "''"),
        outfile=str(mat_out).replace("'", "''"),
        numCPU=int(num_cpu),
    )
    _run_octave_eval(script)


def _run_octave_compute_rms_sparse_portable(
    mat_in: Path, mat_out: Path, num_cpu: int
) -> None:
    """
    Portable sparse regression without needing errpca_pt MEX:
    Force dense branch by converting X->full and using M = spones(X).
    """
    tools = _tools_dir()
    script = (
        "addpath('{tools}');"
        "load('{infile}');"
        "M = spones(X);"
        "Xfull = full(X);"
        "Mfull = full(M);"
        "ndata = sum(Mfull(:));"
        "[rms, errMx] = compute_rms(Xfull, A, S, Mfull, ndata, {numCPU});"
        "save('-mat', '{outfile}', 'rms', 'errMx');"
    ).format(
        tools=str(tools).replace("'", "''"),
        infile=str(mat_in).replace("'", "''"),
        outfile=str(mat_out).replace("'", "''"),
        numCPU=int(num_cpu),
    )
    _run_octave_eval(script)


def _run_octave_compute_rms_sparse_true(
    mat_in: Path, mat_out: Path, num_cpu: int
) -> None:
    """
    True sparse regression: calls compute_rms.m with sparse X so it uses errpca_pt.
    Requires that errpca_pt.* (compiled MEX) exists in tools/ and is discoverable via addpath(tools).
    """
    tools = _tools_dir()
    script = (
        "addpath('{tools}');"
        "load('{infile}');"
        "M = spones(X);"
        "ndata = nnz(X);"
        "[rms, errMx] = compute_rms(X, A, S, M, ndata, {numCPU});"
        "save('-mat', '{outfile}', 'rms', 'errMx');"
    ).format(
        tools=str(tools).replace("'", "''"),
        infile=str(mat_in).replace("'", "''"),
        outfile=str(mat_out).replace("'", "''"),
        numCPU=int(num_cpu),
    )
    _run_octave_eval(script)


def _load_octave_outputs(mat_out: Path) -> tuple[float, Any]:
    out = loadmat(mat_out)
    rms = float(np.asarray(out["rms"]).squeeze())
    errMx = out["errMx"]
    return rms, errMx


# ----------------------------
# Data generators
# ----------------------------


def _mk_dense_case(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n1, n2, k = 6, 5, 3

    X = rng.standard_normal((n1, n2)).astype(np.float64)
    M = (rng.random((n1, n2)) > 0.25).astype(np.float64)
    X0 = X * M  # legacy convention: missing entries already zeroed

    A = rng.standard_normal((n1, k)).astype(np.float64)
    S = rng.standard_normal((k, n2)).astype(np.float64)

    ndata = int(np.sum(M))
    assert ndata > 0

    return {"X": X0, "A": A, "S": S, "M": M, "ndata": ndata}


def _mk_sparse_case(seed: int = 1) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n1, n2, k = 7, 6, 2

    obs = rng.random((n1, n2)) > 0.55
    vals = rng.standard_normal((n1, n2)).astype(np.float64)
    vals[rng.random((n1, n2)) < 0.15] = 0.0  # observed zeros

    eps = np.finfo(np.float64).eps
    X_obs = np.zeros((n1, n2), dtype=np.float64)
    X_obs[obs] = vals[obs]
    X_obs[(obs) & (X_obs == 0.0)] = eps

    X_csr = sp.csr_matrix(X_obs)
    assert X_csr.nnz == int(obs.sum())

    A = rng.standard_normal((n1, k)).astype(np.float64)
    S = rng.standard_normal((k, n2)).astype(np.float64)

    M = X_csr.copy()
    M.data[:] = 1.0

    ndata = int(X_csr.nnz)
    return {"X": X_csr, "A": A, "S": S, "M": M, "ndata": ndata}


# ----------------------------
# Fixture: build errpca_pt MEX directly into tools/
# ----------------------------


def _find_existing_errpca_mex(tools: Path) -> list[Path]:
    # Extensions vary by platform; match anything that starts with errpca_pt.
    return sorted([p for p in tools.glob("errpca_pt.*") if p.is_file()])


@pytest.fixture(scope="session")
def octave_errpca_mex_in_tools() -> bool:
    """
    Best-effort: compile tools/errpca_pt.cpp into a MEX placed directly in tools/.
    Returns True if we believe errpca_pt is available to Octave, else False.

    Behavior:
    - Does nothing if octave/mkoctfile missing.
    - Builds in tools/ with output name 'errpca_pt' (Octave adds its own extension).
    - Uses -DNOTHREADS for maximum portability/stability (sufficient for regression).
    - Cleans up only the artifacts it created (if any).
    """
    if not _octave_available() or not _mkoctfile_available():
        return False

    tools = _tools_dir()
    src = tools / "errpca_pt.cpp"
    if not src.exists():
        # If your file name differs, change it here.
        return False

    # Snapshot existing errpca_pt.* files so we only clean up what we created.
    before = set(_find_existing_errpca_mex(tools))

    # Compile directly into tools/
    # -o errpca_pt ensures function name matches, output lands in cwd (tools)
    cmd = ["mkoctfile", "--mex", "-O", "-DNOTHREADS", "-o", "errpca_pt", str(src)]

    proc = subprocess.run(
        cmd,
        cwd=str(tools),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    after = set(_find_existing_errpca_mex(tools))
    created = sorted(after - before)

    if proc.returncode != 0 or not after:
        # Build failed or produced nothing usable.
        return False

    # Quick Octave-level "which" check to confirm it resolves
    try:
        script = (
            "addpath('{tools}');"
            "w = which('errpca_pt');"
            "if isempty(w), error('errpca_pt not found on path'); end;"
        ).format(tools=str(tools).replace("'", "''"))
        _run_octave_eval(script)
    except Exception:
        return False

    # Register cleanup for created files only
    def _cleanup_created() -> None:
        for p in created:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    # Pytest doesn't have a built-in session-finalizer here without request; do manual via atexit
    import atexit

    atexit.register(_cleanup_created)

    return True


# ----------------------------
# Tests: Python-only correctness
# ----------------------------


def test_sparse_reconstruction_error_structure_and_values() -> None:
    case = _mk_sparse_case(seed=999)
    X: sp.csr_matrix = case["X"]
    A: np.ndarray = case["A"]
    S: np.ndarray = case["S"]

    for num_cpu in (1, 2, 4):
        err = sparse_reconstruction_error(X, A, S, num_cpu=num_cpu)

        assert sp.isspmatrix_csr(err)
        np.testing.assert_array_equal(err.indptr, X.indptr)
        np.testing.assert_array_equal(err.indices, X.indices)
        assert err.nnz == X.nnz

        expected = np.empty_like(X.data, dtype=np.float64)
        for r in range(X.shape[0]):
            start, end = X.indptr[r], X.indptr[r + 1]
            for p in range(start, end):
                c = X.indices[p]
                expected[p] = X.data[p] - float(A[r, :] @ S[:, c])

        np.testing.assert_allclose(err.data, expected, rtol=1e-12, atol=1e-12)


def test_python_compute_rms_dense_invariants() -> None:
    case = _mk_dense_case(seed=7)
    cfg = RmsConfig(n_observed=case["ndata"], num_cpu=1)

    rms, err = compute_rms(case["X"], case["A"], case["S"], case["M"], cfg)

    expected_err = (case["X"] - case["A"] @ case["S"]) * case["M"]
    np.testing.assert_allclose(err, expected_err, rtol=0.0, atol=0.0)

    expected_rms = float(np.sqrt(np.sum(expected_err**2) / case["ndata"]))
    np.testing.assert_allclose(rms, expected_rms, rtol=0.0, atol=0.0)


def test_python_compute_rms_sparse_invariants() -> None:
    case = _mk_sparse_case(seed=8)
    cfg = RmsConfig(n_observed=case["ndata"], num_cpu=2, validate_sparse_mask=True)

    rms, err = compute_rms(case["X"], case["A"], case["S"], case["M"], cfg)

    assert sp.isspmatrix_csr(err)
    np.testing.assert_array_equal(err.indptr, case["X"].indptr)
    np.testing.assert_array_equal(err.indices, case["X"].indices)
    assert err.nnz == case["X"].nnz

    expected_rms = float(np.sqrt(np.sum(err.data**2) / case["ndata"]))
    np.testing.assert_allclose(rms, expected_rms, rtol=0.0, atol=0.0)


def test_compute_rms_dense_raises_on_zero_observed_mask() -> None:
    rng = np.random.default_rng(123)
    X = rng.standard_normal((3, 4))
    M = np.zeros_like(X)
    A = rng.standard_normal((3, 2))
    S = rng.standard_normal((2, 4))
    cfg = RmsConfig(n_observed=None, num_cpu=1)

    with pytest.raises(ValueError, match=r"n_obs > 0"):
        compute_rms(X, A, S, M, cfg)


def test_compute_rms_dense_shape_and_mask_validation() -> None:
    rng = np.random.default_rng(321)
    X = rng.standard_normal((3, 4))
    M = np.ones_like(X)
    A = rng.standard_normal((2, 2))  # wrong rows
    S = rng.standard_normal((2, 4))
    cfg = RmsConfig(n_observed=None, num_cpu=1)

    with pytest.raises(ValueError, match=r"loadings has 2 rows"):
        compute_rms(X, A, S, M, cfg)

    A_ok = rng.standard_normal((3, 2))
    S_bad_cols = rng.standard_normal((2, 3))
    with pytest.raises(ValueError, match=r"scores has 3 columns"):
        compute_rms(X, A_ok, S_bad_cols, M, cfg)

    S_bad_latent = rng.standard_normal((3, 4))
    with pytest.raises(ValueError, match=r"Incompatible latent dims"):
        compute_rms(X, A_ok, S_bad_latent, M, cfg)

    cfg_mismatch = RmsConfig(n_observed=5, num_cpu=1)
    with pytest.raises(ValueError, match=r"n_observed mismatch"):
        compute_rms(X, A_ok, S, M, cfg_mismatch)


def test_compute_rms_sparse_validation_errors() -> None:
    rng = np.random.default_rng(999)
    X_dense = (rng.random((4, 5)) > 0.6).astype(float)
    X = sp.csr_matrix(X_dense)
    A = rng.standard_normal((4, 3))
    S = rng.standard_normal((3, 5))

    cfg_mismatch = RmsConfig(n_observed=X.nnz + 1, num_cpu=1)
    with pytest.raises(ValueError, match=r"n_observed mismatch for sparse"):
        compute_rms(X, A, S, None, cfg_mismatch)

    # Mask structure mismatch when validate_sparse_mask=True
    bad_mask = np.ones_like(X_dense)
    bad_mask[0, 0] = 0.0  # alters structure
    cfg = RmsConfig(n_observed=None, num_cpu=1, validate_sparse_mask=True)
    with pytest.raises(ValueError, match=r"sparsity pattern"):
        compute_rms(X, A, S, bad_mask, cfg)


def test_sparse_reconstruction_error_validates_shapes() -> None:
    rng = np.random.default_rng(42)
    X_dense = (rng.random((3, 4)) > 0.5).astype(float)
    X = sp.csr_matrix(X_dense)
    A = rng.standard_normal((2, 3))
    S = rng.standard_normal((3, 4))

    with pytest.raises(ValueError, match=r"loadings has 2 rows"):
        sparse_reconstruction_error(X, A, S)

    A_ok = rng.standard_normal((3, 3))
    S_bad = rng.standard_normal((3, 3))
    with pytest.raises(ValueError, match=r"scores has 3 columns"):
        sparse_reconstruction_error(X, A_ok, S_bad)

    S_bad_latent = rng.standard_normal((2, 4))
    with pytest.raises(ValueError, match=r"Incompatible latent dims"):
        sparse_reconstruction_error(X, A_ok, S_bad_latent)


# ----------------------------
# Tests: Octave regression
# ----------------------------


def test_octave_regression_dense_compute_rms(tmp_path: Path) -> None:
    case = _mk_dense_case(seed=42)

    cfg = RmsConfig(n_observed=case["ndata"], num_cpu=1)
    py_rms, py_err = compute_rms(case["X"], case["A"], case["S"], case["M"], cfg)

    mat_in = tmp_path / "in_dense.mat"
    mat_out = tmp_path / "out_dense.mat"
    savemat(mat_in, {"X": case["X"], "A": case["A"], "S": case["S"], "M": case["M"]})

    _run_octave_compute_rms_dense(mat_in, mat_out, num_cpu=1)
    oc_rms, oc_err = _load_octave_outputs(mat_out)
    oc_err = np.asarray(oc_err, dtype=np.float64)

    np.testing.assert_allclose(py_rms, oc_rms, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(py_err, oc_err, rtol=1e-12, atol=1e-12)


def test_octave_regression_sparse_compute_rms_portable(tmp_path: Path) -> None:
    """
    Always-on sparse regression without needing errpca_pt MEX.
    """
    case = _mk_sparse_case(seed=123)

    cfg = RmsConfig(n_observed=case["ndata"], num_cpu=2, validate_sparse_mask=True)
    py_rms, py_err = compute_rms(case["X"], case["A"], case["S"], case["M"], cfg)

    mat_in = tmp_path / "in_sparse.mat"
    mat_out = tmp_path / "out_sparse.mat"
    savemat(mat_in, {"X": case["X"], "A": case["A"], "S": case["S"]})

    _run_octave_compute_rms_sparse_portable(mat_in, mat_out, num_cpu=2)
    oc_rms, oc_err = _load_octave_outputs(mat_out)
    oc_err = np.asarray(oc_err, dtype=np.float64)

    np.testing.assert_allclose(py_rms, oc_rms, rtol=1e-12, atol=1e-12)

    rows, cols = case["X"].nonzero()
    py_vals = np.asarray(py_err[rows, cols]).reshape(-1)
    oc_vals = oc_err[rows, cols].reshape(-1)
    np.testing.assert_allclose(py_vals, oc_vals, rtol=1e-12, atol=1e-12)


def test_octave_regression_sparse_true_branch_optional(
    tmp_path: Path,
    octave_errpca_mex_in_tools: bool,
) -> None:
    """
    Optional: true sparse-path regression through Octave compute_rms.m issparse branch.
    Requires that the fixture successfully compiled errpca_pt into tools/.
    """
    if not octave_errpca_mex_in_tools:
        pytest.skip(
            "Could not compile/use errpca_pt MEX in tools/; skipping true sparse-branch regression."
        )

    case = _mk_sparse_case(seed=321)

    cfg = RmsConfig(n_observed=case["ndata"], num_cpu=2, validate_sparse_mask=True)
    py_rms, py_err = compute_rms(case["X"], case["A"], case["S"], case["M"], cfg)

    mat_in = tmp_path / "in_sparse_true.mat"
    mat_out = tmp_path / "out_sparse_true.mat"
    # Pass sparse X; compute_rms.m sparse branch will call errpca_pt
    savemat(mat_in, {"X": case["X"], "A": case["A"], "S": case["S"], "M": case["M"]})

    _run_octave_compute_rms_sparse_true(mat_in, mat_out, num_cpu=2)
    oc_rms, oc_err = _load_octave_outputs(mat_out)

    np.testing.assert_allclose(py_rms, oc_rms, rtol=1e-12, atol=1e-12)

    # Compare residuals on observed coords; Octave likely returns sparse here
    if sp.issparse(oc_err):
        oc_err = oc_err.tocsr()
        rows, cols = case["X"].nonzero()
        py_vals = np.asarray(py_err[rows, cols]).reshape(-1)
        oc_vals = np.asarray(oc_err[rows, cols]).reshape(-1)
        np.testing.assert_allclose(py_vals, oc_vals, rtol=1e-12, atol=1e-12)
    else:
        oc_err = np.asarray(oc_err, dtype=np.float64)
        rows, cols = case["X"].nonzero()
        py_vals = np.asarray(py_err[rows, cols]).reshape(-1)
        oc_vals = oc_err[rows, cols].reshape(-1)
        np.testing.assert_allclose(py_vals, oc_vals, rtol=1e-12, atol=1e-12)
