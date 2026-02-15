# tests/test_rotate.py
"""
Regression + property tests for vbpca_py._rotate.rotate_to_pca.

Directly calls Octave tools/RotateToPCA.m (no static fixtures).

We compare:
- strict data-space invariance (Python + Octave),
- cross-impl agreement in data space (basis-invariant),
- basis-dependent A/S only after Procrustes alignment,
- *new*: required post-conditions implied by MATLAB algebra:
    (i) covS_final ≈ I
    (ii) covA_final ≈ diagonal with descending diagonal entries
- *new*: Av is tested using a basis-invariant scalar diagnostic:
    s_j^T Av_i s_j   (row-covariance contribution for sample j)
  which is invariant to orthogonal latent mixing when (S, Av) transform together.

We also keep a few pure-Python property/validation tests.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.io import loadmat, savemat

import vbpca_py._rotate as rot
from vbpca_py._rotate import RotateParams, _build_cov_s, rotate_to_pca

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tools_dir() -> Path:
    return _repo_root() / "tools"


def _octave_available() -> bool:
    return shutil.which("octave") is not None


def _as_col(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr.T
    return arr


def _orthogonal_procrustes_q(a_src: np.ndarray, a_tgt: np.ndarray) -> np.ndarray:
    m = a_src.T @ a_tgt
    u, _s, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt  # reflection allowed


def _rel_frob_err(x: np.ndarray, y: np.ndarray) -> float:
    num = np.linalg.norm(x - y, ord="fro")
    den = max(np.linalg.norm(y, ord="fro"), 1e-12)
    return float(num / den)


def _mat_cell_to_list(obj: Any) -> list[np.ndarray]:
    if obj is None:
        return []
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return [np.asarray(x, dtype=float) for x in obj.ravel()]
    if isinstance(obj, (list, tuple)):
        return [np.asarray(x, dtype=float) for x in obj]
    return [np.asarray(obj, dtype=float)]


def _to_mat_cell(lst: list[np.ndarray]) -> np.ndarray:
    cell = np.empty((1, len(lst)), dtype=object)
    for i, item in enumerate(lst):
        cell[0, i] = np.asarray(item, dtype=float)
    return cell


def _run_octave_rotatetopca(mat_in: Path, mat_out: Path) -> None:
    tools = _tools_dir()
    addpath = str(tools).replace("\\", "/")
    in_path = str(mat_in).replace("\\", "/")
    out_path = str(mat_out).replace("\\", "/")

    cmd = (
        f"addpath('{addpath}');"
        f"load('{in_path}');"
        "[dMu, A1, Av1, S1, Sv1] = RotateToPCA(A0, Av0, S0, Sv0, Isv, obscombj, update_bias);"
        f"save('-mat','{out_path}','dMu','A1','Av1','S1','Sv1');"
    )

    subprocess.run(
        ["octave", "--quiet", "--eval", cmd],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _cov_a_from_outputs(A: np.ndarray, Av: list[np.ndarray]) -> np.ndarray:
    """CovA = (A' A + sum_i Av[i]) / n_features (n1 in MATLAB)."""
    n_features = A.shape[0]
    cov_a = A.T @ A
    for av_i in Av:
        cov_a = cov_a + np.asarray(av_i, dtype=float)
    cov_a = cov_a / float(n_features)
    cov_a = 0.5 * (cov_a + cov_a.T)
    return cov_a


def _diag_offdiag_norms(M: np.ndarray) -> tuple[float, float]:
    """Return (||diag||_F, ||offdiag||_F)."""
    d = np.diag(np.diag(M))
    off = M - d
    return float(np.linalg.norm(d, ord="fro")), float(np.linalg.norm(off, ord="fro"))


def _av_scalar_contribs(
    S: np.ndarray, Av: list[np.ndarray], js: list[int]
) -> np.ndarray:
    """Return matrix C where C[i, t] = s_j^T Av[i] s_j for j=js[t]."""
    k, n = S.shape
    assert all(0 <= j < n for j in js)
    out = np.empty((len(Av), len(js)), dtype=float)
    for t, j in enumerate(js):
        s = S[:, j].reshape(k, 1)
        for i, av_i in enumerate(Av):
            a = np.asarray(av_i, dtype=float)
            out[i, t] = float((s.T @ a @ s)[0, 0])
    return out


def _sv_scalar_contribs(
    A: np.ndarray, Sv: list[np.ndarray], ixs: list[int]
) -> np.ndarray:
    """
    Optional diagnostic: for a few feature indices i, compare a_i^T Sv[j] a_i.
    This is invariant under consistent latent rotations too.
    """
    n_features, k = A.shape
    assert all(0 <= i < n_features for i in ixs)
    n_cov = len(Sv)
    out = np.empty((len(ixs), n_cov), dtype=float)
    for p, i in enumerate(ixs):
        a_i = A[i, :].reshape(k, 1)
        for j, sv_j in enumerate(Sv):
            s = np.asarray(sv_j, dtype=float)
            out[p, j] = float((a_i.T @ s @ a_i)[0, 0])
    return out


# --------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Case:
    A0: np.ndarray
    S0: np.ndarray
    Mu0: np.ndarray
    Av0: list[np.ndarray]
    Sv0: list[np.ndarray]
    update_bias: bool
    Isv: np.ndarray
    obscombj: list[list[int]]


def _mk_case_per_sample(seed: int = 0) -> _Case:
    rng = np.random.default_rng(seed)
    n_features, k, n = 7, 3, 11

    A0 = rng.standard_normal((n_features, k))
    S0 = rng.standard_normal((k, n))
    Mu0 = rng.standard_normal((n_features, 1))

    Av0 = []
    for _ in range(n_features):
        m = rng.standard_normal((k, k))
        Av0.append(m @ m.T + 0.05 * np.eye(k))

    Sv0 = []
    for _ in range(n):
        m = rng.standard_normal((k, k))
        Sv0.append(m @ m.T + 0.05 * np.eye(k))

    return _Case(
        A0=A0,
        S0=S0,
        Mu0=Mu0,
        Av0=Av0,
        Sv0=Sv0,
        update_bias=True,
        Isv=np.array([], dtype=int),
        obscombj=[],
    )


def _mk_case_pattern(seed: int = 1) -> _Case:
    rng = np.random.default_rng(seed)
    n_features, k, n = 6, 3, 10

    A0 = rng.standard_normal((n_features, k))
    S0 = rng.standard_normal((k, n))
    Mu0 = rng.standard_normal((n_features, 1))

    Av0 = []
    for _ in range(n_features):
        m = rng.standard_normal((k, k))
        Av0.append(m @ m.T + 0.05 * np.eye(k))

    # 3 patterns covering all samples
    obscombj = [list(range(4)), list(range(4, 7)), list(range(7, 10))]
    Isv = np.zeros(n, dtype=int)
    for g, cols in enumerate(obscombj):
        for j in cols:
            Isv[j] = g

    # Sv is per-pattern (len=3)
    Sv0 = []
    for _ in range(len(obscombj)):
        m = rng.standard_normal((k, k))
        Sv0.append(m @ m.T + 0.05 * np.eye(k))

    return _Case(
        A0=A0,
        S0=S0,
        Mu0=Mu0,
        Av0=Av0,
        Sv0=Sv0,
        update_bias=True,
        Isv=Isv,
        obscombj=obscombj,
    )


def _save_case_for_octave(case: _Case, mat_in: Path) -> None:
    # MATLAB-style 1-based indices in obscombj; RotateToPCA only uses lengths,
    # but we still pass MATLAB-convention indices for sanity.
    obs_cell = np.empty((1, len(case.obscombj)), dtype=object)
    for i, cols in enumerate(case.obscombj):
        obs_cell[0, i] = (np.asarray(cols, dtype=int) + 1).reshape(-1, 1)

    payload: dict[str, Any] = {
        "A0": np.asarray(case.A0, dtype=float),
        "S0": np.asarray(case.S0, dtype=float),
        "Mu0": np.asarray(case.Mu0, dtype=float),
        "update_bias": float(1.0 if case.update_bias else 0.0),
        "Isv": np.asarray(case.Isv, dtype=float).reshape(
            -1, 1
        ),  # MATLAB loads as double
        "obscombj": obs_cell,
        "Av0": _to_mat_cell(case.Av0) if case.Av0 else np.empty((0, 0), dtype=object),
        "Sv0": _to_mat_cell(case.Sv0),
    }
    savemat(mat_in, payload)


def _load_octave_out(
    mat_out: Path,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray]]:
    mat = loadmat(mat_out, squeeze_me=True, struct_as_record=False)
    dMu = _as_col(np.asarray(mat["dMu"], dtype=float))
    A1 = np.asarray(mat["A1"], dtype=float)
    S1 = np.asarray(mat["S1"], dtype=float)
    Av1 = _mat_cell_to_list(mat.get("Av1", None))
    Sv1 = _mat_cell_to_list(mat.get("Sv1", None))
    return dMu, A1, Av1, S1, Sv1


def _python_rotate(
    case: _Case,
) -> tuple[
    np.ndarray, np.ndarray, list[np.ndarray] | None, np.ndarray, list[np.ndarray]
]:
    A = case.A0.copy()
    S = case.S0.copy()
    Av = [x.copy() for x in case.Av0] if case.Av0 else []
    Sv = [x.copy() for x in case.Sv0]

    params = RotateParams(
        loading_covariances=Av or [],
        score_covariances=Sv,
        isv=None if case.Isv.size == 0 else case.Isv.astype(int),
        obscombj=None if case.Isv.size == 0 else case.obscombj,  # 0-based for Python
        update_bias=case.update_bias,
    )
    return rotate_to_pca(A, S, params)


# --------------------------------------------------------------------------------------
# Octave-backed regression tests (+ new invariants / diagnostics)
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not _octave_available(), reason="Octave not available on PATH.")
@pytest.mark.parametrize("case_factory", [_mk_case_per_sample, _mk_case_pattern])
def test_rotate_to_pca_matches_octave(tmp_path: Path, case_factory) -> None:
    case = case_factory()

    # --- Python ---
    dMu_py, A1_py, Av1_py, S1_py, Sv1_py = _python_rotate(case)

    # Strict Python data-space invariance
    X0 = case.A0 @ case.S0 + case.Mu0
    X1_py = (
        A1_py @ S1_py + (case.Mu0 + dMu_py)
        if case.update_bias
        else A1_py @ S1_py + case.Mu0
    )
    assert_allclose(X1_py, X0, rtol=1e-10, atol=1e-12)

    # --- Octave ---
    mat_in = tmp_path / "in.mat"
    mat_out = tmp_path / "out.mat"
    _save_case_for_octave(case, mat_in)
    _run_octave_rotatetopca(mat_in, mat_out)
    dMu_oc, A1_oc, Av1_oc, S1_oc, Sv1_oc = _load_octave_out(mat_out)

    # Strict Octave data-space invariance
    X1_oc = (
        A1_oc @ S1_oc + (case.Mu0 + dMu_oc)
        if case.update_bias
        else A1_oc @ S1_oc + case.Mu0
    )
    assert_allclose(X1_oc, X0, rtol=1e-10, atol=1e-10)

    # Cross-impl match in data space (basis-invariant)
    rel = _rel_frob_err(X1_py, X1_oc)
    assert rel < 1e-8, f"data-space mismatch too large: rel_frob_err={rel:.3e}"

    # dMu is basis-invariant
    assert_allclose(dMu_py, dMu_oc, rtol=1e-8, atol=1e-10)

    # Basis-dependent A/S after Procrustes
    q = _orthogonal_procrustes_q(A1_py, A1_oc)
    A1_py_al = A1_py @ q
    S1_py_al = q.T @ S1_py

    err_A = _rel_frob_err(A1_py_al, A1_oc)
    err_S = _rel_frob_err(S1_py_al, S1_oc)

    assert err_A < 2e-15, f"A mismatch too large after Procrustes: {err_A:.3e}"
    assert err_S < 2e-15, f"S mismatch too large after Procrustes: {err_S:.3e}"

    # ------------------------------------------------------------------
    # NEW: Required post-conditions implied by RotateToPCA algebra
    # ------------------------------------------------------------------

    # covS_final should be ~ I after applying R (in both implementations).
    # Build covS_final using the same logic as MATLAB: use S_final and Sv_final.
    isv_py = np.array([], dtype=int) if case.Isv.size == 0 else case.Isv.astype(int)

    covs_py = _build_cov_s(
        S1_py,
        Sv1_py,
        isv_py if isv_py.size == 0 else isv_py,
        case.obscombj if isv_py.size else None,
    )
    covs_oc = _build_cov_s(
        S1_oc,
        Sv1_oc,
        isv_py if isv_py.size == 0 else isv_py,
        case.obscombj if isv_py.size else None,
    )

    I = np.eye(covs_py.shape[0], dtype=float)
    assert_allclose(covs_py, I, rtol=1e-7, atol=1e-9)
    assert_allclose(covs_oc, I, rtol=1e-7, atol=1e-9)

    # covA_final should be ~ diagonal, with descending diagonal entries
    # covA_final = (A' A + sum_i Av[i]) / n_features
    if len(Av1_oc) > 0:
        assert Av1_py is not None and isinstance(Av1_py, list)
        cova_py = _cov_a_from_outputs(A1_py, [np.asarray(x, float) for x in Av1_py])
        cova_oc = _cov_a_from_outputs(A1_oc, [np.asarray(x, float) for x in Av1_oc])

        _, off_py = _diag_offdiag_norms(cova_py)
        _, off_oc = _diag_offdiag_norms(cova_oc)

        # Diagonalization should be strong (tune if needed)
        assert off_py <= 1e-7 * np.linalg.norm(cova_py, ord="fro") + 1e-10
        assert off_oc <= 1e-7 * np.linalg.norm(cova_oc, ord="fro") + 1e-10

        d_py = np.diag(cova_py)
        d_oc = np.diag(cova_oc)

        # Descending order is an intended MATLAB effect after sorting
        assert np.all(d_py[:-1] >= d_py[1:] - 1e-10)
        assert np.all(d_oc[:-1] >= d_oc[1:] - 1e-10)

        # Cross-impl: diagonal entries should match (allow tiny slack)
        assert_allclose(d_py, d_oc, rtol=5e-6, atol=5e-9)

    # ------------------------------------------------------------------
    # NEW: Av diagnostics that are actually meaningful and basis-invariant
    # ------------------------------------------------------------------

    # Compare scalar contributions s_j^T Av_i s_j for a few samples j.
    if len(Av1_oc) > 0:
        assert Av1_py is not None and isinstance(Av1_py, list)
        js = [0, min(1, S1_py.shape[1] - 1), S1_py.shape[1] // 2, S1_py.shape[1] - 1]
        js = sorted(set(js))

        C_py = _av_scalar_contribs(S1_py, [np.asarray(x, float) for x in Av1_py], js)
        C_oc = _av_scalar_contribs(S1_oc, [np.asarray(x, float) for x in Av1_oc], js)

        # These scalars should match closely across impls even if latent basis differs.
        assert_allclose(C_py, C_oc, rtol=2e-5, atol=2e-8)

    # Optional extra: Sv scalar contributions a_i^T Sv_j a_i (also invariant)
    ixs = [0, min(1, A1_py.shape[0] - 1), A1_py.shape[0] - 1]
    ixs = sorted(set(ixs))
    D_py = _sv_scalar_contribs(A1_py, Sv1_py, ixs)
    D_oc = _sv_scalar_contribs(A1_oc, Sv1_oc, ixs)
    assert_allclose(D_py, D_oc, rtol=2e-5, atol=2e-8)


# --------------------------------------------------------------------------------------
# Pure-Python property tests (kept)
# --------------------------------------------------------------------------------------


def test_rotate_python_data_space_invariance_update_bias_true() -> None:
    rng = np.random.default_rng(123)
    n_features, k, n = 8, 3, 9

    A0 = rng.standard_normal((n_features, k))
    S0 = rng.standard_normal((k, n))
    Mu0 = rng.standard_normal((n_features, 1))

    Sv0 = [np.eye(k) for _ in range(n)]
    Av0 = [np.eye(k) for _ in range(n_features)]

    A = A0.copy()
    S = S0.copy()
    Sv = [x.copy() for x in Sv0]
    Av = [x.copy() for x in Av0]

    params = RotateParams(
        loading_covariances=Av,
        score_covariances=Sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    dMu, A1, _Av1, S1, _Sv1 = rotate_to_pca(A, S, params)

    X0 = A0 @ S0 + Mu0
    X1 = A1 @ S1 + (Mu0 + dMu)
    assert_allclose(X1, X0, rtol=1e-10, atol=1e-12)


def test_rotate_python_data_space_invariance_update_bias_false() -> None:
    rng = np.random.default_rng(321)
    n_features, k, n = 5, 3, 7

    A0 = rng.standard_normal((n_features, k))
    S0 = rng.standard_normal((k, n))
    Mu0 = rng.standard_normal((n_features, 1))

    Sv0 = [np.eye(k) for _ in range(n)]

    A = A0.copy()
    S = S0.copy()
    Sv = [x.copy() for x in Sv0]

    params = RotateParams(
        loading_covariances=None,
        score_covariances=Sv,
        isv=None,
        obscombj=None,
        update_bias=False,
    )
    dMu, A1, _Av1, S1, _Sv1 = rotate_to_pca(A, S, params)

    X0 = A0 @ S0 + Mu0
    X1 = A1 @ S1 + Mu0 + dMu  # dMu should be zeros when update_bias=False
    assert_allclose(dMu, np.zeros_like(dMu), atol=0.0)
    assert_allclose(X1, X0, rtol=1e-10, atol=1e-12)


def test_rotate_errors_on_zero_samples() -> None:
    rng = np.random.default_rng(13)
    n_features, k = 4, 2

    loadings = rng.standard_normal((n_features, k))
    scores = np.empty((k, 0))
    sv = [np.eye(k)]

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    with pytest.raises(ValueError, match=r"n_samples > 0"):
        rotate_to_pca(loadings, scores, params)


def test_rotate_covariances_remain_symmetric_and_finite() -> None:
    """Av and Sv must remain symmetric PSD-like matrices after rotation."""
    rng = np.random.default_rng(333)

    n_features, n_components, n_samples = 6, 3, 8

    A0 = rng.standard_normal((n_features, n_components))
    S0 = rng.standard_normal((n_components, n_samples))

    Av0 = []
    for _ in range(n_features):
        m = rng.standard_normal((n_components, n_components))
        Av0.append(m @ m.T + 0.1 * np.eye(n_components))

    Sv0 = []
    for _ in range(n_samples):
        m = rng.standard_normal((n_components, n_components))
        Sv0.append(m @ m.T + 0.1 * np.eye(n_components))

    A = A0.copy()
    S = S0.copy()
    Av = [x.copy() for x in Av0]
    Sv = [x.copy() for x in Sv0]

    params = RotateParams(
        loading_covariances=Av,
        score_covariances=Sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    _dMu, _A1, Av1, _S1, Sv1 = rotate_to_pca(A, S, params)

    tol_sym = 1e-10

    # Check Av
    assert Av1 is not None
    for i, av in enumerate(Av1):
        av = np.asarray(av, dtype=float)
        assert np.all(np.isfinite(av)), f"Av[{i}] contains non-finite values"
        assert np.allclose(av, av.T, atol=tol_sym), f"Av[{i}] not symmetric"

    # Check Sv
    for j, sv in enumerate(Sv1):
        sv = np.asarray(sv, dtype=float)
        assert np.all(np.isfinite(sv)), f"Sv[{j}] contains non-finite values"
        assert np.allclose(sv, sv.T, atol=tol_sym), f"Sv[{j}] not symmetric"


def test_pattern_case_obscombj_covers_all_samples_exactly_once() -> None:
    """Fixture sanity: obscombj must cover [0..n_samples-1] exactly once."""
    case = _mk_case_pattern(seed=1234)

    all_cols: list[int] = []
    for group in case.obscombj:
        all_cols.extend(group)

    all_cols_arr = np.asarray(all_cols, dtype=int)
    expected = np.arange(case.S0.shape[1], dtype=int)

    assert all_cols_arr.size == expected.size
    assert np.array_equal(np.sort(all_cols_arr), expected)


def test_rotate_rejects_invalid_inputs() -> None:
    rng = np.random.default_rng(2024)
    n_features, k, n = 4, 2, 3

    A = rng.standard_normal((n_features, k))
    S = rng.standard_normal((k, n))
    sv = [np.eye(k) for _ in range(n)]

    bad_av_len = [np.eye(k) for _ in range(n_features - 1)]
    with pytest.raises(ValueError, match=r"length must match"):
        rotate_to_pca(A.copy(), S.copy(), RotateParams(bad_av_len, sv))

    bad_av_shape = [np.eye(k) for _ in range(n_features)]
    bad_av_shape[0] = np.eye(k + 1)
    with pytest.raises(ValueError, match=r"shape=.*index=0"):
        rotate_to_pca(A.copy(), S.copy(), RotateParams(bad_av_shape, sv))

    short_sv = [np.eye(k) for _ in range(n - 1)]
    with pytest.raises(ValueError, match=r"must have length equal to n_samples"):
        rotate_to_pca(A.copy(), S.copy(), RotateParams(None, short_sv))

    bad_sv_shape = [np.eye(k) for _ in range(n)]
    bad_sv_shape[1] = np.eye(k + 1)
    with pytest.raises(ValueError, match=r"shape=.*index=1"):
        rotate_to_pca(A.copy(), S.copy(), RotateParams(None, bad_sv_shape))

    # Pattern-mode: missing obscombj
    isv = np.zeros(n, dtype=int)
    with pytest.raises(ValueError, match=r"obscombj must be provided"):
        rotate_to_pca(
            A.copy(),
            S.copy(),
            RotateParams(None, sv, isv=isv, obscombj=None),
        )

    # Pattern-mode: length mismatch
    obscombj = [list(range(n))]
    with pytest.raises(ValueError, match=r"length must match len\(obscombj\)"):
        rotate_to_pca(
            A.copy(),
            S.copy(),
            RotateParams(None, sv + [np.eye(k)], isv=isv, obscombj=obscombj),
        )

    # Pattern-mode: coverage error / duplicates
    obscombj_bad = [list(range(n - 1)), list(range(1, n))]
    with pytest.raises(ValueError, match=r"cover all sample indices"):
        rotate_to_pca(
            A.copy(),
            S.copy(),
            RotateParams(None, [np.eye(k)] * 2, isv=isv, obscombj=obscombj_bad),
        )

    # Pattern-mode: isv length mismatch
    isv_short = np.zeros(n - 1, dtype=int)
    with pytest.raises(ValueError, match=r"isv must have length equal"):
        rotate_to_pca(
            A.copy(),
            S.copy(),
            RotateParams(None, [np.eye(k)] * 2, isv=isv_short, obscombj=obscombj_bad),
        )


def test_rotate_handles_rank_deficient_covs_without_nan() -> None:
    rng = np.random.default_rng(4242)
    n_features, k, n = 4, 3, 3

    # Make S rank-1 to introduce zero eigenvalues in covS
    s_base = rng.standard_normal(n)
    S0 = np.vstack([s_base, np.zeros_like(s_base), np.zeros_like(s_base)])
    A0 = rng.standard_normal((n_features, k))

    Sv0 = [np.zeros((k, k)) for _ in range(n)]

    dMu, A1, Av1, S1, Sv1 = rotate_to_pca(
        A0.copy(), S0.copy(), RotateParams(None, Sv0, isv=None, obscombj=None)
    )

    assert np.all(np.isfinite(dMu))
    assert np.all(np.isfinite(A1))
    assert Av1 is None
    assert np.all(np.isfinite(S1))
    for sv in Sv1:
        sv_arr = np.asarray(sv, dtype=float)
        assert np.all(np.isfinite(sv_arr))
        assert np.allclose(sv_arr, sv_arr.T, atol=1e-12)


def test_rotate_pattern_mode_python_only() -> None:
    case = _mk_case_pattern(seed=99)

    A = case.A0.copy()
    S = case.S0.copy()
    Av = [x.copy() for x in case.Av0]
    Sv = [x.copy() for x in case.Sv0]

    params = RotateParams(
        loading_covariances=Av,
        score_covariances=Sv,
        isv=case.Isv,
        obscombj=case.obscombj,
        update_bias=case.update_bias,
    )

    dMu, A1, Av1, S1, Sv1 = rotate_to_pca(A, S, params)

    X0 = case.A0 @ case.S0 + case.Mu0
    X1 = A1 @ S1 + (case.Mu0 + dMu)
    assert_allclose(X1, X0, rtol=1e-10, atol=1e-12)

    isv = case.Isv
    covs = _build_cov_s(S1, Sv1, isv, case.obscombj)
    assert_allclose(covs, np.eye(covs.shape[0]), rtol=1e-7, atol=1e-9)

    if Av1:
        cova = _cov_a_from_outputs(A1, Av1)
        d_norm, off_norm = _diag_offdiag_norms(cova)
        assert off_norm <= 1e-7 * np.linalg.norm(cova, ord="fro") + 1e-10
        d = np.diag(cova)
        assert np.all(d[:-1] >= d[1:] - 1e-10)


@pytest.mark.parametrize("av_kind", [None, []])
def test_rotate_handles_absent_av(av_kind: list[np.ndarray] | None) -> None:
    rng = np.random.default_rng(777)
    n_features, k, n = 5, 2, 6

    A0 = rng.standard_normal((n_features, k))
    S0 = rng.standard_normal((k, n))
    Sv0 = [np.eye(k) * 0.2 for _ in range(n)]

    A = A0.copy()
    S = S0.copy()
    Sv = [x.copy() for x in Sv0]

    params = RotateParams(
        loading_covariances=av_kind,
        score_covariances=Sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    dMu, A1, Av1, S1, Sv1 = rotate_to_pca(A, S, params)

    X0 = A0 @ S0
    X1 = A1 @ S1 + dMu  # bias update applies when update_bias=True
    assert_allclose(X1, X0, rtol=1e-10, atol=1e-12)
    assert Av1 is None or len(Av1) == 0
    for sv in Sv1:
        sv_arr = np.asarray(sv, dtype=float)
        assert np.all(np.isfinite(sv_arr))
        assert np.allclose(sv_arr, sv_arr.T, atol=1e-12)


def test_rotate_regression_python_only_fixture() -> None:
    # Deterministic small-case regression guard that runs even without Octave.
    A0 = np.array(
        [
            [-2.221253875745, 0.02599965265],
            [-0.538969020353, -1.129192775482],
            [-2.441866645633, 0.765391403162],
            [-0.759709345383, 0.266996194927],
        ],
        dtype=float,
    )
    S0 = np.array(
        [
            [
                0.701780851885,
                0.292121315819,
                -0.198093083842,
                0.658771263358,
                0.519957431003,
            ],
            [
                0.599011418567,
                -1.65158095341,
                -0.392440699326,
                -0.677316958821,
                2.936010765698,
            ],
        ],
        dtype=float,
    )
    Sv0 = [np.eye(2) * 0.2 for _ in range(S0.shape[1])]

    params = RotateParams(
        loading_covariances=None,
        score_covariances=[s.copy() for s in Sv0],
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    dMu, A1, _Av1, S1, Sv1 = rotate_to_pca(A0.copy(), S0.copy(), params)

    expected_dmu = np.array(
        [[-0.872958840486], [-0.396604060762], [-0.839754305948], [-0.256564877028]],
        dtype=float,
    )
    expected_A1 = np.array(
        [
            [0.365309717115, -1.171205809652],
            [-1.578904758864, -1.071504963959],
            [1.483635844814, -0.77664794818],
            [0.503981802265, -0.22161254337],
        ],
        dtype=float,
    )
    expected_S1 = np.array(
        [
            [
                0.023413824745,
                -1.013210637934,
                0.128250782708,
                -0.712222485487,
                1.573768515968,
            ],
            [
                0.579619568117,
                -0.470693016891,
                -1.072330141602,
                0.296931239223,
                0.666472351153,
            ],
        ],
        dtype=float,
    )
    expected_Sv0 = np.array(
        [[0.194479904408, -0.238069440142], [-0.238069440142, 0.552048750368]],
        dtype=float,
    )

    # Eigenvector sign is not unique; align component signs to the fixture.
    A1_aligned = A1.copy()
    S1_aligned = S1.copy()
    Sv1_aligned = [np.asarray(sv, dtype=float).copy() for sv in Sv1]
    for c in range(A1.shape[1]):
        if float(np.dot(A1_aligned[:, c], expected_A1[:, c])) >= 0.0:
            continue
        A1_aligned[:, c] *= -1.0
        S1_aligned[c, :] *= -1.0
        for idx in range(len(Sv1_aligned)):
            sv_i = Sv1_aligned[idx]
            sv_i[c, :] *= -1.0
            sv_i[:, c] *= -1.0
            Sv1_aligned[idx] = sv_i

    assert_allclose(dMu, expected_dmu, rtol=5e-12, atol=5e-12)
    assert_allclose(A1_aligned, expected_A1, rtol=5e-12, atol=5e-12)
    assert_allclose(S1_aligned, expected_S1, rtol=5e-12, atol=5e-12)
    assert_allclose(Sv1_aligned[0], expected_Sv0, rtol=5e-12, atol=5e-12)


def test_rotate_private_validation_helpers_cover_error_paths() -> None:
    with pytest.raises(ValueError, match=r"index array must be 1-D"):
        rot._as_int_array(np.array([[1, 2]], dtype=int))  # noqa: SLF001

    with pytest.raises(ValueError, match=r"loadings \(A\) must be a 2D"):
        rot._validate_a_s_shapes(np.array([1.0, 2.0]), np.zeros((2, 3)))  # noqa: SLF001

    with pytest.raises(ValueError, match=r"scores \(S\) must be a 2D"):
        rot._validate_a_s_shapes(np.zeros((2, 2)), np.array([1.0, 2.0, 3.0]))  # noqa: SLF001

    with pytest.raises(ValueError, match=r"n_components=2"):
        rot._validate_a_s_shapes(np.zeros((3, 2)), np.zeros((3, 4)))  # noqa: SLF001

    with pytest.raises(ValueError, match=r"must be a non-empty list"):
        rot._validate_sv_shapes(tuple([np.eye(2)]), 2)  # type: ignore[arg-type]  # noqa: SLF001


def test_rotate_build_cov_s_validation_branches() -> None:
    scores = np.zeros((2, 3), dtype=float)
    sv = [np.eye(2) for _ in range(3)]

    with pytest.raises(ValueError, match=r"isv must have length"):
        rot._build_cov_s(scores, sv, np.array([0, 1], dtype=int), None)  # noqa: SLF001

    with pytest.raises(ValueError, match=r"must have length equal to n_samples"):
        rot._build_cov_s(scores, sv[:2], np.array([], dtype=int), None)  # noqa: SLF001

    isv = np.array([0, 0, 1], dtype=int)
    with pytest.raises(ValueError, match=r"obscombj must be provided"):
        rot._build_cov_s(scores, [np.eye(2), np.eye(2)], isv, None)  # noqa: SLF001

    with pytest.raises(ValueError, match=r"length must match len\(obscombj\)"):
        rot._build_cov_s(scores, [np.eye(2)], isv, [[0, 1], [2]])  # noqa: SLF001

    with pytest.raises(ValueError, match=r"cover all sample indices"):
        rot._build_cov_s(  # noqa: SLF001
            scores,
            [np.eye(2), np.eye(2)],
            isv,
            [[0, 1], [1, 2]],
        )


def test_rotate_transform_and_weighted_cov_ext_path() -> None:
    left = np.array([[2.0, 0.0], [0.0, 3.0]])
    right = np.array([[1.0, 0.0], [0.0, 4.0]])

    empty_out = rot._transform_covariances([], left, right)  # noqa: SLF001
    assert empty_out == []

    covs = [np.eye(2), np.array([[2.0, 1.0], [1.0, 2.0]])]
    transformed = rot._transform_covariances(covs, left, right)  # noqa: SLF001
    assert len(transformed) == 2
    assert_allclose(transformed[0], left @ covs[0] @ right)

    base = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=float)
    cov, eigvals, eigvecs = rot._weighted_cov_eigh(  # noqa: SLF001
        base,
        covs,
        np.array([1.0, 0.5], dtype=float),
        normalizer=3.0,
    )
    assert cov.shape == (2, 2)
    assert eigvals.shape == (2,)
    assert eigvecs.shape == (2, 2)
    assert_allclose(cov, cov.T, atol=1e-12)
