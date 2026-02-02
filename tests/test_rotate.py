"""Tests for the rotate_to_pca helper in vbpca_py._rotate.

This suite includes:
- Robust MATLAB fixture regression test (basis-invariant + Procrustes alignment)
- Shape validation / error handling tests
- Property tests (diagonalization tendency, edge cases)

Important note about eigenvector ambiguity:
------------------------------------------
The RotateToPCA transform is defined via eigendecompositions. When eigenvalues
are close (or equal), eigenvectors are not unique, and different LAPACK builds
(MATLAB vs NumPy) can return different orthonormal bases within the same
eigenspace. This can manifest as a general orthogonal mixing of latent
dimensions, not just sign flips or permutations.

Therefore, fixture comparisons:
- check data-space invariance (A @ S + Mu consistency) directly, and
- compare basis-dependent outputs (A, S) only after aligning the latent
  basis via an orthogonal Procrustes solve.

Even after Procrustes, covariance comparisons in latent space can be brittle:
Procrustes Q is fit from A, and in nearly-degenerate subspaces A does not
strongly constrain Q; covariances transform quadratically and can amplify small
Q differences.

For Av and Sv we therefore compare basis-invariant data-space quantities:
    A @ Av[i] @ A.T   and   A @ Sv[j] @ A.T
using relative Frobenius norm criteria.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.io import loadmat
from vbpca_py._rotate import RotateParams, _build_cov_s, rotate_to_pca

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def _fixture_path(name: str) -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.joinpath("data").joinpath(name)


def _as_col(x: np.ndarray) -> np.ndarray:
    """Ensure x is (n, 1)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr.T
    return arr


def _mat_cell_to_list_of_arrays(obj: Any) -> list[np.ndarray]:
    """Convert a MATLAB cell array loaded by scipy.io.loadmat into list[np.ndarray].

    Handles:
    - empty cell -> []
    - object arrays of shape (n,) or (1,n) or (n,1)
    """
    if obj is None:
        return []

    # scipy loads MATLAB cell arrays as dtype=object ndarrays
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        flat = obj.ravel()
        out: list[np.ndarray] = []
        for item in flat:
            out.append(np.asarray(item, dtype=float))
        return out

    if isinstance(obj, (list, tuple)):
        return [np.asarray(x, dtype=float) for x in obj]

    return [np.asarray(obj, dtype=float)]


def _orthogonal_procrustes_q(a_src: np.ndarray, a_tgt: np.ndarray) -> np.ndarray:
    """Compute best orthogonal Q minimizing ||a_src Q - a_tgt||_F."""
    m = a_src.T @ a_tgt
    u, _s, vt = np.linalg.svd(m, full_matrices=False)
    q = u @ vt
    # det(Q)=+1 is not enforced; reflections are acceptable across LAPACKs.
    return q


def _apply_q_to_outputs(
    *,
    q: np.ndarray,
    A: np.ndarray,
    S: np.ndarray,
    Av: list[np.ndarray] | None,
    Sv: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray] | None, list[np.ndarray]]:
    """Apply latent orthogonal transform Q consistently.

    Align such that:
      A_aligned = A @ Q
      S_aligned = Q.T @ S
      Av_aligned[i] = Q.T @ Av[i] @ Q
      Sv_aligned[j] = Q.T @ Sv[j] @ Q
    """
    a_al = A @ q
    s_al = q.T @ S

    av_al: list[np.ndarray] | None
    if Av is None:
        av_al = None
    else:
        av_al = [q.T @ np.asarray(av_i, dtype=float) @ q for av_i in Av]

    sv_al = [q.T @ np.asarray(sv_j, dtype=float) @ q for sv_j in Sv]
    return a_al, s_al, av_al, sv_al


def _close(a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> bool:
    return np.allclose(a, b, atol=tol, rtol=tol)


def _rel_frob_err(x: np.ndarray, y: np.ndarray) -> float:
    """Relative Frobenius error ||x-y||_F / max(||y||_F, eps)."""
    num = np.linalg.norm(x - y, ord="fro")
    den = max(np.linalg.norm(y, ord="fro"), 1e-12)
    return float(num / den)


def _data_space_from_av(A: np.ndarray, Av_i: np.ndarray) -> np.ndarray:
    """Basis-invariant data-space quantity for a latent covariance Av[i]."""
    return A @ Av_i @ A.T


def _data_space_from_sv(A: np.ndarray, Sv_j: np.ndarray) -> np.ndarray:
    """Basis-invariant data-space quantity for a latent covariance Sv[j]."""
    return A @ Sv_j @ A.T


# ----------------------------------------------------------------------
# MATLAB fixture regression test
# ----------------------------------------------------------------------


def test_rotate_to_pca_matches_matlab_fixture() -> None:
    """rotate_to_pca should match MATLAB/Octave RotateToPCA.m on a realistic state.

    We compare in three layers:
    (1) Python spec: strict data-space invariance.
    (2) Cross-impl: Python vs MATLAB reconstruction (basis-invariant).
    (3) Basis-dependent outputs:
        - Compare A and S after Procrustes alignment using rel Frobenius.
        - Compare Av and Sv via basis-invariant data-space quantities
          (A @ Av @ A.T and A @ Sv @ A.T), avoiding latent-basis brittleness.
    """
    fixture = _fixture_path("legacy_rotate_to_pca.mat")
    if not fixture.exists():
        pytest.skip("MATLAB rotate_to_pca fixture not available")

    mat = loadmat(fixture, squeeze_me=True, struct_as_record=False)

    # Inputs (pre-rotation)
    A0 = np.asarray(mat["A0"], dtype=float)
    S0 = np.asarray(mat["S0"], dtype=float)
    Mu0 = _as_col(np.asarray(mat["Mu0"], dtype=float))

    Av0 = _mat_cell_to_list_of_arrays(mat.get("Av0", None))
    Sv0 = _mat_cell_to_list_of_arrays(mat.get("Sv0", None))

    Isv0 = mat.get("Isv0", None)
    if Isv0 is None:
        isv_py = None
    else:
        isv_arr = np.asarray(Isv0).astype(int).ravel()
        isv_py = None if isv_arr.size == 0 else isv_arr

    update_bias = bool(np.asarray(mat["update_bias"]).item())

    # MATLAB outputs (post-rotation)
    dMu_mat = _as_col(np.asarray(mat["dMu"], dtype=float))
    A1_mat = np.asarray(mat["A1"], dtype=float)
    S1_mat = np.asarray(mat["S1"], dtype=float)
    Av1_mat = _mat_cell_to_list_of_arrays(mat.get("Av1", None))
    Sv1_mat = _mat_cell_to_list_of_arrays(mat.get("Sv1", None))

    # Run Python on copies
    A_py = A0.copy()
    S_py = S0.copy()
    Av_py = [a.copy() for a in Av0] if len(Av0) > 0 else []
    Sv_py = [s.copy() for s in Sv0]

    params = RotateParams(
        loading_covariances=Av_py if len(Av_py) > 0 else [],
        score_covariances=Sv_py,
        isv=isv_py,
        obscombj=None,
        update_bias=update_bias,
    )

    dMu_py, A1_py, Av1_py, S1_py, Sv1_py = rotate_to_pca(A_py, S_py, params)

    # ------------------------------------------------------------------
    # (1) Enforce strict data-space invariance for Python
    # ------------------------------------------------------------------
    X0 = A0 @ S0 + Mu0
    if update_bias:
        X1_py = A1_py @ S1_py + (Mu0 + dMu_py)
        X1_mat = A1_mat @ S1_mat + (Mu0 + dMu_mat)
    else:
        X1_py = A1_py @ S1_py + Mu0
        X1_mat = A1_mat @ S1_mat + Mu0

    assert_allclose(X1_py, X0, rtol=1e-10, atol=1e-12)

    # ------------------------------------------------------------------
    # (2) Compare Python vs MATLAB in data space (basis-invariant)
    # ------------------------------------------------------------------
    rel_err = _rel_frob_err(X1_py, X1_mat)
    assert rel_err < 2e-2, f"Relative Frobenius error too large: {rel_err:.3e}"

    # dMu is basis-invariant; it should still be quite close.
    assert_allclose(dMu_py, dMu_mat, rtol=1e-8, atol=1e-10)

    # ------------------------------------------------------------------
    # (3) Basis-dependent outputs: align via Procrustes on A
    # ------------------------------------------------------------------
    q = _orthogonal_procrustes_q(A1_py, A1_mat)
    A1_py_al, S1_py_al, _Av1_py_al, _Sv1_py_al = _apply_q_to_outputs(
        q=q,
        A=A1_py,
        S=S1_py,
        Av=Av1_py if Av1_py is not None else None,
        Sv=Sv1_py,
    )

    # A and S after Procrustes: norm-based comparisons.
    tol_basis = 2e-2

    err_A = _rel_frob_err(A1_py_al, A1_mat)
    assert err_A < tol_basis, f"A mismatch too large: rel_frob_err={err_A:.3e}"

    err_S = _rel_frob_err(S1_py_al, S1_mat)
    assert err_S < tol_basis, f"S mismatch too large: rel_frob_err={err_S:.3e}"

    # ------------------------------------------------------------------
    # (3b) Covariances: compare basis-invariant data-space quantities
    # ------------------------------------------------------------------
    # Covariances transform quadratically under latent rotations; comparing them
    # in latent space after a Q fit from A can be brittle. Instead compare:
    #   G_i = A @ Av[i] @ A.T   and   H_j = A @ Sv[j] @ A.T
    #
    # Use the aligned A for Python (A1_py_al) and the MATLAB A (A1_mat).
    tol_cov = 6e-2

    if len(Av1_mat) > 0:
        # Python should have Av list if fixture has it. It can be [] or None depending
        # on how RotateParams is populated; handle both safely.
        assert Av1_py is not None
        assert len(Av1_py) == len(Av1_mat)
        for i in range(len(Av1_mat)):
            g_py = _data_space_from_av(A1_py_al, np.asarray(Av1_py[i], dtype=float))
            g_mat = _data_space_from_av(A1_mat, np.asarray(Av1_mat[i], dtype=float))
            err = _rel_frob_err(g_py, g_mat)
            assert err < tol_cov, f"data-space Av[{i}] mismatch: rel_frob_err={err:.3e}"

    assert len(Sv1_py) == len(Sv1_mat)
    for j in range(len(Sv1_mat)):
        h_py = _data_space_from_sv(A1_py_al, np.asarray(Sv1_py[j], dtype=float))
        h_mat = _data_space_from_sv(A1_mat, np.asarray(Sv1_mat[j], dtype=float))
        err = _rel_frob_err(h_py, h_mat)
        assert err < tol_cov, f"data-space Sv[{j}] mismatch: rel_frob_err={err:.3e}"


# ---------------------------------------------------------------------------
# Still-relevant behavior / property tests (no MATLAB dependency)
# ---------------------------------------------------------------------------


def test_rotate_update_bias_true_centers_scores() -> None:
    """When update_bias=True, S rows should be approximately centered."""
    rng = np.random.default_rng(4)

    n_features = 6
    n_components = 2
    n_samples = 8

    a0 = rng.standard_normal((n_features, n_components))
    s0 = rng.standard_normal((n_components, n_samples))
    sv0 = [np.eye(n_components) for _ in range(n_samples)]

    a_new = a0.copy()
    s_new = s0.copy()
    sv_new = [sv.copy() for sv in sv0]

    params_new = RotateParams(
        loading_covariances=None,
        score_covariances=sv_new,
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    d_mu_new, a_new, _av_new, s_new, sv_new = rotate_to_pca(a_new, s_new, params_new)

    mean_after = np.mean(s_new, axis=1, keepdims=True)

    assert np.allclose(mean_after, 0.0, atol=1e-10)
    assert not np.allclose(a_new, a0)
    assert not np.allclose(s_new, s0)
    assert not np.allclose(d_mu_new, 0.0)


def test_rotate_update_bias_false_does_not_center_scores() -> None:
    """When update_bias=False, S should not be explicitly centered to zero."""
    rng = np.random.default_rng(3)

    n_features = 4
    n_components = 3
    n_samples = 5

    a0 = rng.standard_normal((n_features, n_components))
    s0 = rng.standard_normal((n_components, n_samples))
    sv0 = [np.eye(n_components) for _ in range(n_samples)]

    mean_before = np.mean(s0, axis=1, keepdims=True)

    a_new = a0.copy()
    s_new = s0.copy()
    sv_new = [sv.copy() for sv in sv0]

    params_new = RotateParams(
        loading_covariances=None,
        score_covariances=sv_new,
        isv=None,
        obscombj=None,
        update_bias=False,
    )
    d_mu_new, a_new, _av_new, s_new, sv_new = rotate_to_pca(a_new, s_new, params_new)

    mean_after = np.mean(s_new, axis=1, keepdims=True)

    assert np.allclose(d_mu_new, 0.0)
    assert not np.allclose(a_new, a0)
    assert not np.allclose(mean_after, 0.0, atol=1e-8)
    assert not _close(mean_before, mean_after)


def test_rotate_rank_deficient_cov_s_smoke() -> None:
    """Rotation should behave when cov_s has near-zero eigenvalues."""
    rng = np.random.default_rng(10)

    n_features = 5
    n_components = 3
    n_samples = 4

    a0 = rng.standard_normal((n_features, n_components))

    # S is rank 1
    s0 = np.zeros((n_components, n_samples))
    s0[0, :] = rng.standard_normal(n_samples)

    sv0 = [np.eye(n_components) for _ in range(n_samples)]

    a_new = a0.copy()
    s_new = s0.copy()
    sv_new = [sv.copy() for sv in sv0]

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv_new,
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    d_mu, a_new, _, s_new, sv_new = rotate_to_pca(a_new, s_new, params)

    assert d_mu.shape == (n_features, 1)
    assert np.all(np.isfinite(a_new))
    assert np.all(np.isfinite(s_new))
    for sv in sv_new:
        assert np.all(np.isfinite(sv))


def test_rotate_makes_cov_s_more_diagonal() -> None:
    """After rotation, cov(S) should be at least as diagonal as before."""
    rng = np.random.default_rng(21)

    n_features = 6
    n_components = 3
    n_samples = 10

    a0 = rng.standard_normal((n_features, n_components))
    s0 = rng.standard_normal((n_components, n_samples))

    sv0 = []
    for _ in range(n_samples):
        m = rng.standard_normal((n_components, n_components))
        sv0.append(m @ m.T + 0.1 * np.eye(n_components))

    s_centered = s0 - np.mean(s0, axis=1, keepdims=True)
    isv = np.array([], dtype=int)
    cov_before = _build_cov_s(s_centered, [sv.copy() for sv in sv0], isv, None)

    off_before = cov_before - np.diag(np.diag(cov_before))
    off_before_norm = np.linalg.norm(off_before)

    a_new = a0.copy()
    s_new = s0.copy()
    sv_new = [sv.copy() for sv in sv0]

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv_new,
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    _d_mu, a_new, _av_new, s_new, sv_new = rotate_to_pca(a_new, s_new, params)

    cov_after = _build_cov_s(s_new, sv_new, isv, None)
    off_after = cov_after - np.diag(np.diag(cov_after))
    off_after_norm = np.linalg.norm(off_after)

    assert off_after_norm <= off_before_norm + 1e-8


def test_rotate_single_component_edge_case() -> None:
    """Rotation should behave sensibly with a single latent component."""
    rng = np.random.default_rng(22)

    n_features = 5
    n_components = 1
    n_samples = 7

    a0 = rng.standard_normal((n_features, n_components))
    s0 = rng.standard_normal((n_components, n_samples))

    sv0 = [np.eye(n_components) for _ in range(n_samples)]
    av0 = [np.array([[1.0]]) for _ in range(n_features)]

    a_new = a0.copy()
    s_new = s0.copy()
    sv_new = [sv.copy() for sv in sv0]
    av_new = [av.copy() for av in av0]

    params = RotateParams(
        loading_covariances=av_new,
        score_covariances=sv_new,
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    d_mu, a_new, av_new, s_new, sv_new = rotate_to_pca(a_new, s_new, params)

    assert d_mu.shape == (n_features, 1)
    assert a_new.shape == (n_features, n_components)
    assert s_new.shape == (n_components, n_samples)

    for av in av_new or []:
        assert av.shape == (n_components, n_components)
        assert av[0, 0] >= 0.0
    for sv in sv_new:
        assert sv.shape == (n_components, n_components)
        assert sv[0, 0] >= 0.0


# ---------------------------------------------------------------------------
# Validation / error-handling tests (kept)
# ---------------------------------------------------------------------------


def test_rotate_errors_on_mismatched_scores_shape() -> None:
    rng = np.random.default_rng(11)

    n_features = 4
    n_components = 3
    n_samples = 5

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components + 1, n_samples))
    sv = [np.eye(n_components) for _ in range(n_samples)]

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    with pytest.raises(
        ValueError, match=r"must have shape \(n_components, n_samples\)"
    ):
        rotate_to_pca(loadings, scores, params)


def test_rotate_errors_on_non_2d_a_or_s() -> None:
    rng = np.random.default_rng(12)

    n_features = 4
    n_components = 2
    n_samples = 3

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))
    sv = [np.eye(n_components) for _ in range(n_samples)]

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    with pytest.raises(ValueError, match=r"loadings \(A\) must be a 2D array"):
        rotate_to_pca(loadings.reshape(-1), scores, params)

    with pytest.raises(ValueError, match=r"scores \(S\) must be a 2D array"):
        rotate_to_pca(loadings, scores.reshape(-1), params)


def test_rotate_errors_on_zero_samples() -> None:
    rng = np.random.default_rng(13)

    n_features = 4
    n_components = 2

    loadings = rng.standard_normal((n_features, n_components))
    scores = np.empty((n_components, 0))
    sv = [np.eye(n_components)]

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    with pytest.raises(ValueError, match=r"n_samples > 0"):
        rotate_to_pca(loadings, scores, params)


def test_rotate_errors_on_av_length_mismatch() -> None:
    rng = np.random.default_rng(14)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))
    sv = [np.eye(n_components) for _ in range(n_samples)]

    av = [np.eye(n_components)]  # wrong length

    params = RotateParams(
        loading_covariances=av,
        score_covariances=sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    with pytest.raises(ValueError, match=r"loading_covariances \(Av\) length"):
        rotate_to_pca(loadings, scores, params)


def test_rotate_errors_on_av_shape_mismatch() -> None:
    rng = np.random.default_rng(15)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))
    sv = [np.eye(n_components) for _ in range(n_samples)]

    av = [np.eye(n_components) for _ in range(n_features)]
    av[1] = np.eye(n_components + 1)  # wrong shape

    params = RotateParams(
        loading_covariances=av,
        score_covariances=sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    with pytest.raises(ValueError, match=r"found shape="):
        rotate_to_pca(loadings, scores, params)


def test_rotate_errors_on_sv_none_or_empty() -> None:
    rng = np.random.default_rng(16)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))

    params_none = RotateParams(
        loading_covariances=None,
        score_covariances=None,  # type: ignore[arg-type]
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    with pytest.raises(ValueError, match=r"score_covariances"):
        rotate_to_pca(loadings, scores, params_none)

    params_empty = RotateParams(
        loading_covariances=None,
        score_covariances=[],
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    with pytest.raises(ValueError, match=r"score_covariances"):
        rotate_to_pca(loadings, scores, params_empty)


def test_rotate_errors_on_sv_shape_mismatch() -> None:
    rng = np.random.default_rng(17)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))

    sv = [np.eye(n_components) for _ in range(n_samples)]
    sv[2] = np.eye(n_components + 1)

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    with pytest.raises(ValueError, match=r"found shape="):
        rotate_to_pca(loadings, scores, params)


def test_rotate_pattern_mode_requires_obscombj() -> None:
    rng = np.random.default_rng(18)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))

    sv = [np.eye(n_components) for _ in range(2)]  # pretend per-pattern
    isv = np.array([0, 1, 0, 1], dtype=int)

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=isv,
        obscombj=None,
        update_bias=True,
    )

    with pytest.raises(ValueError, match=r"obscombj must be provided"):
        rotate_to_pca(loadings, scores, params)


def test_rotate_pattern_mode_sv_len_mismatch() -> None:
    rng = np.random.default_rng(19)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))

    obscombj = [[0, 2], [1, 3]]
    isv = np.array([0, 1, 0, 1], dtype=int)

    sv = [np.eye(n_components)]  # wrong length for pattern mode

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=isv,
        obscombj=obscombj,
        update_bias=True,
    )

    with pytest.raises(ValueError, match=r"length must match len\(obscombj\)"):
        rotate_to_pca(loadings, scores, params)


def test_rotate_pattern_mode_obscombj_coverage_error() -> None:
    rng = np.random.default_rng(20)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))

    obscombj = [[0, 1], [0, 2]]  # duplicate 0, missing 3
    isv = np.array([0, 1, 0, 1], dtype=int)
    sv = [np.eye(n_components) for _ in range(len(obscombj))]

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=isv,
        obscombj=obscombj,
        update_bias=True,
    )

    with pytest.raises(ValueError, match=r"must cover all sample indices"):
        rotate_to_pca(loadings, scores, params)
