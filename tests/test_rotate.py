"""Tests for the rotate_to_pca helper in vbpca_py._rotate."""

from __future__ import annotations

import numpy as np
import pytest

from vbpca_py._rotate import RotateParams, _build_cov_s, rotate_to_pca


def _close(a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True if arrays are numerically close."""
    return np.allclose(a, b, atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# Reference implementation (ported from legacy rotate_to_pca)
# ---------------------------------------------------------------------------


def _legacy_rotate_to_pca(
    a: np.ndarray,
    av: list[np.ndarray] | None,
    s: np.ndarray,
    sv: list[np.ndarray],
    isv: np.ndarray | list[int] | None,
    obscombj: list[list[int]] | None,
    *,
    update_bias: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray] | None,
    np.ndarray,
    list[np.ndarray],
]:
    """Legacy rotate_to_pca behavior, matching the original implementation.

    This is used only in tests as a reference to validate the new
    implementation in vbpca_py._rotate.
    """
    # Convert index structures to Python list-like for len() semantics.
    isv_list = [] if isv is None else list(isv)

    n1 = a.shape[0]
    n2 = s.shape[1]

    if update_bias:
        m_s = np.mean(s, axis=1, keepdims=True)
        d_mu = a @ m_s
        s = s - m_s
    else:
        # In the original code this was a scalar 0; we return a zero vector
        # for comparability.
        d_mu = np.zeros((n1, 1), dtype=a.dtype)

    cov_s = s @ s.T

    if len(isv_list) == 0:
        for j in range(n2):
            cov_s += sv[j]
    else:
        if obscombj is None:
            msg = "obscombj required when isv is non-empty in legacy path."
            raise ValueError(msg)
        nobscomb = len(obscombj)
        for j in range(nobscomb):
            cov_s += len(obscombj[j]) * sv[j]

    cov_s /= n2

    eigvals, v_s = np.linalg.eigh(cov_s)
    d_mat = np.diag(eigvals)
    sqrt_d = np.sqrt(d_mat)
    ra = v_s @ sqrt_d

    a = a @ ra

    cov_a = a.T @ a

    if av:
        for i in range(n1):
            av[i] = ra.T @ av[i] @ ra
            cov_a += av[i]

    cov_a /= n1

    eigvals_a, v_a = np.linalg.eigh(cov_a)
    idx = np.argsort(-eigvals_a)
    eigvals_a = eigvals_a[idx]
    v_a = v_a[:, idx]

    a = a @ v_a

    if av:
        for i in range(n1):
            av[i] = v_a.T @ av[i] @ v_a

    epsilon = 1e-10
    d_diag = np.sqrt(np.diag(d_mat))
    d_inv = np.diag(np.where(d_diag > epsilon, 1.0 / d_diag, 0.0))
    r = v_a.T @ d_inv @ v_s.T

    s = r @ s

    for j in range(len(sv)):
        sv[j] = r @ sv[j] @ r.T

    return d_mu, a, av, s, sv


# ---------------------------------------------------------------------------
# Existing behavior / regression tests
# ---------------------------------------------------------------------------


def test_rotate_basic_no_covariances() -> None:
    """Basic case: no Av, one Sv per sample, no pattern sharing."""
    rng = np.random.default_rng(0)

    n_features = 5
    n_components = 3
    n_samples = 7

    a0 = rng.standard_normal((n_features, n_components))
    s0 = rng.standard_normal((n_components, n_samples))

    # Simple Sv: identity per sample
    sv0 = [np.eye(n_components) for _ in range(n_samples)]

    # New implementation inputs
    a_new = a0.copy()
    s_new = s0.copy()
    sv_new = [sv.copy() for sv in sv0]

    # Legacy inputs: Av as empty list (falsy) to skip Av branch
    a_ref = a0.copy()
    s_ref = s0.copy()
    sv_ref = [sv.copy() for sv in sv0]
    av_ref: list[np.ndarray] | None = []

    # Call new implementation
    params_new = RotateParams(
        loading_covariances=None,
        score_covariances=sv_new,
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    d_mu_new, a_new, _av_new, s_new, sv_new = rotate_to_pca(a_new, s_new, params_new)

    # Call legacy reference
    d_mu_ref, a_ref, av_ref, s_ref, sv_ref = _legacy_rotate_to_pca(
        a_ref,
        av_ref,
        s_ref,
        sv_ref,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    # Shapes
    assert a_new.shape == a_ref.shape == (n_features, n_components)
    assert s_new.shape == s_ref.shape == (n_components, n_samples)
    assert len(sv_new) == len(sv_ref) == n_samples

    # Numerical closeness
    assert _close(d_mu_new, d_mu_ref)
    assert _close(a_new, a_ref)
    assert _close(s_new, s_ref)
    for j in range(n_samples):
        assert _close(sv_new[j], sv_ref[j])


def test_rotate_with_loading_covariances() -> None:
    """Case with Av present: Av and A should rotate consistently."""
    rng = np.random.default_rng(1)

    n_features = 4
    n_components = 2
    n_samples = 6

    a0 = rng.standard_normal((n_features, n_components))
    s0 = rng.standard_normal((n_components, n_samples))

    # Make Av SPD-ish per row
    av0: list[np.ndarray] = []
    for _ in range(n_features):
        m = rng.standard_normal((n_components, n_components))
        av0.append(m @ m.T + 0.1 * np.eye(n_components))

    # Sv per sample: slightly perturbed identity
    sv0: list[np.ndarray] = []
    for _ in range(n_samples):
        m = rng.standard_normal((n_components, n_components))
        sv0.append(np.eye(n_components) + 0.05 * (m @ m.T))

    # New impl inputs
    a_new = a0.copy()
    s_new = s0.copy()
    av_new = [m.copy() for m in av0]
    sv_new = [m.copy() for m in sv0]

    # Legacy impl inputs
    a_ref = a0.copy()
    s_ref = s0.copy()
    av_ref = [m.copy() for m in av0]
    sv_ref = [m.copy() for m in sv0]

    params_new = RotateParams(
        loading_covariances=av_new,
        score_covariances=sv_new,
        isv=None,
        obscombj=None,
        update_bias=False,
    )
    d_mu_new, a_new, _av_new, s_new, sv_new = rotate_to_pca(a_new, s_new, params_new)

    d_mu_ref, a_ref, av_ref, s_ref, sv_ref = _legacy_rotate_to_pca(
        a_ref,
        av_ref,
        s_ref,
        sv_ref,
        isv=None,
        obscombj=None,
        update_bias=False,
    )

    # d_mu should be zeros (no bias update) in both cases
    assert np.allclose(d_mu_new, 0.0)
    assert np.allclose(d_mu_ref, 0.0)

    # Shapes
    assert a_new.shape == a_ref.shape == (n_features, n_components)
    assert s_new.shape == s_ref.shape == (n_components, n_samples)
    assert len(av_new or []) == len(av_ref or []) == n_features
    assert len(sv_new) == len(sv_ref) == n_samples

    # Values
    assert _close(a_new, a_ref)
    assert _close(s_new, s_ref)

    for i in range(n_features):
        assert _close(av_new[i], av_ref[i])

    for j in range(n_samples):
        assert _close(sv_new[j], sv_ref[j])


def test_rotate_pattern_mode_matches_legacy() -> None:
    """Pattern-indexed Sv (obscombj/isv) should match legacy behavior."""
    rng = np.random.default_rng(2)

    n_features = 3
    n_components = 2
    n_samples = 4

    a0 = rng.standard_normal((n_features, n_components))
    s0 = rng.standard_normal((n_components, n_samples))

    # Two patterns: columns [0, 2] and [1, 3]
    obscombj = [[0, 2], [1, 3]]
    isv = np.array([0, 1, 0, 1], dtype=int)

    # Sv per pattern
    sv0: list[np.ndarray] = []
    for _ in range(len(obscombj)):
        m = rng.standard_normal((n_components, n_components))
        sv0.append(m @ m.T + 0.2 * np.eye(n_components))

    # New impl inputs
    a_new = a0.copy()
    s_new = s0.copy()
    sv_new = [m.copy() for m in sv0]

    # Legacy impl inputs
    a_ref = a0.copy()
    s_ref = s0.copy()
    sv_ref = [m.copy() for m in sv0]

    params_new = RotateParams(
        loading_covariances=None,
        score_covariances=sv_new,
        isv=isv,
        obscombj=obscombj,
        update_bias=True,
    )
    d_mu_new, a_new, av_new, s_new, sv_new = rotate_to_pca(a_new, s_new, params_new)

    d_mu_ref, a_ref, av_ref, s_ref, sv_ref = _legacy_rotate_to_pca(
        a_ref,
        [],
        s_ref,
        sv_ref,
        isv=isv,
        obscombj=obscombj,
        update_bias=True,
    )

    assert av_new is None
    assert av_ref is not None  # legacy returns an empty list

    assert _close(d_mu_new, d_mu_ref)
    assert _close(a_new, a_ref)
    assert _close(s_new, s_ref)
    for k in range(len(sv_new)):
        assert _close(sv_new[k], sv_ref[k])


def test_rotate_update_bias_false_preserves_score_means() -> None:
    """When update_bias=False, S should not be explicitly centered."""
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

    # Means may change slightly due to rotations, but they should not
    # be explicitly centered to zero as in update_bias=True.
    mean_after = np.mean(s_new, axis=1, keepdims=True)

    # d_mu is zero vector
    assert np.allclose(d_mu_new, 0.0)
    # And we did actually change something
    assert not np.allclose(a_new, a0)

    # Means should not be (numerically) near zero in general, and should
    # differ from explicit centering behavior.
    assert not np.allclose(mean_after, 0.0, atol=1e-8)

    # Just sanity check that we didn't accidentally leave scores unchanged.
    assert not _close(mean_before, mean_after)


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

    # Each component's mean over samples should be very close to zero.
    assert np.allclose(mean_after, 0.0, atol=1e-10)
    # Non-trivial rotation: not everything is zero or identity.
    assert not np.allclose(a_new, a0)
    assert not np.allclose(s_new, s0)
    # d_mu should not be identically zero (in general) when update_bias=True
    assert not np.allclose(d_mu_new, 0.0)


def test_rotate_rank_deficient_cov_s() -> None:
    """Rotation must behave properly when cov_s has zero eigenvalues."""
    rng = np.random.default_rng(10)

    n_features = 5
    n_components = 3
    n_samples = 4

    a0 = rng.standard_normal((n_features, n_components))

    # S is rank 1: only first row nonzero
    s0 = np.zeros((n_components, n_samples))
    s0[0, :] = rng.standard_normal(n_samples)

    # Identity Sv so no extra structure fills rank
    sv0 = [np.eye(n_components) for _ in range(n_samples)]

    a_new = a0.copy()
    s_new = s0.copy()
    sv_new = [sv.copy() for sv in sv0]

    a_ref = a0.copy()
    s_ref = s0.copy()
    sv_ref = [sv.copy() for sv in sv0]

    params_new = RotateParams(
        loading_covariances=None,
        score_covariances=sv_new,
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    _d_mu_new, a_new, _, s_new, sv_new = rotate_to_pca(a_new, s_new, params_new)

    _d_mu_ref, a_ref, _, s_ref, sv_ref = _legacy_rotate_to_pca(
        a_ref,
        [],
        s_ref,
        sv_ref,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    assert _close(a_new, a_ref)
    assert _close(s_new, s_ref)
    for j in range(len(sv_new)):
        assert _close(sv_new[j], sv_ref[j])


def test_rotate_no_av_but_sv_modified() -> None:
    """Ensure Sv rotates correctly even when Av=None."""
    rng = np.random.default_rng(7)

    n_features = 4
    n_components = 2
    n_samples = 5

    a0 = rng.standard_normal((n_features, n_components))
    s0 = rng.standard_normal((n_components, n_samples))

    # Non-identity Sv
    sv0: list[np.ndarray] = []
    for _ in range(n_samples):
        m = rng.standard_normal((n_components, n_components))
        sv0.append(m @ m.T + 0.1 * np.eye(n_components))

    a_new = a0.copy()
    s_new = s0.copy()
    sv_new = [sv.copy() for sv in sv0]

    a_ref = a0.copy()
    s_ref = s0.copy()
    sv_ref = [sv.copy() for sv in sv0]

    params_new = RotateParams(
        loading_covariances=None,
        score_covariances=sv_new,
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    _d_mu_new, a_new, av_new, s_new, sv_new = rotate_to_pca(a_new, s_new, params_new)

    _d_mu_ref, a_ref, av_ref, s_ref, sv_ref = _legacy_rotate_to_pca(
        a_ref,
        [],
        s_ref,
        sv_ref,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    assert av_new is None
    assert av_ref is not None

    for j in range(n_samples):
        assert _close(sv_new[j], sv_ref[j])


# ---------------------------------------------------------------------------
# Additional validation / error-handling tests
# ---------------------------------------------------------------------------


def test_rotate_errors_on_mismatched_scores_shape() -> None:
    """S must have matching n_components."""
    rng = np.random.default_rng(11)

    n_features = 4
    n_components = 3
    n_samples = 5

    loadings = rng.standard_normal((n_features, n_components))
    # Wrong: scores has k+1 rows
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
    """A and S must be 2D."""
    rng = np.random.default_rng(12)

    n_features = 4
    n_components = 2
    n_samples = 3

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))
    sv = [np.eye(n_components) for _ in range(n_samples)]

    # A not 2D
    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    with pytest.raises(ValueError, match="loadings \\(A\\) must be a 2D array"):
        rotate_to_pca(loadings.reshape(-1), scores, params)

    # S not 2D
    with pytest.raises(ValueError, match="scores \\(S\\) must be a 2D array"):
        rotate_to_pca(loadings, scores.reshape(-1), params)


def test_rotate_errors_on_zero_samples() -> None:
    """S must have at least one sample."""
    rng = np.random.default_rng(13)

    n_features = 4
    n_components = 2

    loadings = rng.standard_normal((n_features, n_components))
    scores = np.empty((n_components, 0))
    sv: list[np.ndarray] = [np.eye(n_components)]  # never actually used

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    with pytest.raises(ValueError, match="n_samples > 0"):
        rotate_to_pca(loadings, scores, params)


def test_rotate_errors_on_av_length_mismatch() -> None:
    """Av length must match n_features."""
    rng = np.random.default_rng(14)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))
    sv = [np.eye(n_components) for _ in range(n_samples)]

    # Wrong length for Av: 1 instead of n_features
    av = [np.eye(n_components)]

    params = RotateParams(
        loading_covariances=av,
        score_covariances=sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    with pytest.raises(ValueError, match="loading_covariances \\(Av\\) length"):
        rotate_to_pca(loadings, scores, params)


def test_rotate_errors_on_av_shape_mismatch() -> None:
    """Each Av[i] must be (k, k)."""
    rng = np.random.default_rng(15)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))
    sv = [np.eye(n_components) for _ in range(n_samples)]

    # One Av has wrong shape
    av: list[np.ndarray] = [np.eye(n_components) for _ in range(n_features)]
    av[1] = np.eye(n_components + 1)

    params = RotateParams(
        loading_covariances=av,
        score_covariances=sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    with pytest.raises(ValueError, match="loading_covariances\\[i\\]"):
        rotate_to_pca(loadings, scores, params)


def test_rotate_errors_on_sv_none_or_empty() -> None:
    """Sv must be a non-empty list of (k, k) matrices."""
    rng = np.random.default_rng(16)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))

    # Sv is None
    params_none = RotateParams(
        loading_covariances=None,
        score_covariances=None,  # type: ignore[arg-type]
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    with pytest.raises(ValueError, match="score_covariances \\(Sv\\)"):
        rotate_to_pca(loadings, scores, params_none)

    # Sv is empty list
    params_empty = RotateParams(
        loading_covariances=None,
        score_covariances=[],
        isv=None,
        obscombj=None,
        update_bias=True,
    )
    with pytest.raises(ValueError, match="score_covariances \\(Sv\\)"):
        rotate_to_pca(loadings, scores, params_empty)


def test_rotate_errors_on_sv_shape_mismatch() -> None:
    """Each Sv[j] must be (k, k)."""
    rng = np.random.default_rng(17)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))

    sv: list[np.ndarray] = [np.eye(n_components) for _ in range(n_samples)]
    sv[2] = np.eye(n_components + 1)

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    with pytest.raises(ValueError, match="score_covariances\\[j\\]"):
        rotate_to_pca(loadings, scores, params)


def test_rotate_pattern_mode_requires_obscombj() -> None:
    """Non-empty isv requires obscombj."""
    rng = np.random.default_rng(18)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))
    sv = [np.eye(n_components) for _ in range(2)]  # per-pattern (pretend)

    isv = np.array([0, 1, 0, 1], dtype=int)

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=isv,
        obscombj=None,
        update_bias=True,
    )

    with pytest.raises(ValueError, match="obscombj must be provided"):
        rotate_to_pca(loadings, scores, params)


def test_rotate_pattern_mode_sv_len_mismatch() -> None:
    """Pattern-mode Sv length must match len(obscombj)."""
    rng = np.random.default_rng(19)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))

    # obscombj has 2 patterns
    obscombj = [[0, 2], [1, 3]]
    isv = np.array([0, 1, 0, 1], dtype=int)

    # But Sv has wrong length (1 instead of 2)
    sv = [np.eye(n_components)]

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=isv,
        obscombj=obscombj,
        update_bias=True,
    )

    with pytest.raises(ValueError, match="length must match len\\(obscombj\\)"):
        rotate_to_pca(loadings, scores, params)


def test_rotate_pattern_mode_obscombj_coverage_error() -> None:
    """Obscombj must cover each sample exactly once."""
    rng = np.random.default_rng(20)

    n_features = 3
    n_components = 2
    n_samples = 4

    loadings = rng.standard_normal((n_features, n_components))
    scores = rng.standard_normal((n_components, n_samples))

    # Duplicate index 0, missing index 3
    obscombj = [[0, 1], [0, 2]]
    isv = np.array([0, 1, 0, 1], dtype=int)

    sv = [np.eye(n_components) for _ in range(len(obscombj))]

    params = RotateParams(
        loading_covariances=None,
        score_covariances=sv,
        isv=isv,
        obscombj=obscombj,
        update_bias=True,
    )

    with pytest.raises(ValueError, match="must cover all sample indices"):
        rotate_to_pca(loadings, scores, params)


# ---------------------------------------------------------------------------
# Additional math / property tests
# ---------------------------------------------------------------------------


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

    # Compute cov_s before rotation (with explicit centering).
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

    # Recompute cov_s after rotation using same helper.
    s_centered_after = s_new  # already centered by update_bias=True
    cov_after = _build_cov_s(s_centered_after, sv_new, isv, None)
    off_after = cov_after - np.diag(np.diag(cov_after))
    off_after_norm = np.linalg.norm(off_after)

    # Off-diagonal magnitude should not increase substantially.
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

    # Basic sanity checks: shapes and finite values.
    assert d_mu.shape == (n_features, 1)
    assert a_new.shape == (n_features, n_components)
    assert s_new.shape == (n_components, n_samples)
    for av in av_new or []:
        assert av.shape == (n_components, n_components)
    for sv in sv_new:
        assert sv.shape == (n_components, n_components)

    # Variances/covariances should be non-negative (1x1 PSD).
    for av in av_new or []:
        assert av[0, 0] >= 0.0
    for sv in sv_new:
        assert sv[0, 0] >= 0.0


def test_rotate_pattern_mode_with_av_present() -> None:
    """Pattern-mode Sv with Av present should still match legacy behavior."""
    rng = np.random.default_rng(23)

    n_features = 4
    n_components = 2
    n_samples = 6

    a0 = rng.standard_normal((n_features, n_components))
    s0 = rng.standard_normal((n_components, n_samples))

    # Two patterns: [0, 2, 4] and [1, 3, 5]
    obscombj = [[0, 2, 4], [1, 3, 5]]
    isv = np.array([0, 1, 0, 1, 0, 1], dtype=int)

    # Av SPD-ish per row
    av0: list[np.ndarray] = []
    for _ in range(n_features):
        m = rng.standard_normal((n_components, n_components))
        av0.append(m @ m.T + 0.1 * np.eye(n_components))

    # Sv per pattern
    sv0: list[np.ndarray] = []
    for _ in range(len(obscombj)):
        m = rng.standard_normal((n_components, n_components))
        sv0.append(m @ m.T + 0.2 * np.eye(n_components))

    # New implementation inputs
    a_new = a0.copy()
    s_new = s0.copy()
    av_new = [m.copy() for m in av0]
    sv_new = [m.copy() for m in sv0]

    # Legacy implementation inputs
    a_ref = a0.copy()
    s_ref = s0.copy()
    av_ref = [m.copy() for m in av0]
    sv_ref = [m.copy() for m in sv0]

    params = RotateParams(
        loading_covariances=av_new,
        score_covariances=sv_new,
        isv=isv,
        obscombj=obscombj,
        update_bias=True,
    )
    d_mu_new, a_new, av_new, s_new, sv_new = rotate_to_pca(a_new, s_new, params)

    d_mu_ref, a_ref, av_ref, s_ref, sv_ref = _legacy_rotate_to_pca(
        a_ref,
        av_ref,
        s_ref,
        sv_ref,
        isv=isv,
        obscombj=obscombj,
        update_bias=True,
    )

    assert _close(d_mu_new, d_mu_ref)
    assert _close(a_new, a_ref)
    assert _close(s_new, s_ref)
    for i in range(n_features):
        assert _close(av_new[i], av_ref[i])
    for k in range(len(sv_new)):
        assert _close(sv_new[k], sv_ref[k])
