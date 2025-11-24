"""Tests for the rotate_to_pca helper in vbpca_py._rotate."""

from __future__ import annotations

import numpy as np

from vbpca_py._rotate import rotate_to_pca


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
    np.ndarray, np.ndarray, list[np.ndarray] | None, np.ndarray, list[np.ndarray]
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
            raise ValueError("obscombj required when isv is non-empty in legacy path.")
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
# Tests
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

    # Call both implementations
    d_mu_new, a_new, av_new, s_new, sv_new = rotate_to_pca(
        a_new,
        None,
        s_new,
        sv_new,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

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

    d_mu_new, a_new, av_new, s_new, sv_new = rotate_to_pca(
        a_new,
        av_new,
        s_new,
        sv_new,
        isv=None,
        obscombj=None,
        update_bias=False,
    )

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

    d_mu_new, a_new, av_new, s_new, sv_new = rotate_to_pca(
        a_new,
        None,
        s_new,
        sv_new,
        isv=isv,
        obscombj=obscombj,
        update_bias=True,
    )

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
    """When update_bias=False, S should not be centered."""
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

    d_mu_new, a_new, av_new, s_new, sv_new = rotate_to_pca(
        a_new,
        None,
        s_new,
        sv_new,
        isv=None,
        obscombj=None,
        update_bias=False,
    )

    # Means may change slightly due to rotations, but they should not
    # be explicitly centered to zero as in update_bias=True.
    mean_after = np.mean(s_new, axis=1, keepdims=True)

    # d_mu is zero vector
    assert np.allclose(d_mu_new, 0.0)

    # Means should not be (numerically) near zero in general, and should
    # differ from explicit centering behavior.
    assert not np.allclose(mean_after, 0.0, atol=1e-8)

    # We don't require equality with mean_before (rotations can change it),
    # just ensure it's not forced to zero.


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

    d_mu_new, a_new, av_new, s_new, sv_new = rotate_to_pca(
        a_new,
        None,
        s_new,
        sv_new,
        isv=None,
        obscombj=None,
        update_bias=True,
    )

    mean_after = np.mean(s_new, axis=1, keepdims=True)

    # Each component's mean over samples should be very close to zero.
    assert np.allclose(mean_after, 0.0, atol=1e-10)
    # Non-trivial rotation: not everything is zero or identity.
    assert not np.allclose(a_new, a0)
    assert not np.allclose(s_new, s0)


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

    d_mu_new, a_new, _, s_new, sv_new = rotate_to_pca(
        a_new, None, s_new, sv_new, isv=None, obscombj=None, update_bias=True
    )

    d_mu_ref, a_ref, _, s_ref, sv_ref = _legacy_rotate_to_pca(
        a_ref, [], s_ref, sv_ref, isv=None, obscombj=None, update_bias=True
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
    sv0 = []
    for _ in range(n_samples):
        m = rng.standard_normal((n_components, n_components))
        sv0.append(m @ m.T + 0.1 * np.eye(n_components))

    a_new = a0.copy()
    s_new = s0.copy()
    sv_new = [sv.copy() for sv in sv0]

    a_ref = a0.copy()
    s_ref = s0.copy()
    sv_ref = [sv.copy() for sv in sv0]

    d_mu_new, a_new, av_new, s_new, sv_new = rotate_to_pca(
        a_new, None, s_new, sv_new, isv=None, obscombj=None, update_bias=True
    )

    d_mu_ref, a_ref, av_ref, s_ref, sv_ref = _legacy_rotate_to_pca(
        a_ref, [], s_ref, sv_ref, isv=None, obscombj=None, update_bias=True
    )

    assert av_new is None
    for j in range(n_samples):
        assert _close(sv_new[j], sv_ref[j])
