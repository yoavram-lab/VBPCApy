"""Latent-space PCA rotation for VB PCA.

This module implements :func:`rotate_to_pca`, which performs the standard
"PCA rotation" of the latent space in Variational Bayesian PCA:

* Re-center the latent scores ``S`` (optionally), adjusting the mean
  parameter via ``d_mu = A @ mean(S, axis=1)``.
* Form the posterior covariance of the scores,
  ``covS = (S Sᵀ + sum(Sv_j)) / n_samples`` (with or without pattern
  sharing via ``isv`` / ``obscombj``).
* Rotate the latent space so that ``covS`` becomes diagonal, and then
  further rotate using the posterior covariance of ``A``. This results
  in a numerically nicer, more "PCA-like" representation while leaving
  the data space unchanged.

The rotation is applied consistently to:

* loading matrix ``A``,
* row-wise loading covariances ``Av`` (if provided),
* score matrix ``S``,
* per-sample (or per-pattern) score covariances ``Sv``.

All operations are done with NumPy/BLAS (no Python-level loops over
entries), and only small ``(n_components, n_components)`` eigendecom-
positions are used. This makes the routine scalable to large
``(n_features, n_samples)`` as long as the latent dimension is modest.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------

ERR_A_SHAPE = "a must be a 2D array of shape (n_features, n_components)."
ERR_S_SHAPE = "s must be a 2D array of shape (n_components, n_samples)."
ERR_AV_LENGTH = "av length must match the number of features (rows of a)."
ERR_AV_SHAPE = "each av[i] must have shape (n_components, n_components)."
ERR_SV_LENGTH = "sv must be non-empty list of (n_components, n_components) matrices."
ERR_SV_SHAPE = "each sv[j] must have shape (n_components, n_components)."
ERR_OBSCOMB_REQUIRED = "obscombj must be provided when isv / pattern mode is used."
ERR_OBSCOMB_COVERAGE = "obscombj must cover all sample indices exactly once."


# Small numerical floor for eigenvalues to protect against negative noise.
_EPS_EIG = 1e-12


def _as_int_array(idx: Iterable[int] | np.ndarray | None) -> np.ndarray:
    """Convert an index-like iterable to a 1D int array (possibly empty)."""
    if idx is None:
        return np.array([], dtype=int)
    arr = np.asarray(idx, dtype=int)
    if arr.ndim != 1:
        raise ValueError("index array must be 1-D.")
    return arr


def _validate_shapes(
    a: np.ndarray,
    s: np.ndarray,
    av: list[np.ndarray] | None,
    sv: list[np.ndarray],
) -> tuple[int, int, int]:
    """Validate basic shapes and return (n_features, n_components, n_samples)."""
    if a.ndim != 2:
        raise ValueError(ERR_A_SHAPE)
    if s.ndim != 2:
        raise ValueError(ERR_S_SHAPE)

    n_features, n_components = a.shape
    k_s, n_samples = s.shape

    if k_s != n_components:
        raise ValueError(
            f"s must have shape (n_components, n_samples) with "
            f"n_components={n_components}, got {s.shape}."
        )

    # Validate Av, if provided
    if av is not None:
        if len(av) != n_features:
            raise ValueError(ERR_AV_LENGTH)
        for idx, av_i in enumerate(av):
            av_i_arr = np.asarray(av_i)
            if av_i_arr.shape != (n_components, n_components):
                raise ValueError(
                    f"{ERR_AV_SHAPE} (index {idx}, shape={av_i_arr.shape})"
                )

    # Validate Sv
    if not isinstance(sv, list) or len(sv) == 0:
        raise ValueError(ERR_SV_LENGTH)
    for idx, sv_j in enumerate(sv):
        sv_j_arr = np.asarray(sv_j)
        if sv_j_arr.shape != (n_components, n_components):
            raise ValueError(f"{ERR_SV_SHAPE} (index {idx}, shape={sv_j_arr.shape})")

    return n_features, n_components, n_samples


def _build_cov_s(
    s: np.ndarray,
    sv: list[np.ndarray],
    isv: np.ndarray,
    obscombj: list[list[int]] | None,
) -> np.ndarray:
    """Compute the posterior covariance of S in latent space.

    covS = (S Sᵀ + sum_j Sv_j) / n_samples

    When ``isv`` is empty, Sv is assumed to be per-sample (len(Sv) == n_samples).
    When ``isv`` is non-empty, Sv is assumed to be per-pattern with obscombj
    mapping patterns to sample indices; covS then accumulates
    ``len(obscombj[k]) * Sv[k]``.
    """
    k, n_samples = s.shape
    cov_s = s @ s.T  # (k, k)

    if isv.size == 0:
        # One Sv per sample.
        if len(sv) != n_samples:
            raise ValueError(
                "When isv is empty, sv must have length equal to n_samples."
            )
        for sv_j in sv:
            cov_s += np.asarray(sv_j, dtype=float)
    else:
        # Pattern mode: sv is per pattern; obscombj required.
        if obscombj is None:
            raise ValueError(ERR_OBSCOMB_REQUIRED)
        if len(sv) != len(obscombj):
            raise ValueError(
                "When isv is non-empty, sv length must match len(obscombj)."
            )

        # Optionally, sanity check coverage of sample indices.
        all_cols: list[int] = []
        for cols in obscombj:
            all_cols.extend(cols)
        all_cols_arr = np.asarray(all_cols, dtype=int)
        if all_cols_arr.size != n_samples or not np.array_equal(
            np.sort(all_cols_arr),
            np.arange(n_samples, dtype=int),
        ):
            raise ValueError(ERR_OBSCOMB_COVERAGE)

        for sv_k, cols in zip(sv, obscombj, strict=True):
            cov_s += len(cols) * np.asarray(sv_k, dtype=float)

    cov_s /= float(n_samples)
    return cov_s


def _eigh_psd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose a symmetric PSD matrix with small numerical safeguards.

    Returns eigenvalues and eigenvectors such that:

        matrix ≈ v @ diag(w) @ v.T

    Small negative eigenvalues (from numerical noise) are clipped to zero.
    """
    w, v = np.linalg.eigh(matrix)
    # Clip small negatives to zero to avoid sqrt of negative.
    w_clipped = np.clip(w, 0.0, None)
    return w_clipped, v


def _build_ra(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
) -> np.ndarray:
    """Build RA = V_S * sqrt(D) using broadcasting to avoid forming D explicitly."""
    # sqrt of eigenvalues, with clip for numerical stability
    sqrt_eig = np.sqrt(np.clip(eigvals, 0.0, None))
    # VS: (k, k); multiply each column by sqrt_eig
    return eigvecs * sqrt_eig[np.newaxis, :]


def _build_r(
    eigvals_s: np.ndarray,
    v_s: np.ndarray,
    v_a: np.ndarray,
) -> np.ndarray:
    """Build rotation R = V_A.T * D_inv * V_S.T in a numerically robust way.

    D_inv is diag(1 / sqrt(eigvals_s)), with a small epsilon threshold to
    avoid division by zero when eigenvalues are extremely small.
    """
    # Invert sqrt of eigenvalues with threshold.
    sqrt_eig = np.sqrt(np.clip(eigvals_s, 0.0, None))
    inv_sqrt = np.zeros_like(sqrt_eig)
    mask = sqrt_eig > _EPS_EIG
    inv_sqrt[mask] = 1.0 / sqrt_eig[mask]

    # Form D_inv explicitly. k is small, so this is not a bottleneck.
    d_inv = np.diag(inv_sqrt)
    return v_a.T @ d_inv @ v_s.T


def rotate_to_pca(
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
    """Rotate the VB-PCA latent space to a PCA-like orientation.

    This function implements the "rotate to PCA" step used in VB PCA
    algorithms (e.g., Illin & Raiko 2010). It applies an orthogonal
    transformation in latent space that:

    - Optionally recenters the latent scores S (zero mean per component),
      producing an adjustment to the mean, ``d_mu``.
    - Diagonalizes the posterior covariance of the scores.
    - Uses the posterior covariance of the loadings to further rotate the
      latent basis.

    The transformation is applied consistently to:

    - ``a`` (loadings),
    - ``av`` (per-row loading covariances),
    - ``s`` (scores),
    - ``sv`` (score covariances, per sample or per pattern).

    All operations are performed in-place where possible to avoid
    unnecessary memory allocations.

    Args:
        a:
            Loading matrix of shape ``(n_features, n_components)``.
        av:
            Optional list of per-row loading covariances. If provided, it
            must have length ``n_features``, and each entry must be an
            array of shape ``(n_components, n_components)``.
        s:
            Score matrix of shape ``(n_components, n_samples)``.
        sv:
            List of score covariances. The interpretation depends on
            ``isv`` / ``obscombj``:

            * If ``isv`` is empty or ``None``, ``sv`` must have length
              ``n_samples`` and ``sv[j]`` is the covariance for column ``j``.
            * If ``isv`` is non-empty, ``sv`` must have length
              ``len(obscombj)`` and each ``sv[k]`` corresponds to a missing
              pattern, with ``obscombj[k]`` listing the sample indices that
              share that pattern.

        isv:
            Optional index mapping for score covariance patterns. This is
            kept for compatibility with the original code, but only
            ``obscombj`` is used to build the covariance of S; ``isv`` is
            not actively indexed here. If not used, pass ``None`` or an
            empty sequence.
        obscombj:
            Optional list of lists of sample indices. Required when using
            pattern-mode Sv (non-empty ``isv``). Each sub-list gives the
            columns of S that share the same missingness / covariance
            structure.
        update_bias:
            If True, S is centered along the sample axis and a mean
            adjustment is computed as ``d_mu = a @ mean(S, axis=1)``.
            If False, S is left unchanged and ``d_mu`` is a zero vector.

    Returns:
        d_mu:
            Mean adjustment vector of shape ``(n_features, 1)``. Add this
            to the current mean parameter ``Mu`` if ``update_bias`` was
            True; otherwise it is all zeros.
        a_rot:
            Rotated loading matrix (same ndarray object as input ``a``).
        av_rot:
            Rotated loading covariances (same list object as ``av`` if
            provided). ``None`` if ``av`` was ``None``.
        s_rot:
            Rotated score matrix (same ndarray object as input ``s``).
        sv_rot:
            Rotated score covariances (same list object as input ``sv``).

    Raises:
        ValueError:
            If shapes or covariance structures are inconsistent.
    """
    # Normalize index structures.
    isv_arr = _as_int_array(isv)

    # Validate basic shapes.
    n_features, n_components, n_samples = _validate_shapes(a, s, av, sv)

    # ----------------------------------------------------------------------
    # 1. Optional centering of S and bias adjustment d_mu
    # ----------------------------------------------------------------------
    if update_bias:
        # mean over samples, shape (k, 1)
        mean_s = np.mean(s, axis=1, keepdims=True)
        d_mu = a @ mean_s  # (n_features, 1)
        s -= mean_s
    else:
        d_mu = np.zeros((n_features, 1), dtype=a.dtype)

    # ----------------------------------------------------------------------
    # 2. Build posterior covariance of S
    # ----------------------------------------------------------------------
    cov_s = _build_cov_s(s, sv, isv_arr, obscombj)

    # Symmetrize cov_s to mitigate accumulated numerical asymmetry
    cov_s = 0.5 * (cov_s + cov_s.T)

    # Eigen-decomposition of cov_s (k x k)
    eigvals_s, v_s = _eigh_psd(cov_s)

    # RA = V_S * sqrt(D), no explicit diag.
    ra = _build_ra(eigvals_s, v_s)

    # Rotate A and Av by RA
    # A <- A * RA
    a[:] = a @ ra

    if av is not None and len(av) > 0:
        for i in range(n_features):
            av_i = np.asarray(av[i], dtype=float)
            av[i] = ra.T @ av_i @ ra

    # ----------------------------------------------------------------------
    # 3. Build posterior covariance of A in rotated basis
    # ----------------------------------------------------------------------
    cov_a = a.T @ a  # (k, k)
    if av is not None and len(av) > 0:
        for av_i in av:
            cov_a += np.asarray(av_i, dtype=float)
    cov_a /= float(n_features)

    # Symmetrize to ensure numerical symmetry
    cov_a = 0.5 * (cov_a + cov_a.T)

    # Eigen-decomposition of cov_a, sort eigenvalues descending
    eigvals_a, v_a = _eigh_psd(cov_a)
    order = np.argsort(-eigvals_a)
    eigvals_a = eigvals_a[order]
    v_a = v_a[:, order]

    # Second rotation: A <- A * V_A
    a[:] = a @ v_a

    if av is not None and len(av) > 0:
        for i in range(n_features):
            av_i = np.asarray(av[i], dtype=float)
            av[i] = v_a.T @ av_i @ v_a

    # ----------------------------------------------------------------------
    # 4. Build overall latent rotation R and apply to S, Sv
    # ----------------------------------------------------------------------
    r = _build_r(eigvals_s, v_s, v_a)

    # S <- R @ S
    s[:] = r @ s

    # Sv[j] <- R @ Sv[j] @ R.T
    for j, sv_j in enumerate(sv):
        sv_j_arr = np.asarray(sv_j, dtype=float)
        sv[j] = r @ sv_j_arr @ r.T

    return d_mu, a, av, s, sv
