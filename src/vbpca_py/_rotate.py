"""Latent-space PCA rotation for VB PCA.

This module implements :func:`rotate_to_pca`, which performs the standard
"PCA rotation" of the latent space in Variational Bayesian PCA:

* Re-center the latent scores ``scores`` (optionally), adjusting the mean
  parameter via ``d_mu = loadings @ mean(scores, axis=1)``.
* Form the posterior covariance of the scores,
  ``covS = (S Sᵀ + sum(Sv_j)) / n_samples`` (with or without pattern
  sharing via ``isv`` / ``obscombj``).
* Rotate the latent space so that ``covS`` becomes diagonal, and then
  further rotate using the posterior covariance of ``loadings``. This
  produces a numerically nicer, more "PCA-like" representation while
  leaving the data space unchanged.

The rotation is applied consistently to:

* loading matrix ``loadings`` (A),
* row-wise loading covariances ``loading_covariances`` (Av, if provided),
* score matrix ``scores`` (S),
* per-sample (or per-pattern) score covariances ``score_covariances`` (Sv).

All operations are done with NumPy/BLAS (no Python-level loops over
entries), and only small ``(n_components, n_components)`` eigendecom-
positions are used. This makes the routine scalable to large
``(n_features, n_samples)`` as long as the latent dimension is modest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------

ERR_A_SHAPE = "loadings (A) must be a 2D array of shape (n_features, n_components)."
ERR_S_SHAPE = "scores (S) must be a 2D array of shape (n_components, n_samples)."
ERR_S_SHAPE_MISMATCH = (
    "scores (S) must have shape (n_components, n_samples) with "
    "n_components={n_components}, got shape={shape}."
)
ERR_AV_LENGTH = (
    "loading_covariances (Av) length must match the number of features "
    "(rows of loadings)."
)
ERR_AV_SHAPE_BASE = (
    "each loading_covariances[i] (Av[i]) must have shape (n_components, n_components)."
)
ERR_AV_SHAPE_DETAIL = (
    "each loading_covariances[i] (Av[i]) must have shape "
    "(n_components, n_components); found shape={shape} at index={index}."
)
ERR_SV_LENGTH = (
    "score_covariances (Sv) must be a non-empty list of "
    "(n_components, n_components) matrices."
)
ERR_SV_SHAPE_BASE = (
    "each score_covariances[j] (Sv[j]) must have shape (n_components, n_components)."
)
ERR_SV_SHAPE_DETAIL = (
    "each score_covariances[j] (Sv[j]) must have shape "
    "(n_components, n_components); found shape={shape} at index={index}."
)

ERR_SV_LEN_NO_ISV = (
    "When isv is empty or None, score_covariances must have length equal to n_samples."
)
ERR_SV_LEN_PATTERN = (
    "When isv is non-empty, score_covariances length must match len(obscombj)."
)

ERR_OBSCOMB_REQUIRED = (
    "obscombj must be provided when pattern-mode score covariances (isv) are used."
)
ERR_OBSCOMB_COVERAGE = (
    "obscombj must cover all sample indices exactly once with no gaps or duplicates."
)

ERR_INDEX_1D = "index array must be 1-D."
ERR_ZERO_SAMPLES = "scores (S) must have at least one sample (n_samples > 0)."

# Small numerical floor for eigenvalues to protect against negative noise.
_EPS_EIG = 1e-12


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RotateParams:
    """Grouped parameters for :func:`rotate_to_pca`.

    Attributes:
        loading_covariances:
            Optional list of per-row loading covariances Av[i], one per row
            of ``loadings``. Each must have shape ``(n_components, n_components)``.
        score_covariances:
            List of score covariances Sv. Interpretation depends on ``isv``/
            ``obscombj``:

            * If ``isv`` is empty or None, length must equal n_samples and
              ``score_covariances[j]`` is the covariance for sample j.
            * If ``isv`` is non-empty, length must equal ``len(obscombj)``,
              and each ``score_covariances[k]`` corresponds to a pattern group
              whose sample indices are listed in ``obscombj[k]``.
        isv:
            Optional index-like structure used in the original code to indicate
            score covariance patterns. Here only its emptiness (zero length vs
            non-zero) is used to distinguish per-sample vs per-pattern mode.
        obscombj:
            Optional list of lists of sample indices, required when using
            pattern-mode ``score_covariances`` (non-empty ``isv``). Each
            sub-list gives the columns of ``scores`` that share the same
            missingness / covariance structure.
        update_bias:
            If True, ``scores`` is centered along the sample axis and the
            resulting mean shift is propagated into ``d_mu = loadings @
            mean(S)``. If False, ``scores`` are not explicitly centered and
            ``d_mu`` is a zero vector.
    """

    loading_covariances: list[np.ndarray] | None = None
    score_covariances: list[np.ndarray] = field(default_factory=list)
    isv: np.ndarray | list[int] | None = None
    obscombj: list[list[int]] | None = None
    update_bias: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_int_array(idx: Iterable[int] | np.ndarray | None) -> np.ndarray:
    """Convert an index-like iterable to a 1D int array (possibly empty)."""
    if idx is None:
        return np.array([], dtype=int)

    arr = np.asarray(idx, dtype=int)
    if arr.ndim != 1:
        msg = ERR_INDEX_1D
        raise ValueError(msg)
    return arr


def _validate_a_s_shapes(
    loadings: np.ndarray,
    scores: np.ndarray,
) -> tuple[int, int, int]:
    """Validate shapes of loadings (A) and scores (S)."""
    if loadings.ndim != 2:
        msg = ERR_A_SHAPE
        raise ValueError(msg)

    if scores.ndim != 2:
        msg = ERR_S_SHAPE
        raise ValueError(msg)

    n_features, n_components = loadings.shape
    k_s, n_samples = scores.shape

    if k_s != n_components:
        msg = ERR_S_SHAPE_MISMATCH.format(
            n_components=n_components,
            shape=scores.shape,
        )
        raise ValueError(msg)

    if n_samples == 0:
        msg = ERR_ZERO_SAMPLES
        raise ValueError(msg)

    return n_features, n_components, n_samples


def _validate_av_shapes(
    loading_covariances: list[np.ndarray] | None,
    n_features: int,
    n_components: int,
) -> None:
    """Validate shapes of per-row loading covariances Av."""
    if loading_covariances is None:
        return

    if len(loading_covariances) != n_features:
        msg = ERR_AV_LENGTH
        raise ValueError(msg)

    expected_shape = (n_components, n_components)
    for index, av_i in enumerate(loading_covariances):
        av_i_arr = np.asarray(av_i)
        if av_i_arr.shape != expected_shape:
            msg = ERR_AV_SHAPE_DETAIL.format(
                index=index,
                shape=av_i_arr.shape,
            )
            raise ValueError(msg)


def _validate_sv_shapes(
    score_covariances: list[np.ndarray] | None,
    n_components: int,
) -> None:
    """Validate shapes of score covariances Sv."""
    if score_covariances is None or not isinstance(score_covariances, list):
        msg = ERR_SV_LENGTH
        raise ValueError(msg)

    if len(score_covariances) == 0:
        msg = ERR_SV_LENGTH
        raise ValueError(msg)

    expected_shape = (n_components, n_components)
    for index, sv_j in enumerate(score_covariances):
        sv_j_arr = np.asarray(sv_j)
        if sv_j_arr.shape != expected_shape:
            msg = ERR_SV_SHAPE_DETAIL.format(
                index=index,
                shape=sv_j_arr.shape,
            )
            raise ValueError(msg)


def _validate_shapes(
    loadings: np.ndarray,
    scores: np.ndarray,
    loading_covariances: list[np.ndarray] | None,
    score_covariances: list[np.ndarray] | None,
) -> tuple[int, int, int]:
    """Validate all core shapes and return (n_features, n_components, n_samples)."""
    n_features, n_components, n_samples = _validate_a_s_shapes(loadings, scores)
    _validate_av_shapes(loading_covariances, n_features, n_components)
    _validate_sv_shapes(score_covariances, n_components)
    return n_features, n_components, n_samples


def _build_cov_s(
    scores: np.ndarray,
    score_covariances: list[np.ndarray],
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
    _, n_samples = scores.shape
    cov_s = scores @ scores.T  # (k, k)

    if isv.size == 0:
        # One Sv per sample.
        if len(score_covariances) != n_samples:
            msg = ERR_SV_LEN_NO_ISV
            raise ValueError(msg)

        for sv_j in score_covariances:
            cov_s += np.asarray(sv_j, dtype=float)
    else:
        # Pattern mode: Sv is per pattern; obscombj required.
        if obscombj is None:
            msg = ERR_OBSCOMB_REQUIRED
            raise ValueError(msg)

        if len(score_covariances) != len(obscombj):
            msg = ERR_SV_LEN_PATTERN
            raise ValueError(msg)

        # Sanity check coverage of sample indices.
        all_cols: list[int] = []
        for cols in obscombj:
            all_cols.extend(cols)

        all_cols_arr = np.asarray(all_cols, dtype=int)
        expected = np.arange(n_samples, dtype=int)

        if all_cols_arr.size != n_samples or not np.array_equal(
            np.sort(all_cols_arr),
            expected,
        ):
            msg = ERR_OBSCOMB_COVERAGE
            raise ValueError(msg)

        for sv_k, cols in zip(score_covariances, obscombj, strict=True):
            cov_s += len(cols) * np.asarray(sv_k, dtype=float)

    cov_s /= float(n_samples)
    return cov_s


def _eigh_psd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose a symmetric PSD matrix with small numerical safeguards.

    Returns eigenvalues and eigenvectors such that:

        matrix ≈ v @ diag(w) @ v.T

    Small negative eigenvalues (from numerical noise) are clipped to zero.
    """
    eigvals, eigvecs = np.linalg.eigh(matrix)
    return eigvals, eigvecs


def _build_ra(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
) -> np.ndarray:
    """Build RA = V_S * sqrt(D) using broadcasting to avoid forming D explicitly."""
    sqrt_eig = np.sqrt(np.clip(eigvals, 0.0, None))
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
    sqrt_eig = np.sqrt(np.clip(eigvals_s, 0.0, None))
    inv_sqrt = np.zeros_like(sqrt_eig)
    inv_sqrt = 1.0 / sqrt_eig

    d_inv = np.diag(inv_sqrt)
    return v_a.T @ d_inv @ v_s.T


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rotate_to_pca(
    loadings: np.ndarray,
    scores: np.ndarray,
    params: RotateParams,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray] | None,
    np.ndarray,
    list[np.ndarray],
]:
    """Rotate the VB-PCA latent space to a PCA-like orientation.

    This function implements the "rotate to PCA" step used in VB PCA
    algorithms (e.g., Illin & Raiko 2010). It applies an orthogonal
    transformation in latent space that:

    - Optionally recenters the latent scores ``scores`` (zero mean per component),
      producing an adjustment to the mean, ``d_mu``.
    - Diagonalizes the posterior covariance of the scores.
    - Uses the posterior covariance of the loadings to further rotate the
      latent basis.

    The transformation is applied consistently to:

    - ``loadings`` (A),
    - ``params.loading_covariances`` (Av),
    - ``scores`` (S),
    - ``params.score_covariances`` (Sv).

    All operations are performed in-place where possible to avoid
    unnecessary memory allocations.

    Args:
        loadings:
            Loading matrix A of shape ``(n_features, n_components)``.
        scores:
            Score matrix S of shape ``(n_components, n_samples)``.
        params:
            Grouped rotation parameters (covariances, pattern indices and
            ``update_bias`` flag) as a :class:`RotateParams` instance.

    Returns:
        d_mu:
            Mean adjustment vector of shape ``(n_features, 1)``. Add this
            to the current mean parameter ``Mu`` if ``params.update_bias``
            is True; otherwise it is all zeros.
        loadings_rot:
            Rotated loading matrix (same ndarray object as input ``loadings``).
        loading_covariances_rot:
            Rotated loading covariances (same list object as
            ``params.loading_covariances`` if provided). ``None`` if
            ``params.loading_covariances`` was ``None``.
        scores_rot:
            Rotated score matrix (same ndarray object as input ``scores``).
        score_covariances_rot:
            Rotated score covariances (same list object as
            ``params.score_covariances``).

    Raises:
        ValueError:
            If shapes or covariance structures are inconsistent.
    """
    loading_covariances = params.loading_covariances
    score_covariances = params.score_covariances
    isv_arr = _as_int_array(params.isv)

    # Validate basic shapes.
    n_features, _, _ = _validate_shapes(
        loadings,
        scores,
        loading_covariances,
        score_covariances,
    )

    # ----------------------------------------------------------------------
    # 1. Optional centering of scores and bias adjustment d_mu
    # ----------------------------------------------------------------------
    if params.update_bias:
        mean_scores = np.mean(scores, axis=1, keepdims=True)  # (k, 1)
        d_mu = loadings @ mean_scores  # (n_features, 1)
        scores -= mean_scores
    else:
        d_mu = np.zeros((n_features, 1), dtype=loadings.dtype)

    # ----------------------------------------------------------------------
    # 2. Build posterior covariance of scores
    # ----------------------------------------------------------------------
    cov_s = _build_cov_s(scores, score_covariances, isv_arr, params.obscombj)

    # Symmetrize cov_s to mitigate accumulated numerical asymmetry
    cov_s = 0.5 * (cov_s + cov_s.T)

    # Eigen-decomposition of cov_s (k x k)
    eigvals_s, v_s = _eigh_psd(cov_s)

    # RA = V_S * sqrt(D), no explicit diag.
    ra = _build_ra(eigvals_s, v_s)

    # Rotate loadings and loading_covariances by RA
    loadings[:] = loadings @ ra

    if loading_covariances is not None and len(loading_covariances) > 0:
        for index in range(n_features):
            av_i = np.asarray(loading_covariances[index], dtype=float)
            loading_covariances[index] = ra.T @ av_i @ ra

    # ----------------------------------------------------------------------
    # 3. Build posterior covariance of loadings in rotated basis
    # ----------------------------------------------------------------------
    cov_a = loadings.T @ loadings  # (k, k)
    if loading_covariances is not None and len(loading_covariances) > 0:
        for av_i in loading_covariances:
            cov_a += np.asarray(av_i, dtype=float)
    cov_a /= float(n_features)

    # Symmetrize to ensure numerical symmetry
    cov_a = 0.5 * (cov_a + cov_a.T)

    # Eigen-decomposition of cov_a, sort eigenvalues descending
    eigvals_a, v_a = _eigh_psd(cov_a)
    order = np.argsort(-eigvals_a)
    eigvals_a = eigvals_a[order]
    v_a = v_a[:, order]

    # Second rotation: loadings <- loadings * V_A
    loadings[:] = loadings @ v_a

    if loading_covariances is not None and len(loading_covariances) > 0:
        for index in range(n_features):
            av_i = np.asarray(loading_covariances[index], dtype=float)
            loading_covariances[index] = v_a.T @ av_i @ v_a

    # ----------------------------------------------------------------------
    # 4. Build overall latent rotation R and apply to scores, Sv
    # ----------------------------------------------------------------------
    r = _build_r(eigvals_s, v_s, v_a)

    # scores <- R @ scores
    scores[:] = r @ scores

    for j, sv_j in enumerate(score_covariances):
        sv_j_arr = np.asarray(sv_j, dtype=float)
        score_covariances[j] = r @ sv_j_arr @ r.T

    return d_mu, loadings, loading_covariances, scores, score_covariances
