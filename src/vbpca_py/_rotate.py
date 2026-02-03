# vbpca_py/_rotate.py
"""
Latent-space PCA rotation for VB PCA.

Implements rotate_to_pca, a faithful + shape-safe translation of MATLAB
RotateToPCA.m while remaining idiomatic Python.

Key behavioral notes:
- Eigenvectors are not unique under degenerate eigenvalues; comparisons
  must be basis-invariant or use Procrustes alignment in tests.
- We guard against rank-deficient covS by flooring eigenvalues in the
  (1/sqrt(eig)) factor used to build the final R rotation.
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
ERR_AV_SHAPE_DETAIL = (
    "each loading_covariances[i] (Av[i]) must have shape "
    "(n_components, n_components); found shape={shape} at index={index}."
)
ERR_SV_LENGTH = (
    "score_covariances (Sv) must be a non-empty list of "
    "(n_components, n_components) matrices."
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

# Numerical floor for eigenvalues used in 1/sqrt(eig) to avoid inf/NaN in rank-deficient cases.
_EPS_EIG = 1e-20


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RotateParams:
    """Grouped parameters for rotate_to_pca."""

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
        raise ValueError(ERR_INDEX_1D)
    return arr


def _validate_a_s_shapes(
    loadings: np.ndarray,
    scores: np.ndarray,
) -> tuple[int, int, int]:
    if loadings.ndim != 2:
        raise ValueError(ERR_A_SHAPE)
    if scores.ndim != 2:
        raise ValueError(ERR_S_SHAPE)

    n_features, n_components = loadings.shape
    k_s, n_samples = scores.shape

    if k_s != n_components:
        raise ValueError(
            ERR_S_SHAPE_MISMATCH.format(n_components=n_components, shape=scores.shape)
        )
    if n_samples == 0:
        raise ValueError(ERR_ZERO_SAMPLES)

    return n_features, n_components, n_samples


def _validate_av_shapes(
    loading_covariances: list[np.ndarray] | None,
    n_features: int,
    n_components: int,
) -> None:
    if loading_covariances is None:
        return

    if len(loading_covariances) != n_features:
        raise ValueError(ERR_AV_LENGTH)

    expected_shape = (n_components, n_components)
    for index, av_i in enumerate(loading_covariances):
        av_i_arr = np.asarray(av_i)
        if av_i_arr.shape != expected_shape:
            raise ValueError(
                ERR_AV_SHAPE_DETAIL.format(index=index, shape=av_i_arr.shape)
            )


def _validate_sv_shapes(
    score_covariances: list[np.ndarray] | None, n_components: int
) -> None:
    if (
        score_covariances is None
        or not isinstance(score_covariances, list)
        or len(score_covariances) == 0
    ):
        raise ValueError(ERR_SV_LENGTH)

    expected_shape = (n_components, n_components)
    for index, sv_j in enumerate(score_covariances):
        sv_j_arr = np.asarray(sv_j)
        if sv_j_arr.shape != expected_shape:
            raise ValueError(
                ERR_SV_SHAPE_DETAIL.format(index=index, shape=sv_j_arr.shape)
            )


def _validate_shapes(
    loadings: np.ndarray,
    scores: np.ndarray,
    loading_covariances: list[np.ndarray] | None,
    score_covariances: list[np.ndarray] | None,
) -> tuple[int, int, int]:
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
    """covS = (S Sᵀ + sum_j Sv_j) / n_samples, with pattern-mode weighting."""
    _, n_samples = scores.shape
    cov_s = scores @ scores.T

    if isv.size == 0:
        if len(score_covariances) != n_samples:
            raise ValueError(ERR_SV_LEN_NO_ISV)
        for sv_j in score_covariances:
            cov_s += np.asarray(sv_j, dtype=float)
    else:
        if obscombj is None:
            raise ValueError(ERR_OBSCOMB_REQUIRED)
        if len(score_covariances) != len(obscombj):
            raise ValueError(ERR_SV_LEN_PATTERN)

        all_cols: list[int] = []
        for cols in obscombj:
            all_cols.extend(cols)

        all_cols_arr = np.asarray(all_cols, dtype=int)
        expected = np.arange(n_samples, dtype=int)
        if all_cols_arr.size != n_samples or not np.array_equal(
            np.sort(all_cols_arr), expected
        ):
            raise ValueError(ERR_OBSCOMB_COVERAGE)

        for sv_k, cols in zip(score_covariances, obscombj, strict=True):
            cov_s += len(cols) * np.asarray(sv_k, dtype=float)

    cov_s /= float(n_samples)
    return cov_s


def _eigh_psd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose symmetric PSD matrix, clipping tiny negative eigenvalues (recommended)."""
    w, v = np.linalg.eigh(matrix)
    w = np.clip(w, 0.0, None)
    return w, v


def _build_ra(eigvals: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """RA = V_S * sqrt(D) using broadcasting."""
    sqrt_eig = np.sqrt(np.clip(eigvals, 0.0, None))
    return eigvecs * sqrt_eig[np.newaxis, :]


def _build_r(eigvals_s: np.ndarray, v_s: np.ndarray, v_a: np.ndarray) -> np.ndarray:
    """R = V_A.T * diag(1/sqrt(eigvals_s)) * V_S.T with eigenvalue floor (required)."""
    sqrt_eig = np.sqrt(np.clip(eigvals_s, 0.0, None))
    inv_sqrt = 1.0 / np.maximum(sqrt_eig, _EPS_EIG)
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
    np.ndarray, np.ndarray, list[np.ndarray] | None, np.ndarray, list[np.ndarray]
]:
    """Rotate the VB-PCA latent space to a PCA-like orientation (RotateToPCA.m)."""
    loading_covariances = params.loading_covariances
    score_covariances = params.score_covariances
    isv_arr = _as_int_array(params.isv)

    n_features, _, _ = _validate_shapes(
        loadings,
        scores,
        loading_covariances,
        score_covariances,
    )

    # 1) Optional centering of S and dMu propagation
    if params.update_bias:
        mean_scores = np.mean(scores, axis=1, keepdims=True)
        d_mu = loadings @ mean_scores
        scores -= mean_scores
    else:
        d_mu = np.zeros((n_features, 1), dtype=loadings.dtype)

    # 2) covS and eigen-decomposition
    cov_s = _build_cov_s(scores, score_covariances, isv_arr, params.obscombj)
    cov_s = 0.5 * (cov_s + cov_s.T)
    eigvals_s, v_s = _eigh_psd(cov_s)

    # RA = VS*sqrt(D)
    ra = _build_ra(eigvals_s, v_s)

    # Apply RA to A and Av
    loadings[:] = loadings @ ra
    if loading_covariances is not None and len(loading_covariances) > 0:
        for i in range(n_features):
            av_i = np.asarray(loading_covariances[i], dtype=float)
            loading_covariances[i] = ra.T @ av_i @ ra

    # 3) covA in rotated basis, eigen-decompose, sort descending
    cov_a = loadings.T @ loadings
    if loading_covariances is not None and len(loading_covariances) > 0:
        for av_i in loading_covariances:
            cov_a += np.asarray(av_i, dtype=float)
    cov_a /= float(n_features)
    cov_a = 0.5 * (cov_a + cov_a.T)

    eigvals_a, v_a = _eigh_psd(cov_a)
    order = np.argsort(-eigvals_a)
    v_a = v_a[:, order]

    # Apply VA to A and Av
    loadings[:] = loadings @ v_a
    if loading_covariances is not None and len(loading_covariances) > 0:
        for i in range(n_features):
            av_i = np.asarray(loading_covariances[i], dtype=float)
            loading_covariances[i] = v_a.T @ av_i @ v_a

    # 4) Build and apply R to S and Sv
    r = _build_r(eigvals_s, v_s, v_a)
    scores[:] = r @ scores

    for j, sv_j in enumerate(score_covariances):
        sv_j_arr = np.asarray(sv_j, dtype=float)
        score_covariances[j] = r @ sv_j_arr @ r.T

    return d_mu, loadings, loading_covariances, scores, score_covariances
