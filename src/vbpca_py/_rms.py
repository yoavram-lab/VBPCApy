"""
RMS computation utilities for Variational Bayesian PCA.

This module provides a vectorized, shape-safe implementation of
compute_rms, which computes both:

1. The elementwise reconstruction error matrix
   E = (data - loadings @ scores) ⊙ mask   (dense or sparse), and

2. The root-mean-square (RMS) of its observed entries:
      rms = sqrt( sum(E_ij^2) / n_observed )

Notation mapping to the original paper / MATLAB code:
- data     ↔ X
- loadings ↔ A
- scores   ↔ S
- mask     ↔ M

Here:
- data is the observed data matrix (dense or sparse).
- loadings and scores are factor matrices such that
  loadings @ scores has the same shape as data.
- mask is a binary mask of the same shape as data
  (1 = observed, 0 = missing).
- config.n_observed is the *number of observed entries*, supplied
  by the caller.
- config.num_cpu configures CPU threads used on the sparse path.
- For sparse data, reconstruction errors are computed by the optimized
  C++ backend via :func:`vbpca_py._sparse_error.sparse_reconstruction_error`.

This function is hot-path (called every EM/VB iteration) so unnecessary
copies are avoided where possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ._sparse_error import sparse_reconstruction_error

# ---------------------------------------------------------------------------
# Configuration container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RmsConfig:
    """Configuration options for :func:`compute_rms`.

    Attributes:
    ----------
    n_observed:
        Number of observed (non-missing) entries in the data.
    num_cpu:
        Number of CPU threads to use for the sparse path. Values < 1
        are coerced to 1.
    """

    n_observed: int
    num_cpu: int = 1


# ---------------------------------------------------------------------------
# Error message constants
# ---------------------------------------------------------------------------

ERR_FACTORS_2D = "loadings and scores must be 2-D arrays."
ERR_LOADINGS_ROWS = "loadings has {load_rows} rows but data has {data_rows} rows."
ERR_SCORES_COLS = "scores has {score_cols} columns but data has {data_cols} columns."
ERR_LATENT_DIM = (
    "Incompatible latent dimensions: "
    "loadings.shape[1]={k_loadings}, scores.shape[0]={k_scores}."
)


def compute_rms(
    data: np.ndarray | sp.spmatrix,
    loadings: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray | sp.spmatrix,
    config: RmsConfig,
) -> tuple[float, np.ndarray | sp.spmatrix]:
    """
    Compute the reconstruction RMS error and the error matrix.

    Parameters
    ----------
    data : ndarray or sparse matrix (CSR preferred)
        Observed data matrix of shape (n_features, n_samples).
        If sparse, must represent missing entries implicitly
        (i.e., absent entries treated as missing).

    loadings : ndarray
        Factor loadings with shape (n_features, k).

    scores : ndarray
        Factor scores with shape (k, n_samples).

    mask : ndarray or sparse matrix
        Mask matrix of shape equal to data. Ones mark observed entries,
        zeros mark missing. May be dense or sparse.

    config : RmsConfig
        Configuration for the RMS computation, including the number of
        observed entries and the number of CPU threads for the sparse path.

    Returns:
    -------
    rms : float
        Root-mean-square reconstruction error over observed entries.

    err : ndarray or sparse matrix
        Elementwise error matrix (data - loadings @ scores) ⊙ mask.
        Sparse if data was sparse.

    Raises:
    ------
    ValueError
        If shapes are incompatible.
    """
    # Handle trivial empty case early
    if data.size == 0:
        empty_err: np.ndarray | sp.spmatrix
        if sp.issparse(data):
            empty_err = sp.csr_matrix(data.shape)
        else:
            empty_err = np.empty_like(data)
        return float("nan"), empty_err

    # Normalize num_cpu from config
    num_cpu = max(int(config.num_cpu), 1)

    _validate_shapes(data, loadings, scores)

    if sp.issparse(data):
        err = _sparse_error_matrix(data, loadings, scores, mask, num_cpu)
    else:
        err = _dense_error_matrix(data, loadings, scores, mask)

    rms = _rms_from_error(err, config.n_observed)
    return rms, err


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_shapes(
    data: np.ndarray | sp.spmatrix,
    loadings: np.ndarray,
    scores: np.ndarray,
) -> None:
    """Raise ValueError if shapes of data, loadings, scores are incompatible."""
    if loadings.ndim != 2 or scores.ndim != 2:
        raise ValueError(ERR_FACTORS_2D)

    n_features, n_samples = data.shape
    load_rows, k_loadings = loadings.shape
    k_scores, score_cols = scores.shape

    if load_rows != n_features:
        raise ValueError(
            ERR_LOADINGS_ROWS.format(load_rows=load_rows, data_rows=n_features)
        )
    if score_cols != n_samples:
        raise ValueError(
            ERR_SCORES_COLS.format(score_cols=score_cols, data_cols=n_samples)
        )
    if k_loadings != k_scores:
        raise ValueError(
            ERR_LATENT_DIM.format(k_loadings=k_loadings, k_scores=k_scores)
        )


def _sparse_error_matrix(
    data: sp.spmatrix,
    loadings: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray | sp.spmatrix,
    num_cpu: int,
) -> sp.csr_matrix:
    """Compute masked sparse error matrix for sparse data."""
    # Ensure CSR for data
    if not sp.isspmatrix_csr(data):
        data = data.tocsr()

    # Compute sparse reconstruction error using C++ helper
    err = sparse_reconstruction_error(data, loadings, scores, num_cpu=num_cpu)

    # Normalise to CSR before applying the mask.
    if not sp.isspmatrix_csr(err):
        err = err.tocsr()

    # Apply mask (CSR preferred, but dense masks are allowed)
    if sp.issparse(mask):
        if not sp.isspmatrix_csr(mask):
            mask = mask.tocsr()
        err = err.multiply(mask)
    else:
        # Dense mask; csr_matrix.multiply handles this without densifying
        err = err.multiply(mask)

    # Ensure we return CSR
    if not sp.isspmatrix_csr(err):
        err = err.tocsr()

    return err


def _dense_error_matrix(
    data: np.ndarray,
    loadings: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray | sp.spmatrix,
) -> np.ndarray:
    """Compute masked dense error matrix for dense data."""
    residual = data - loadings @ scores

    if sp.issparse(mask):
        mask = mask.toarray()

    return residual * mask


def _rms_from_error(
    err: np.ndarray | sp.spmatrix,
    n_observed: int,
) -> float:
    """Compute RMS from an error matrix and number of observed entries."""
    if sp.issparse(err):
        return float(np.sqrt(np.sum(err.data**2) / n_observed))
    return float(np.sqrt(np.sum(err**2) / n_observed))
