"""
RMS computation utilities for Variational Bayesian PCA.

This module provides a vectorized, shape-safe implementation of
`compute_rms`, which computes both:

1. The elementwise reconstruction error matrix
   E = (X - A @ S) ⊙ M   (dense or sparse), and

2. The root-mean-square (RMS) of its observed entries:
      rms = sqrt( sum(E_ij^2) / n_data )

Here:
- `X` is the observed data matrix (dense or sparse).
- `A` and `S` are factor matrices such that A @ S has shape X.
- `M` is a binary mask of the same shape as X (1 = observed, 0 = missing).
- `n_data` is the *number of observed entries*, supplied by the caller.
- For sparse X, reconstruction errors are computed by the optimized
  C++ backend via `vbpca_py._sparse_error.sparse_reconstruction_error`.

This function is hot-path (called every EM/VB iteration) so unnecessary
copies are avoided where possible.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._sparse_error import sparse_reconstruction_error


def compute_rms(
    X: np.ndarray | sp.spmatrix,
    A: np.ndarray,
    S: np.ndarray,
    M: np.ndarray | sp.spmatrix,
    n_data: int,
    *,
    num_cpu: int = 1,
) -> tuple[float, np.ndarray | sp.spmatrix]:
    """
    Compute the reconstruction RMS error and the error matrix.

    Parameters
    ----------
    X : ndarray or sparse matrix (CSR preferred)
        Observed data matrix of shape (n_features, n_samples).
        If sparse, must represent missing entries implicitly
        (i.e., absent entries treated as missing).

    A : ndarray
        Factor loadings with shape (n_features, k).

    S : ndarray
        Factor scores with shape (k, n_samples).

    M : ndarray or sparse matrix
        Mask matrix of shape equal to X. Ones mark observed entries,
        zeros mark missing. May be dense or sparse.

    n_data : int
        Number of observed entries. Provided explicitly by the caller.

    num_cpu : int, optional
        CPU threads to use on the sparse path. Values < 1 are coerced to 1.

    Returns
    -------
    rms : float
        Root-mean-square reconstruction error over observed entries.

    err : ndarray or sparse matrix
        Elementwise error matrix (X − A @ S) ⊙ M. Sparse if X was sparse.

    Raises
    ------
    ValueError
        If shapes are incompatible.
    """
    # Handle trivial empty case
    if X.size == 0:
        # Return a harmless empty error matrix of matching shape
        empty_err: np.ndarray | sp.spmatrix
        if sp.issparse(X):
            empty_err = sp.csr_matrix(X.shape)
        else:
            empty_err = np.empty_like(X)
        return np.nan, empty_err

    # Normalize num_cpu
    if num_cpu < 1:
        num_cpu = 1

    # ---- Shape validation -------------------------------------------------
    if A.ndim != 2 or S.ndim != 2:
        raise ValueError("A and S must be 2-D arrays.")

    n_features, n_samples = X.shape
    a_features, k1 = A.shape
    k2, s_samples = S.shape

    if a_features != n_features:
        raise ValueError(f"A has {a_features} rows but X has {n_features} rows.")
    if s_samples != n_samples:
        raise ValueError(f"S has {s_samples} columns but X has {n_samples} columns.")
    if k1 != k2:
        raise ValueError(
            f"Incompatible latent dimensions: A.shape[1]={k1}, S.shape[0]={k2}."
        )

    # ---- Sparse X path ----------------------------------------------------
    if sp.issparse(X):
        # Ensure CSR for X
        if not sp.isspmatrix_csr(X):
            X = X.tocsr()

        # Compute sparse reconstruction error using C++ helper
        err = sparse_reconstruction_error(X, A, S, num_cpu=num_cpu)

        # Some sparse ops / backends may return COO or other formats.
        # Normalise to CSR before applying the mask.
        if not sp.isspmatrix_csr(err):
            err = err.tocsr()

        # Apply mask M (CSR preferred, but dense masks are allowed)
        if sp.issparse(M):
            if not sp.isspmatrix_csr(M):
                M = M.tocsr()
            err = err.multiply(M)
        else:
            # Dense mask; csr_matrix.multiply handles this without densifying
            err = err.multiply(M)

        # Ensure final sparse error matrix is CSR
        if not sp.isspmatrix_csr(err):
            err = err.tocsr()

    # ---- Dense X path -----------------------------------------------------
    else:
        # Residual
        residual = X - A @ S

        # Apply mask
        if sp.issparse(M):
            M = M.toarray()
        err = residual * M

    # ---- RMS --------------------------------------------------------------
    if sp.issparse(err):
        # err.data already stores only nonzero residuals
        rms = float(np.sqrt(np.sum(err.data**2) / n_data))
    else:
        rms = float(np.sqrt(np.sum(err**2) / n_data))

    return rms, err
