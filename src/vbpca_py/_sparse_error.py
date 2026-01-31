"""Thin wrapper around the C++ errpca_pt extension, with a clearer name."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from vbpca_py.errpca_pt import errpca_pt as _errpca_pt_ext


def sparse_reconstruction_error(
    x_csr: sp.csr_matrix,
    loadings: np.ndarray,
    scores: np.ndarray,
    num_cpu: int = 1,
) -> sp.csr_matrix:
    """Compute sparse reconstruction error matrix (X - A @ S) on nonzeros of X.

    Parameters
    ----------
    x_csr:
        Input data in CSR format (shape (n_rows, n_cols)).
    loadings, scores:
        Factor matrices such that ``loadings @ scores`` approximates ``x_csr``.
        In the notation of the original MATLAB code and paper, these correspond
        to A (loadings) and S (scores).
    num_cpu:
        Number of worker threads to use inside the C++ routine.

    Returns:
    -------
    err_csr:
        CSR matrix with the same sparsity structure as ``x_csr`` containing
        reconstruction errors.
    """
    if not sp.isspmatrix_csr(x_csr):
        msg = "x_csr must be a scipy.sparse.csr_matrix"
        raise TypeError(msg)

    x_data = x_csr.data.astype(np.float64, copy=False)
    x_indices = x_csr.indices.astype(np.int32, copy=False)
    x_indptr = x_csr.indptr.astype(np.int32, copy=False)

    loadings = np.asarray(loadings, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    # All heavy lifting is done in the C++ extension: CSR -> Eigen, A @ S,
    # threaded error computation on the sparsity pattern of X, CSR back.
    result = _errpca_pt_ext(x_data, x_indices, x_indptr, loadings, scores, int(num_cpu))

    return sp.csr_matrix(
        (
            np.asarray(result["data"]),
            np.asarray(result["indices"]),
            np.asarray(result["indptr"]),
        ),
        shape=tuple(result["shape"]),
    )
