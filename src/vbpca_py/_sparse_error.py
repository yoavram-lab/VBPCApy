# src/vbpca_py/_sparse_error.py
"""
Thin wrapper around the C++ errpca_pt extension, with a clearer name.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import scipy.sparse as sp

from vbpca_py.errpca_pt import errpca_pt as _errpca_pt_ext


def sparse_reconstruction_error(
    X_csr: sp.csr_matrix,
    A: np.ndarray,
    S: np.ndarray,
    num_cpu: int = 1,
) -> sp.csr_matrix:
    """Compute sparse reconstruction error matrix (X - A @ S) on nonzeros of X.

    Parameters
    ----------
    X_csr:
        Input data in CSR format (shape (n_rows, n_cols)).
    A, S:
        Factor matrices such that A @ S approximates X_csr.
    num_cpu:
        Number of worker threads to use inside the C++ routine.

    Returns
    -------
    err_csr:
        CSR matrix with the same sparsity structure as X_csr containing
        reconstruction errors.
    """
    if not sp.isspmatrix_csr(X_csr):
        raise TypeError("X_csr must be a scipy.sparse.csr_matrix")

    X_data = X_csr.data.astype(np.float64, copy=False)
    X_indices = X_csr.indices.astype(np.int32, copy=False)
    X_indptr = X_csr.indptr.astype(np.int32, copy=False)

    A = np.asarray(A, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)

    result = _errpca_pt_ext(X_data, X_indices, X_indptr, A, S, int(num_cpu))

    return sp.csr_matrix(
        (np.asarray(result["data"]), np.asarray(result["indices"]),
         np.asarray(result["indptr"])),
        shape=tuple(result["shape"]),
    )
