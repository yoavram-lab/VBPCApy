"""Thin wrapper around the C++ errpca_pt extension, with a clearer name.

Guarantees:
- Input is CSR.
- Output is CSR with the SAME sparsity structure as x_csr.
- Residuals computed only on stored entries: Err = X - A @ S on the CSR pattern.
"""

from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

from vbpca_py.errpca_pt import errpca_pt as _errpca_pt_ext


def _resolve_num_cpu(num_cpu: int) -> int:
    """Resolve worker count for sparse kernels.

    Semantics:
    - ``num_cpu > 0``: use the provided value.
    - ``num_cpu <= 0``: read ``VBPCA_NUM_THREADS`` if set to a positive integer,
      else fall back to ``1``.

    Returns:
                Effective worker count used for sparse kernels.
    """
    if num_cpu > 0:
        return num_cpu

    env_val = os.environ.get("VBPCA_NUM_THREADS", "").strip()
    if env_val:
        try:
            parsed = int(env_val)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed

    return 1


def sparse_reconstruction_error(
    x_csr: sp.csr_matrix,
    loadings: np.ndarray,
    scores: np.ndarray,
    num_cpu: int = 1,
) -> sp.csr_matrix:
    """Compute sparse reconstruction error matrix (X - A @ S) on stored entries of X.

    Parameters
    ----------
    x_csr:
        Input data in CSR format, shape (n_rows, n_cols). The sparsity pattern
        defines the observed set (legacy MATLAB semantics).
    loadings:
        A matrix of shape (n_rows, n_components).
    scores:
        S matrix of shape (n_components, n_cols).
    num_cpu:
        Worker threads used by the C++ routine.

    Returns:
    -------
    err_csr:
        CSR matrix with the same sparsity structure as x_csr containing residuals.

    Raises:
        TypeError: If ``x_csr`` is not a CSR matrix.
        ValueError: If shapes of ``loadings`` or ``scores`` are incompatible
            with ``x_csr``.
    """
    if not sp.isspmatrix_csr(x_csr):
        err_msg = "x_csr must be a scipy.sparse.csr_matrix"
        raise TypeError(err_msg)

    n_rows, n_cols = x_csr.shape
    loadings = np.asarray(loadings, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    if loadings.shape[0] != n_rows:
        err_msg = f"loadings has {loadings.shape[0]} rows but data has {n_rows} rows."
        raise ValueError(err_msg)
    if scores.shape[1] != n_cols:
        err_msg = f"scores has {scores.shape[1]} columns but data has {n_cols} columns."
        raise ValueError(err_msg)
    if loadings.shape[1] != scores.shape[0]:
        err_msg = (
            "Incompatible latent dims: "
            f"loadings.shape[1]={loadings.shape[1]}, scores.shape[0]={scores.shape[0]}."
        )
        raise ValueError(err_msg)

    num_cpu = _resolve_num_cpu(int(num_cpu))

    # Ensure expected dtypes for the extension
    x_data = x_csr.data.astype(np.float64, copy=False)
    x_indices = x_csr.indices.astype(np.int32, copy=False)
    x_indptr = x_csr.indptr.astype(np.int32, copy=False)

    result = _errpca_pt_ext(x_data, x_indices, x_indptr, loadings, scores, num_cpu)

    data = np.asarray(result["data"], dtype=np.float64)
    indices = np.asarray(result["indices"], dtype=np.int32)
    indptr = np.asarray(result["indptr"], dtype=np.int32)
    shape = tuple(result["shape"])

    # Defensive: ensure the extension produced a shape consistent with the input
    if shape != x_csr.shape:
        err_msg = (
            f"Extension returned shape {shape}, expected {x_csr.shape}. "
            "Check that scores.shape[1] matches x_csr.shape[1]."
        )
        raise ValueError(err_msg)

    # CSR construction; structure should match input by design
    return sp.csr_matrix((data, indices, indptr), shape=shape)
