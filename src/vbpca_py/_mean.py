"""
Mean-subtraction utilities for Variational Bayesian PCA.

This module provides :func:`subtract_mu`, a helper that subtracts a row-wise
mean vector from dense or sparse data matrices and an optional probe matrix.

Behavior is aligned to the original MATLAB implementation:

    SubtractMu(Mu, X, M, Xprobe, Mprobe, update_bias)

Dense:
    X = X - repmat(Mu, 1, n2) .* M
    Xprobe updated similarly with Mprobe.

Sparse:
    X = subtract_mu(X, Mu)  (mex / C++), which subtracts Mu[row] from each
    stored entry in that row and replaces exact zeros with EPS.
    Mask is ignored because missing entries are implicit in sparse storage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .subtract_mu_from_sparse import subtract_mu_from_sparse

Matrix = np.ndarray | sp.spmatrix

__all__ = ["ProbeMatrices", "subtract_mu"]

# ============================================================
# Error messages
# ============================================================

ERR_MU_SHAPE = "mu must have shape (n_rows,) or (n_rows, 1)."
ERR_MASK_SHAPE = "mask must have the same shape as x."
ERR_MASK_SPARSE_WITH_DENSE_X = "mask must be dense when x is dense."
ERR_MASK_PROBE_REQUIRED = "mask_probe must be provided when x_probe is not None."
ERR_PROBE_ROWS = "x_probe must have the same number of rows as x."
ERR_PROBE_SPARSE_TYPE = "x_probe must be sparse when x is sparse."


# ============================================================
# Small containers
# ============================================================


@dataclass(slots=True)
class ProbeMatrices:
    """Container for probe matrix and its mask.

    Attributes:
        x:
            Probe data matrix with the same number of rows as x.
        mask:
            Mask for the probe matrix. Required (non-None) in the dense path
            when update_bias is True. Ignored for sparse inputs, matching
            MATLAB behavior where missing entries are implicit.
    """

    x: Matrix
    mask: Matrix | None


# ============================================================
# Internal helpers
# ============================================================


def _normalize_mu_column(mu: np.ndarray, n_rows: int) -> np.ndarray:
    """Return mu as a column vector of shape (n_rows, 1).

    Raises:
        ValueError: If ``mu`` cannot be reshaped to ``(n_rows, 1)``.
    """
    mu_arr = np.asarray(mu, dtype=float)

    if mu_arr.ndim == 1:
        if mu_arr.shape[0] != n_rows:
            raise ValueError(ERR_MU_SHAPE)
        return mu_arr[:, np.newaxis]

    if mu_arr.ndim == 2 and mu_arr.shape == (n_rows, 1):
        return mu_arr.astype(float, copy=False)

    raise ValueError(ERR_MU_SHAPE)


def _ensure_dense_mask(mask: Matrix, shape: tuple[int, int]) -> np.ndarray:
    """Return a dense mask of the requested shape.

    Raises:
        ValueError: If ``mask`` does not match ``shape``.
    """
    mask_arr = mask.toarray() if sp.isspmatrix(mask) else np.asarray(mask)  # type: ignore[union-attr]

    if mask_arr.shape != shape:
        raise ValueError(ERR_MASK_SHAPE)

    return mask_arr


def _subtract_dense(mu_col: np.ndarray, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract row-wise mu from dense x with a dense mask.

    Returns:
        Dense matrix with mean subtracted on observed entries.
    """
    return x - mu_col * mask


def _subtract_sparse(mu_col: np.ndarray, x: sp.spmatrix) -> sp.csr_matrix:
    """Subtract row-wise mu from sparse CSR x using the C++ helper.

    Returns:
        CSR matrix with mean subtracted on stored entries.
    """
    if not sp.isspmatrix_csr(x):
        x = x.tocsr()

    n_rows, n_cols = x.shape
    data = x.data.astype(float, copy=False)
    indices = x.indices.astype(np.int32, copy=False)
    indptr = x.indptr.astype(np.int32, copy=False)
    shape_tuple: tuple[int, int] = (n_rows, n_cols)

    new_data = subtract_mu_from_sparse(
        data,
        indices,
        indptr,
        shape_tuple,
        mu_col[:, 0],  # 1D Mu
    )

    return sp.csr_matrix((new_data, indices, indptr), shape=x.shape)


def _is_matlab_empty_matrix(mat: Matrix) -> bool:
    """Match MATLAB isempty semantics: empty if any dimension is zero.

    Returns:
        True if any dimension is zero.
    """
    # Works for both dense and sparse (SciPy sparse has .shape).
    n_rows, n_cols = mat.shape
    return n_rows == 0 or n_cols == 0


def _subtract_sparse_branch(
    mu_col: np.ndarray,
    x: sp.spmatrix,
    probe: ProbeMatrices | None,
    n_rows: int,
) -> tuple[sp.spmatrix, Matrix | None]:
    """Handle sparse subtraction branch.

    Returns:
        Mean-subtracted sparse data and optional probe output.

    Raises:
        ValueError: If probe sparsity or row shape is invalid.
    """
    x_out = _subtract_sparse(mu_col, x)

    x_probe_out: Matrix | None = None
    if probe is not None:
        x_probe = probe.x

        # MATLAB: if isempty(Xprobe) do nothing.
        if not _is_matlab_empty_matrix(x_probe):
            if not sp.isspmatrix(x_probe):
                raise ValueError(ERR_PROBE_SPARSE_TYPE)

            n_rows_probe, _ = x_probe.shape
            if n_rows_probe != n_rows:
                raise ValueError(ERR_PROBE_ROWS)

            x_probe_out = _subtract_sparse(mu_col, x_probe)

    return x_out, x_probe_out


def _subtract_dense_branch(
    mu_col: np.ndarray,
    x: np.ndarray,
    mask: Matrix,
    probe: ProbeMatrices | None,
    shape: tuple[int, int],
) -> tuple[np.ndarray, Matrix | None]:
    """Handle dense subtraction branch.

    Returns:
        Mean-subtracted dense data and optional probe output.
    """
    mask_dense = _ensure_dense_mask(mask, shape)
    x_out = _subtract_dense(mu_col, x, mask_dense)

    x_probe_out = None
    if probe is not None:
        x_probe = np.array(probe.x, dtype=float, copy=False)
        mask_probe = _ensure_dense_mask(probe.mask, x_probe.shape)
        x_probe_out = _subtract_dense(mu_col, x_probe, mask_probe)

    return x_out, x_probe_out


# ============================================================
# Public API
# ============================================================


def subtract_mu(
    mu: np.ndarray,
    x: Matrix,
    mask: Matrix,
    probe: ProbeMatrices | None = None,
    *,
    update_bias: bool = True,
) -> tuple[Matrix, Matrix | None]:
    """Subtract a row-wise mean vector from data and optional probe matrices.

    Mirrors MATLAB SubtractMu.m behavior for dense and sparse inputs.

    Returns:
        Tuple of mean-subtracted data and optional probe matrix.

    Raises:
        ValueError: If shapes or sparsity expectations are violated.
    """
    # Fast path: do nothing (and do not validate shapes), mirroring MATLAB.
    if not update_bias:
        probe_x = probe.x if probe is not None else None
        return x, probe_x

    n_rows, n_cols = x.shape
    mu_col = _normalize_mu_column(mu, n_rows)

    # ------------------------------------------------------------------
    # Sparse branch (mask ignored, matching MATLAB behavior)
    # ------------------------------------------------------------------
    if sp.isspmatrix(x):
        return _subtract_sparse_branch(mu_col, x, probe, n_rows)

    # ------------------------------------------------------------------
    # Dense branch
    # ------------------------------------------------------------------
    if not sp.isspmatrix(mask):
        x_arr = np.array(x, dtype=float, copy=False)
        return _subtract_dense_branch(mu_col, x_arr, mask, probe, (n_rows, n_cols))

    # ------------------------------------------------------------------
    # Mask is sparse but x is dense: mirror MATLAB by rejecting
    # ------------------------------------------------------------------
    raise ValueError(ERR_MASK_SPARSE_WITH_DENSE_X)
