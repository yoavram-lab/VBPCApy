"""Mean-subtraction utilities for Variational Bayesian PCA.

This module provides :func:`subtract_mu`, a small helper that subtracts a
row-wise mean vector from dense or sparse data matrices and an optional
probe matrix.

For sparse CSR inputs, the heavy lifting is delegated to the C++
extension :mod:`vbpca_py.subtract_mu_from_sparse`, which operates
directly on CSR components and preserves the sparsity structure
(replacing exact zeros with a small epsilon for numerical stability).

For dense inputs, the behavior mirrors the original MATLAB/Python
implementation used in pca_full:

* For the main data matrix x:

    x_out = x - (mu * mask)

  where mu is broadcast along columns and mask is a 0/1 (or
  boolean) mask of the same shape as x. Entries with mask == 0 are
  left unchanged.

* For the probe matrix x_probe (if provided):

    x_probe_out = x_probe - (mu * mask_probe)

  with an analogous mask.
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
ERR_MASK_PROBE_REQUIRED = "mask_probe must be provided when x_probe is not None."
ERR_PROBE_ROWS = "x_probe must have the same number of rows as x."


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
            when update_bias is True. Ignored for sparse inputs, matching the
            legacy behavior where missing entries are implicit.
    """

    x: Matrix
    mask: Matrix | None


# ============================================================
# Internal helpers
# ============================================================


def _normalize_mu_column(mu: np.ndarray, n_rows: int) -> np.ndarray:
    """Return mu as a column vector of shape (n_rows, 1)."""
    mu_arr = np.asarray(mu, dtype=float)

    if mu_arr.ndim == 1:
        if mu_arr.shape[0] != n_rows:
            raise ValueError(ERR_MU_SHAPE)
        return mu_arr[:, np.newaxis]

    if mu_arr.ndim == 2 and mu_arr.shape == (n_rows, 1):
        return mu_arr.astype(float, copy=False)

    raise ValueError(ERR_MU_SHAPE)


def _ensure_dense_mask(mask: Matrix, shape: tuple[int, int]) -> np.ndarray:
    """Return a dense mask of the requested shape."""
    mask_arr = mask.toarray() if sp.isspmatrix(mask) else np.asarray(mask)  # type: ignore[union-attr]

    if mask_arr.shape != shape:
        raise ValueError(ERR_MASK_SHAPE)

    return mask_arr


def _subtract_dense(
    mu_col: np.ndarray,
    x: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Subtract row-wise mu from dense x with a dense mask.

    mu_col has shape (n_rows, 1), mask and x have shape (n_rows, n_cols).
    The result mirrors x - (mu * mask) from the legacy implementation.
    """
    # Mask is usually 0/1 or boolean; broadcasting works as expected.
    return x - mu_col * mask


def _subtract_sparse(
    mu_col: np.ndarray,
    x: sp.spmatrix,
) -> sp.csr_matrix:
    """Subtract row-wise mu from sparse CSR x using the C++ helper.

    The mask is intentionally ignored, matching the original behavior:
    all stored entries in a given row are shifted by the corresponding
    mu value.
    """
    if not sp.isspmatrix_csr(x):
        x = x.tocsr()

    n_rows, n_cols = x.shape
    data = x.data.astype(float, copy=False)
    indices = x.indices.astype(np.int32, copy=False)
    indptr = x.indptr.astype(np.int32, copy=False)
    shape_tuple: tuple[int, int] = (n_rows, n_cols)

    # C++ helper returns a new data array with the same nnz.
    new_data = subtract_mu_from_sparse(
        data,
        indices,
        indptr,
        shape_tuple,
        mu_col[:, 0],  # pass as 1D (length n_rows)
    )

    return sp.csr_matrix((new_data, indices, indptr), shape=x.shape)


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

    This is the modern, typed version of the original inlined
    subtract_mu from pca_full. It supports both dense and sparse
    inputs, an optional probe matrix, and an explicit update_bias
    flag.

    Args:
        mu:
            Mean vector of shape (n_rows,) or (n_rows, 1).
        x:
            Data matrix of shape (n_rows, n_cols), dense or sparse.
        mask:
            Mask with the same shape as x. Entries equal to zero mark
            positions where the mean should *not* be subtracted. For
            sparse x, this mask is currently ignored (to match the
            original MATLAB/Python behavior), since missing entries are
            implicit.
        probe:
            Optional container with a probe/test data matrix and its mask.
            The probe matrix must have the same number of rows as x.
            For dense probes, probe.mask must be provided when update_bias
            is True. For sparse probes, the mask is ignored.
        update_bias:
            If False, x and probe.x are returned unchanged and no
            work is performed.

    Returns:
        x_out:
            Matrix x with row-wise mean subtracted (if update_bias).
        x_probe_out:
            Probe matrix with mean subtracted, or None if no probe
            was provided.

    Raises:
        ValueError:
            If shapes of mu, x, or masks are incompatible.
    """
    # Fast path: do nothing (and do not validate shapes).
    if not update_bias:
        probe_x = probe.x if probe is not None else None
        return x, probe_x

    n_rows, n_cols = x.shape
    mu_col = _normalize_mu_column(mu, n_rows)

    # ------------------------------------------------------------------
    # Sparse branch: ignore mask, match legacy behavior.
    # ------------------------------------------------------------------
    if sp.isspmatrix(x):
        x_out = _subtract_sparse(mu_col, x)

        x_probe_out: Matrix | None = None
        if probe is not None and probe.x.size > 0:
            x_probe = probe.x
            if not sp.isspmatrix(x_probe):
                raise ValueError(ERR_PROBE_ROWS)
            n_rows_probe, _ = x_probe.shape
            if n_rows_probe != n_rows:
                raise ValueError(ERR_PROBE_ROWS)
            x_probe_out = _subtract_sparse(mu_col, x_probe)

        return x_out, x_probe_out

    # ------------------------------------------------------------------
    # Dense branch
    # ------------------------------------------------------------------
    x_arr = np.asarray(x, dtype=float)
    mask_arr = _ensure_dense_mask(mask, (n_rows, n_cols))

    x_out = _subtract_dense(mu_col, x_arr, mask_arr)

    x_probe_out: Matrix | None = None
    if probe is not None and probe.x.size > 0:
        x_probe_arr = np.asarray(probe.x, dtype=float)
        n_rows_probe, n_cols_probe = x_probe_arr.shape
        if n_rows_probe != n_rows:
            raise ValueError(ERR_PROBE_ROWS)

        if probe.mask is None:
            raise ValueError(ERR_MASK_PROBE_REQUIRED)

        mask_probe_arr = _ensure_dense_mask(probe.mask, (n_rows_probe, n_cols_probe))
        x_probe_out = _subtract_dense(mu_col, x_probe_arr, mask_probe_arr)

    return x_out, x_probe_out
