"""Remove empty rows and columns from a data matrix.

This module provides :func:`rmempty`, a preprocessing helper that removes
rows and columns which are entirely "empty" and updates an optional
initialization dictionary accordingly.

Definitions of "empty":

* Dense arrays:
  - A row is empty if all entries are NaN.
  - A column is empty if all entries are NaN.

* Modern sparse inputs use stored observations, including numeric zeros.
  Stored NaNs are missing. Historical sparse handling uses signed sums.
* Explicit observation masks use their Boolean values in either mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    import numpy.typing as npt

    from .results import IndexArray

InitType = Any
Dense = np.ndarray
Sparse = sp.csr_matrix
Matrix = Dense | Sparse


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _nonempty_indices(
    x: Matrix,
    compat_mode: str,
    *,
    is_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices of non-empty rows and columns.

    Dense data use NaNs, modern sparse data use stored observations, and
    explicit masks use nonzero values. Historical sparse data use signed sums.

    Returns:
        A tuple ``(ir, ic)`` where ``ir`` are the kept row indices and
        ``ic`` are the kept column indices (both 1D integer arrays).
    """
    if sp.isspmatrix(x):
        x_csr = sp.csr_matrix(x)
        if is_mask or compat_mode == "modern":
            observed = x_csr.data != 0 if is_mask else ~np.isnan(x_csr.data)
            return _sparse_observed_axes(x_csr, observed)
        col_sums = np.asarray(x_csr.sum(axis=0)).ravel()
        row_sums = np.asarray(x_csr.sum(axis=1)).ravel()
        ic = np.where(col_sums != 0)[0]
        ir = np.where(row_sums != 0)[0]
    else:
        mask = np.asarray(x, dtype=bool) if is_mask else ~np.isnan(np.asarray(x))
        ic = np.where(mask.sum(axis=0) > 0)[0]
        ir = np.where(mask.sum(axis=1) > 0)[0]

    return ir.astype(int), ic.astype(int)


def _sparse_observed_axes(
    x: Sparse, observed: npt.NDArray[np.bool_]
) -> tuple[IndexArray, IndexArray]:
    """Return occupied axes without a full-shape mask or changing storage.

    Column counting is bounded by stored entries for very wide sparse inputs.
    The prefix sum is needed only when stored NaNs or false mask entries must
    be excluded. Numeric zeros in data remain observed.

    Returns:
        Retained row and column indices.
    """
    columns: npt.NDArray[np.int32 | np.int64]
    if observed.all():
        ir = np.flatnonzero(np.diff(x.indptr))
        columns = x.indices
    else:
        prefix = np.empty(x.nnz + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(observed, dtype=np.int64, out=prefix[1:])
        ir = np.flatnonzero(np.diff(prefix[x.indptr]))
        columns = x.indices[observed]
    ic: npt.NDArray[np.int32 | np.int64]
    if x.shape[1] > columns.size:
        ic = np.unique(columns)
    else:
        ic = np.flatnonzero(np.bincount(columns, minlength=x.shape[1]))
    return ir.astype(int, copy=False), ic.astype(int, copy=False)


def _slice_matrix_like(
    mat: Matrix | None,
    ir: np.ndarray,
    ic: np.ndarray,
    n_rows_orig: int,
    n_cols_orig: int,
) -> Matrix | None:
    """Slice a matrix or sparse matrix using kept row/column indices.

    If both dimensions are unchanged, return ``mat`` unchanged.

    Returns:
        The sliced matrix (or the original object if no slicing is needed),
        or ``None`` if ``mat`` was ``None``.
    """
    if mat is None:
        return None

    slice_rows = ir.size < n_rows_orig
    slice_cols = ic.size < n_cols_orig

    # Nothing to slice
    if not slice_rows and not slice_cols:
        return mat

    result: Matrix

    if sp.isspmatrix(mat):
        # For sparse matrices, preserve sparsity:
        # - using ir[:, None] & ic for row+col selection
        #   keeps CSR instead of densifying.
        if slice_rows and slice_cols:
            result = mat[ir[:, None], ic]
        elif slice_rows:
            result = mat[ir, :]
        else:
            result = mat[:, ic]
    # Dense ndarray
    elif slice_rows and slice_cols:
        result = mat[np.ix_(ir, ic)]
    elif slice_rows:
        result = mat[ir, :]
    else:
        result = mat[:, ic]

    return result


def _update_init_dict(
    init: InitType,
    ir: np.ndarray,
    ic: np.ndarray,
    n_rows_orig: int,
    n_cols_orig: int,
) -> InitType:
    """Update an init dict in-place to match removed rows/columns.

    If ``init`` is not a dict (or is None), it is returned unchanged.

    Returns:
        The updated init object (dict or passthrough of the original).
    """
    if init is None or not isinstance(init, dict):
        return init

    n_rows_kept = ir.size
    n_cols_kept = ic.size

    # Update A and Av along rows
    if n_rows_kept < n_rows_orig:
        if "A" in init and init["A"] is not None:
            init["A"] = init["A"][ir, :]
        if "Av" in init and init["Av"] is not None:
            av = init["Av"]
            if isinstance(av, list):
                init["Av"] = [av[i] for i in ir]
            else:
                init["Av"] = av[ir, :]

    # Update S and Sv along columns
    if n_cols_kept < n_cols_orig:
        if "S" in init and init["S"] is not None:
            init["S"] = init["S"][:, ic]
        if "Sv" in init and init["Sv"] is not None:
            sv = init["Sv"]
            if isinstance(sv, list):
                init["Sv"] = [sv[j] for j in ic]
            else:
                init["Sv"] = sv[:, ic]

    return init


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def remove_empty_entries(
    x: Matrix,
    x_probe: Matrix | None,
    init: InitType,
    *,
    compat_mode: str = "strict_legacy",
    mask: Matrix | None = None,
) -> tuple[Matrix, Matrix | None, np.ndarray, np.ndarray, InitType]:
    """Remove rows and columns that are entirely empty and update ``init``.

    Args:
        x:
            Input matrix ``(n_rows, n_cols)``.
        x_probe:
            Optional probe matrix sliced identically to ``x``.
        init:
            Optional initialization object. If it is a dict, recognized
            keys ``"A"``, ``"Av"``, ``"S"``, and ``"Sv"`` are sliced
            consistently with the retained rows and columns. Non-dict
            values are passed through unchanged.
        compat_mode:
            Compatibility mode for sparse empty-row/column handling.
            ``"strict_legacy"`` preserves sum-based legacy behavior;
            ``"modern"`` retains stored observations, including numeric zeros.
        mask:
            Optional boolean mask to drive emptiness detection; when provided
            it must match the shape of ``x``. False entries are unobserved,
            independently of data values or the compatibility mode.

    Returns:
        A tuple ``(x_out, x_probe_out, ir, ic, init_out)`` where matrices
        and init payload are sliced consistently with the retained rows and
        columns.

    Raises:
        ValueError: If ``compat_mode`` is not one of
            ``{"strict_legacy", "modern"}``.
        ValueError: If ``mask`` shape does not match ``x``.
        RuntimeError: If slicing yields ``None`` for non-null input.
    """
    mode = compat_mode.strip().lower()
    if mode not in {"strict_legacy", "modern"}:
        msg = "compat_mode must be one of {'strict_legacy', 'modern'}."
        raise ValueError(msg)

    n_rows_orig, n_cols_orig = x.shape

    # Identify kept rows/columns
    basis = mask if mask is not None else x
    ir, ic = _nonempty_indices(basis, mode, is_mask=mask is not None)

    # Fast path: nothing to remove
    if ir.size == n_rows_orig and ic.size == n_cols_orig:
        return x, x_probe, ir, ic, init

    # Slice data
    x_out = _slice_matrix_like(x, ir, ic, n_rows_orig, n_cols_orig)
    if x_out is None:
        msg = "x_out cannot be None for non-null input"
        raise RuntimeError(msg)
    x_probe_out = _slice_matrix_like(x_probe, ir, ic, n_rows_orig, n_cols_orig)

    # Update init dictionary (if dict-like)
    init_out = _update_init_dict(init, ir, ic, n_rows_orig, n_cols_orig)

    return x_out, x_probe_out, ir, ic, init_out
