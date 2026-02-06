"""Remove empty rows and columns from a data matrix.

This module provides :func:`rmempty`, a preprocessing helper that removes
rows and columns which are entirely "empty" and updates an optional
initialization dictionary accordingly.

Definitions of "empty":

* Dense arrays:
  - A row is empty if all entries are NaN.
  - A column is empty if all entries are NaN.

* Sparse arrays:
  - A row is empty if its sum is exactly zero.
  - A column is empty if its sum is exactly zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

InitType = Any
Matrix = np.ndarray | sp.spmatrix


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _nonempty_indices(x: Matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return indices of non-empty rows and columns.

    For dense ``x``, emptiness means all-NaN.
    For sparse ``x``, emptiness means zero-sum row/column.

    Returns:
        A tuple ``(ir, ic)`` where ``ir`` are the kept row indices and
        ``ic`` are the kept column indices (both 1D integer arrays).
    """
    if sp.isspmatrix(x):
        col_sums = np.asarray(x.sum(axis=0)).ravel()
        row_sums = np.asarray(x.sum(axis=1)).ravel()
        ic = np.where(col_sums != 0)[0]
        ir = np.where(row_sums != 0)[0]
    else:
        mask = ~np.isnan(x)
        ic = np.where(mask.sum(axis=0) > 0)[0]
        ir = np.where(mask.sum(axis=1) > 0)[0]

    return ir.astype(int), ic.astype(int)


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
            result = mat[ir[:, None], ic]  # type: ignore[index]
        elif slice_rows:
            result = mat[ir, :]  # type: ignore[index]
        elif slice_cols:
            result = mat[:, ic]  # type: ignore[index]
        else:  # safety fallback, though logically unreachable
            result = mat
    # Dense ndarray
    elif slice_rows and slice_cols:
        result = mat[np.ix_(ir, ic)]
    elif slice_rows:
        result = mat[ir, :]
    elif slice_cols:
        result = mat[:, ic]
    else:  # safety fallback
        result = mat

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

    Returns:
        A tuple ``(x_out, x_probe_out, ir, ic, init_out)`` where matrices
        and init payload are sliced consistently with the retained rows and
        columns.
    """
    n_rows_orig, n_cols_orig = x.shape

    # Identify kept rows/columns
    ir, ic = _nonempty_indices(x)

    # Fast path: nothing to remove
    if ir.size == n_rows_orig and ic.size == n_cols_orig:
        return x, x_probe, ir, ic, init

    # Slice data
    x_out = _slice_matrix_like(x, ir, ic, n_rows_orig, n_cols_orig)
    x_probe_out = _slice_matrix_like(x_probe, ir, ic, n_rows_orig, n_cols_orig)

    # Update init dictionary (if dict-like)
    init_out = _update_init_dict(init, ir, ic, n_rows_orig, n_cols_orig)

    return x_out, x_probe_out, ir, ic, init_out
