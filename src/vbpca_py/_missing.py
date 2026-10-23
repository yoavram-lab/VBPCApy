# src/vbpca_py/_missing.py
"""
Helpers for analysing missing-value patterns.

This module provides utilities for identifying *unique* patterns of
missing/observed entries across columns of a mask matrix. It is used
upstream to group columns that share the same missingness structure,
which can then share computations such as covariance patterns.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from ._types import Float64Array, IndexArray, InputArray, InputMask, InputMatrix
from .options import CompatMode

logger = logging.getLogger(__name__)

MASK_WRONG_DIM = "mask must be a 2D array."
Matrix = Float64Array | sp.spmatrix


def _missing_patterns(
    mask: InputArray,
    *,
    verbose: bool = False,
) -> tuple[int, list[list[int]], IndexArray]:
    """Identify unique missingness patterns across columns.

    Columns that share the same pattern of missing/observed entries
    (i.e. identical columns in ``mask``) are grouped together.

    The function is agnostic to the exact coding of missingness:
    it simply treats each column as a pattern vector and finds unique
    columns via ``np.unique(mask.T, axis=0)``. Typical conventions
    might be:
    - 1 = observed, 0 = missing, or
    - True = observed, False = missing.

    Args:
        mask:
            2D array of shape ``(n_rows, n_cols)`` indicating which entries
            are missing/observed. Any dtype is allowed; only pattern equality
            matters.
        verbose:
            If ``True``, emits debug-level logging about the patterns found.
            Defaults to ``False``.

    Returns:
        n_patterns:
            Number of unique missingness patterns across columns.
        pattern_columns:
            A list of length ``n_patterns`` where
            ``pattern_columns[j]`` is a list of column indices (0-based)
            that share pattern ``j``.
        column_pattern_index:
            1D integer array of length ``n_cols`` where
            ``column_pattern_index[i]`` is the pattern index (0-based)
            for column ``i``.

    Raises:
        ValueError: If ``mask`` is not a 2D array.

    Notes:
        This behaviour differs slightly from the original MATLAB
        ``miscomb`` helper, which returned empty groupings when all
        columns had unique patterns. Here, the groupings and per-column
        pattern indices are *always* returned for consistency.
    """
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(MASK_WRONG_DIM)

    _n_rows, n_cols = mask.shape  # n_rows is unused but kept for clarity

    # Degenerate case: no columns.
    if n_cols == 0:
        if verbose:
            logger.debug("[_missing_patterns] matrix has zero columns; no patterns.")
        column_pattern_index = np.empty(0, dtype=int)
        return 0, [], column_pattern_index

    # Each column's pattern is its n_rows-length vector; we look for
    # unique columns of mask, so we apply np.unique to mask.T.
    unique_patterns, column_pattern_index = np.unique(
        mask.T, axis=0, return_inverse=True
    )
    n_patterns = int(unique_patterns.shape[0])

    # Build groups of columns per pattern.
    pattern_columns: list[list[int]] = [[] for _ in range(n_patterns)]
    for col_idx, pattern_idx in enumerate(column_pattern_index):
        pattern_columns[int(pattern_idx)].append(col_idx)

    if verbose:
        logger.debug(
            "[_missing_patterns] found %d missingness pattern(s) across %d column(s).",
            n_patterns,
            n_cols,
        )
        logger.debug(
            "[_missing_patterns] column pattern index (first 50): %s",
            column_pattern_index[:50],
        )

    return n_patterns, pattern_columns, column_pattern_index


def _xprobe_sparse(
    x: sp.csr_matrix,
    fraction: float,
    rng: np.random.Generator,
    mask: InputMask | None = None,
    *,
    compat_mode: CompatMode = CompatMode.MODERN,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Hold out probe entries from a sparse matrix.

    Returns:
        Tuple of (x_masked, xprobe) as CSR matrices.

    Raises:
        ValueError: If the mask shape is invalid or has no observed entries.
    """
    x_csr = sp.csr_matrix(x, copy=True)
    if compat_mode == CompatMode.MODERN:
        x_csr.sum_duplicates()
    if mask is None:
        observed = x_csr.copy()
        observed.data = np.ones(observed.nnz, dtype=bool)
    elif sp.issparse(mask):
        observed = sp.csr_matrix(mask, dtype=bool, copy=True)
        observed.eliminate_zeros()
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != x_csr.shape:
            msg = "mask must have the same shape as x"
            raise ValueError(msg)
        observed = sp.csr_matrix(mask_arr)

    if observed.shape != x_csr.shape:
        msg = "mask must have the same shape as x"
        raise ValueError(msg)

    rows, cols = observed.nonzero()
    if compat_mode == CompatMode.MODERN and len(rows):
        # A mask can request absent zeros, but stored NaNs remain missing.
        observed.sum_duplicates()
        rows, cols = observed.nonzero()
        present = ~np.isnan(np.asarray(x_csr[rows, cols]).ravel())
        rows, cols = rows[present], cols[present]
    n_observed = len(rows)
    if n_observed == 0:
        msg = "cannot create a probe set from data with no observed entries"
        raise ValueError(msg)
    n_probe = min(n_observed, max(1, round(n_observed * fraction)))
    probe_idx = rng.choice(n_observed, size=n_probe, replace=False)

    rows, cols = rows[probe_idx], cols[probe_idx]
    sp_vals = np.array(x_csr[rows, cols]).ravel()

    # Construct directly from coordinates so explicitly observed zero values
    # remain structurally present in the sparse probe matrix.
    xprobe = sp.csr_matrix((sp_vals, (rows, cols)), shape=x_csr.shape)

    if compat_mode == CompatMode.MODERN:
        # Import locally because solver preparation also imports this module.
        from ._full_update import _filter_csr_entries  # noqa: PLC0415

        coordinates = x_csr.tocoo(copy=False)
        probe_indicator = xprobe.copy()
        probe_indicator.data = np.ones(probe_indicator.nnz, dtype=bool)
        held_out = np.asarray(
            probe_indicator[coordinates.row, coordinates.col], dtype=bool
        ).ravel()
        x_csr = _filter_csr_entries(x_csr, ~held_out)
    else:
        # Retain historical subtraction/zero removal for frozen reproductions.
        probe_indicator = xprobe.copy()
        probe_indicator.data = np.ones(probe_indicator.nnz, dtype=float)
        x_csr -= x_csr.multiply(probe_indicator)
        x_csr.eliminate_zeros()
    x_csr.sort_indices()
    return x_csr, xprobe


def _xprobe_dense(
    x: Float64Array,
    fraction: float,
    rng: np.random.Generator,
    mask: InputMask | None = None,
) -> tuple[Float64Array, Float64Array]:
    """Hold out probe entries from a dense matrix.

    Returns:
        Tuple of (x_masked, xprobe) as dense arrays.

    Raises:
        ValueError: If the mask is sparse, has the wrong shape, or is empty.
    """
    x_dense = np.array(x, dtype=float, copy=True)
    if mask is None:
        observed = ~np.isnan(x_dense)
    else:
        if sp.issparse(mask):
            msg = "mask must be dense when x is dense"
            raise ValueError(msg)
        observed = np.array(mask, dtype=bool, copy=True)
        if observed.shape != x_dense.shape:
            msg = "mask must have the same shape as x"
            raise ValueError(msg)
        observed &= ~np.isnan(x_dense)

    obs_rows, obs_cols = np.nonzero(observed)
    if len(obs_rows) == 0:
        msg = "cannot create a probe set from data with no observed entries"
        raise ValueError(msg)
    n_probe = max(1, round(len(obs_rows) * fraction))
    probe_idx = rng.choice(len(obs_rows), size=n_probe, replace=False)

    probe_rows = obs_rows[probe_idx]
    probe_cols = obs_cols[probe_idx]

    xprobe = np.full(x_dense.shape, np.nan, dtype=float)
    xprobe[probe_rows, probe_cols] = x_dense[probe_rows, probe_cols]
    x_dense[probe_rows, probe_cols] = np.nan
    return x_dense, xprobe


def make_xprobe_mask(
    x: InputMatrix,
    fraction: float = 0.10,
    rng: np.random.Generator | None = None,
    *,
    mask: InputMask | None = None,
    compat_mode: str | CompatMode = CompatMode.MODERN,
) -> tuple[Float64Array | sp.csr_matrix, Float64Array | sp.csr_matrix]:
    """Hold out a fraction of observed entries as probe data.

    Selects a random subset of observed entries (non-NaN for dense,
    stored non-NaN entries for sparse) and returns a modified data matrix
    with those entries masked out plus a probe matrix containing only the
    held-out values.

    Args:
        x: Two-dimensional dense or sparse data. The returned matrices keep
            the same orientation; use the layout required by the consuming API.
        fraction: Fraction of observed entries to hold out (default 0.10).
            Must be in ``(0, 1)``.
        rng: NumPy random generator.  If ``None``, a new default generator
            is created.
        mask: Optional observation mask. When supplied, probe entries are
            sampled only from positions marked observed, including explicitly
            stored zero values in sparse data.
            This restricts probe sampling; unselected entries outside the mask
            remain in the returned training copy.
        compat_mode: ``"modern"`` (default) preserves unselected stored zeros
            and excludes sparse NaNs from sampling. ``"strict_legacy"`` retains
            historical sparse sampling/subtraction, which can remove unselected
            zeros and sample NaNs. Dense sampling is the same in both modes.

    Returns:
        x_masked: Copy of *x* with probe entries set to NaN (dense) or
            removed (sparse).
        xprobe: Matrix of the same shape as *x* containing only the
            held-out probe values; all other entries are NaN (dense) or
            absent (sparse).

    Raises:
        ValueError: If *fraction* is not in ``(0, 1)``, the mode is unknown,
            the mask is invalid, or there are no eligible observations.
        TypeError: If the mode is not a string.
    """  # noqa: DOC502 - TypeError is raised by CompatMode.parse.
    if not 0.0 < fraction < 1.0:
        msg = f"fraction must be in (0, 1), got {fraction}"
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    mode = CompatMode.parse(compat_mode)

    if sp.issparse(x):
        return _xprobe_sparse(sp.csr_matrix(x), fraction, rng, mask, compat_mode=mode)
    return _xprobe_dense(np.asarray(x), fraction, rng, mask)
