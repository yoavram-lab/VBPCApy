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

logger = logging.getLogger(__name__)

MASK_WRONG_DIM = "mask must be a 2D array."


def _missing_patterns(
    mask: np.ndarray,
    *,
    verbose: bool = False,
) -> tuple[int, list[list[int]], np.ndarray]:
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
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Hold out probe entries from a sparse matrix.

    Returns:
        Tuple of (x_masked, xprobe) as CSR matrices.
    """
    x_csr = sp.csr_matrix(x, copy=True)
    n_probe = max(1, round(x_csr.nnz * fraction))
    probe_idx = rng.choice(x_csr.nnz, size=n_probe, replace=False)

    rows, cols = x_csr.nonzero()
    sp_rows = rows[probe_idx]
    sp_cols = cols[probe_idx]
    sp_vals = np.array(x_csr[sp_rows, sp_cols]).ravel()

    xprobe_sp = sp.lil_matrix(x_csr.shape, dtype=float)
    for r, c, v in zip(sp_rows, sp_cols, sp_vals, strict=True):
        xprobe_sp[r, c] = v
    xprobe = sp.csr_matrix(xprobe_sp)

    for r, c in zip(sp_rows, sp_cols, strict=True):
        x_csr[r, c] = 0.0
    x_csr.eliminate_zeros()
    x_csr.sort_indices()
    return x_csr, xprobe


def _xprobe_dense(
    x: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out probe entries from a dense matrix.

    Returns:
        Tuple of (x_masked, xprobe) as dense arrays.
    """
    x_dense = np.array(x, dtype=float, copy=True)
    obs_rows, obs_cols = np.nonzero(~np.isnan(x_dense))
    n_probe = max(1, round(len(obs_rows) * fraction))
    probe_idx = rng.choice(len(obs_rows), size=n_probe, replace=False)

    probe_rows = obs_rows[probe_idx]
    probe_cols = obs_cols[probe_idx]

    xprobe = np.full(x_dense.shape, np.nan, dtype=float)
    xprobe[probe_rows, probe_cols] = x_dense[probe_rows, probe_cols]
    x_dense[probe_rows, probe_cols] = np.nan
    return x_dense, xprobe


def make_xprobe_mask(
    x: np.ndarray | sp.csr_matrix,
    fraction: float = 0.10,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | sp.csr_matrix, np.ndarray | sp.csr_matrix]:
    """Hold out a fraction of observed entries as probe data.

    Selects a random subset of observed entries (non-NaN for dense,
    structurally non-zero for sparse) and returns a modified data matrix
    with those entries masked out plus a probe matrix containing only the
    held-out values.

    Args:
        x: Data matrix of shape ``(n_features, n_samples)``.
            Dense or sparse (CSR).
        fraction: Fraction of observed entries to hold out (default 0.10).
            Must be in ``(0, 1)``.
        rng: NumPy random generator.  If ``None``, a new default generator
            is created.

    Returns:
        x_masked: Copy of *x* with probe entries set to NaN (dense) or
            removed (sparse).
        xprobe: Matrix of the same shape as *x* containing only the
            held-out probe values; all other entries are NaN (dense) or
            absent (sparse).

    Raises:
        ValueError: If *fraction* is not in ``(0, 1)``.
    """
    if not 0.0 < fraction < 1.0:
        msg = f"fraction must be in (0, 1), got {fraction}"
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    if sp.issparse(x):
        return _xprobe_sparse(sp.csr_matrix(x), fraction, rng)
    return _xprobe_dense(np.asarray(x), fraction, rng)
