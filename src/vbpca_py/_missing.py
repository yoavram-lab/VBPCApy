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
