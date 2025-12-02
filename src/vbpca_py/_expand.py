"""Internal helpers for expanding rows and columns back to full size.

These functions are used at the end of ``pca_full`` to reinsert rows/columns
that were removed by ``rmempty``. Indices are **0-based** throughout.
"""

import numpy as np

# ============================================================
# Common index / shape validation errors
# ============================================================

ERR_KEPT_INDICES_1D = "kept_indices must be a 1D array or list of integer indices."

ERR_KEPT_INDICES_RANGE = "kept_indices contains values outside the valid range."

ERR_SCORES_2D = "scores must be a 2D array of shape (n_components, n_kept_cols)."

ERR_SCORES_KEPT_MISMATCH = "Number of kept columns does not match scores.shape[1]."

ERR_COVS_LIST_LENGTH = (
    "In per-column covariance mode, score_covs must have "
    "length equal to the number of kept columns."
)

ERR_COV_INDICES_1D = "score_cov_indices must be a 1D array or list."

ERR_COV_INDICES_LENGTH = (
    "score_cov_indices length must match the number of kept columns."
)

ERR_COV_INDICES_RANGE = (
    "score_cov_indices contains an index out of range for score_covs."
)

ERR_DIAG_COV_1D = "Diagonal covariance mode requires a 1D array of length n_components."

ERR_ROWS_2D = "rows must be a 2D array."

ERR_ROWS_KEPT_MISMATCH = "Number of kept rows does not match rows.shape[0]."

ERR_ROW_VARIANCES_LENGTH = (
    "row_variances must be scalar or length equal to number of components."
)

ERR_ROW_COV_LIST_LENGTH = (
    "When row_covs is a list, its length must equal the number of kept rows."
)

ERR_ROW_COV_ARRAY_SHAPE = (
    "When row_covs is an array, it must have shape (n_kept_rows, n_components)."
)


def _compute_kept_and_missing_indices(
    kept_indices: np.ndarray | list[int],
    n_total: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate kept indices and compute missing indices.

    Args:
        kept_indices: 0-based indices of kept entries (rows or columns).
        n_total: Total number of entries in the original dimension.

    Returns:
        kept_idx: 1D array of kept indices (0-based).
        missing_idx: 1D array of missing indices (0-based).
    """
    kept_idx = np.asarray(kept_indices, dtype=int)
    if kept_idx.ndim != 1:
        raise ValueError(ERR_KEPT_INDICES_1D)

    if np.any(kept_idx < 0) or np.any(kept_idx >= n_total):
        raise ValueError(ERR_KEPT_INDICES_RANGE)

    all_idx = np.arange(n_total, dtype=int)
    mask_kept = np.zeros(n_total, dtype=bool)
    mask_kept[kept_idx] = True
    missing_idx = all_idx[~mask_kept]

    return kept_idx, missing_idx


# ============================================================
# Internal helpers for columns
# ============================================================


def _expand_score_covs_list_per_column(
    score_covs: list[np.ndarray],
    kept_cols_idx: np.ndarray,
    missing_cols_idx: np.ndarray,
    n_components: int,
    dtype: np.dtype,
) -> tuple[list[np.ndarray], list[int] | None]:
    """Expand list-based per-column covariances to all columns."""
    n_kept_cols = kept_cols_idx.size
    if len(score_covs) != n_kept_cols:
        raise ValueError(ERR_COVS_LIST_LENGTH)

    n_total_cols = kept_cols_idx.size + missing_cols_idx.size
    expanded_score_covs: list[np.ndarray] = [None] * n_total_cols  # type: ignore[assignment]

    # Assign learned covariances to their original positions.
    for j, col in enumerate(kept_cols_idx):
        expanded_score_covs[int(col)] = score_covs[j]

    # Fill missing columns with identity covariance.
    identity = np.eye(n_components, dtype=dtype)
    for col in missing_cols_idx:
        expanded_score_covs[int(col)] = identity

    # Per-column mode does not use a pattern index mapping.
    expanded_score_cov_indices: list[int] | None = []

    return expanded_score_covs, expanded_score_cov_indices


def _expand_score_covs_list_unique_pattern(
    score_covs: list[np.ndarray],
    score_cov_indices: np.ndarray,
    kept_cols_idx: np.ndarray,
    n_total_cols: int,
    n_components: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Expand list-based covariances in unique-pattern mode."""
    n_kept_cols = kept_cols_idx.size
    if score_cov_indices.ndim != 1 or score_cov_indices.size != n_kept_cols:
        raise ValueError(ERR_COV_INDICES_LENGTH)

    # Start from existing unique covariances and append identity for new cols.
    expanded_score_covs = list(score_covs)
    # Use dtype from first covariance; assumes all covs consistent.
    base_dtype = np.asarray(score_covs[0]).dtype
    identity_cov = np.eye(n_components, dtype=base_dtype)
    expanded_score_covs.append(identity_cov)
    identity_idx = len(expanded_score_covs) - 1

    # Initialize all columns to use the identity covariance.
    expanded_score_cov_indices: list[int] = [identity_idx] * n_total_cols

    # Preserve learned mapping for kept columns.
    for j, col in enumerate(kept_cols_idx):
        idx = int(score_cov_indices[j])
        if idx < 0 or idx >= len(score_covs):
            raise ValueError(ERR_COV_INDICES_RANGE)
        expanded_score_cov_indices[int(col)] = idx

    return expanded_score_covs, expanded_score_cov_indices


def _expand_score_covs_diagonal(
    score_covs: np.ndarray,
    n_components: int,
    n_total_cols: int,
    kept_cols_idx: np.ndarray,
) -> tuple[np.ndarray, None]:
    """Expand diagonal covariance specification to all columns."""
    if score_covs.ndim != 1 or score_covs.shape[0] != n_components:
        raise ValueError(ERR_DIAG_COV_1D)

    expanded_score_covs = np.ones((n_components, n_total_cols), dtype=score_covs.dtype)
    expanded_score_covs[:, kept_cols_idx] = score_covs[:, np.newaxis]
    return expanded_score_covs, None


def _add_m_cols(
    scores: np.ndarray,
    score_covs: np.ndarray | list[np.ndarray],
    kept_cols: np.ndarray | list[int],
    n_total_cols: int,
    score_cov_indices: np.ndarray | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray | list[np.ndarray], np.ndarray | list[int] | None]:
    """Expand scores and covariances to the original number of columns.

    Used at the end of ``pca_full`` to reinsert columns removed by ``rmempty``.
    Indices are **0-based** throughout.

    Args:
        scores:
            Component scores of shape ``(n_components, n_kept_cols)`` for the
            columns that survived ``rmempty``.
        score_covs:
            Either
            - a list of covariance matrices (per-column or per-pattern), or
            - a 1D ndarray of diagonal variances of length ``n_components``.
        kept_cols:
            0-based indices of the kept columns (length ``n_kept_cols``),
            referring to positions in the original ``n_total_cols`` columns.
        n_total_cols:
            Total number of columns in the original data matrix.
        score_cov_indices:
            Optional mapping from kept columns into ``score_covs`` (0-based).
            If provided and non-empty, this indicates the "unique pattern"
            mode, and the returned ``expanded_score_cov_indices`` will be a
            length-``n_total_cols`` mapping for all columns.

    Returns:
        expanded_scores:
            Scores expanded to shape ``(n_components, n_total_cols)``, with
            zeros in newly reinserted columns.
        expanded_score_covs:
            Expanded covariance structure, which can be:
            - a list of length ``n_total_cols`` (per-column covariance mode),
            - a list of unique covariances (pattern mode) with
              ``expanded_score_cov_indices`` as mapping, or
            - an ndarray of shape ``(n_components, n_total_cols)`` in diagonal
              mode.
        expanded_score_cov_indices:
            Updated mapping from column index to covariance index (0-based)
            in the unique-patterns case, or ``[]`` / ``None`` otherwise.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(ERR_SCORES_2D)

    n_components, n_kept_cols = scores.shape

    kept_cols_idx, missing_cols_idx = _compute_kept_and_missing_indices(
        kept_cols, n_total_cols
    )

    if kept_cols_idx.size != n_kept_cols:
        raise ValueError(ERR_SCORES_KEPT_MISMATCH)

    # Expand scores to full width with zeros for missing columns.
    expanded_scores = np.zeros((n_components, n_total_cols), dtype=scores.dtype)
    expanded_scores[:, kept_cols_idx] = scores

    # Handle score_covs depending on its representation.
    if isinstance(score_covs, list):
        if score_cov_indices is None or len(score_cov_indices) == 0:
            expanded_score_covs, expanded_score_cov_indices = (
                _expand_score_covs_list_per_column(
                    score_covs=score_covs,
                    kept_cols_idx=kept_cols_idx,
                    missing_cols_idx=missing_cols_idx,
                    n_components=n_components,
                    dtype=scores.dtype,
                )
            )
        else:
            score_cov_indices_arr = np.asarray(score_cov_indices, dtype=int)
            expanded_score_covs, expanded_score_cov_indices = (
                _expand_score_covs_list_unique_pattern(
                    score_covs=score_covs,
                    score_cov_indices=score_cov_indices_arr,
                    kept_cols_idx=kept_cols_idx,
                    n_total_cols=n_total_cols,
                    n_components=n_components,
                )
            )
    else:
        score_covs_arr = np.asarray(score_covs)
        expanded_score_covs, expanded_score_cov_indices = _expand_score_covs_diagonal(
            score_covs=score_covs_arr,
            n_components=n_components,
            n_total_cols=n_total_cols,
            kept_cols_idx=kept_cols_idx,
        )

    return expanded_scores, expanded_score_covs, expanded_score_cov_indices


# ============================================================
# Internal helpers for rows
# ============================================================


def _normalize_row_variances(
    n_components: int,
    row_variances: np.ndarray | list[float] | float | None,
) -> np.ndarray:
    """Normalize row_variances into a 1D array of length n_components."""
    if n_components == 0:
        return np.array([], dtype=float)

    if row_variances is None:
        return np.full(n_components, np.inf, dtype=float)

    variances = np.asarray(row_variances, dtype=float)
    if variances.size == 1:
        return np.full(n_components, variances.item(), dtype=float)
    if variances.size != n_components:
        raise ValueError(ERR_ROW_VARIANCES_LENGTH)
    return variances


def _init_expanded_rows(
    rows: np.ndarray,
    n_total_rows: int,
    kept_rows_idx: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Initialize the expanded row matrix, inserting kept rows."""
    n_kept_rows, n_components = rows.shape

    if n_components == 0:
        return np.empty((n_total_rows, 0), dtype=rows.dtype)

    if np.isinf(variances).any():
        expanded_rows = np.full((n_total_rows, n_components), np.nan, dtype=rows.dtype)
    else:
        expanded_rows = np.zeros((n_total_rows, n_components), dtype=rows.dtype)

    if n_kept_rows > 0:
        expanded_rows[kept_rows_idx, :] = rows

    return expanded_rows


def _expand_row_covs_list(
    row_covs: list[np.ndarray],
    n_total_rows: int,
    kept_rows_idx: np.ndarray,
    missing_rows_idx: np.ndarray,
    variances: np.ndarray,
) -> list[np.ndarray]:
    """Expand list-based per-row covariance matrices to all rows."""
    if len(row_covs) != kept_rows_idx.size:
        raise ValueError(ERR_ROW_COV_LIST_LENGTH)

    expanded_row_covs: list[np.ndarray] = [None] * n_total_rows  # type: ignore[assignment]

    # Assign existing covariances to kept rows.
    for j, row_idx in enumerate(kept_rows_idx):
        expanded_row_covs[int(row_idx)] = row_covs[j]

    # Default covariance for missing rows.
    default_cov = np.diag(variances) if variances.size > 0 else np.zeros((0, 0))

    for row_idx in missing_rows_idx:
        expanded_row_covs[int(row_idx)] = default_cov

    return expanded_row_covs


def _expand_row_covs_array(
    row_covs: np.ndarray,
    n_components: int,
    n_total_rows: int,
    kept_rows_idx: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Expand array-like per-row diagonal variances to all rows."""
    if row_covs.ndim != 2 or row_covs.shape[1] != n_components:
        raise ValueError(ERR_ROW_COV_ARRAY_SHAPE)

    expanded_row_covs = np.tile(variances, (n_total_rows, 1))
    expanded_row_covs[kept_rows_idx, :] = row_covs
    return expanded_row_covs


def _add_m_rows(
    rows: np.ndarray,
    row_covs: list[np.ndarray] | np.ndarray | list | None,
    kept_rows: np.ndarray | list[int],
    n_total_rows: int,
    row_variances: np.ndarray | list[float] | float | None = None,
) -> tuple[np.ndarray, list[np.ndarray] | np.ndarray | list]:
    """Expand rows and associated covariances to the original number of rows.

    This mirrors ``_add_m_cols`` but for rows, and is used at the end of
    ``pca_full`` to reinsert rows removed by ``rmempty`` (for ``A``, ``Mu``,
    and their covariances).

    Args:
        rows:
            Array of shape ``(n_kept_rows, n_components)`` (for A/Av) or
            ``(n_kept_rows, 1)`` (for Mu/Muv).
        row_covs:
            Either:
            - a list of per-row covariance matrices (common for Av/Muv), or
            - a 2D ndarray (n_kept_rows, n_components) for diagonal-like
              variances, or
            - ``None`` / empty list if no covariances are tracked.
        kept_rows:
            0-based indices of rows that were kept (length ``n_kept_rows``),
            referring to positions in the original ``n_total_rows`` rows.
        n_total_rows:
            Total number of rows in the original data matrix.
        row_variances:
            Variance specification for *newly added rows*:
            - If None, defaults to ``+inf`` (rows initialized to NaN).
            - If scalar, broadcast to all components.
            - If 1D array, length must equal the number of components.

    Returns:
        expanded_rows:
            Array of shape ``(n_total_rows, n_components)`` (or
            ``(n_total_rows, 1)`` for Mu), with rows inserted at kept_positions
            and default values elsewhere.
        expanded_row_covs:
            Expanded covariance structure:
            - list of length ``n_total_rows`` when ``row_covs`` is a list,
            - 2D ndarray (n_total_rows, n_components) when ``row_covs`` is an
              array,
            - empty list if no covariances are tracked.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2:
        raise ValueError(ERR_ROWS_2D)

    n_kept_rows, n_components = rows.shape

    kept_rows_idx, missing_rows_idx = _compute_kept_and_missing_indices(
        kept_rows, n_total_rows
    )

    if kept_rows_idx.size != n_kept_rows:
        raise ValueError(ERR_ROWS_KEPT_MISMATCH)

    variances = _normalize_row_variances(n_components, row_variances)
    expanded_rows = _init_expanded_rows(
        rows=rows,
        n_total_rows=n_total_rows,
        kept_rows_idx=kept_rows_idx,
        variances=variances,
    )

    # Decide if we have covariance information to expand.
    has_covs = row_covs is not None and (
        not isinstance(row_covs, (list, tuple)) or len(row_covs) > 0
    )

    if not has_covs or n_components == 0:
        return expanded_rows, []

    if isinstance(row_covs, list):
        expanded_row_covs = _expand_row_covs_list(
            row_covs=row_covs,
            n_total_rows=n_total_rows,
            kept_rows_idx=kept_rows_idx,
            missing_rows_idx=missing_rows_idx,
            variances=variances,
        )
    else:
        row_covs_arr = np.asarray(row_covs)
        expanded_row_covs = _expand_row_covs_array(
            row_covs=row_covs_arr,
            n_components=n_components,
            n_total_rows=n_total_rows,
            kept_rows_idx=kept_rows_idx,
            variances=variances,
        )

    return expanded_rows, expanded_row_covs
