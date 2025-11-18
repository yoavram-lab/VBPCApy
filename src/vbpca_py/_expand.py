"""Internal helpers for expanding rows and columns back to full size.

These functions are used at the end of ``pca_full`` to reinsert rows/columns
that were removed by ``rmempty``. Indices are **0-based** throughout.
"""

import numpy as np


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
        raise ValueError("kept_indices must be a 1D array or list of indices.")

    if np.any(kept_idx < 0) or np.any(kept_idx >= n_total):
        raise ValueError("kept_indices contains indices out of range [0, n_total).")

    all_idx = np.arange(n_total, dtype=int)
    mask_kept = np.zeros(n_total, dtype=bool)
    mask_kept[kept_idx] = True
    missing_idx = all_idx[~mask_kept]

    return kept_idx, missing_idx


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
        raise ValueError(
            "scores must be a 2D array of shape (n_components, n_kept_cols)."
        )

    n_components, n_kept_cols = scores.shape

    kept_cols_idx, missing_cols_idx = _compute_kept_and_missing_indices(
        kept_cols, n_total_cols
    )

    if kept_cols_idx.size != n_kept_cols:
        raise ValueError(
            f"Number of kept columns in kept_cols ({kept_cols_idx.size}) must "
            f"match scores.shape[1] ({n_kept_cols})."
        )

    # Expand scores to full width with zeros for missing columns.
    expanded_scores = np.zeros((n_components, n_total_cols), dtype=scores.dtype)
    expanded_scores[:, kept_cols_idx] = scores

    # Handle score_covs depending on its representation.
    if isinstance(score_covs, list):
        # Full covariance matrices (per-column or per-pattern).
        if score_cov_indices is None or len(score_cov_indices) == 0:
            # Per-column covariance mode: score_covs assumed to have length
            # n_kept_cols, one covariance per kept column.
            if len(score_covs) != n_kept_cols:
                raise ValueError(
                    "In per-column covariance mode (empty score_cov_indices), "
                    "score_covs must have length equal to the number of kept "
                    "columns."
                )

            expanded_score_covs: list[np.ndarray] = [None] * n_total_cols  # type: ignore[assignment]
            # Assign learned covariances to their original positions.
            for j, col in enumerate(kept_cols_idx):
                expanded_score_covs[col] = score_covs[j]

            # Fill missing columns with identity covariance.
            for col in missing_cols_idx:
                expanded_score_covs[col] = np.eye(n_components, dtype=scores.dtype)

            expanded_score_cov_indices: list[int] | None = []
        else:
            # Unique-pattern mode: score_covs is a list of unique covariances
            # and score_cov_indices maps kept columns to 0-based indices.
            score_cov_indices_arr = np.asarray(score_cov_indices, dtype=int)
            if (
                score_cov_indices_arr.ndim != 1
                or score_cov_indices_arr.size != n_kept_cols
            ):
                raise ValueError(
                    "score_cov_indices must be a 1D array/list whose length "
                    "equals the number of kept columns when provided."
                )

            expanded_score_covs = list(score_covs)
            # Append identity covariance for newly added columns.
            identity_cov = np.eye(n_components, dtype=scores.dtype)
            expanded_score_covs.append(identity_cov)
            identity_idx = len(expanded_score_covs) - 1  # 0-based index

            # Initialize all columns to use the identity covariance.
            expanded_score_cov_indices = [identity_idx] * n_total_cols

            # Preserve learned mapping for kept columns.
            for j, col in enumerate(kept_cols_idx):
                idx = int(score_cov_indices_arr[j])
                if idx < 0 or idx >= len(score_covs):
                    raise ValueError(
                        "score_cov_indices contains index out of range for score_covs."
                    )
                expanded_score_cov_indices[col] = idx

    else:
        # Diagonal covariance mode: score_covs is a 1D array of length n_components.
        score_covs_arr = np.asarray(score_covs)
        if score_covs_arr.ndim != 1 or score_covs_arr.shape[0] != n_components:
            raise ValueError(
                "In diagonal covariance mode, score_covs must be a 1D array "
                "of length equal to n_components."
            )

        expanded_score_covs = np.ones(
            (n_components, n_total_cols), dtype=score_covs_arr.dtype
        )
        expanded_score_covs[:, kept_cols_idx] = score_covs_arr[:, np.newaxis]
        expanded_score_cov_indices = None

    return expanded_scores, expanded_score_covs, expanded_score_cov_indices


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
        raise ValueError("rows must be a 2D array.")

    n_kept_rows, n_components = rows.shape

    kept_rows_idx, missing_rows_idx = _compute_kept_and_missing_indices(
        kept_rows, n_total_rows
    )

    if kept_rows_idx.size != n_kept_rows:
        raise ValueError(
            f"Number of kept rows in kept_rows ({kept_rows_idx.size}) must "
            f"match rows.shape[0] ({n_kept_rows})."
        )

    # Handle row_variances and its broadcasting.
    if n_components > 0:
        if row_variances is None:
            variances = np.full(n_components, np.inf, dtype=float)
        else:
            variances = np.asarray(row_variances, dtype=float)
            if variances.size == 1:
                variances = np.full(n_components, variances.item(), dtype=float)
            elif variances.size != n_components:
                raise ValueError(
                    "Length of row_variances must be 1 or equal to the "
                    "number of components."
                )
    else:
        variances = np.array([], dtype=float)

    # Initialize expanded_rows.
    if n_components == 0:
        expanded_rows = np.empty((n_total_rows, 0), dtype=rows.dtype)
    else:
        if np.isinf(variances).any():
            expanded_rows = np.full(
                (n_total_rows, n_components), np.nan, dtype=rows.dtype
            )
        else:
            expanded_rows = np.zeros((n_total_rows, n_components), dtype=rows.dtype)

        expanded_rows[kept_rows_idx, :] = rows

    # Handle expanded_row_covs.
    if row_covs and n_components > 0:
        if isinstance(row_covs, list):
            # Per-row covariance matrices.
            if len(row_covs) != n_kept_rows:
                raise ValueError(
                    "When row_covs is a list, its length must equal "
                    "the number of kept rows."
                )

            expanded_row_covs: list[np.ndarray] = [None] * n_total_rows  # type: ignore[assignment]
            # Assign existing covariances to kept rows.
            for j, row_idx in enumerate(kept_rows_idx):
                expanded_row_covs[row_idx] = row_covs[j]

            # Default covariance for missing rows.
            default_cov = np.diag(variances) if variances.size > 0 else np.zeros((0, 0))
            for row_idx in missing_rows_idx:
                expanded_row_covs[row_idx] = default_cov
        else:
            # Array-like covariances: treat as diagonal variances per row.
            row_covs_arr = np.asarray(row_covs)
            if row_covs_arr.ndim != 2 or row_covs_arr.shape[1] != n_components:
                raise ValueError(
                    "When row_covs is an array, it must have shape "
                    "(n_kept_rows, n_components)."
                )

            expanded_row_covs = np.tile(variances, (n_total_rows, 1))
            expanded_row_covs[kept_rows_idx, :] = row_covs_arr
    else:
        expanded_row_covs = []

    return expanded_rows, expanded_row_covs
