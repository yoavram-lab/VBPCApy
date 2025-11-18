# tests/test_expand.py
"""Tests for the _expand module functionality.

This module contains tests for:
- _compute_kept_and_missing_indices: computing kept and missing index arrays
- _add_m_rows: adding missing rows to data matrices with covariances
- _add_m_cols: adding missing columns to score matrices with covariances
"""

import re

import numpy as np
import pytest

from vbpca_py._expand import (
    ERR_COV_INDICES_LENGTH,
    ERR_COV_INDICES_RANGE,
    ERR_COVS_LIST_LENGTH,
    ERR_DIAG_COV_1D,
    ERR_KEPT_INDICES_RANGE,
    ERR_ROW_COV_ARRAY_SHAPE,
    ERR_ROW_VARIANCES_LENGTH,
    ERR_ROWS_2D,
    ERR_ROWS_KEPT_MISMATCH,
    ERR_SCORES_2D,
    ERR_SCORES_KEPT_MISMATCH,
    _add_m_cols,
    _add_m_rows,
    _compute_kept_and_missing_indices,
)

# ---------------------------------------------------------------------------
# _compute_kept_and_missing_indices
# ---------------------------------------------------------------------------


def test_compute_kept_and_missing_indices_basic() -> None:
    """Test basic functionality of _compute_kept_and_missing_indices."""
    kept, missing = _compute_kept_and_missing_indices([0, 2], 4)
    np.testing.assert_array_equal(kept, np.array([0, 2]))
    np.testing.assert_array_equal(missing, np.array([1, 3]))


def test_compute_kept_and_missing_indices_out_of_range() -> None:
    """Test _compute_kept_and_missing_indices raises error for out-of-range indices."""
    with pytest.raises(ValueError, match=re.escape(ERR_KEPT_INDICES_RANGE)):
        _compute_kept_and_missing_indices([0, 4], 4)


# ---------------------------------------------------------------------------
# _add_m_rows: happy paths
# ---------------------------------------------------------------------------


def test_add_m_rows_list_covs_with_missing_rows() -> None:
    """Test _add_m_rows with list covariances when some rows are missing."""
    # rows: 2 kept rows, 3 components
    rows = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # list covariances for each kept row
    row_covs = [
        np.eye(3),
        2.0 * np.eye(3),
    ]
    # kept_rows are 0-based indices into total rows
    kept_rows = [0, 2]
    n_total_rows = 4
    row_variances = [3.0, 3.0, 3.0]

    expanded_rows, expanded_row_covs = _add_m_rows(
        rows=rows,
        row_covs=row_covs,
        kept_rows=kept_rows,
        n_total_rows=n_total_rows,
        row_variances=row_variances,
    )

    # Rows: kept rows in positions 0 and 2, others default (zeros because Va finite)
    assert expanded_rows.shape == (4, 3)
    np.testing.assert_array_equal(expanded_rows[0], rows[0])
    np.testing.assert_array_equal(expanded_rows[2], rows[1])
    np.testing.assert_array_equal(expanded_rows[1], np.zeros(3))
    np.testing.assert_array_equal(expanded_rows[3], np.zeros(3))

    # Covariances: kept rows from list, missing rows diag(Va)
    assert isinstance(expanded_row_covs, list)
    assert len(expanded_row_covs) == 4
    np.testing.assert_array_equal(expanded_row_covs[0], row_covs[0])
    np.testing.assert_array_equal(expanded_row_covs[2], row_covs[1])
    np.testing.assert_array_equal(expanded_row_covs[1], np.diag(row_variances))
    np.testing.assert_array_equal(expanded_row_covs[3], np.diag(row_variances))


def test_add_m_rows_all_rows_observed() -> None:
    """Test _add_m_rows when all rows are observed (no missing rows)."""
    rows = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    row_covs = [
        np.eye(2),
        2.0 * np.eye(2),
        3.0 * np.eye(2),
    ]
    kept_rows = [0, 1, 2]
    n_total_rows = 3
    row_variances = [5.0, 5.0]

    expanded_rows, expanded_row_covs = _add_m_rows(
        rows=rows,
        row_covs=row_covs,
        kept_rows=kept_rows,
        n_total_rows=n_total_rows,
        row_variances=row_variances,
    )

    np.testing.assert_array_equal(expanded_rows, rows)
    assert isinstance(expanded_row_covs, list)
    assert len(expanded_row_covs) == 3
    for i in range(3):
        np.testing.assert_array_equal(expanded_row_covs[i], row_covs[i])


def test_add_m_rows_no_observed_rows_with_finite_variances() -> None:
    """Test _add_m_rows when no rows are observed but finite variances are provided.

    Since row_covs is empty, we treat this as 'no covariances tracked', so only
    expanded_rows are returned and expanded_row_covs is an empty list.
    """
    rows = np.empty((0, 2))  # n_kept_rows = 0, n_components = 2
    row_covs: list[np.ndarray] = []
    kept_rows: list[int] = []
    n_total_rows = 3
    row_variances = [4.0, 4.0]

    expanded_rows, expanded_row_covs = _add_m_rows(
        rows=rows,
        row_covs=row_covs,
        kept_rows=kept_rows,
        n_total_rows=n_total_rows,
        row_variances=row_variances,
    )

    # All rows newly added, finite variances => zeros
    assert expanded_rows.shape == (3, 2)
    np.testing.assert_array_equal(expanded_rows, np.zeros((3, 2)))
    # No covariances tracked when row_covs is empty
    assert expanded_row_covs == []


def test_add_m_rows_no_components() -> None:
    """Test _add_m_rows when there are no components (empty feature space)."""
    rows = np.empty((2, 0))  # shape (2, 0)
    row_covs: list[np.ndarray] = []
    kept_rows = [0, 1]
    n_total_rows = 2
    row_variances: list[float] = []

    expanded_rows, expanded_row_covs = _add_m_rows(
        rows=rows,
        row_covs=row_covs,
        kept_rows=kept_rows,
        n_total_rows=n_total_rows,
        row_variances=row_variances,
    )

    assert expanded_rows.shape == (2, 0)
    assert expanded_row_covs == []


def test_add_m_rows_array_covariances_branch() -> None:
    """Test _add_m_rows with array covariances instead of list covariances."""
    rows = np.array([[1.0, 2.0], [3.0, 4.0]])
    # row_covs array: per-row, per-component variances
    row_covs = np.array([[1.0, 2.0], [3.0, 4.0]])
    kept_rows = [1, 3]
    n_total_rows = 4
    row_variances = [10.0, 20.0]

    expanded_rows, expanded_row_covs = _add_m_rows(
        rows=rows,
        row_covs=row_covs,
        kept_rows=kept_rows,
        n_total_rows=n_total_rows,
        row_variances=row_variances,
    )

    assert expanded_rows.shape == (4, 2)
    np.testing.assert_array_equal(expanded_rows[1], rows[0])
    np.testing.assert_array_equal(expanded_rows[3], rows[1])

    assert isinstance(expanded_row_covs, np.ndarray)
    assert expanded_row_covs.shape == (4, 2)
    # kept rows recovered
    np.testing.assert_array_equal(expanded_row_covs[1], row_covs[0])
    np.testing.assert_array_equal(expanded_row_covs[3], row_covs[1])
    # missing rows use default variances
    np.testing.assert_array_equal(expanded_row_covs[0], np.array(row_variances))
    np.testing.assert_array_equal(expanded_row_covs[2], np.array(row_variances))


def test_add_m_rows_no_covariances() -> None:
    """Test _add_m_rows when no covariances are provided (empty list)."""
    rows = np.array([[1.0, 2.0]])
    row_covs: list[np.ndarray] = []
    kept_rows = [0]
    n_total_rows = 3
    row_variances = [1.0, 1.0]

    expanded_rows, expanded_row_covs = _add_m_rows(
        rows=rows,
        row_covs=row_covs,
        kept_rows=kept_rows,
        n_total_rows=n_total_rows,
        row_variances=row_variances,
    )

    assert expanded_rows.shape == (3, 2)
    assert expanded_row_covs == []


# ---------------------------------------------------------------------------
# _add_m_rows: error cases
# ---------------------------------------------------------------------------


def test_add_m_rows_bad_rows_dim() -> None:
    """Test _add_m_rows raises error when rows array is not 2D."""
    with pytest.raises(ValueError, match=re.escape(ERR_ROWS_2D)):
        _add_m_rows(
            rows=np.array([1.0, 2.0]),  # 1D
            row_covs=None,
            kept_rows=[0],
            n_total_rows=1,
            row_variances=None,
        )


def test_add_m_rows_kept_rows_mismatch() -> None:
    """Test _add_m_rows raises error when kept_rows length doesn't match rows."""
    rows = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match=re.escape(ERR_ROWS_KEPT_MISMATCH)):
        _add_m_rows(
            rows=rows,
            row_covs=None,
            kept_rows=[0, 1],  # mismatched length
            n_total_rows=2,
            row_variances=None,
        )


def test_add_m_rows_bad_row_variances_length() -> None:
    """Test _add_m_rows raises error when row var len doesn't match number of comps."""
    rows = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match=re.escape(ERR_ROW_VARIANCES_LENGTH)):
        _add_m_rows(
            rows=rows,
            row_covs=None,
            kept_rows=[0],
            n_total_rows=2,
            row_variances=[1.0, 2.0, 3.0],  # wrong length
        )


def test_add_m_rows_bad_row_cov_array_shape() -> None:
    """Test _add_m_rows raises error when row_covs array has wrong shape."""
    rows = np.array([[1.0, 2.0]])
    bad_row_covs = np.array([[1.0, 2.0, 3.0]])  # wrong width
    with pytest.raises(ValueError, match=re.escape(ERR_ROW_COV_ARRAY_SHAPE)):
        _add_m_rows(
            rows=rows,
            row_covs=bad_row_covs,
            kept_rows=[0],
            n_total_rows=1,
            row_variances=[1.0, 1.0],
        )


# ---------------------------------------------------------------------------
# _add_m_cols: happy paths
# ---------------------------------------------------------------------------


def test_add_m_cols_per_column_covs_with_missing() -> None:
    """Test _add_m_cols with per-column covariances when some columns are missing."""
    scores = np.array([[1.0, 2.0]])  # n_components = 1, n_kept_cols = 2
    score_covs = [np.array([[1.0]]), np.array([[2.0]])]
    kept_cols = [0, 2]
    n_total_cols = 4

    expanded_scores, expanded_covs, expanded_indices = _add_m_cols(
        scores=scores,
        score_covs=score_covs,
        kept_cols=kept_cols,
        n_total_cols=n_total_cols,
        score_cov_indices=None,
    )

    assert expanded_scores.shape == (1, 4)
    np.testing.assert_array_equal(expanded_scores[0, kept_cols], scores[0])
    np.testing.assert_array_equal(expanded_scores[0, [1, 3]], np.array([0.0, 0.0]))

    assert isinstance(expanded_covs, list)
    assert len(expanded_covs) == 4
    np.testing.assert_array_equal(expanded_covs[0], score_covs[0])
    np.testing.assert_array_equal(expanded_covs[2], score_covs[1])
    # missing columns have identity
    np.testing.assert_array_equal(expanded_covs[1], np.eye(1))
    np.testing.assert_array_equal(expanded_covs[3], np.eye(1))
    assert expanded_indices == []


def test_add_m_cols_unique_pattern_mode() -> None:
    """Test _add_m_cols with unique pattern mode using covariance indices."""
    scores = np.array([[1.0, 2.0, 3.0]])  # 3 kept cols
    # two unique covariance patterns
    score_covs = [np.array([[1.0]]), np.array([[5.0]])]
    score_cov_indices = [0, 1, 0]  # pattern assignment for kept cols
    kept_cols = [0, 2, 4]
    n_total_cols = 6

    expanded_scores, expanded_covs, expanded_indices = _add_m_cols(
        scores=scores,
        score_covs=score_covs,
        kept_cols=kept_cols,
        n_total_cols=n_total_cols,
        score_cov_indices=score_cov_indices,
    )

    assert expanded_scores.shape == (1, 6)
    np.testing.assert_array_equal(expanded_scores[0, kept_cols], scores[0])

    assert isinstance(expanded_covs, list)
    # one extra identity cov appended
    assert len(expanded_covs) == 3

    # mapping length equals n_total_cols
    assert len(expanded_indices) == n_total_cols
    # kept columns preserve mapping
    assert expanded_indices[0] == 0
    assert expanded_indices[2] == 1
    assert expanded_indices[4] == 0
    # missing columns should use identity index (last one)
    identity_idx = len(expanded_covs) - 1
    for col in [1, 3, 5]:
        assert expanded_indices[col] == identity_idx


def test_add_m_cols_diagonal_mode() -> None:
    """Test _add_m_cols with diagonal covariance mode."""
    # Two components, one kept column
    scores = np.array([[1.0], [2.0]])  # shape: (n_components=2, n_kept_cols=1)
    diag_cov = np.array([0.5, 2.0])  # length must equal n_components
    kept_cols = [1]
    n_total_cols = 3

    expanded_scores, expanded_covs, expanded_indices = _add_m_cols(
        scores=scores,
        score_covs=diag_cov,
        kept_cols=kept_cols,
        n_total_cols=n_total_cols,
        score_cov_indices=None,
    )

    assert expanded_scores.shape == (2, 3)
    assert expanded_covs.shape == (2, 3)

    # Kept column gets given variances
    np.testing.assert_array_equal(expanded_covs[:, 1], diag_cov)
    # Others are ones
    np.testing.assert_array_equal(expanded_covs[:, 0], np.ones(2))
    np.testing.assert_array_equal(expanded_covs[:, 2], np.ones(2))

    # Scores are correctly placed in kept column, zeros elsewhere
    np.testing.assert_array_equal(expanded_scores[:, 1], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(expanded_scores[:, 0], np.zeros(2))
    np.testing.assert_array_equal(expanded_scores[:, 2], np.zeros(2))

    assert expanded_indices is None


# ---------------------------------------------------------------------------
# _add_m_cols: error cases
# ---------------------------------------------------------------------------


def test_add_m_cols_bad_scores_dim() -> None:
    """Test _add_m_cols raises error when scores array is not 2D."""
    with pytest.raises(ValueError, match=re.escape(ERR_SCORES_2D)):
        _add_m_cols(
            scores=np.array([1.0, 2.0]),  # 1D
            score_covs=np.array([1.0]),
            kept_cols=[0],
            n_total_cols=1,
            score_cov_indices=None,
        )


def test_add_m_cols_kept_cols_mismatch() -> None:
    """Test _add_m_cols raises error when kept_cols length doesn't match scores columns."""
    scores = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match=re.escape(ERR_SCORES_KEPT_MISMATCH)):
        _add_m_cols(
            scores=scores,
            score_covs=[np.array([[1.0]])],
            kept_cols=[0],  # only 1 kept col, scores has 2
            n_total_cols=2,
            score_cov_indices=None,
        )


def test_add_m_cols_per_column_covs_length_mismatch() -> None:
    """Test _add_m_cols raises when per-column covariances list length is wrong."""
    scores = np.array([[1.0, 2.0]])
    # wrong length: only one cov for two kept cols
    score_covs = [np.array([[1.0]])]
    with pytest.raises(ValueError, match=re.escape(ERR_COVS_LIST_LENGTH)):
        _add_m_cols(
            scores=scores,
            score_covs=score_covs,
            kept_cols=[0, 1],
            n_total_cols=2,
            score_cov_indices=None,
        )


def test_add_m_cols_bad_cov_indices_length() -> None:
    """Test _add_m_cols raises when cov_indices length doesn't match kept_cols."""
    scores = np.array([[1.0, 2.0]])
    score_covs = [np.array([[1.0]])]
    score_cov_indices = [0, 0, 0]  # too long
    with pytest.raises(ValueError, match=re.escape(ERR_COV_INDICES_LENGTH)):
        _add_m_cols(
            scores=scores,
            score_covs=score_covs,
            kept_cols=[0, 1],
            n_total_cols=2,
            score_cov_indices=score_cov_indices,
        )


def test_add_m_cols_bad_cov_indices_range() -> None:
    """Test _add_m_cols raises error when cov_indices values are out of range."""
    scores = np.array([[1.0]])
    score_covs = [np.array([[1.0]])]
    score_cov_indices = [5]  # out of range
    with pytest.raises(ValueError, match=re.escape(ERR_COV_INDICES_RANGE)):
        _add_m_cols(
            scores=scores,
            score_covs=score_covs,
            kept_cols=[0],
            n_total_cols=1,
            score_cov_indices=score_cov_indices,
        )


def test_add_m_cols_bad_diag_cov_shape() -> None:
    """Test _add_m_cols raises error when diagonal covariance is not 1D."""
    scores = np.array([[1.0, 2.0]])
    bad_diag_cov = np.array([[1.0, 2.0]])  # 2D instead of 1D
    with pytest.raises(ValueError, match=re.escape(ERR_DIAG_COV_1D)):
        _add_m_cols(
            scores=scores,
            score_covs=bad_diag_cov,
            kept_cols=[0, 1],
            n_total_cols=2,
            score_cov_indices=None,
        )


def test_add_m_cols_out_of_range_kept_cols() -> None:
    """Test _add_m_cols raises error when kept_cols indices are out of range."""
    scores = np.array([[1.0]])
    with pytest.raises(ValueError, match=re.escape(ERR_KEPT_INDICES_RANGE)):
        _add_m_cols(
            scores=scores,
            score_covs=np.array([1.0]),
            kept_cols=[1],  # out of range for n_total_cols=1
            n_total_cols=1,
            score_cov_indices=None,
        )
