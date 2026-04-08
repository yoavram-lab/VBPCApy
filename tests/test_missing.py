# tests/test_missing.py
"""Tests for the _missing_patterns helper."""

import numpy as np
import pytest

import vbpca_py._missing as missing_mod
from vbpca_py._missing import _missing_patterns

# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_missing_patterns_basic_grouping() -> None:
    """Two distinct patterns, each appearing multiple times."""
    # mask shape: (rows, cols)
    # Columns 0, 2 share pattern [1, 0]
    # Columns 1, 3 share pattern [0, 1]
    mask = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])

    n_patterns, pattern_columns, column_pattern_index = _missing_patterns(mask)

    assert n_patterns == 2
    assert len(pattern_columns) == 2

    # Each column index should appear exactly once across groups
    all_cols = sorted(c for group in pattern_columns for c in group)
    assert all_cols == [0, 1, 2, 3]

    # Check grouping: patterns are unique up to ordering, so we sort within groups
    sorted_groups = [sorted(g) for g in pattern_columns]
    assert sorted(sorted_groups) == [[0, 2], [1, 3]]

    # column_pattern_index should be length n_cols
    assert column_pattern_index.shape == (4,)
    # Check that columns with same pattern share the same index
    assert column_pattern_index[0] == column_pattern_index[2]
    assert column_pattern_index[1] == column_pattern_index[3]
    assert column_pattern_index[0] != column_pattern_index[1]


def test_missing_patterns_all_identical_columns() -> None:
    """All columns share the same missingness pattern -> one group."""
    mask = np.array([
        [1, 1, 1],
        [0, 0, 0],
    ])

    n_patterns, pattern_columns, column_pattern_index = _missing_patterns(mask)

    assert n_patterns == 1
    assert len(pattern_columns) == 1
    assert sorted(pattern_columns[0]) == [0, 1, 2]

    assert column_pattern_index.shape == (3,)
    assert np.all(column_pattern_index == 0)


def test_missing_patterns_all_unique_columns() -> None:
    """Every column has a unique pattern -> n_patterns == n_cols, each group size 1."""
    # 3 columns, all different
    mask = np.array([
        [1, 0, 1],
        [0, 1, 1],
    ])

    n_patterns, pattern_columns, column_pattern_index = _missing_patterns(mask)

    # Each column is unique
    assert n_patterns == 3
    assert len(pattern_columns) == 3

    # Each group should contain exactly one column index
    group_sizes = [len(g) for g in pattern_columns]
    assert group_sizes == [1, 1, 1]

    # column_pattern_index should be a permutation of [0, 1, 2]
    assert sorted(column_pattern_index.tolist()) == [0, 1, 2]


def test_missing_patterns_non_boolean_mask() -> None:
    """Mask can be numeric; only pattern equality matters."""
    # numeric mask (e.g., 1 = observed, 9 = missing)
    mask = np.array([
        [1, 9, 1],
        [9, 9, 1],
    ])

    n_patterns, pattern_columns, column_pattern_index = _missing_patterns(mask)

    # Columns 0 and 2 differ: [1,9] vs [1,1]; col1 is [9,9]
    assert n_patterns == 3
    assert len(pattern_columns) == 3
    assert column_pattern_index.shape == (3,)


def test_missing_patterns_zero_columns() -> None:
    """Matrix with zero columns should yield zero patterns and empty outputs."""
    mask = np.empty((5, 0))  # 5 rows, 0 columns

    n_patterns, pattern_columns, column_pattern_index = _missing_patterns(mask)

    assert n_patterns == 0
    assert pattern_columns == []
    assert column_pattern_index.shape == (0,)


def test_missing_patterns_zero_columns_verbose_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mask = np.empty((3, 0), dtype=int)
    caplog.set_level("DEBUG", logger=missing_mod.logger.name)

    n_patterns, pattern_columns, column_pattern_index = _missing_patterns(
        mask,
        verbose=True,
    )

    assert n_patterns == 0
    assert pattern_columns == []
    assert column_pattern_index.size == 0
    assert "zero columns" in caplog.text


def test_missing_patterns_verbose_logs_non_degenerate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
    caplog.set_level("DEBUG", logger=missing_mod.logger.name)

    n_patterns, _pattern_columns, idx = _missing_patterns(mask, verbose=True)

    assert n_patterns == 2
    assert idx.shape == (3,)
    assert "found" in caplog.text
    assert "column pattern index" in caplog.text


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.array(1.0),  # scalar
        np.array([1, 0, 1]),  # 1D
        np.array([[[1], [0]]]),  # 3D
    ],
)
def test_missing_patterns_bad_dim_raises(bad_mask: np.ndarray) -> None:
    """_missing_patterns should raise when mask is not 2D."""
    with pytest.raises(ValueError, match=r"mask must be a 2D array."):
        _missing_patterns(bad_mask)
