"""Tests for vbpca_py._rmempty.remove_empty_entries."""

import numpy as np
import scipy.sparse as sp

from vbpca_py._remove_empty import remove_empty_entries


def test_dense_no_empty_rows_or_columns() -> None:
    """If there are no empty rows/cols, inputs should be returned unchanged."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    x_probe = x + 1.0

    init = {
        "A": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "Av": [np.eye(2), np.eye(2)],
        "S": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "Sv": [np.eye(2), np.eye(2)],
    }

    x_out, x_probe_out, ir, ic, init_out = remove_empty_entries(x, x_probe, init)

    # Indices should cover all rows/cols
    assert np.array_equal(ir, np.array([0, 1]))
    assert np.array_equal(ic, np.array([0, 1]))

    # Data matrices unchanged
    assert isinstance(x_out, np.ndarray)
    assert np.array_equal(x_out, x)
    assert isinstance(x_probe_out, np.ndarray)
    assert np.array_equal(x_probe_out, x_probe)

    # Init should be structurally unchanged
    assert init_out is init
    assert np.array_equal(init_out["A"], init["A"])
    assert np.array_equal(init_out["S"], init["S"])
    assert len(init_out["Av"]) == 2
    assert len(init_out["Sv"]) == 2
    assert np.allclose(init_out["Av"][0], np.eye(2))
    assert np.allclose(init_out["Sv"][1], np.eye(2))


def test_dense_remove_empty_rows_and_columns() -> None:
    """Dense X: rows/cols that are all-NaN should be removed and init sliced."""
    # Row 1 is all NaN, column 1 is all NaN
    x = np.array(
        [
            [1.0, np.nan, 2.0],
            [np.nan, np.nan, np.nan],
            [3.0, np.nan, 4.0],
        ],
    )
    x_probe = x + 10.0

    # A: (n_rows, k), S: (k, n_cols)
    k = 2
    a = np.arange(3 * k, dtype=float).reshape(3, k)
    s = np.arange(k * 3, dtype=float).reshape(k, 3)

    av = [np.eye(k) * (i + 1.0) for i in range(3)]
    sv = [np.eye(k) * (j + 1.0) for j in range(3)]

    init = {
        "A": a.copy(),
        "Av": list(av),
        "S": s.copy(),
        "Sv": list(sv),
    }

    x_out, x_probe_out, ir, ic, init_out = remove_empty_entries(x, x_probe, init)

    # We expect to keep rows 0 and 2, and columns 0 and 2.
    assert np.array_equal(ir, np.array([0, 2]))
    assert np.array_equal(ic, np.array([0, 2]))

    # Check X slicing (all-NaN row/col removed)
    expected_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert x_out.shape == (2, 2)
    assert np.allclose(x_out, expected_x, equal_nan=False)

    # Probe sliced in the same way
    assert x_probe_out is not None
    expected_x_probe = expected_x + 10.0
    assert x_probe_out.shape == (2, 2)
    assert np.allclose(x_probe_out, expected_x_probe, equal_nan=False)

    # Init fields sliced consistently
    a_expected = a[ir, :]
    s_expected = s[:, ic]
    assert np.array_equal(init_out["A"], a_expected)
    assert np.array_equal(init_out["S"], s_expected)

    # Av and Sv lists filtered by row/col indices
    assert len(init_out["Av"]) == 2
    assert np.allclose(init_out["Av"][0], av[0])
    assert np.allclose(init_out["Av"][1], av[2])

    assert len(init_out["Sv"]) == 2
    assert np.allclose(init_out["Sv"][0], sv[0])
    assert np.allclose(init_out["Sv"][1], sv[2])


def test_sparse_remove_empty_rows_and_columns() -> None:
    """Sparse X: rows/cols with zero sums should be removed."""
    # 3x4 matrix; row 1 is all zeros, columns 1 and 2 are all zeros
    # Non-zeros at (0,0), (0,3), (2,0), (2,3).
    data = np.array([1.0, 2.0, 3.0, 4.0])
    indices = np.array([0, 3, 0, 3])
    indptr = np.array([0, 2, 2, 4])  # row 0: 0-2, row1: 2-2, row2: 2-4
    x = sp.csr_matrix((data, indices, indptr), shape=(3, 4))

    # Use a probe matrix that mirrors X
    x_probe = x.copy()

    a = np.arange(3 * 2, dtype=float).reshape(3, 2)
    s = np.arange(2 * 4, dtype=float).reshape(2, 4)
    av = [np.eye(2) * (i + 1.0) for i in range(3)]
    sv = [np.eye(2) * (j + 1.0) for j in range(4)]

    init = {
        "A": a.copy(),
        "Av": list(av),
        "S": s.copy(),
        "Sv": list(sv),
    }

    # Snapshot originals before mutation
    a_orig = init["A"].copy()
    s_orig = init["S"].copy()
    av_orig = list(init["Av"])
    sv_orig = list(init["Sv"])

    x_out, x_probe_out, ir, ic, init_out = remove_empty_entries(x, x_probe, init)

    # Keep rows 0 and 2, columns 0 and 3
    assert np.array_equal(ir, np.array([0, 2]))
    assert np.array_equal(ic, np.array([0, 3]))

    assert sp.isspmatrix_csr(x_out)
    assert x_out.shape == (2, 2)

    # The non-zero pattern should compress into a 2x2 matrix:
    # original:
    # [1 0 0 2]
    # [0 0 0 0]
    # [3 0 0 4]
    # after removing row1, col1,2:
    # [1 2]
    # [3 4]
    expected_dense = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(x_out.toarray(), expected_dense)

    assert x_probe_out is not None
    assert sp.isspmatrix_csr(x_probe_out)
    assert np.allclose(x_probe_out.toarray(), expected_dense)

    # Init arrays/lists sliced consistently (compare to *originals*)
    assert np.array_equal(init_out["A"], a_orig[ir, :])


def test_init_non_dict_is_passed_through() -> None:
    """If init is not a dict, it should be returned unchanged."""
    x = np.array([[np.nan, 1.0], [2.0, np.nan]])
    x_probe = None
    init = 42  # arbitrary non-dict

    _, _, ir, ic, init_out = remove_empty_entries(x, x_probe, init)

    # Here both rows and both cols have at least one non-NaN entry.
    assert np.array_equal(ir, np.array([0, 1]))
    assert np.array_equal(ic, np.array([0, 1]))
    assert init_out == init


def test_x_probe_none_is_preserved() -> None:
    """When x_probe is None, it should remain None after processing."""
    x = np.array([[np.nan, 1.0], [np.nan, np.nan]])
    x_probe = None
    init = None

    x_out, x_probe_out, ir, ic, init_out = remove_empty_entries(x, x_probe, init)

    # Row 1 is empty, col 0 has only NaN, so we keep only the (0,1) entry.
    assert np.array_equal(ir, np.array([0]))
    assert np.array_equal(ic, np.array([1]))
    assert x_out.shape == (1, 1)
    assert x_probe_out is None
    assert init_out is None
