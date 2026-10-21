"""Preparation follows observation presence rather than numeric magnitude."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from scipy import sparse

from vbpca_py._pca_full import pca_full
from vbpca_py._remove_empty import _nonempty_indices, remove_empty_entries


def _stored(values, rows, columns, shape):
    return sparse.coo_matrix((values, (rows, columns)), shape=shape).tocsr()


def _snapshot(matrix):
    return [getattr(matrix, key).copy() for key in ("data", "indices", "indptr")]


def _assert_sparse_unchanged(matrix, before):
    for key, expected in zip(("data", "indices", "indptr"), before, strict=True):
        np.testing.assert_array_equal(getattr(matrix, key), expected)


def test_modern_stored_zeros_are_observations_and_nan_entries_are_missing():
    data = _stored([0.0, 0.0, np.nan], [0, 0, 1], [0, 2, 1], (3, 4))
    before = _snapshot(data)
    result, _, rows, columns, _ = remove_empty_entries(
        data, None, None, compat_mode="modern"
    )
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(columns, [0, 2])
    assert result.shape == (1, 2)
    assert result.nnz == 2
    np.testing.assert_array_equal(result.data, [0.0, 0.0])
    _assert_sparse_unchanged(data, before)


def test_zero_only_sparse_input_keeps_historical_mode_explicit():
    data = _stored([0.0, 0.0], [0, 0], [0, 2], (1, 3))
    result, _, rows, columns, _ = remove_empty_entries(
        data, None, None, compat_mode="strict_legacy"
    )
    assert rows.size == columns.size == 0
    assert result.shape == (0, 0)


@pytest.mark.parametrize("compat_mode", ["strict_legacy", "modern"])
@pytest.mark.parametrize("mask_sparse", [False, True])
@pytest.mark.parametrize("data_sparse", [False, True])
def test_boolean_masks_define_empty_axes(compat_mode, mask_sparse, data_sparse):
    data = np.ones((2, 3))
    if data_sparse:
        data = sparse.csr_matrix(data)
    mask = np.array([[True, False, False], [False, False, False]])
    if mask_sparse:
        # Include false entries in the stored mask: storage alone is not truth.
        mask = _stored([True, False, False], [0, 0, 1], [0, 1, 2], (2, 3))
        before = _snapshot(mask)
        for value in (mask.data, mask.indices, mask.indptr):
            value.setflags(write=False)
    else:
        before = mask.copy()
        mask.setflags(write=False)
    result, _, rows, columns, _ = remove_empty_entries(
        data, None, None, compat_mode=compat_mode, mask=mask
    )
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(columns, [0])
    assert result.shape == (1, 1)
    if mask_sparse:
        _assert_sparse_unchanged(mask, before)
    else:
        np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("mask_sparse", [False, True])
def test_entirely_false_mask_has_no_observations(mask_sparse):
    mask = np.zeros((2, 3), dtype=bool)
    if mask_sparse:
        mask = _stored([False, False], [0, 1], [0, 2], (2, 3))
    result, _, rows, columns, _ = remove_empty_entries(
        np.ones((2, 3)), None, None, compat_mode="modern", mask=mask
    )
    assert result.shape == (0, 0)
    assert rows.size == columns.size == 0


def test_dense_nan_and_sparse_observation_axes_agree_including_zero_rows():
    dense = np.array([
        [np.nan, 0.0, np.nan, 0.0],
        [np.nan] * 4,
        [2.0, np.nan, np.nan, -2.0],
    ])
    rows, columns = np.nonzero(~np.isnan(dense))
    stored = _stored(dense[rows, columns], rows, columns, dense.shape)
    results = [
        remove_empty_entries(table, None, None, compat_mode="modern")
        for table in (dense, stored)
    ]
    np.testing.assert_array_equal(results[0][2], results[1][2])
    np.testing.assert_array_equal(results[0][3], results[1][3])
    assert results[0][0].shape == results[1][0].shape == (2, 3)


def test_observed_zero_feature_has_equivalent_dense_and_sparse_fits():
    dense = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
    rows, columns = np.indices(dense.shape)
    stored = _stored(dense.ravel(), rows.ravel(), columns.ravel(), dense.shape)
    fitted = [
        pca_full(
            table,
            1,
            compat_mode="modern",
            maxiters=4,
            niter_broadprior=0,
            random_state=26,
            runtime_tuning="off",
            num_cpu=1,
            return_diagnostics=False,
        )
        for table in (dense, stored)
    ]
    for result in fitted:
        assert np.isfinite(result["A"]).all()
        assert np.isfinite(result["Mu"]).all()
        np.testing.assert_allclose(result["Mu"][0, 0], 0.0, rtol=0, atol=1e-10)
    reconstructions = [r["A"] @ r["S"] + r["Mu"] for r in fitted]
    np.testing.assert_allclose(
        reconstructions[0], reconstructions[1], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(fitted[0]["V"], fitted[1]["V"], rtol=1e-8, atol=1e-10)


def test_wide_sparse_axis_detection_does_not_allocate_a_full_column_count():
    data = _stored([0.0, 3.0], [0, 1], [0, 999_999], (2, 1_000_000))
    tracemalloc.start()
    try:
        rows, columns = _nonempty_indices(data, "modern")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(rows, [0, 1])
    np.testing.assert_array_equal(columns, [0, 999_999])
    # A full int64 column-count vector would use eight MB; allow ample room
    # for small metadata without depending on exact counts. SciPy's subsequent
    # matrix slicing has its own per-column indexing metadata.
    assert peak < 1_000_000
