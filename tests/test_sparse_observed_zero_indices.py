"""Modern sparse observations include stored zero values in noise updates."""

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py._full_update import (
    _exclude_probe_from_training,
    _filter_csr_entries,
    _normalize_sparse_data,
    _observed_indices_with_mode,
)
from vbpca_py._pca_full import pca_full

_OFF = dict.fromkeys(
    ("angle", "earlystop", "rms_plateau", "cost", "composite", "slowing_down"), False
)


def _stored(values, mask):
    rows, columns = np.nonzero(mask)
    return sp.csr_matrix((values[rows, columns], (rows, columns)), shape=values.shape)


@pytest.mark.parametrize("keep", [[True] * 4, [False] * 4, [True, False, True, False]])
def test_csr_filter_preserves_zero_entries_empty_rows_and_ownership(
    keep, monkeypatch
) -> None:
    matrix = sp.csr_matrix(
        ([0.0, 2.0, 0.0, 4.0], ([0, 1, 1, 3], [0, 2, 4, 1])), shape=(4, 5)
    )
    originals = [array.copy() for array in (matrix.data, matrix.indices, matrix.indptr)]
    for array in (matrix.data, matrix.indices, matrix.indptr):
        array.flags.writeable = False

    def reject_dense(*args, **kwargs):
        pytest.fail("CSR filtering must not densify")

    monkeypatch.setattr(sp.csr_matrix, "toarray", reject_dense)
    monkeypatch.setattr(sp.csr_matrix, "todense", reject_dense)
    selected = np.asarray(keep)
    result = _filter_csr_entries(matrix, selected)
    assert result.shape == matrix.shape
    np.testing.assert_array_equal(result.data, originals[0][selected])
    np.testing.assert_array_equal(result.indices, originals[1][selected])
    np.testing.assert_array_equal(
        result.indptr, [0, sum(keep[:1]), sum(keep[:3]), sum(keep[:3]), sum(keep)]
    )
    for actual, original in zip(
        (result.data, result.indices, result.indptr),
        (matrix.data, matrix.indices, matrix.indptr),
        strict=True,
    ):
        assert not np.shares_memory(actual, original)
    for actual, original in zip(
        (matrix.data, matrix.indices, matrix.indptr), originals, strict=True
    ):
        np.testing.assert_array_equal(actual, original)


def test_csr_filter_accepts_empty_storage() -> None:
    result = _filter_csr_entries(sp.csr_matrix((4, 5)), np.zeros(0, dtype=bool))
    assert result.shape == (4, 5)
    assert result.nnz == 0
    np.testing.assert_array_equal(result.indptr, np.zeros(5, dtype=int))


@pytest.mark.parametrize("mode", ["modern", "strict_legacy"])
def test_sparse_probe_exclusion_preserves_the_selected_zero_policy(mode) -> None:
    values = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]])
    original = _stored(values, np.ones(values.shape, dtype=bool))
    probe = sp.csr_matrix(([1.0], ([0], [1])), shape=values.shape)
    training, _ = _exclude_probe_from_training(original, probe, None, compat_mode=mode)
    assert training.nnz == (5 if mode == "modern" else 3)
    assert (0, 1) not in set(zip(*training.nonzero(), strict=True))
    assert original.nnz == 6
    np.testing.assert_array_equal(original.toarray(), values)


@pytest.mark.parametrize("mode", ["modern", "strict_legacy"])
def test_nan_removal_retains_other_observed_zero_entries(mode) -> None:
    data = sp.csr_matrix(([0.0, np.nan, 2.0], ([0, 0, 1], [0, 1, 1])), shape=(2, 2))
    result = _normalize_sparse_data(data, mode)
    assert result.nnz == 2
    assert result[0, 0] == (0.0 if mode == "modern" else np.finfo(float).eps)
    assert np.isnan(data.data[1])
    assert data.nnz == 3


@pytest.mark.parametrize("storage", [np.asarray, sp.csr_matrix, sp.csc_matrix])
@pytest.mark.parametrize("mode", ["modern", "strict_legacy"])
def test_coordinate_membership_uses_mask_only_in_modern_mode(storage, mode) -> None:
    data = np.arange(12, dtype=float).reshape(3, 4)
    data[2, 3] = 0.0
    mask = np.ones_like(data, dtype=bool)
    mask[0, 1] = False
    sparse = _stored(data, np.ones_like(mask))
    snapshot = sparse.copy()
    rows, columns = _observed_indices_with_mode(sparse, storage(mask), mode)
    expected = np.nonzero(mask if mode == "modern" else data)
    assert set(zip(rows, columns, strict=True)) == set(zip(*expected, strict=True))
    assert rows.dtype == columns.dtype == np.intp
    np.testing.assert_array_equal(sparse.data, snapshot.data)
    np.testing.assert_array_equal(sparse.indices, snapshot.indices)
    np.testing.assert_array_equal(sparse.indptr, snapshot.indptr)


@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("writeback", ["python", "bulk"])
@pytest.mark.parametrize("threads", [1, 4])
def test_modern_sparse_fit_matches_dense_with_observed_zeros(
    algorithm, bias, masked, sharing, writeback, threads
) -> None:
    rng = np.random.default_rng(303)
    data = rng.normal(size=(7, 9))
    data[0, 0] = data[2, 1] = data[3, 4] = 0.0
    mask = np.ones(data.shape, dtype=bool)
    if masked:
        mask[1, 2:4] = False
        mask[5, 5] = False
    sparse = _stored(data, mask)
    original = sparse.copy()
    options = {
        "algorithm": algorithm,
        "bias": bias,
        "uniquesv": sharing,
        "maxiters": 4,
        "niter_broadprior": 0,
        "random_state": 303,
        "compat_mode": "modern",
        "convergence_criteria": _OFF,
        "record_cost": True,
        "runtime_tuning": "off",
        "cov_writeback_mode": writeback,
        "num_cpu": threads,
    }
    expected = pca_full(data, 2, mask=mask, **options)
    actual = pca_full(sparse, 2, **options)
    _compare_fits(actual, expected)
    for name in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(sparse, name), getattr(original, name))


def _compare_fits(actual, expected):
    signs = np.where(np.sum(actual["A"] * expected["A"], axis=0) >= 0, 1.0, -1.0)
    np.testing.assert_allclose(
        actual["A"] * signs, expected["A"], rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        actual["S"] * signs[:, None], expected["S"], rtol=1e-9, atol=1e-10
    )
    for key in (
        "Mu",
        "Muv",
        "Va",
        "Vmu",
        "V",
        "Xrec",
        "Vr",
        "Vpred",
        "RMS",
        "PRMS",
        "Cost",
    ):
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=1e-9, atol=1e-10, err_msg=key
        )
    for key in ("rms", "prms", "cost"):
        np.testing.assert_allclose(
            actual["lc"][key], expected["lc"][key], rtol=1e-9, atol=1e-10, err_msg=key
        )
    for key in ("Av",):
        assert len(actual[key]) == len(expected[key])
        for actual_block, expected_block in zip(
            actual[key], expected[key], strict=True
        ):
            np.testing.assert_allclose(
                signs[:, None] * actual_block * signs,
                expected_block,
                rtol=1e-9,
                atol=1e-10,
            )
    for column in range(actual["S"].shape[1]):
        actual_index = column if actual["Isv"] is None else actual["Isv"][column]
        expected_index = column if expected["Isv"] is None else expected["Isv"][column]
        np.testing.assert_allclose(
            signs[:, None] * actual["Sv"][actual_index] * signs,
            expected["Sv"][expected_index],
            rtol=1e-9,
            atol=1e-10,
        )


@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
@pytest.mark.parametrize("probe_zero", [False, True])
def test_zero_and_nonzero_sparse_probes_match_dense(algorithm, probe_zero) -> None:
    data = np.random.default_rng(303).normal(size=(7, 9))
    data[0, 0] = 0.0
    data[3, 4] = 0.0 if probe_zero else 2.0
    mask = np.ones(data.shape, dtype=bool)
    probe_mask = np.zeros_like(mask)
    probe_mask[3, 4] = True
    probe = np.full_like(data, np.nan)
    probe[probe_mask] = data[probe_mask]
    options = {
        "maxiters": 4,
        "niter_broadprior": 0,
        "random_state": 303,
        "bias": False,
        "compat_mode": "modern",
        "algorithm": algorithm,
        "convergence_criteria": _OFF,
        "record_cost": True,
    }
    expected = pca_full(data, 2, mask=mask, xprobe=probe, **options)
    actual = pca_full(
        _stored(data, mask), 2, xprobe=_stored(data, probe_mask), **options
    )
    _compare_fits(actual, expected)


@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
@pytest.mark.parametrize("bias", [False, True])
def test_nan_and_observed_zero_sparse_fit_matches_dense(algorithm, bias) -> None:
    data = np.random.default_rng(303).normal(size=(7, 9))
    data[0, 0] = data[3, 4] = 0.0
    data[1, 1] = np.nan
    options = {
        "maxiters": 4,
        "niter_broadprior": 0,
        "random_state": 303,
        "bias": bias,
        "compat_mode": "modern",
        "algorithm": algorithm,
        "convergence_criteria": _OFF,
        "record_cost": True,
    }
    expected = pca_full(data, 2, mask=np.isfinite(data), **options)
    actual = pca_full(_stored(data, np.ones(data.shape, dtype=bool)), 2, **options)
    _compare_fits(actual, expected)
