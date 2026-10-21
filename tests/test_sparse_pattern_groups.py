"""Sparse missingness groups must partition sample columns, not feature rows."""

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py._full_update import _patterns_from_csc
from vbpca_py._pca_full import pca_full

_OFF = dict.fromkeys(
    ("angle", "earlystop", "rms_plateau", "cost", "composite", "slowing_down"), False
)


def test_sparse_groups_are_sample_columns_without_densification(monkeypatch) -> None:
    mask = sp.csc_matrix([
        [True, True, False, False, True],
        [False, False, False, True, False],
        [True, True, False, False, True],
    ])
    originals = [array.copy() for array in (mask.data, mask.indices, mask.indptr)]
    for array in (mask.data, mask.indices, mask.indptr):
        array.flags.writeable = False

    def reject_dense(*args, **kwargs):
        pytest.fail("Pattern grouping must not densify the sparse mask")

    monkeypatch.setattr(sp.csc_matrix, "toarray", reject_dense)
    monkeypatch.setattr(sp.csc_matrix, "todense", reject_dense)
    count, groups, index = _patterns_from_csc(mask)
    assert count == 3
    assert groups == [[0, 1, 4], [2], [3]]
    np.testing.assert_array_equal(index, [0, 0, 1, 2, 0])
    for actual, expected in zip(
        (mask.data, mask.indices, mask.indptr), originals, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(3, 11), (11, 3), (5, 5), (0, 4), (3, 0)])
def test_groups_partition_all_samples_and_preserve_pattern_identity(shape) -> None:
    mask = np.random.default_rng(299).random(shape) > 0.4
    count, groups, index = _patterns_from_csc(sp.csc_matrix(mask))
    assert count == len(groups)
    assert sorted(column for group in groups for column in group) == list(
        range(shape[1])
    )
    assert len(index) == shape[1]
    first_seen = {}
    for column in range(shape[1]):
        key = tuple(np.flatnonzero(mask[:, column]))
        expected = first_seen.setdefault(key, len(first_seen))
        assert index[column] == expected
        assert column in groups[expected]
    assert count == len(first_seen)


@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
@pytest.mark.parametrize("compat", ["modern", "strict_legacy"])
@pytest.mark.parametrize("writeback", ["python", "bulk"])
@pytest.mark.parametrize("threads", [1, 2, 4])
@pytest.mark.parametrize("sharing", ["uniquesv", "auto_pattern_masked"])
def test_sparse_shared_fit_matches_unshared_path(
    algorithm, compat, writeback, threads, sharing
) -> None:
    rng = np.random.default_rng(299)
    values = rng.normal(size=(7, 9))
    mask = np.ones(values.shape, dtype=bool)
    mask[1, [1, 4, 7]] = False
    mask[3, [2, 5, 8]] = False
    values[0, 0] = 0.0
    rows, columns = np.nonzero(mask)
    data = sp.csr_matrix((values[rows, columns], (rows, columns)), shape=values.shape)
    data_before = data.copy()
    options = {
        "algorithm": algorithm,
        "compat_mode": compat,
        "cov_writeback_mode": writeback,
        "num_cpu": threads,
        "maxiters": 4,
        "niter_broadprior": 0,
        "random_state": 299,
        "convergence_criteria": _OFF,
        "record_cost": True,
        "runtime_tuning": "off",
        "return_diagnostics": True,
    }
    expected = pca_full(data, 2, **options)
    actual = pca_full(data, 2, **options, **{sharing: True})
    assert actual["Isv"] is not None
    # Eigensolvers can choose opposite signs for the same component. Apply
    # the same sign change to scores and covariance blocks, preserving AS.
    signs = np.where(np.sum(actual["A"] * expected["A"], axis=0) >= 0, 1.0, -1.0)
    np.testing.assert_allclose(
        actual["A"] * signs, expected["A"], rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        actual["S"] * signs[:, None], expected["S"], rtol=1e-9, atol=1e-10
    )
    for key in (
        "Mu",
        "V",
        "Muv",
        "Va",
        "Vmu",
        "Xrec",
        "Vr",
        "Vpred",
        "RMS",
        "Cost",
    ):
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=1e-9, atol=1e-10, err_msg=key
        )
    for key in ("rms", "prms", "cost"):
        np.testing.assert_allclose(
            actual["lc"][key], expected["lc"][key], rtol=1e-9, atol=1e-10, err_msg=key
        )
    assert len(actual["lc"]["rms"]) == 5
    assert np.isfinite(actual["lc"]["cost"][1:]).all()
    for column in range(data.shape[1]):
        if len(expected["Sv"]):
            np.testing.assert_allclose(
                signs[:, None] * actual["Sv"][actual["Isv"][column]] * signs,
                expected["Sv"][column],
                rtol=1e-9,
                atol=1e-10,
            )
    for actual_block, expected_block in zip(actual["Av"], expected["Av"], strict=True):
        np.testing.assert_allclose(
            signs[:, None] * actual_block * signs, expected_block, rtol=1e-9, atol=1e-10
        )
    for actual_array, original in zip(
        (data.data, data.indices, data.indptr),
        (data_before.data, data_before.indices, data_before.indptr),
        strict=True,
    ):
        np.testing.assert_array_equal(actual_array, original)
