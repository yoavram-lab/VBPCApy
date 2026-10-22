"""Unit changes preserve observed scaling, constants and sparse observation support."""

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py.preprocessing import (
    MissingAwareMinMaxScaler,
    MissingAwareStandardScaler,
)


@pytest.mark.parametrize(
    "scaler_class", [MissingAwareStandardScaler, MissingAwareMinMaxScaler]
)
@pytest.mark.parametrize("storage", ["dense_nan", "dense_mask", "csr", "csc"])
@pytest.mark.parametrize("unit", [1e-100, 1e-10, 1.0, 1e10, 1e100])
def test_observed_scaling_is_invariant_under_positive_unit_changes(
    scaler_class, storage, unit
):
    x = np.array([
        [-1.0, -2.0, 0.1, 5.0],
        [0.0, 2.0, 0.1, -8.0],
        [1.0, 8.0, 0.1, 9.0],
        [9.0, -2.0, 7.0, 0.0],
        [9.0, 2.0, 0.1, 3.0],
    ])
    observed = np.array([
        [True, True, True, False],
        [True, True, True, False],
        [True, False, True, False],
        [False, True, False, False],
        [False, True, True, False],
    ])
    units = np.array([unit, 1 / unit, unit, 1.0])
    scaled = x * units
    rows, cols = np.nonzero(observed)
    original = sp.csr_matrix((scaled[rows, cols], (rows, cols)), shape=x.shape)
    if storage in {"csr", "csc"}:
        values = original if storage == "csr" else original.tocsc()
        mask = None
    elif storage == "dense_nan":
        values = np.where(observed, scaled, np.nan)
        mask = None
    else:
        values = scaled
        mask = observed
    scaler = scaler_class().fit(values, mask=mask)
    transformed = scaler.transform(values, mask=mask)
    reconstructed = scaler.inverse_transform(transformed, mask=mask)
    if storage in {"csr", "csc"}:
        assert sp.isspmatrix_csr(transformed)
        assert sp.isspmatrix_csr(reconstructed)
        # Numerical zeros remain stored observations, not missing cells.
        for result in (transformed, reconstructed):
            np.testing.assert_array_equal(result.indptr, original.indptr)
            np.testing.assert_array_equal(result.indices, original.indices)
            assert result.nnz == observed.sum()
        actual = np.asarray(transformed[rows, cols]).ravel()
        back = np.asarray(reconstructed[rows, cols]).ravel()
    else:
        np.testing.assert_array_equal(np.isnan(transformed), ~observed)
        np.testing.assert_array_equal(np.isnan(reconstructed), ~observed)
        actual = transformed[observed]
        back = reconstructed[observed]

    # References are exact moments/ranges of the three manufactured columns.
    expected = np.zeros(x.shape)
    if scaler_class is MissingAwareStandardScaler:
        expected[:3, 0] = [-np.sqrt(1.5), 0.0, np.sqrt(1.5)]
        expected[:, 1] = [-1.0, 1.0, 0.0, -1.0, 1.0]
        np.testing.assert_allclose(
            scaler.scale_[:2] / units[:2], [np.sqrt(2 / 3), 2.0], rtol=1e-14
        )
        np.testing.assert_array_equal(scaler.var_[2:], [0.0, np.nan])
        np.testing.assert_array_equal(scaler.scale_[2:], [1.0, 1.0])
        assert np.isnan(scaler.mean_[3])
    else:
        expected[:3, 0] = [0.0, 0.5, 1.0]
        expected[:, 1] = [0.0, 1.0, 0.0, 0.0, 1.0]
        np.testing.assert_allclose(
            scaler.data_range_[:2] / units[:2], [2.0, 4.0], rtol=1e-14
        )
        np.testing.assert_array_equal(scaler.data_range_[2:], [1.0, 1.0])
        assert np.isnan(scaler.data_min_[3])
        assert np.isnan(scaler.data_max_[3])
    np.testing.assert_allclose(actual, expected[observed], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(back / units[cols], x[observed], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def test_standard_scaler_retains_variation_around_a_large_offset(storage):
    x = np.array([[1e8], [1e8 + 1], [1e8 + 2]])
    values = x if storage == "dense" else getattr(sp, storage + "_matrix")(x)
    scaler = MissingAwareStandardScaler().fit(values)
    result = scaler.transform(values)
    actual = result if storage == "dense" else result.toarray()
    np.testing.assert_allclose(scaler.mean_, [1e8 + 1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scaler.var_, [2 / 3], rtol=1e-14)
    np.testing.assert_allclose(
        actual[:, 0], [-np.sqrt(1.5), 0.0, np.sqrt(1.5)], rtol=1e-14, atol=0.0
    )
