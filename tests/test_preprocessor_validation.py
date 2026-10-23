"""Preprocessing boundaries reject ambiguity before indexing or broadcasting."""

import copy
import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from vbpca_py import preprocessing as p

CLASSES = (
    p.MissingAwareStandardScaler,
    p.MissingAwareMinMaxScaler,
    p.MissingAwareOneHotEncoder,
    p.MissingAwareSparseOneHotEncoder,
    p.MissingAwareOrdinalEncoder,
    p.AutoEncoder,
    p.MissingAwareLogTransformer,
    p.MissingAwareWinsorizer,
    p.MissingAwarePowerTransformer,
)
SPARSE_CLASSES = (
    p.MissingAwareStandardScaler,
    p.MissingAwareMinMaxScaler,
    p.MissingAwareSparseOneHotEncoder,
    p.AutoEncoder,
)


def _data():
    return np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [1.0, 3.0]])


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "bad",
    [1.0, np.ones(3), np.ones((2, 2, 2)), np.empty((0, 2)), np.empty((2, 0))],
)
def test_failed_refit_rejects_non_2d_or_empty_data_and_clears_state(cls, bad):
    model = cls().fit(_data())
    with pytest.raises(ValueError):
        model.fit(bad)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert not hasattr(model, "n_features_in_")


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("count", [1, 3])
def test_transform_checks_feature_count_before_slicing_or_broadcasting(cls, count):
    model = cls().fit(_data())
    with pytest.raises(ValueError, match=r"features.*expecting 2 features"):
        model.transform(np.ones((4, count)))
    assert model.n_features_in_ == 2
    check_is_fitted(model)


@pytest.mark.parametrize("cls", CLASSES)
def test_inverse_checks_encoded_width_before_decoding(cls):
    model = cls().fit(_data())
    encoded = model.transform(_data())
    for count in [max(encoded.shape[1] - 1, 0), encoded.shape[1] + 1]:
        with pytest.raises(ValueError, match="fitted output width"):
            model.inverse_transform(np.ones((4, count)))
    with pytest.raises(ValueError):
        model.inverse_transform(np.ones(4))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("value", [np.inf, -np.inf, 1.0 + 2.0j])
def test_observed_infinite_and_complex_values_rejected_at_public_boundaries(cls, value):
    model = cls().fit(_data())
    bad = _data().astype(complex if np.iscomplexobj(value) else float)
    bad[0, 0] = value
    with pytest.raises(ValueError, match=r"infinity|Complex"):
        model.transform(bad)
    encoded = model.transform(_data())
    width = encoded.shape[1]
    bad_inverse = np.ones((4, width), dtype=bad.dtype)
    bad_inverse[0, 0] = value
    with pytest.raises(ValueError, match=r"infinity|Complex"):
        model.inverse_transform(bad_inverse)
    with pytest.raises(ValueError, match=r"infinity|Complex"):
        model.fit(bad)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@pytest.mark.parametrize("cls", CLASSES)
def test_feature_names_checked_before_column_selection_and_reset_on_refit(cls):
    pd = pytest.importorskip("pandas")
    data = pd.DataFrame(_data(), columns=["alpha", "beta"])
    model = cls().fit(data)
    np.testing.assert_array_equal(model.feature_names_in_, data.columns)
    with pytest.raises(ValueError, match=r"Feature names|feature names"):
        model.transform(data[["beta", "alpha"]])
    with pytest.raises(ValueError, match=r"Feature names|feature names"):
        model.transform(data.rename(columns={"beta": "gamma"}))
    model.fit(_data())
    assert not hasattr(model, "feature_names_in_")


@pytest.mark.parametrize("cls", CLASSES)
def test_masks_checked_during_fit_transform_and_inverse(cls):
    model = cls().fit(_data())
    for operation in (
        lambda: model.fit(_data(), mask=np.ones((4, 1))),
        lambda: model.transform(_data(), mask=np.ones((4, 1))),
        lambda: model.inverse_transform(model.transform(_data()), mask=np.ones((4, 1))),
    ):
        model.fit(_data())
        with pytest.raises(ValueError, match="mask"):
            operation()


@pytest.mark.parametrize("cls", SPARSE_CLASSES)
@pytest.mark.parametrize(
    "kind", [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix, sp.csr_array]
)
def test_sparse_types_observed_zeros_nans_and_caller_ownership(cls, kind):
    values = np.array([[0.0, 2.0], [1.0, np.nan], [2.0, 3.0]])
    rows, cols = np.indices(values.shape)
    data = kind((values.ravel(), (rows.ravel(), cols.ravel())), shape=values.shape)
    original = copy.deepcopy(data)
    data.data.flags.writeable = False
    model = cls().fit(data)
    encoded = model.transform(data)
    inverse = model.inverse_transform(encoded)
    assert sp.issparse(encoded)
    assert sp.issparse(inverse)
    assert inverse.nnz == 5
    assert not np.isnan(inverse.data).any()
    restored = inverse.tocsr()
    # Both zero values and transformed zeros remain explicit observations.
    assert 0 in restored.indices[restored.indptr[0] : restored.indptr[1]]
    np.testing.assert_allclose(restored.toarray(), np.nan_to_num(values), atol=1e-12)
    np.testing.assert_array_equal(data.data, original.data)
    if hasattr(data, "indptr"):
        np.testing.assert_array_equal(data.indptr, original.indptr)
        np.testing.assert_array_equal(data.indices, original.indices)


@pytest.mark.parametrize("cls", SPARSE_CLASSES)
def test_sparse_mask_is_explicitly_refused_instead_of_silently_ignored(cls):
    data = sp.csr_matrix(_data())
    model = cls().fit(data)
    for operation in (
        lambda: model.fit(data, mask=np.ones(data.shape)),
        lambda: model.transform(data, mask=np.ones(data.shape)),
        lambda: model.inverse_transform(
            model.transform(data), mask=sp.csr_matrix(data.shape)
        ),
    ):
        model.fit(data)
        with pytest.raises(ValueError, match="mask must be None"):
            operation()


@pytest.mark.parametrize("cls", SPARSE_CLASSES)
@pytest.mark.parametrize(
    "fault", ["decreasing", "negative", "out_of_bounds", "float_indices"]
)
def test_invalid_compressed_sparse_buffers_rejected_before_scipy_operations(cls, fault):
    data = sp.csr_matrix(np.ones((4, 2)))
    if fault == "decreasing":
        data.indptr[2] = 1
    elif fault == "negative":
        data.indices[0] = -1
    elif fault == "out_of_bounds":
        data.indices[0] = 2
    else:
        data.indices = data.indices.astype(float)
    with pytest.raises(ValueError, match="Invalid sparse"):
        cls().fit(data)


@pytest.mark.parametrize("center", [False, True])
def test_multi_column_sparse_encoding_uses_separate_category_blocks(center):
    values = np.array([[0.0, 2.0, np.nan], [1.0, 3.0, np.nan], [2.0, 2.0, np.nan]])
    rows, cols = np.nonzero(~np.isnan(values))
    data = sp.csr_matrix((values[rows, cols], (rows, cols)), shape=values.shape)
    encoder = p.MissingAwareSparseOneHotEncoder(mean_center=center).fit(data)
    assert encoder.categories_ == [0.0, 1.0, 2.0, 2.0, 3.0]
    np.testing.assert_array_equal(encoder.feature_groups_, [0, 0, 0, 1, 1])
    encoded = encoder.transform(data)
    assert encoded.shape == (3, 5)
    assert encoded.nnz == 6
    restored = encoder.inverse_transform(encoded)
    assert restored.shape == values.shape
    np.testing.assert_array_equal(restored.indptr, data.indptr)
    np.testing.assert_array_equal(restored.indices, data.indices)
    np.testing.assert_allclose(restored.data, data.data)
    # Dense zeros are observed too; dense and stored-entry inputs agree.
    dense_encoder = p.MissingAwareSparseOneHotEncoder(mean_center=center).fit(values)
    dense_encoded = dense_encoder.transform(values)
    np.testing.assert_allclose(dense_encoded.toarray(), encoded.toarray())


def test_sparse_unknown_category_does_not_leak_from_another_column():
    data = sp.csr_matrix([[1.0, 2.0], [2.0, 3.0]])
    unknown = sp.csr_matrix([[3.0, 1.0]])
    ignored = p.MissingAwareSparseOneHotEncoder().fit(data).transform(unknown)
    assert ignored.nnz == 0
    with pytest.raises(ValueError, match=r"Unknown category.*column 0"):
        p.MissingAwareSparseOneHotEncoder(handle_unknown="raise").fit(data).transform(
            unknown
        )


def test_large_sparse_shape_never_densifies_and_keeps_centered_stored_zero(monkeypatch):
    values = sp.csr_matrix(([0.0, 7.0], ([0, 99999], [0, 16])), shape=(100000, 17))

    def forbid(*_args, **_kwargs):
        pytest.fail("sparse preprocessing must not densify")

    monkeypatch.setattr(sp.csr_matrix, "toarray", forbid)
    monkeypatch.setattr(sp.csr_matrix, "todense", forbid)
    encoder = p.MissingAwareSparseOneHotEncoder(mean_center=True).fit(values)
    encoded = encoder.transform(values)
    restored = encoder.inverse_transform(encoded)
    assert encoded.nnz == restored.nnz == 2
    np.testing.assert_array_equal(restored.indices, values.indices)
    np.testing.assert_array_equal(restored.indptr, values.indptr)
    np.testing.assert_array_equal(restored.data, values.data)


@pytest.mark.parametrize("cls", CLASSES)
def test_all_preprocessors_pass_every_applicable_generic_sklearn_check(cls):
    # SciPy dispatch configuration is fixed at import. A fresh CPU process
    # ensures the Array API checks actually execute on current sklearn too.
    code = (
        "from sklearn.utils.estimator_checks import check_estimator\n"
        f"from vbpca_py.preprocessing import {cls.__name__}\n"
        f"results = check_estimator({cls.__name__}())\n"
        "unpassed = [r for r in results if r['status'] != 'passed']\n"
        "assert results and not unpassed, unpassed\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "SCIPY_ARRAY_API": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("cls", CLASSES)
def test_duplicate_named_inputs_refused_and_failed_fit_does_not_keep_names(cls):
    pd = pytest.importorskip("pandas")
    duplicate = pd.DataFrame(_data(), columns=["same", "same"])
    model = cls()
    with pytest.raises(
        ValueError, match=r"duplicate feature names|unique column names"
    ):
        model.fit(duplicate)
    assert not hasattr(model, "feature_names_in_")
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@pytest.mark.parametrize(
    "cls",
    [p.MissingAwareOneHotEncoder, p.MissingAwareOrdinalEncoder, p.AutoEncoder],
)
def test_string_categories_remain_valid_but_object_infinity_and_complex_do_not(cls):
    model = cls().fit([["a", "d"], ["b", "e"], ["a", "f"]])
    model.transform([["a", "d"]])
    for value in (np.inf, complex(1.0, 0.0)):
        bad = np.array([[value, "d"]], dtype=object)
        with pytest.raises(ValueError, match=r"infinity|Complex"):
            model.transform(bad)


@pytest.mark.parametrize(
    "cls",
    [
        p.MissingAwareStandardScaler,
        p.MissingAwareMinMaxScaler,
        p.MissingAwareLogTransformer,
        p.MissingAwareWinsorizer,
        p.MissingAwarePowerTransformer,
    ],
)
def test_masked_infinity_is_excluded_from_dense_numeric_statistics(cls):
    data = _data()
    data[0, 0] = np.inf
    mask = np.ones(data.shape, dtype=bool)
    mask[0, 0] = False
    canonical = np.where(mask, data, np.nan)
    actual = cls().fit(data, mask=mask).transform(data, mask=mask)
    expected = cls().fit_transform(canonical)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_autoencoder_inverse_mask_uses_original_variables_after_expansion():
    values = np.array([[0.0, "a"], [1.0, "b"], [2.0, "c"]], dtype=object)
    model = p.AutoEncoder(column_types=["continuous", "categorical"]).fit(values)
    encoded = model.transform(values)
    assert encoded.shape == (3, 4)
    mask = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
    restored = model.inverse_transform(encoded, mask=mask)
    assert np.isnan(restored[0, 1])
    assert np.isnan(restored[1, 0])
    np.testing.assert_allclose(
        restored[mask[:, 0], 0].astype(float),
        values[mask[:, 0], 0].astype(float),
        atol=2e-12,
    )
    np.testing.assert_array_equal(restored[mask[:, 1], 1], values[mask[:, 1], 1])
    with pytest.raises(ValueError, match="mask shape"):
        model.inverse_transform(encoded, mask=np.ones(encoded.shape, dtype=bool))


@pytest.mark.parametrize(
    "cls",
    [p.MissingAwareOneHotEncoder, p.MissingAwareSparseOneHotEncoder, p.AutoEncoder],
)
def test_wholly_missing_category_blocks_can_be_decoded_from_zero_width(cls):
    data = np.full((3, 2), np.nan)
    model = cls().fit(data)
    encoded = model.transform(data)
    assert encoded.shape == (3, 0)
    restored = model.inverse_transform(encoded)
    assert restored.shape == data.shape
    if sp.issparse(restored):
        assert restored.nnz == 0
    else:
        assert np.isnan(restored).all()


@pytest.mark.parametrize("cls", SPARSE_CLASSES)
def test_sparse_duplicate_cancellation_keeps_zero_observed_and_caller_unchanged(cls):
    # Two stored contributions sum to one observed zero, not missingness.
    data = sp.csr_matrix(
        (np.array([1.0, -1.0, 2.0]), np.array([0, 0, 0]), np.array([0, 2, 3])),
        shape=(2, 1),
    )
    before = copy.deepcopy(data)
    model = cls().fit(data)
    restored = model.inverse_transform(model.transform(data))
    assert restored.nnz == 2
    np.testing.assert_array_equal(restored.data, [0.0, 2.0])
    np.testing.assert_array_equal(data.data, before.data)
    np.testing.assert_array_equal(data.indices, before.indices)
    np.testing.assert_array_equal(data.indptr, before.indptr)


@pytest.mark.parametrize("cls", SPARSE_CLASSES)
@pytest.mark.parametrize("kind", [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix])
@pytest.mark.parametrize("values", [[np.inf, -np.inf], [1e308, 1e308]])
def test_sparse_duplicates_cannot_hide_raw_infinity_or_sum_overflow(cls, kind, values):
    if kind is sp.coo_matrix:
        data = kind((values, ([0, 0], [0, 0])), shape=(2, 1))
    else:
        pointer = [0, 2, 2] if kind is sp.csr_matrix else [0, 2]
        data = kind((values, [0, 0], pointer), shape=(2, 1))
    before = copy.deepcopy(data)
    with (
        pytest.raises(ValueError, match="infinity"),
        np.errstate(over="ignore", invalid="ignore"),
    ):
        cls().fit(data)
    np.testing.assert_array_equal(data.data, before.data)
    np.testing.assert_array_equal(
        data.indices if hasattr(data, "indices") else data.row,
        before.indices if hasattr(before, "indices") else before.row,
    )


@pytest.mark.parametrize("cls", [p.MissingAwareOneHotEncoder, p.AutoEncoder])
def test_inverse_infinity_is_ignored_only_inside_explicitly_missing_variable(cls):
    values = np.array([["a", "x"], ["b", "y"], ["c", "z"]], dtype=object)
    model = cls().fit(values)
    encoded = model.transform(values)
    encoded[0, 0] = np.inf
    mask = np.ones(values.shape, dtype=bool)
    with pytest.raises(ValueError, match="infinity"):
        model.inverse_transform(encoded, mask=mask)
    mask[0, 0] = False
    restored = model.inverse_transform(encoded, mask=mask)
    assert np.isnan(restored[0, 0])
    np.testing.assert_array_equal(restored[mask], values[mask])


@pytest.mark.parametrize("cls", SPARSE_CLASSES)
def test_one_dimensional_sparse_input_is_refused_before_conversion(cls, monkeypatch):
    data = sp.coo_array(([1.0], ([0],)), shape=(3,))
    # Shape refusal must precede any conversion of even malformed coordinates.
    data.coords = (np.array([-1]),)

    def forbid(*_args, **_kwargs):
        pytest.fail("unsupported sparse dimensions must be refused before conversion")

    monkeypatch.setattr(sp.coo_array, "tocsr", forbid)
    with pytest.raises(ValueError, match="Expected 2D sparse input"):
        cls().fit(data)


@pytest.mark.parametrize("cls", CLASSES)
def test_native_input_tags_match_missing_and_representation_checks(cls):
    from typing import get_type_hints

    from sklearn.utils import Tags, get_tags

    tags = get_tags(cls())
    assert get_type_hints(cls().__sklearn_tags__)["return"] is Tags
    assert tags.input_tags.allow_nan
    assert tags.input_tags.two_d_array
    assert tags.requires_fit
    assert not tags.non_deterministic
    assert tags.input_tags.sparse == (cls in SPARSE_CLASSES)
    categorical = cls in {
        p.MissingAwareOneHotEncoder,
        p.MissingAwareOrdinalEncoder,
        p.AutoEncoder,
    }
    assert tags.input_tags.string == categorical
    assert tags.input_tags.categorical == categorical
    assert tags.transformer_tags.preserves_dtype == ["float64"]


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_declared_floating_output_dtype_matches_dense_numeric_transform(cls, dtype):
    from sklearn.utils import get_tags

    configurable = cls in {
        p.MissingAwareOneHotEncoder,
        p.MissingAwareSparseOneHotEncoder,
    }
    model = cls(dtype=dtype) if configurable else cls()
    output = model.fit_transform(_data().astype(dtype))
    expected = np.dtype(dtype if configurable else np.float64)
    assert output.dtype == expected
    assert get_tags(model).transformer_tags.preserves_dtype == [expected.name]


@pytest.mark.parametrize(
    "cls", [p.MissingAwareOneHotEncoder, p.MissingAwareSparseOneHotEncoder]
)
@pytest.mark.parametrize("center", [False, True])
def test_configured_complex_indicator_storage_roundtrips_real_categories(cls, center):
    values = np.array([[0.0, 2.0], [1.0, 2.0], [0.0, np.nan], [2.0, 0.0]])
    model = cls(dtype=np.complex64, mean_center=center).fit(values)
    encoded = model.transform(values)
    original = encoded.copy()
    assert encoded.dtype == np.complex64
    restored = model.inverse_transform(encoded)
    if sp.issparse(restored):
        rows, cols = np.nonzero(~np.isnan(values))
        np.testing.assert_array_equal(
            np.asarray(restored[rows, cols]).ravel(), values[rows, cols]
        )
        np.testing.assert_array_equal(encoded.data, original.data)
        assert restored.nnz == len(rows)
    else:
        np.testing.assert_array_equal(
            restored[~np.isnan(values)], values[~np.isnan(values)]
        )
        np.testing.assert_array_equal(encoded, original)


@pytest.mark.parametrize(
    "cls", [p.MissingAwareOneHotEncoder, p.MissingAwareSparseOneHotEncoder]
)
@pytest.mark.parametrize("imaginary", [1.0, np.nan, np.inf])
def test_complex_indicator_compatibility_cannot_hide_imaginary_data(cls, imaginary):
    model = cls(dtype=np.complex64).fit(_data())
    encoded = model.transform(_data())
    if sp.issparse(encoded):
        encoded.data[0] = complex(1.0, imaginary)
    else:
        encoded[0, 0] = complex(1.0, imaginary)
    with pytest.raises(ValueError, match=r"imaginary|infinity"):
        model.inverse_transform(encoded)
