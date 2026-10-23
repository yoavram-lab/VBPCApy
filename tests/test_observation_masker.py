"""Masks belong to original variables and must be consumed before expansion."""

import copy
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import config_context
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from vbpca_py import (
    VBPCA,
    AutoEncoder,
    MissingAwareStandardScaler,
    ObservationMasker,
)

SPARSE_TYPES = (
    sp.csr_matrix,
    sp.csc_matrix,
    sp.coo_matrix,
    sp.csr_array,
    sp.csc_array,
    sp.coo_array,
)


def _data():
    return np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [1.0, 3.0]])


@pytest.mark.parametrize("dtype", [bool, np.int32, np.float32, np.float64, object])
@pytest.mark.parametrize("method", ["separate", "combined"])
def test_dense_masks_own_output_preserve_observed_zeros_and_never_mutate(dtype, method):
    x = _data().astype(dtype)
    mask = np.ones(x.shape, dtype=bool)
    mask[1, 1] = False
    before = x.copy()
    x.flags.writeable = False
    mask.flags.writeable = False
    model = ObservationMasker()
    output = (
        model.fit_transform(x, mask=mask)
        if method == "combined"
        else model.fit(x, mask=mask).transform(x, mask=mask)
    )
    np.testing.assert_array_equal(output[mask], x[mask])
    assert np.isnan(output[1, 1])
    assert output[0, 0] == 0
    assert not np.shares_memory(output, x)
    np.testing.assert_array_equal(x, before)
    expected_dtype = (
        np.dtype(dtype) if dtype in {np.float32, np.float64, object} else float
    )
    assert output.dtype == expected_dtype
    assert set(vars(model)) == {"n_features_in_"}
    # Calling transform without a fresh mask never reuses fit's mask.
    np.testing.assert_array_equal(model.transform(x), x)


@pytest.mark.parametrize("dtype", ["U8", "S8", object])
def test_string_masks_insert_real_nan_without_conflating_literal_nan_labels(dtype):
    x = np.array([["nan", "blue"], ["green", "red"]], dtype=dtype)
    mask = np.array([[True, False], [True, True]])
    output = ObservationMasker().fit_transform(x, mask=mask)
    assert output.dtype == object
    assert np.isnan(output[0, 1])
    np.testing.assert_array_equal(output[mask], x[mask])
    assert output[0, 0] == x[0, 0]


def test_nan_stays_missing_and_only_observed_infinity_is_refused():
    x = np.array([[0.0, np.inf], [np.nan, 2.0], [1.0, -np.inf]])
    mask = np.array([[True, False], [True, True], [True, False]])
    output = ObservationMasker().fit_transform(x, mask=mask)
    np.testing.assert_array_equal(output, [[0, np.nan], [np.nan, 2], [1, np.nan]])
    with pytest.raises(ValueError, match="infinity"):
        ObservationMasker().fit(x)


@pytest.mark.parametrize("sparse_type", SPARSE_TYPES)
@pytest.mark.parametrize("mask_kind", ["none", "dense", "sparse"])
def test_sparse_masks_preserve_stored_and_requested_zeros_without_densification(
    sparse_type, mask_kind, monkeypatch
):
    x = sparse_type(
        (np.array([0.0, 2.0, np.nan, 4.0]), ([0, 0, 1, 2], [0, 2, 1, 2])), shape=(3, 4)
    )
    before = x.copy()
    selected = np.array([
        [True, True, False, False],
        [False, True, False, False],
        [False, False, True, False],
    ])
    mask = None
    if mask_kind == "dense":
        mask = selected
    elif mask_kind == "sparse":
        mask = sparse_type(selected)

    def forbid(*args, **kwargs):
        pytest.fail("sparse observations/masks must never be densified")

    for cls in SPARSE_TYPES:
        monkeypatch.setattr(cls, "toarray", forbid)
        monkeypatch.setattr(cls, "todense", forbid)
    output = ObservationMasker().fit_transform(x, mask=mask)
    assert isinstance(output, sp.csr_matrix)
    coo = output.tocoo()
    entries = dict(zip(zip(coo.row, coo.col, strict=True), coo.data, strict=True))
    expected = {(0, 0): 0.0, (0, 2): 2.0, (2, 2): 4.0}
    if mask_kind != "none":
        expected = {(0, 0): 0.0, (0, 1): 0.0, (2, 2): 4.0}
    assert entries == expected
    np.testing.assert_array_equal(x.data, before.data)
    if x.format == "coo":
        np.testing.assert_array_equal(x.row, before.row)
        np.testing.assert_array_equal(x.col, before.col)
    else:
        np.testing.assert_array_equal(x.indices, before.indices)
        np.testing.assert_array_equal(x.indptr, before.indptr)


@pytest.mark.parametrize("sparse_type", SPARSE_TYPES)
def test_boolean_mask_duplicates_are_observation_union_not_replicate_counts(
    sparse_type,
):
    x = sparse_type((np.array([2.0, -2.0, 7.0]), ([0, 0, 1], [0, 0, 1])), shape=(2, 3))
    mask = sparse_type(
        (np.array([False, True, True, False]), ([0, 0, 0, 1], [0, 0, 2, 1])),
        shape=x.shape,
    )
    output = ObservationMasker().fit_transform(x, mask=mask)
    assert output.nnz == 2
    np.testing.assert_array_equal(output.indices, [0, 2])
    np.testing.assert_array_equal(output.data, [0, 0])


@pytest.mark.parametrize("sparse_type", SPARSE_TYPES)
def test_raw_duplicate_infinity_is_refused_before_aggregation_when_observed(
    sparse_type,
):
    raw = sp.coo_matrix(([np.inf, -np.inf, 1.0], ([0, 0, 1], [0, 0, 1])), shape=(2, 2))
    if sparse_type in {sp.csr_matrix, sp.csr_array} or sparse_type in {
        sp.csc_matrix,
        sp.csc_array,
    }:
        x = sparse_type(([np.inf, -np.inf, 1.0], [0, 0, 1], [0, 2, 3]), shape=(2, 2))
    else:
        x = sparse_type(raw)
    with pytest.raises(ValueError, match="infinity"):
        ObservationMasker().fit(x)
    # Exclusion is meaningful for sparse input just as for dense input.
    selected = sparse_type(np.array([[False, True], [False, True]]))
    output = ObservationMasker().fit_transform(x, mask=selected)
    np.testing.assert_array_equal(output.data, [0, 1])
    assert output.nnz == 2


@pytest.mark.parametrize("sparse_type", SPARSE_TYPES)
def test_sparse_readonly_buffers_and_missing_entries_remain_unchanged(sparse_type):
    x = sparse_type(([0.0, np.nan, 2.0], ([0, 1, 2], [0, 1, 1])), shape=(3, 2))
    x.data.flags.writeable = False
    if x.format != "coo":
        x.indices.flags.writeable = False
        x.indptr.flags.writeable = False
    output = ObservationMasker().fit_transform(x)
    np.testing.assert_array_equal(output.data, [0, 2])
    np.testing.assert_array_equal(x.data, [0, np.nan, 2])


@pytest.mark.parametrize("method", ["fit", "transform", "fit_transform"])
@pytest.mark.parametrize("bad", [np.ones((4, 2)), [[1, 1]] * 4, "all", 1])
def test_explicit_masks_must_be_boolean(method, bad):
    model = ObservationMasker().fit(_data())
    with pytest.raises(TypeError, match="Boolean dtype"):
        getattr(model, method)(_data(), mask=bad)


@pytest.mark.parametrize("mask", [np.ones(4, dtype=bool), np.ones((3, 2), dtype=bool)])
def test_mask_shape_errors_are_explicit(mask):
    with pytest.raises(ValueError, match=r"mask shape.*data shape"):
        ObservationMasker().fit_transform(_data(), mask=mask)


@pytest.mark.parametrize("sparse_type", SPARSE_TYPES)
def test_sparse_mask_type_shape_and_dense_data_compatibility(sparse_type):
    x = sparse_type(_data())
    with pytest.raises(TypeError, match="Boolean dtype"):
        ObservationMasker().fit(x, mask=sparse_type(np.ones(x.shape)))
    with pytest.raises(ValueError, match="mask shape"):
        ObservationMasker().fit(x, mask=sparse_type(np.ones((1, 2), dtype=bool)))
    with pytest.raises(ValueError, match="mask must be dense"):
        ObservationMasker().fit(_data(), mask=sparse_type(np.ones(x.shape, dtype=bool)))


@pytest.mark.parametrize("target", ["data", "mask"])
@pytest.mark.parametrize("sparse_type", SPARSE_TYPES)
def test_malformed_sparse_buffers_are_refused_before_conversion(
    target, sparse_type, monkeypatch
):
    raw = sparse_type(np.ones((3, 2), dtype=bool if target == "mask" else float))
    if raw.format == "coo":
        raw.row[0] = -1
    else:
        raw.indices[0] = -1
    x = raw if target == "data" else sp.csr_matrix(_data()[:3])
    mask = raw if target == "mask" else None

    def forbid(*args, **kwargs):
        pytest.fail("malformed sparse buffers reached conversion")

    monkeypatch.setattr(sparse_type, "tocsr", forbid)
    with pytest.raises(ValueError, match="Invalid sparse"):
        ObservationMasker().fit(x, mask=mask)


def test_failed_refit_clears_state_and_successful_copy_pickle_clone_work():
    model = ObservationMasker()
    assert vars(model) == {}
    with pytest.raises(NotFittedError):
        model.transform(_data())
    model.fit(_data())
    for restored in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        np.testing.assert_array_equal(restored.transform(_data()), _data())
    assert vars(clone(model)) == {}
    with pytest.raises(ValueError, match="mask shape"):
        model.fit(_data(), mask=np.ones((2, 2), dtype=bool))
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert vars(model) == {}


@pytest.mark.parametrize("bad", [np.ones(3), np.empty((0, 2)), np.empty((2, 0))])
def test_nonempty_two_dimensional_data_is_required(bad):
    with pytest.raises(ValueError):
        ObservationMasker().fit(bad)


def test_named_feature_order_and_duplicates_are_checked():
    pd = pytest.importorskip("pandas")
    x = pd.DataFrame(_data(), columns=["a", "b"])
    model = ObservationMasker().fit(x)
    np.testing.assert_array_equal(model.feature_names_in_, ["a", "b"])
    with pytest.raises(ValueError, match="feature names"):
        model.transform(x[["b", "a"]])
    with pytest.raises(
        ValueError, match=r"duplicate feature names|unique column names"
    ):
        model.fit(x.rename(columns={"b": "a"}))
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def _pipeline():
    return Pipeline([
        ("observe", ObservationMasker()),
        (
            "encode",
            AutoEncoder(column_types=["categorical", "continuous"], binary="both"),
        ),
        (
            "model",
            VBPCA(
                0,
                maxiters=2,
                niter_broadprior=0,
                verbose=0,
                num_cpu=1,
                runtime_tuning="off",
                random_state=19,
            ),
        ),
    ])


def _mixed_data(*, strings=False):
    raw = np.column_stack([np.tile([0.0, 1.0, 2.0], 8), np.arange(24.0) + 100])
    if strings:
        raw = raw.astype(object)
        raw[:, 0] = np.tile(["west", "east", "north"], 8)
    mask = np.ones(raw.shape, dtype=bool)
    mask[np.arange(24) % 4 == 0, 0] = False
    mask[np.arange(24) % 4 == 1, 1] = False
    return raw, mask, np.where(mask, raw, 999999.0)


@pytest.mark.parametrize("strings", [False, True])
def test_cv_routes_train_and_test_masks_before_expansion_without_leakage(strings):
    raw, mask, placeholders = _mixed_data(strings=strings)
    with config_context(enable_metadata_routing=True):
        result = cross_validate(
            _pipeline(),
            placeholders,
            cv=KFold(3),
            params={"observation_mask": mask},
            return_estimator=True,
            error_score="raise",
        )
        for fitted, (train, test), score in zip(
            result["estimator"], KFold(3).split(raw), result["test_score"], strict=True
        ):
            schema = fitted.named_steps["encode"].encoding_schema_
            assert len(schema.feature_groups) == 4  # original mask width was two
            assert schema.variables[1].offset[0] == pytest.approx(
                raw[train, 1][mask[train, 1]].mean()
            )
            reference = _pipeline().fit(np.where(mask[train], raw[train], np.nan))
            expected = reference.score(np.where(mask[test], raw[test], np.nan))
            assert score == pytest.approx(expected)
            assert fitted.transform(
                placeholders[test], observation_mask=mask[test]
            ).shape == (8, 0)
            # A new mask is used for inference; the training mask was not cached.
            np.testing.assert_array_equal(
                fitted.named_steps["observe"].transform(raw[test]), raw[test]
            )


def test_gridsearch_routes_masks_and_scores_against_canonical_nan_reference():
    raw, mask, placeholders = _mixed_data()
    grid = {"model__n_components": [0, 1]}
    with config_context(enable_metadata_routing=True):
        actual = GridSearchCV(_pipeline(), grid, cv=3, error_score="raise").fit(
            placeholders, observation_mask=mask
        )
        reference = GridSearchCV(_pipeline(), grid, cv=3, error_score="raise").fit(
            np.where(mask, raw, np.nan)
        )
        np.testing.assert_allclose(
            actual.cv_results_["mean_test_score"],
            reference.cv_results_["mean_test_score"],
        )
        assert actual.best_params_ == reference.best_params_
        assert actual.score(placeholders, observation_mask=mask) == pytest.approx(
            reference.score(np.where(mask, raw, np.nan))
        )


def test_explicit_mask_routing_requires_opt_in_but_canonical_nan_pipeline_does_not():
    raw, mask, placeholders = _mixed_data()
    with config_context(enable_metadata_routing=False):
        model = _pipeline().fit(np.where(mask, raw, np.nan))
        assert np.isfinite(model.score(np.where(mask, raw, np.nan)))
        with pytest.raises(ValueError, match="does not accept"):
            _pipeline().fit(placeholders, observation_mask=mask)
        # Traditional prefixed fit parameters remain usable for training only.
        prefixed = _pipeline().fit(placeholders, observe__mask=mask)
        assert (
            prefixed.named_steps["encode"].encoding_schema_
            == model.named_steps["encode"].encoding_schema_
        )


def test_sparse_pipeline_cv_retains_observed_zeros_and_slices_sparse_masks():
    raw = np.arange(48.0).reshape(24, 2)
    raw[::3] = 0.0
    data = sp.csr_matrix(raw)  # zeros intentionally absent from this input
    mask = sp.csr_matrix(np.ones(raw.shape, dtype=bool))
    pipe = Pipeline([
        ("observe", ObservationMasker()),
        ("scale", MissingAwareStandardScaler()),
        (
            "model",
            VBPCA(
                0,
                maxiters=2,
                niter_broadprior=0,
                verbose=0,
                num_cpu=1,
                runtime_tuning="off",
                random_state=19,
            ),
        ),
    ])
    with config_context(enable_metadata_routing=True):
        actual = cross_validate(
            pipe,
            data,
            cv=3,
            params={"observation_mask": mask},
            return_estimator=True,
            error_score="raise",
        )
        reference = cross_validate(
            pipe, raw, cv=3, return_estimator=True, error_score="raise"
        )
    np.testing.assert_allclose(actual["test_score"], reference["test_score"])
    for fitted, (train, test) in zip(
        actual["estimator"], KFold(3).split(raw), strict=True
    ):
        np.testing.assert_allclose(
            fitted.named_steps["scale"].mean_, raw[train].mean(axis=0)
        )
        observed = fitted.named_steps["observe"].transform(data[test], mask=mask[test])
        assert observed.nnz == raw[test].size


def test_all_applicable_generic_sklearn_checks_pass_without_exceptions():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from sklearn.utils.estimator_checks import check_estimator\n"
                "from vbpca_py import ObservationMasker\n"
                "results = check_estimator(ObservationMasker())\n"
                "unpassed = [r for r in results if r['status'] != 'passed']\n"
                "assert results and not unpassed, unpassed\n"
            ),
        ],
        env={**os.environ, "SCIPY_ARRAY_API": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("sparse", [False, True])
def test_an_empty_observation_set_is_represented_without_retaining_old_masks(sparse):
    x = sp.csr_matrix(_data()) if sparse else _data()
    mask = np.zeros(x.shape, dtype=bool)
    output = ObservationMasker().fit_transform(x, mask=mask)
    assert output.nnz == 0 if sparse else np.isnan(output).all()


@pytest.mark.parametrize("target", ["data", "mask"])
def test_one_dimensional_sparse_buffers_are_refused_before_conversion(
    target, monkeypatch
):
    bad = sp.coo_array(([True], ([0],)), shape=(3,))
    bad.coords = (np.array([-1]),)

    def forbid(*args, **kwargs):
        pytest.fail("non-2D sparse buffers reached conversion")

    monkeypatch.setattr(sp.coo_array, "tocsr", forbid)
    x, mask = (bad, None) if target == "data" else (sp.csr_matrix(_data()), bad)
    with pytest.raises(ValueError, match="Expected 2D sparse input"):
        ObservationMasker().fit(x, mask=mask)


def test_documented_masked_pipeline_example_executes():
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts/example_masked_pipeline.py"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "new-row projection passed" in completed.stdout


@pytest.mark.parametrize("shape", [(513, 257), (2, 70_000)])
def test_dense_validation_scratch_is_bounded_and_checks_distant_observations(
    shape, monkeypatch
):
    data = np.ones(shape)
    data[-1, -1] = np.inf
    mask = np.ones(shape, dtype=bool)
    original = np.isinf

    def bounded(values, *args, **kwargs):
        assert np.size(values) <= 65_536, "validation allocated full-table scratch"
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np, "isinf", bounded)
    with pytest.raises(ValueError, match="infinity"):
        ObservationMasker().fit(data, mask=mask)
    mask[-1, -1] = False
    result = ObservationMasker().fit_transform(data, mask=mask)
    assert result.shape == shape
    assert np.isnan(result[-1, -1])
    assert result[0, 0] == 1
