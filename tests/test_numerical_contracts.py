"""Numerical dtype, ownership and serialization contracts for the 0.5 helpers."""

from __future__ import annotations

import copy
import pickle
from dataclasses import FrozenInstanceError, fields
from typing import get_type_hints

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py import VBPCA
from vbpca_py import _full_update as updates
from vbpca_py import _pca_full as solver
from vbpca_py._categorical_scores import categorical_scores
from vbpca_py._cost import CostParams, CovarianceInfo, ObservationInfo
from vbpca_py._monitoring import InitResult, InitShapes
from vbpca_py._noise_statistics import NoiseComplement, prepare_noise_complement
from vbpca_py._rms import RmsConfig
from vbpca_py._runtime_policy import RuntimePolicyConfig, RuntimeThreadConfig
from vbpca_py._types import Float64Array, IndexArray
from vbpca_py.encoding import EncodedVariable, EncodingSchema
from vbpca_py.errpca_pt import errpca_pt
from vbpca_py.preprocessing import MissingAwareOneHotEncoder, _sparse_col_counts


def _settings():
    return {
        "maxiters": 3,
        "niter_broadprior": 0,
        "verbose": 0,
        "num_cpu": 1,
        "runtime_tuning": "off",
        "random_state": 318,
        "compat_mode": "modern",
    }


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("sparse", [False, True])
def test_preparation_owns_normalized_working_data(dtype, sparse):
    source = np.arange(96, dtype=dtype).reshape(8, 12)[::2, ::2]
    source.flags.writeable = False
    x = sp.csr_matrix(source) if sparse else source
    mask = (
        sp.csr_matrix(np.ones(x.shape, dtype=bool))
        if sparse
        else np.ones(x.shape, dtype=np.int16)
    )
    original = x.copy()
    state = solver._solver_options_from_raw(_settings())
    prepared = solver._prepare_problem(x, state, mask_override=mask)
    data = prepared.x_data
    values = data.data if sp.issparse(data) else data
    assert values.dtype == (dtype if sparse else np.float64)
    assert values.flags.writeable
    np.testing.assert_array_equal(
        x.toarray() if sparse else x, original.toarray() if sparse else original
    )
    if sparse:
        assert not np.shares_memory(data.data, x.data)
    else:
        assert not np.shares_memory(data, source)
        assert prepared.mask.dtype == np.bool_
        assert prepared.mask.flags.writeable
    assert prepared.ix_obs.dtype.kind == "i"
    assert prepared.jx_obs.dtype.kind == "i"
    assert get_type_hints(solver.ModelState)["a"] == Float64Array
    assert get_type_hints(solver.PreparedProblem)["ix_obs"] == IndexArray


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_public_fit_accepts_raw_dtypes_and_retains_inputs(dtype):
    x = np.arange(32, dtype=dtype).reshape(4, 8)
    before = x.copy()
    model = VBPCA(1, **_settings()).fit(x, mask=np.ones(x.shape, dtype=np.int16))
    assert model.components_.dtype == np.float64
    assert model.scores_.dtype == np.float64
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, object])
def test_configured_encoder_dtype_stays_configurable(dtype):
    x = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
    encoder = MissingAwareOneHotEncoder(dtype=dtype)
    with np.errstate(all="raise"):
        encoded = encoder.fit_transform(x)
    assert encoded.dtype == dtype
    restored = encoder.inverse_transform(encoded)
    np.testing.assert_array_equal(restored, x)


def test_native_sparse_residual_result_has_heterogeneous_buffers():
    data = np.array([0.0, 2.0, 3.0])
    indices = np.array([0, 2, 1], dtype=np.int32)
    indptr = np.array([0, 2, 3], dtype=np.int32)
    result = errpca_pt(data, indices, indptr, np.ones((2, 1)), np.ones((1, 3)), 1)
    assert result["data"].dtype == np.float64
    assert result["indices"].dtype == np.int32
    assert result["indptr"].dtype == np.int32
    assert result["shape"] == (2, 3)
    for key, original in (("data", data), ("indices", indices), ("indptr", indptr)):
        assert not np.shares_memory(result[key], original)
    np.testing.assert_array_equal(result["data"], [-1.0, 1.0, 2.0])


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_sparse_count_storage_preserves_signed_index_width(index_dtype):
    x = sp.csc_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]]))
    x.indices = x.indices.astype(index_dtype)
    x.indptr = x.indptr.astype(index_dtype)
    counts = _sparse_col_counts(x)
    assert counts.dtype == index_dtype
    np.testing.assert_array_equal(counts, [1, 1, 2])


@pytest.mark.parametrize(
    ("dtype", "count_dtype"),
    [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.longdouble, np.longdouble),
        (np.int32, np.int64),
        (np.uint32, np.uint64),
    ],
)
def test_sparse_preparation_retains_mask_sum_dtype(dtype, count_dtype):
    x = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 4]], dtype=dtype))
    state = solver._solver_options_from_raw(_settings())
    prepared = solver._prepare_problem(x, state)
    assert prepared.n_obs_row.dtype == count_dtype
    np.testing.assert_array_equal(prepared.n_obs_row, [2, 2])


def test_categorical_labels_can_be_strings():
    prediction = np.array([[0.8, 0.1], [0.2, 0.9]])
    truth = np.eye(2)
    expected = categorical_scores(prediction, truth, [0, 0])
    assert categorical_scores(prediction, truth, ["variable", "variable"]) == expected


def test_plan_owns_coordinates_and_preserves_readonly_flags_on_roundtrip():
    rows = np.array([0, 1], dtype=np.int64)
    columns = np.array([2, 3], dtype=np.int32)
    plan = NoiseComplement((2, 5), 8, rows, columns)
    rows[:] = 1
    columns[:] = 0
    np.testing.assert_array_equal(plan.rows, [0, 1])
    np.testing.assert_array_equal(plan.columns, [2, 3])
    assert plan.rows.dtype == np.int32
    assert not plan.rows.flags.writeable
    assert not plan.columns.flags.writeable
    for restored in (copy.deepcopy(plan), pickle.loads(pickle.dumps(plan))):
        assert not restored.rows.flags.writeable
        assert not restored.columns.flags.writeable
        assert not np.shares_memory(plan.rows, restored.rows)
        np.testing.assert_array_equal(restored.rows, plan.rows)
        with pytest.raises(ValueError, match="read-only"):
            restored.rows[0] = 1
    with pytest.raises(FrozenInstanceError):
        plan.observed_count = 7
    with pytest.raises(AttributeError):
        _ = plan.__dict__


@pytest.mark.parametrize("dimension_type", [int, np.int64, np.uint64])
def test_plan_normalizes_dimensions_before_large_coordinate_arithmetic(dimension_type):
    size = 2**31 - 1
    plan = NoiseComplement(
        (dimension_type(size), dimension_type(size)),
        size * size - 2,
        [size - 1, size - 1],
        [0, 1],
    )
    assert plan.shape == (size, size)
    assert all(type(dimension) is int for dimension in plan.shape)
    np.testing.assert_array_equal(plan.columns, [0, 1])


@pytest.mark.parametrize(
    ("rows", "columns", "observed", "message"),
    [
        ([0.5], [1], 9, "signed integer"),
        ([0], [5], 9, "bounded"),
        ([0, 0], [1, 1], 8, "unique"),
        ([0], [1, 2], 9, "lengths"),
        ([0], [1], 8, "count"),
    ],
)
def test_plan_rejects_invalid_coordinate_contracts(rows, columns, observed, message):
    with pytest.raises(ValueError, match=message):
        NoiseComplement((2, 5), observed, rows, columns)


def test_factory_plan_and_covariance_views_do_not_add_payload_copies():
    observed = np.ones((10, 10), dtype=bool)
    observed[0, 1] = False
    rows, columns = np.nonzero(observed)
    plan = prepare_noise_complement(observed.shape, rows, columns)
    assert plan is not None
    assert plan.rows.flags.owndata
    assert not plan.rows.flags.writeable
    rows[:] = 0
    np.testing.assert_array_equal(plan.rows, [0])
    np.testing.assert_array_equal(plan.columns, [1])
    source = np.arange(48, dtype=np.float64).reshape(3, 4, 4)[:, ::2, ::2]
    source.flags.writeable = False
    stacked = updates._covariances_stack_cached(source, {}, "score")
    assert np.shares_memory(stacked, source)
    assert not stacked.flags.writeable
    blocks = updates._covariances_list(stacked)
    assert all(np.shares_memory(block, source) for block in blocks)
    cache = {}
    packed = updates._covariances_stack_cached(blocks, cache, "score")
    assert packed.dtype == np.float64
    assert not np.shares_memory(packed, source)
    assert updates._covariances_stack_cached(blocks, cache, "score") is packed
    with pytest.raises(ValueError, match="3-D"):
        updates._covariances_stack_cached(np.ones((2, 2)), {}, "score")


def test_encoding_metadata_snapshots_sequence_membership():
    columns, categories, offset = [0, 1], ["a", "b"], [0.0, 0.0]
    variable = EncodedVariable("categorical", columns, categories, offset=offset)
    variables = [variable]
    schema = EncodingSchema(variables)
    columns.clear()
    categories.clear()
    offset.clear()
    variables.clear()
    assert variable.columns == (0, 1)
    assert variable.categories == ("a", "b")
    assert variable.offset == (0.0, 0.0)
    schema.validate(2)
    for restored in (copy.deepcopy(schema), pickle.loads(pickle.dumps(schema))):
        assert restored == schema
        assert not hasattr(restored, "__dict__")


@pytest.mark.parametrize(
    "record",
    [
        InitShapes(3, 4, 1, 4),
        RmsConfig(),
        RuntimePolicyConfig(1, "off"),
        RuntimeThreadConfig(1, 1, 1, 1, 1, 1),
    ],
)
def test_immutable_scalar_records_are_slotted_serializable_and_frozen(record):
    with pytest.raises(AttributeError):
        _ = record.__dict__
    with pytest.raises(FrozenInstanceError):
        setattr(record, fields(record)[0].name, None)
    assert pickle.loads(pickle.dumps(record)) == record
    assert copy.deepcopy(record) == record


@pytest.mark.parametrize(
    "record",
    [
        solver.PreparedProblem,
        solver.ModelState,
        solver.TrainingState,
        updates.ScoreState,
        updates.NoiseState,
        updates.InitContext,
        InitResult,
        CostParams,
        CovarianceInfo,
        ObservationInfo,
    ],
)
def test_known_working_record_annotations_resolve(record):
    hints = get_type_hints(record)
    assert hints
    assert "__dict__" not in record.__slots__


@pytest.mark.parametrize("shape", [(1.5, 2), (True, 2), (-1, 2), (2,), (2**31, 1)])
def test_plan_rejects_invalid_shape_metadata(shape):
    with pytest.raises(ValueError, match="shape"):
        NoiseComplement(
            shape, 0, np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        )


@pytest.mark.parametrize("count", [8.0, True])
def test_plan_rejects_noninteger_count_metadata(count):
    with pytest.raises(TypeError, match="count"):
        NoiseComplement((2, 5), count, [0, 1], [2, 3])
