"""Initialization axes remain explicit across removal, sharing and MAT loading."""

from copy import deepcopy
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.io import savemat

from vbpca_py._monitoring import InitShapes, init_params
from vbpca_py._pca_full import pca_full
from vbpca_py._remove_empty import remove_empty_entries


def _payload(p=5, n=8, q=2, layout="stack"):
    rng = np.random.default_rng(327)
    base = np.eye(q) * 0.4 + np.ones((q, q)) * 0.1
    av = np.stack([base * (i + 1) for i in range(p)])
    sv = np.stack([base * (i + 2) for i in range(n)])
    if layout == "list":
        av, sv = list(av), list(sv)
    elif layout == "diagonal":
        av = np.diagonal(av, axis1=1, axis2=2).copy()
        sv = np.diagonal(sv, axis1=1, axis2=2).T.copy()
    return {
        "A": rng.normal(size=(p, q)),
        "S": rng.normal(size=(q, n)),
        "Mu": np.arange(p, dtype=float)[:, None] / 10,
        "Muv": np.full((p, 1), 0.2),
        "Av": av,
        "Sv": sv,
        "V": 1.0,
    }


def _slice_payload(payload, rows, cols, layout):
    result = deepcopy(payload)
    for key in ("A", "Mu", "Muv"):
        result[key] = result[key][rows]
    result["S"] = result["S"][:, cols]
    if layout == "list":
        result["Av"] = [result["Av"][i] for i in rows]
        result["Sv"] = [result["Sv"][i] for i in cols]
    else:
        result["Av"] = result["Av"][rows]
        result["Sv"] = (
            result["Sv"][:, cols] if layout == "diagonal" else result["Sv"][cols]
        )
    return result


@pytest.mark.parametrize("layout", ["diagonal", "stack", "list"])
def test_removal_slices_all_initialization_axes(layout):
    data = np.ones((5, 8))
    data[1, :] = np.nan
    data[:, 2] = np.nan
    payload = _payload(layout=layout)
    expected = _slice_payload(payload, [0, 2, 3, 4], [0, 1, 3, 4, 5, 6, 7], layout)
    _, _, _, _, actual = remove_empty_entries(data, None, payload)
    for key in ("A", "Av", "S", "Sv", "Mu", "Muv"):
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("layout", ["diagonal", "stack", "list"])
@pytest.mark.parametrize("n", [2, 6])
def test_per_sample_covariances_choose_first_actual_sample_for_each_pattern(layout, n):
    payload = _payload(n=n, layout=layout)
    target = np.array([0, 0]) if n == 2 else np.array([1, 1, 0, 1, 0, 0])
    m = len(np.unique(target))
    result = init_params(payload, InitShapes(5, n, 2, m), target)
    expected = payload["Sv"]
    for pattern, block in enumerate(result.sv):
        sample = np.flatnonzero(target == pattern)[0]
        reference = (
            np.diag(expected[:, sample]) if layout == "diagonal" else expected[sample]
        )
        np.testing.assert_array_equal(block, reference)


@pytest.mark.parametrize("layout", ["diagonal", "stack", "list"])
def test_source_and_destination_pattern_indices_have_distinct_roles(layout):
    payload = _payload(n=3, layout=layout)
    payload["S"] = np.zeros((2, 6))
    payload["Isv"] = np.array([2, 0, 2, 1, 0, 1])
    target = np.array([1, 0, 1, 0, 0, 0])
    result = init_params(payload, InitShapes(5, 6, 2, 2), target)
    for actual, source_block in zip(result.sv, (0, 2), strict=True):
        expected = (
            np.diag(payload["Sv"][:, source_block])
            if layout == "diagonal"
            else payload["Sv"][source_block]
        )
        np.testing.assert_array_equal(actual, expected)


def _fit(data, init):
    return pca_full(
        data,
        2,
        init=init,
        maxiters=1,
        niter_broadprior=0,
        runtime_tuning="off",
        num_cpu=1,
        random_state=327,
        compat_mode="modern",
        return_diagnostics=False,
    )


def _supplied_payload(payload, representation, layout, tmp_path):
    if representation == "mapping":
        return payload
    if representation == "proxy":
        return MappingProxyType(payload)
    if representation == "struct":
        return SimpleNamespace(_fieldnames=list(payload), **payload)
    filename = tmp_path / "initialization.mat"
    saved = dict(payload)
    if layout == "list":
        for key in ("Av", "Sv"):
            cells = np.empty((1, len(payload[key])), dtype=object)
            for i, block in enumerate(payload[key]):
                cells[0, i] = block
            saved[key] = cells
    savemat(filename, saved if representation == "mat_flat" else {"init": saved})
    return str(filename)


@pytest.mark.parametrize("layout", ["diagonal", "stack", "list"])
@pytest.mark.parametrize(
    "representation", ["mapping", "proxy", "struct", "mat_flat", "mat_struct"]
)
@pytest.mark.parametrize("sparse_input", [False, True])
def test_complete_warm_start_matches_independently_prepared_fit(
    tmp_path, layout, representation, sparse_input
):
    data = np.random.default_rng(328).normal(size=(5, 8))
    rows = np.array([0, 2, 3, 4])
    cols = np.array([0, 1, 3, 4, 5, 6, 7])
    payload = _payload(layout=layout)
    expected = _fit(
        data[np.ix_(rows, cols)], _slice_payload(payload, rows, cols, layout)
    )
    before = deepcopy(payload)
    for value in payload.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
        elif isinstance(value, list):
            for block in value:
                block.setflags(write=False)
    supplied = _supplied_payload(payload, representation, layout, tmp_path)
    mask = np.zeros(data.shape, dtype=bool)
    mask[np.ix_(rows, cols)] = True
    if sparse_input:
        ri, ci = np.nonzero(mask)
        values = sp.csr_matrix((data[ri, ci], (ri, ci)), shape=data.shape)
    else:
        values = np.where(mask, data, np.nan)
    actual = _fit(values, supplied)
    for key in ("A", "Mu", "Muv"):
        np.testing.assert_allclose(
            actual[key][rows], expected[key], rtol=1e-11, atol=1e-12
        )
    np.testing.assert_allclose(
        actual["S"][:, cols], expected["S"], rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(actual["Av"])[rows], expected["Av"], rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(actual["Sv"])[cols], expected["Sv"], rtol=1e-11, atol=1e-12
    )
    for key in before:
        np.testing.assert_array_equal(payload[key], before[key])


@pytest.mark.parametrize("layout", ["diagonal", "stack", "list"])
@pytest.mark.parametrize("index_shape", ["flat", "row", "column"])
def test_removal_slices_source_index_without_slicing_its_covariance_bank(
    layout, index_shape
):
    data = np.ones((5, 8))
    data[1] = np.nan
    data[:, 2] = np.nan
    payload = _payload(layout=layout)
    payload["Sv"] = _payload(n=3, layout=layout)["Sv"]
    mapping = np.array([2, 0, 1, 0, 1, 2, 0, 2])
    payload["Isv"] = (
        mapping
        if index_shape == "flat"
        else mapping.reshape((1, -1) if index_shape == "row" else (-1, 1))
    )
    source_bank = deepcopy(payload["Sv"])
    _, _, _, columns, result = remove_empty_entries(data, None, payload)
    np.testing.assert_array_equal(result["Isv"], mapping[columns])
    np.testing.assert_array_equal(result["Sv"], source_bank)
    target = np.array([1, 1, 0, 0, 1, 0, 1])
    initialized = init_params(result, InitShapes(4, 7, 2, 2), target)
    for pattern, actual in enumerate(initialized.sv):
        first_sample = np.flatnonzero(target == pattern)[0]
        source = mapping[columns[first_sample]]
        expected = (
            np.diag(source_bank[:, source])
            if layout == "diagonal"
            else source_bank[source]
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("key", ["Av", "Sv"])
def test_bad_full_covariance_cross_axes_fail_clearly(key):
    payload = _payload()
    count = 5 if key == "Av" else 8
    payload[key] = np.ones((count, 2, 3))
    with pytest.raises(ValueError, match=key):
        init_params(payload, InitShapes(5, 8, 2, 8), None)


def test_diagonal_covariance_axes_remain_explicit_when_dimensions_coincide():
    av = np.array([[1.0, 0.2], [0.3, 2.0]])
    sv = np.array([[0.4, 0.6], [0.8, 1.0]])
    result = init_params({"Av": av, "Sv": sv}, InitShapes(2, 2, 2, 2), None)
    np.testing.assert_array_equal(result.av, [np.diag(av[0]), np.diag(av[1])])
    np.testing.assert_array_equal(result.sv, [np.diag(sv[:, 0]), np.diag(sv[:, 1])])


@pytest.mark.parametrize(
    "bad",
    [
        np.array([0, 1]),
        np.array([0, 1, 2, 3, 4, 8]),
        np.array([0, -1, 0, 1, 0, 1]),
        np.array([0, 0.5, 0, 1, 0, 1]),
        np.array([0, np.nan, 0, 1, 0, 1]),
        np.array([True, False, True, False, True, False]),
        np.zeros((2, 3), dtype=int),
    ],
)
def test_invalid_source_covariance_indices_fail_clearly(bad):
    payload = _payload(n=6)
    payload["Isv"] = bad
    with pytest.raises(ValueError, match="Isv"):
        init_params(payload, InitShapes(5, 6, 2, 6), None)


def test_integral_mat_style_source_indices_keep_bank_semantics():
    payload = _payload(n=2)
    payload["S"] = np.zeros((2, 6))
    payload["Isv"] = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    result = init_params(payload, InitShapes(5, 6, 2, 6), None)
    np.testing.assert_array_equal(
        result.sv, np.asarray(payload["Sv"])[[1, 0, 1, 0, 1, 0]]
    )


@pytest.mark.parametrize("layout", ["diagonal", "stack", "list"])
def test_one_component_mat_struct_preserves_matrix_axes(tmp_path, layout):
    payload = _payload(p=3, n=4, q=1, layout=layout)
    supplied = _supplied_payload(payload, "mat_struct", layout, tmp_path)
    expected = init_params(payload, InitShapes(3, 4, 1, 4), None)
    actual = init_params(supplied, InitShapes(3, 4, 1, 4), None)
    for key in ("a", "s", "mu", "muv", "av", "sv", "v"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))


def _score_blocks_by_sample(result, selected):
    bank = np.asarray(result["Sv"])
    mapping = result.get("Isv")
    return bank[selected] if mapping is None else bank[np.asarray(mapping)[selected]]


def test_fitted_pattern_bank_roundtrip_after_feature_and_sample_removal():
    data = np.random.default_rng(329).normal(size=(5, 8))
    data[0, ::2] = np.nan
    source = pca_full(
        data,
        2,
        maxiters=2,
        uniquesv=True,
        niter_broadprior=0,
        random_state=329,
        runtime_tuning="off",
        num_cpu=1,
        return_diagnostics=False,
        compat_mode="modern",
    )
    payload = {
        key: source[key] for key in ("A", "S", "Mu", "Muv", "V", "Av", "Sv", "Isv")
    }
    rows, columns = np.array([0, 2, 3, 4]), np.array([0, 1, 3, 4, 5, 6, 7])
    changed = data.copy()
    changed[1] = np.nan
    changed[:, 2] = np.nan
    prepared = {
        "A": payload["A"][rows],
        "S": payload["S"][:, columns],
        "Mu": payload["Mu"][rows],
        "Muv": payload["Muv"][rows],
        "V": payload["V"],
        "Av": np.asarray(payload["Av"])[rows],
        "Sv": payload["Sv"],
        "Isv": payload["Isv"][columns],
    }
    settings = {
        "maxiters": 1,
        "uniquesv": True,
        "niter_broadprior": 0,
        "random_state": 329,
        "runtime_tuning": "off",
        "num_cpu": 1,
        "return_diagnostics": False,
        "compat_mode": "modern",
    }
    expected = pca_full(changed[np.ix_(rows, columns)], 2, init=prepared, **settings)
    actual = pca_full(changed, 2, init=payload, **settings)
    for key in ("A", "Mu", "Muv"):
        np.testing.assert_allclose(
            actual[key][rows], expected[key], rtol=1e-11, atol=1e-12
        )
    np.testing.assert_allclose(
        actual["S"][:, columns], expected["S"], rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        _score_blocks_by_sample(actual, columns),
        _score_blocks_by_sample(expected, np.arange(7)),
        rtol=1e-11,
        atol=1e-12,
    )


@pytest.mark.parametrize("layout", ["diagonal", "stack", "list"])
def test_zero_rank_initialization_keeps_factor_axes_after_removal(layout):
    data = np.ones((5, 8))
    data[1] = np.nan
    data[:, 2] = np.nan
    payload = _payload(q=0, layout=layout)
    _, _, _, _, prepared = remove_empty_entries(data, None, payload)
    result = init_params(prepared, InitShapes(4, 7, 0, 1), np.zeros(7, dtype=int))
    assert result.a.shape == (4, 0)
    assert result.s.shape == (0, 7)
    assert all(block.shape == (0, 0) for block in result.av)
    assert all(block.shape == (0, 0) for block in result.sv)


def test_normalization_retains_existing_struct_learning_curve_policy():
    payload = _payload()
    payload["lc"] = SimpleNamespace(rms=[1.0, 0.9])
    supplied = SimpleNamespace(_fieldnames=list(payload), **payload)
    result = _fit(np.ones((5, 8)), supplied)
    assert len(result["lc"]["rms"]) == 1
