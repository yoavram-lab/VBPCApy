"""Initialization values belong to the caller until a fit takes its own copy."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone

from vbpca_py import _pca_full as solver
from vbpca_py._pca_full import pca_full
from vbpca_py.legacy import VBPCA


def _warm_start(p=4, n=7, q=2):
    rng = np.random.default_rng(82)
    return {
        "A": rng.normal(size=(p, q)),
        "S": rng.normal(size=(q, n)),
        "Mu": rng.normal(size=(p, 1)),
        "Muv": np.full((p, 1), 0.2),
        "Av": [np.eye(q) * 0.1 for _ in range(p)],
        "Sv": [np.eye(q) * 0.1 for _ in range(n)],
        "V": 1.0,
    }


def _arrays(payload):
    for value in payload.values():
        if isinstance(value, np.ndarray):
            yield value
        elif isinstance(value, list):
            yield from value


def _assert_payload_unchanged(payload, before):
    assert payload.keys() == before.keys()
    for name in payload:
        if isinstance(payload[name], list):
            assert len(payload[name]) == len(before[name])
            for actual, expected in zip(payload[name], before[name], strict=True):
                np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_array_equal(payload[name], before[name])


def _fit(data, payload, algorithm="vb"):
    return pca_full(
        data,
        2,
        init=payload,
        algorithm=algorithm,
        maxiters=2,
        niter_broadprior=0,
        random_state=83,
        runtime_tuning="off",
        num_cpu=1,
        return_diagnostics=False,
        compat_mode="modern",
    )


@pytest.mark.parametrize("algorithm", ["ppca", "map", "vb"])
@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("representation", ["dict", "mappingproxy", "struct"])
def test_warm_start_ownership_and_repeated_fits(algorithm, readonly, representation):
    data = np.random.default_rng(84).normal(size=(4, 7))
    payload = _warm_start()
    before = deepcopy(payload)
    if readonly:
        for array in _arrays(payload):
            array.setflags(write=False)
    if representation == "mappingproxy":
        supplied = MappingProxyType(payload)
    elif representation == "struct":
        supplied = SimpleNamespace(_fieldnames=list(payload), **payload)
    else:
        supplied = payload
    first = _fit(data, supplied, algorithm)
    _assert_payload_unchanged(payload, before)
    second = _fit(data, supplied, algorithm)
    _assert_payload_unchanged(payload, before)
    for name in ("A", "S", "Mu", "Muv", "V", "Av", "Sv"):
        np.testing.assert_array_equal(first[name], second[name])
    outputs = [first[name] for name in ("A", "S", "Mu", "Muv")]
    outputs.extend(first["Av"])
    outputs.extend(first["Sv"])
    for output in outputs:
        for original in _arrays(payload):
            assert not np.shares_memory(output, original)
    if readonly:
        assert all(not array.flags.writeable for array in _arrays(payload))


@pytest.mark.parametrize("sparse_input", [False, True])
def test_empty_row_and_column_preparation_only_changes_owned_payload(sparse_input):
    data = np.random.default_rng(85).normal(size=(5, 8))
    mask = np.ones_like(data, dtype=bool)
    mask[1, :] = False
    mask[:, 2] = False
    if sparse_input:
        rows, cols = np.nonzero(mask)
        data = sparse.coo_matrix(
            (data[rows, cols], (rows, cols)), shape=data.shape
        ).tocsr()
    else:
        data = np.where(mask, data, np.nan)
    payload = _warm_start(5, 8)
    # Factor-only starts use default mean initialization after feature removal.
    del payload["Mu"], payload["Muv"]
    before = deepcopy(payload)
    result = _fit(data, payload)
    _assert_payload_unchanged(payload, before)
    assert result["A"].shape == (5, 2)
    assert result["S"].shape == (2, 8)


def test_failed_fit_does_not_partially_slice_caller_dictionary(monkeypatch):
    data = np.random.default_rng(86).normal(size=(5, 8))
    data[1, :] = np.nan
    payload = _warm_start(5, 8)
    before = deepcopy(payload)

    def fail_after_preparation(**kwargs):
        raise RuntimeError("failed after preparation")

    monkeypatch.setattr(solver, "_initialize_model", fail_after_preparation)
    with pytest.raises(RuntimeError, match="failed after preparation"):
        _fit(data, payload)
    _assert_payload_unchanged(payload, before)


def test_estimator_preserves_constructor_identity_and_caller_arrays():
    payload = _warm_start()
    before = deepcopy(payload)
    data = np.random.default_rng(87).normal(size=(4, 7))
    estimator = VBPCA(
        2,
        init=payload,
        maxiters=2,
        niter_broadprior=0,
        random_state=88,
        runtime_tuning="off",
        num_cpu=1,
        return_diagnostics=False,
        compat_mode="modern",
    )
    assert estimator.get_params()["init"] is payload
    copied = clone(estimator)
    assert copied.get_params()["init"] is not payload
    estimator.fit(data)
    assert estimator.get_params()["init"] is payload
    _assert_payload_unchanged(payload, before)
