"""Storage, ownership and fixed-policy guards for dense mask handling."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import vbpca_py._full_update as updates
import vbpca_py._rms as rms_module
from vbpca_py._pca_full import pca_full
from vbpca_py._rms import RmsConfig, compute_rms
from vbpca_py._sparsity import _as_dense_mask
from vbpca_py.model_selection import SelectionConfig, select_n_components

_OFF = dict.fromkeys(
    ("angle", "earlystop", "rms_plateau", "cost", "composite", "slowing_down"), False
)


def _layout(array: np.ndarray, layout: str) -> np.ndarray:
    if layout == "strided":
        backing = np.empty(tuple(2 * axis for axis in array.shape), dtype=array.dtype)
        view = backing[::2, ::2]
        view[:] = array
        return view
    if layout == "reversed":
        return array[::-1, ::-1]
    return np.array(array, order=layout)


@pytest.mark.parametrize("layout", ["C", "F", "strided", "reversed"])
@pytest.mark.parametrize("dtype", [bool, np.int8, np.float32, np.float64])
def test_mask_storage_and_rms_preserve_numeric_semantics(layout, dtype) -> None:
    rng = np.random.default_rng(292)
    x = _layout(rng.normal(size=(7, 9)), layout)
    x[0, 0] = 0
    a = rng.normal(size=(7, 3))
    s = rng.normal(size=(3, 9))
    values = rng.random(x.shape) > 0.2
    mask = _layout(values.astype(dtype), layout)
    if dtype is not bool:
        # The low-level RMS helper historically supports numeric weights.
        mask[1, 1] = 2
        mask[2, 2] = -1
    expected_mask = np.asarray(mask, dtype=float)
    expected_err = (x - a @ s) * expected_mask
    expected_rms = np.sqrt(np.sum(expected_err**2) / np.count_nonzero(expected_mask))
    snapshots = [array.copy() for array in (x, mask, a, s)]
    for array in (x, mask, a, s):
        array.flags.writeable = False
    normalized = _as_dense_mask(mask)
    if dtype is bool:
        assert normalized.dtype == bool
        assert np.shares_memory(normalized, mask)
        assert normalized.strides == mask.strides
    else:
        assert normalized.dtype == np.float64
    actual_rms, actual_err = compute_rms(x, a, s, mask, RmsConfig())
    assert actual_rms == expected_rms
    assert_array_equal(actual_err, expected_err)
    assert not any(np.shares_memory(actual_err, array) for array in (x, mask, a, s))
    for array, snapshot in zip((x, mask, a, s), snapshots, strict=True):
        assert_array_equal(array, snapshot)


@pytest.mark.parametrize("layout", ["C", "F", "strided", "reversed"])
@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_preparation_owns_one_float_copy(monkeypatch, layout, dtype) -> None:
    x = _layout(np.arange(1, 64).reshape(7, 9).astype(dtype), layout)
    snapshot = x.copy()
    x.flags.writeable = False
    allocated = []
    original_array = np.array

    def trace_array(value, *args, **kwargs):
        result = original_array(value, *args, **kwargs)
        if value is x:
            allocated.append(result)
        return result

    def keep_entries(data, probe, init, **_kwargs):
        assert data.flags.c_contiguous
        assert data.flags.writeable
        assert data.dtype == np.float64
        assert not np.shares_memory(data, x)
        assert np.shares_memory(data, allocated[0])
        data[0, 0] = -123  # Mutating the owned buffer must leave x untouched.
        return data, probe, np.arange(7), np.arange(9), init

    monkeypatch.setattr(np, "array", trace_array)
    monkeypatch.setattr(updates, "remove_empty_entries", keep_entries)
    updates._prepare_data(x, {"init": "random"})
    assert len(allocated) == 1
    assert_array_equal(x, snapshot)


@pytest.mark.parametrize(
    "values", [[[True, True]], [[True, False]], [[1, 2]], [[1, -1]], [[1, np.nan]]]
)
def test_fast_path_retains_positive_mask_rule(values) -> None:
    mask = np.array(values)
    assert updates._dense_mask_is_fully_observed(mask) == bool(
        np.all(np.asarray(mask, dtype=float) > 0)
    )


@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
@pytest.mark.parametrize("compat", ["modern", "strict_legacy"])
@pytest.mark.parametrize("patterns", [False, True])
@pytest.mark.parametrize("threads", [1, 2, 4])
def test_fit_matches_float_mask_path(
    monkeypatch, algorithm, compat, patterns, threads
) -> None:
    rng = np.random.default_rng(293)
    x = _layout(rng.normal(size=(13, 17)), "strided")
    x[0, 0] = 0
    mask = rng.random(x.shape) > 0.2
    mask[0] = True
    mask[:, 0] = True
    # Some columns share masks so the pattern-sharing path has useful groups.
    mask[:, 3:6] = mask[:, 2, None]
    probe = np.full(x.shape, np.nan)
    probe[1, 1] = x[1, 1]
    snapshots = [array.copy() for array in (x, mask, probe)]
    for array in (x, mask, probe):
        array.flags.writeable = False
    options = {
        "maxiters": 6,
        "niter_broadprior": 0,
        "convergence_criteria": _OFF,
        "algorithm": algorithm,
        "compat_mode": compat,
        "uniquesv": patterns,
        "num_cpu": threads,
        "runtime_tuning": "off",
        "random_state": 293,
        "record_cost": True,
        "verbose": 0,
        "xprobe": probe,
    }
    actual = pca_full(x, 3, mask=mask, **options)
    monkeypatch.setattr(updates, "_as_dense_mask", lambda m: np.asarray(m, dtype=float))
    monkeypatch.setattr(
        rms_module, "_as_dense_mask", lambda m: np.asarray(m, dtype=float)
    )
    expected = pca_full(x, 3, mask=mask, **options)
    for key in (
        "A",
        "S",
        "Mu",
        "Muv",
        "Av",
        "Sv",
        "V",
        "Va",
        "Vmu",
        "Xrec",
        "Vr",
        "Vpred",
        "ExplainedVar",
        "ExplainedVarRatio",
    ):
        assert_array_equal(actual[key], expected[key])
    for key in ("rms", "prms", "cost"):
        assert_array_equal(actual["lc"][key], expected["lc"][key])
    for array, snapshot in zip((x, mask, probe), snapshots, strict=True):
        assert_array_equal(array, snapshot)


@pytest.mark.parametrize("patterns", [False, True])
def test_fit_does_not_expand_boolean_masks_to_float(monkeypatch, patterns) -> None:
    rng = np.random.default_rng(294)
    x = rng.normal(size=(13, 17))
    mask = rng.random(x.shape) > 0.2
    mask[0] = True
    mask[:, 0] = True
    original = np.asarray
    conversions = []

    def traced(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if (
            isinstance(value, np.ndarray)
            and value.dtype == bool
            and value.shape == x.shape
            and result.dtype == np.float64
        ):
            conversions.append(result.nbytes)
        return result

    monkeypatch.setattr(np, "asarray", traced)
    pca_full(
        x,
        3,
        mask=mask,
        maxiters=4,
        niter_broadprior=0,
        convergence_criteria=_OFF,
        uniquesv=patterns,
        runtime_tuning="off",
        num_cpu=1,
        random_state=294,
        verbose=0,
        return_diagnostics=False,
    )
    assert conversions == []


@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
@pytest.mark.parametrize("compat", ["modern", "strict_legacy"])
def test_probe_values_cannot_change_fixed_iteration_training(algorithm, compat) -> None:
    rng = np.random.default_rng(295)
    x = rng.normal(size=(11, 15))
    mask = np.ones(x.shape, dtype=bool)
    x[1, 1] = 0
    results = []
    for heldout_value in (0.0, 1e6):
        data = x.copy()
        data[1, 1] = heldout_value
        probe = np.full(x.shape, np.nan)
        probe[1, 1] = heldout_value
        for array in (data, probe, mask):
            array.flags.writeable = False
        results.append(
            pca_full(
                data,
                3,
                mask=mask,
                xprobe=probe,
                algorithm=algorithm,
                compat_mode=compat,
                maxiters=6,
                niter_broadprior=0,
                convergence_criteria=_OFF,
                runtime_tuning="off",
                num_cpu=2,
                random_state=295,
                verbose=0,
                record_cost=True,
            )
        )
        assert data[1, 1] == heldout_value
        assert probe[1, 1] == heldout_value
        assert mask.all()
    for key in ("A", "S", "Mu", "Av", "Sv", "Muv", "V", "Va", "Vmu"):
        assert_array_equal(results[0][key], results[1][key])
    assert_array_equal(results[0]["lc"]["cost"], results[1]["lc"]["cost"])
    assert_array_equal(results[0]["lc"]["rms"], results[1]["lc"]["rms"])


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_selection_matches_float_mask_path(monkeypatch, threads) -> None:
    rng = np.random.default_rng(296)
    x = rng.normal(size=(11, 15))
    x[0, 0] = 0
    mask = rng.random(x.shape) > 0.2
    mask[0] = True
    mask[:, 0] = True
    options = {
        "mask": mask,
        "components": [1, 2, 3],
        "config": SelectionConfig(
            metric="prms", compute_explained_variance=False, return_best_model=True
        ),
        "maxiters": 5,
        "niter_broadprior": 0,
        "convergence_criteria": _OFF,
        "runtime_tuning": "off",
        "num_cpu": threads,
        "random_state": 296,
        "verbose": 0,
    }
    actual = select_n_components(x, **options)
    monkeypatch.setattr(updates, "_as_dense_mask", lambda m: np.asarray(m, dtype=float))
    monkeypatch.setattr(
        rms_module, "_as_dense_mask", lambda m: np.asarray(m, dtype=float)
    )
    expected = select_n_components(x, **options)
    assert actual[0] == expected[0]
    for candidate, reference in zip(actual[2], expected[2], strict=True):
        for key in ("k", "rms", "prms", "cost"):
            assert_array_equal(candidate[key], reference[key])
    assert_array_equal(actual[3].components_, expected[3].components_)
    assert_array_equal(actual[3].scores_, expected[3].scores_)
