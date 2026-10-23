"""Typed categorical inputs retain ordinary-string numerical behavior."""

from __future__ import annotations

import doctest
import json
import pickle

import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat
from sklearn.base import clone

from vbpca_py import options
from vbpca_py._monitoring import _normalize_init
from vbpca_py._pca_full import _build_options, _select_algorithm, pca_full
from vbpca_py._runtime_policy import (
    normalize_accessor_mode,
    normalize_cov_writeback_mode,
    normalize_runtime_tuning_mode,
)
from vbpca_py.legacy import VBPCA
from vbpca_py.options import (
    AccessorMode,
    Algorithm,
    BiasUpdateOrder,
    CompatMode,
    CostDiagnostics,
    CovarianceWriteback,
    Criterion,
    ExplainedVarianceSolver,
    RuntimeTuning,
    StatisticsMode,
    VarianceUpdateOrder,
)

OPTION_TYPES = (
    AccessorMode,
    Algorithm,
    BiasUpdateOrder,
    CompatMode,
    CostDiagnostics,
    CovarianceWriteback,
    Criterion,
    ExplainedVarianceSolver,
    RuntimeTuning,
    StatisticsMode,
    VarianceUpdateOrder,
)


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_option_parsing_string_serialization_and_identity(option_type):
    for member in option_type:
        assert option_type.parse(member) is member
        assert option_type.parse(f" {member.value.upper()} ") is member
        assert str(member) == member.value
        assert hash(member) == hash(member.value)
        assert json.loads(json.dumps({"option": member})) == {"option": member.value}
        assert pickle.loads(pickle.dumps(member)) is member
    with pytest.raises(ValueError, match=option_type.__name__):
        option_type.parse("not-supported")
    with pytest.raises(TypeError, match="requires a string"):
        option_type.parse(3)


def test_executable_option_docstring():
    failures, count = doctest.testmod(options, raise_on_error=True)
    assert failures == 0
    assert count > 0


def test_enum_configuration_matches_legacy_string_resolution():
    enum_values = {
        "algorithm": Algorithm.VB,
        "compat_mode": CompatMode.MODERN,
        "bias_update_order": BiasUpdateOrder.AUTO,
        "variance_update_order": VarianceUpdateOrder.AUTO,
        "cost_diagnostics": CostDiagnostics.ENDPOINT,
        "dense_sufficient_statistics": StatisticsMode.OBSERVED,
        "noise_sufficient_statistics": StatisticsMode.COMPLEMENT,
        "explained_var_solver": ExplainedVarianceSolver.GRAM,
        "runtime_tuning": RuntimeTuning.OFF,
        "cov_writeback_mode": CovarianceWriteback.PYTHON,
        "accessor_mode": AccessorMode.LEGACY,
    }
    common = {
        "num_cpu": 1,
        "criterion_order": [Criterion.ANGLE, Criterion.RMS_PLATEAU],
        "convergence_criteria": {Criterion.COST: False},
    }
    resolved = _build_options({**common, **enum_values})
    strings = _build_options({
        **common,
        **{key: value.value for key, value in enum_values.items()},
    })
    assert resolved.keys() == strings.keys()
    for key, value in resolved.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, strings[key])
        else:
            assert value == strings[key]
    assert resolved["bias_update_order"] == "post_factor"
    assert resolved["variance_update_order"] == "post_rotation"


@pytest.mark.parametrize("algorithm", list(Algorithm))
@pytest.mark.parametrize("sparse_input", [False, True])
def test_enum_and_string_solver_calls_return_identical_posteriors(
    algorithm, sparse_input
):
    rng = np.random.default_rng(60)
    data = rng.normal(size=(6, 15))
    data[0, 0] = 0.0
    mask = rng.random(data.shape) > 0.25
    mask[0, 0] = True
    if sparse_input:
        rows, cols = np.nonzero(mask)
        data = sparse.coo_matrix(
            (data[rows, cols], (rows, cols)), shape=data.shape
        ).tocsr()
    else:
        data = np.where(mask, data, np.nan)
        data.setflags(write=False)
    common = {
        "maxiters": 3,
        "niter_broadprior": 0,
        "random_state": 61,
        "num_cpu": 1,
        "return_diagnostics": False,
    }
    typed = pca_full(
        data,
        2,
        **common,
        algorithm=algorithm,
        compat_mode=CompatMode.MODERN,
        runtime_tuning=RuntimeTuning.OFF,
    )
    ordinary = pca_full(
        data,
        2,
        **common,
        algorithm=algorithm.value,
        compat_mode="modern",
        runtime_tuning="off",
    )
    for name in ("A", "S", "Mu", "V", "Av", "Sv", "Va", "Vmu", "Muv"):
        np.testing.assert_array_equal(typed[name], ordinary[name])


def test_existing_estimator_keeps_enum_parameters_during_clone():
    model = VBPCA(2, algorithm=Algorithm.MAP, compat_mode=CompatMode.MODERN)
    copied = clone(model)
    for fitted in (model, copied):
        assert fitted.get_params()["algorithm"] is Algorithm.MAP
        assert fitted.get_params()["compat_mode"] is CompatMode.MODERN


def test_legacy_normalization_and_runtime_fallbacks_are_preserved():
    assert _select_algorithm({"algorithm": "VB"}) == (True, True)
    with pytest.raises(ValueError, match="Wrong value"):
        _select_algorithm({"algorithm": " VB "})
    assert _build_options({"COMPAT_MODE": " MODERN "})["compat_mode"] == "modern"
    assert normalize_runtime_tuning_mode("invalid") == "off"
    assert normalize_cov_writeback_mode("invalid") == "auto"
    assert normalize_accessor_mode("invalid") == "auto"
    assert normalize_cov_writeback_mode(None) == "auto"
    assert normalize_accessor_mode(None) == "auto"
    # Strict explicit parsing must not inherit the legacy fallback policy.
    with pytest.raises(ValueError, match="RuntimeTuning"):
        RuntimeTuning.parse("invalid")


def test_initialization_remains_random_mapping_or_mat_filename(tmp_path):
    assert _normalize_init("RANDOM") == {}
    assert _normalize_init(None) == {}
    warm = {"A": np.array([[1.0], [2.0]]), "S": np.array([[3.0, 4.0]])}
    normalized = _normalize_init(warm)
    np.testing.assert_array_equal(normalized["A"], warm["A"])
    assert normalized is not warm
    filename = tmp_path / "warm-start.mat"
    savemat(filename, warm)
    loaded = _normalize_init(str(filename))
    np.testing.assert_array_equal(loaded["A"], warm["A"])
    np.testing.assert_array_equal(loaded["S"], warm["S"])
