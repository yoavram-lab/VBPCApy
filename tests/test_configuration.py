"""Typed options preserve numerical behavior and sklearn parameter semantics."""

from __future__ import annotations

import copy
import inspect
import json
import pickle
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py import VBPCA, PCAOptions
from vbpca_py._pca_full import _solver_options_from_raw, pca_full
from vbpca_py._runtime_profiles import new_profile
from vbpca_py.configuration import _replace_config
from vbpca_py.options import Algorithm


def _settings():
    return {
        "maxiters": 3,
        "niter_broadprior": 0,
        "random_state": 19,
        "verbose": 0,
        "num_cpu": 1,
        "runtime_tuning": "off",
        "rmsstop": None,
        "minangle": 0.0,
    }


@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
@pytest.mark.parametrize("compat_mode", ["strict_legacy", "modern"])
@pytest.mark.parametrize("layout", ["dense", "nan", "masked", "sparse"])
def test_typed_and_legacy_numerical_routes_match(algorithm, compat_mode, layout):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(5, 9))
    x[0, 0] = 0.0
    mask = None
    if layout == "nan":
        x[1, ::3] = np.nan
    elif layout == "masked":
        mask = np.ones(x.shape, dtype=bool)
        mask[1, ::3] = False
    elif layout == "sparse":
        # Construct a fully stored pattern, including the observed zero.
        x = sp.csr_matrix(
            (x.ravel(), np.tile(np.arange(9), 5), np.arange(0, 46, 9)), shape=x.shape
        )
        assert x.nnz == 45
    settings = {**_settings(), "algorithm": algorithm, "compat_mode": compat_mode}
    legacy = pca_full(x, 2, mask=mask, **settings)
    typed = pca_full(x, 2, mask=mask, options=PCAOptions.from_kwargs(**settings))
    for key in ("A", "S", "Mu", "V", "Xrec", "Vr", "RMS"):
        np.testing.assert_allclose(typed[key], legacy[key], rtol=2e-12, atol=2e-12)
    assert typed["lc"]["rms"] == legacy["lc"]["rms"]


def test_configuration_owns_small_policies_and_borrows_payloads():
    criteria = {"angle": False}
    composite = {"angle": 0.5}
    order = ["angle", "cost"]
    threshold = np.array([3.0, 0.01])
    probe = np.ones((3, 4))
    initial = {"A": np.ones((3, 1))}
    config = PCAOptions.from_kwargs(
        convergence_criteria=criteria,
        composite_stop=composite,
        criterion_order=order,
        rmsstop=threshold,
        xprobe=probe,
        init=initial,
    )
    criteria["angle"] = True
    composite["angle"] = 2.0
    order.clear()
    threshold[:] = 9
    assert config.convergence_criteria["angle"] is False
    assert config.composite_stop == {"angle": 0.5}
    assert config.criterion_order == ("angle", "cost")
    assert config.rmsstop == (3.0, 0.01)
    assert config.xprobe is probe
    assert config.init is initial
    with pytest.raises(FrozenInstanceError):
        config.maxiters = 2
    with pytest.raises(AttributeError):
        _ = config.__dict__
    with pytest.raises(TypeError):
        config.composite_stop["angle"] = 2.0
    first, second = config.to_kwargs(), config.to_kwargs()
    first["convergence_criteria"]["angle"] = True
    first["rmsstop"][:] = 5
    assert second["convergence_criteria"]["angle"] is False
    np.testing.assert_array_equal(second["rmsstop"], [3.0, 0.01])
    assert not any(key.startswith("_") for key in first)
    assert first["init"] is initial
    for cloned in (copy.deepcopy(config), pickle.loads(pickle.dumps(config))):
        assert cloned.composite_stop == config.composite_stop
        assert cloned.rmsstop == config.rmsstop
        assert cloned.convergence_criteria == config.convergence_criteria


def test_constructor_parameters_are_complete_and_keep_identity():
    signature = inspect.signature(VBPCA)
    solver_names = {item.name for item in fields(PCAOptions)}
    assert set(signature.parameters) == (solver_names - {"xprobe"}) | {
        "n_components",
        "tol",
        "xprobe_fraction",
    }
    assert all(p.kind is not p.VAR_KEYWORD for p in signature.parameters.values())
    values = {
        "verbose": True,
        "rmsstop": np.array([3, 0.01]),
        "criterion_order": ["angle", "cost"],
        "convergence_criteria": {"angle": False},
        "init": {"A": np.ones((3, 1))},
        "random_state": np.random.default_rng(2),
    }
    estimator = VBPCA(1, **values)
    parameters = estimator.get_params(deep=False)
    for name, value in values.items():
        assert getattr(estimator, name) is value
        assert parameters[name] is value
    assert estimator.verbose is True
    assert not any(name.startswith("_") for name in parameters)
    assert estimator.set_params(maxiters=2) is estimator
    assert estimator.maxiters == 2


def test_native_sklearn_clone_and_grid_search():
    pytest.importorskip("sklearn")
    from sklearn.base import BaseEstimator, clone
    from sklearn.model_selection import GridSearchCV

    assert VBPCA.get_params is BaseEstimator.get_params
    assert VBPCA.set_params is BaseEstimator.set_params
    estimator = VBPCA(1, **_settings(), convergence_criteria={"angle": False})
    cloned = clone(estimator)
    assert cloned.get_params() == estimator.get_params()
    assert cloned.convergence_criteria is not estimator.convergence_criteria
    cloned.convergence_criteria["angle"] = True
    assert estimator.convergence_criteria["angle"] is False
    # This tests sklearn parameter plumbing only. The current wrapper still uses
    # features x samples; statistical held-out projection is separate work.
    search = GridSearchCV(
        estimator,
        {"maxiters": [2, 3], "algorithm": [Algorithm.VB, Algorithm.MAP]},
        scoring=lambda fitted, _x: -fitted.rms_,
        cv=2,
        error_score="raise",
    ).fit(np.random.default_rng(2).normal(size=(6, 10)))
    assert len(search.cv_results_["params"]) == 4
    assert search.best_estimator_.maxiters in {2, 3}


@pytest.mark.parametrize("route", ["direct", "legacy", "export", "replace", "pickle"])
@pytest.mark.parametrize("explicit", [False, True])
def test_thread_request_roundtrips_resolve_only_at_fit(
    monkeypatch, tmp_path, route, explicit
):
    monkeypatch.setenv("VBPCA_NUM_THREADS", "2")
    monkeypatch.setattr("vbpca_py._runtime_policy._default_num_cpu", lambda: 7)
    for name in ("SCORE", "LOADINGS", "NOISE", "RMS"):
        monkeypatch.delenv(f"VBPCA_{name}_THREADS", raising=False)
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps({**new_profile(), "default_threads": {"num_cpu_rms": 3}})
    )
    config = PCAOptions.from_kwargs(runtime_tuning="off", runtime_profile=profile)
    assert config.num_cpu is None
    if route == "direct":
        config = PCAOptions(runtime_tuning="off", runtime_profile=profile)
    elif route == "export":
        config = PCAOptions.from_kwargs(**config.to_kwargs())
    elif route == "replace":
        config = replace(config, xprobe=np.ones((3, 4)))
    elif route == "pickle":
        config = pickle.loads(pickle.dumps(config))
    if explicit:
        config = replace(config, num_cpu=4)
    monkeypatch.setenv("VBPCA_NUM_THREADS", "5")
    monkeypatch.setattr("vbpca_py._runtime_policy._default_num_cpu", lambda: 10)
    state = _solver_options_from_raw(config.to_kwargs())
    assert state["num_cpu"] == (4 if explicit else 10)
    assert state.num_cpu_user_set is explicit
    assert config.to_kwargs()["num_cpu"] == (4 if explicit else None)
    assert (
        _replace_config(state.config, {"record_cost": True}).num_cpu
        == state.config.num_cpu
    )
    assert state.config is not config
    from vbpca_py._runtime_policy import (
        RuntimeWorkloadProfile,
        resolve_runtime_thread_config,
    )

    workload = RuntimeWorkloadProfile(3, 4, 1, 12, is_sparse=False)
    assert resolve_runtime_thread_config(dict(state), workload=workload).rms == (
        4 if explicit else 5
    )
    # Profiles are environment-fingerprinted. Refresh for the new environment
    # before checking that inherited requests retain profile precedence.
    monkeypatch.delenv("VBPCA_NUM_THREADS")
    profile.write_text(
        json.dumps({**new_profile(), "default_threads": {"num_cpu_rms": 3}})
    )
    resolved = resolve_runtime_thread_config(dict(state), workload=workload)
    assert resolved.rms == (4 if explicit else 3)


@pytest.mark.parametrize("probe", ["none", "fit", "generated"])
@pytest.mark.parametrize("explicit", [False, True])
def test_estimator_probes_preserve_thread_precedence(
    monkeypatch, tmp_path, probe, explicit
):
    for name in ("NUM", "SCORE", "LOADINGS", "NOISE", "RMS"):
        monkeypatch.delenv(f"VBPCA_{name}_THREADS", raising=False)
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps({**new_profile(), "default_threads": {"num_cpu_rms": 3}})
    )
    settings = _settings()
    settings.pop("num_cpu")
    x = np.random.default_rng(11).normal(size=(4, 8))
    xprobe = np.full(x.shape, np.nan)
    xprobe[0, 0] = x[0, 0]
    model = VBPCA(
        1,
        **settings,
        runtime_report=True,
        runtime_profile=path,
        num_cpu=4 if explicit else None,
        xprobe_fraction=0.1 if probe == "generated" else 0.0,
    ).fit(x, xprobe=xprobe if probe == "fit" else None)
    report = model.runtime_report_
    assert report["num_cpu_user_set"] is explicit
    assert report["kernel_values"]["rms"] == (4 if explicit else 3)
    assert report["kernel_sources"]["rms"] == (
        "global_num_cpu" if explicit else "profile"
    )
    assert model.get_options()["num_cpu"] == (4 if explicit else None)
    assert not any(name.startswith("_") for name in model.get_options())


def test_invalid_names_and_ambiguous_configuration_fail_early():
    x = np.ones((3, 4))
    with pytest.raises(TypeError, match="maxiter"):
        VBPCA(1, maxiter=2)
    with pytest.raises(TypeError, match="xprobe"):
        VBPCA(1, xprobe=x)
    with pytest.raises(ValueError, match="maxiter"):
        VBPCA(1).set_params(maxiter=2)
    for bad in ({"maxiter": 2}, {"autosave": 0}, {"_noise_cache": object()}):
        with pytest.raises(TypeError, match="Unknown pca_full option"):
            pca_full(x, 1, **bad)
    with pytest.raises(TypeError, match="either"):
        pca_full(x, 1, options=PCAOptions(), verbose=0)
    with pytest.raises(TypeError, match="PCAOptions"):
        pca_full(x, 1, options={})
    out = pca_full(x, 1, MAXITERS=1, NITER_BROADPRIOR=0, VERBOSE=0)
    assert out["lc"]["rms"]


def test_runtime_state_and_repeated_fits_do_not_leak():
    config = PCAOptions.from_kwargs(**_settings())
    first = _solver_options_from_raw(config.to_kwargs())
    second = _solver_options_from_raw(config.to_kwargs())
    first["_dense_score_use_complement"] = True
    first.setdefault("_runtime_preflight", []).append({"scope": "test"})
    cloned = first.copy()
    cloned["_runtime_preflight"].append({"scope": "clone"})
    assert len(first["_runtime_preflight"]) == 1
    assert first.pop("_runtime_preflight") == [{"scope": "test"}]
    assert first.pop("_runtime_preflight", []) == []
    assert "_runtime_preflight" not in second
    assert not second.dense_score_use_complement
    with pytest.raises(KeyError):
        first["unknown"] = 1
    x = np.random.default_rng(11).normal(size=(4, 8))
    x.flags.writeable = False
    original = x.copy()
    model = VBPCA(1, **_settings()).fit(x)
    fitted = model.components_.copy()
    parameters = model.get_params()
    model.fit(x)
    np.testing.assert_allclose(model.components_, fitted)
    np.testing.assert_array_equal(x, original)
    assert model.get_params() == parameters


def test_legacy_numeric_fallbacks_and_generator_boundary():
    config = PCAOptions.from_kwargs(maxiters=None, minangle=None, verbose=None)
    assert config.maxiters == 0
    assert config.minangle == float("inf")
    assert config.verbose == 0
    generator = np.random.default_rng(4)
    assert PCAOptions(random_state=generator).random_state is generator
    assert PCAOptions.from_kwargs(random_state=np.int64(4)).random_state == 4
    with pytest.raises(TypeError, match="random_state"):
        PCAOptions.from_kwargs(random_state="bad")
    with pytest.raises(TypeError, match="booleans"):
        PCAOptions.from_kwargs(convergence_criteria={"angle": 1})


def test_parameter_api_works_without_optional_sklearn():
    script = """
import sys
sys.modules["sklearn"] = None
from vbpca_py import VBPCA
model = VBPCA(1, verbose=True)
assert model.get_params()["verbose"] is True
assert model.set_params(maxiters=2) is model
try:
    model.set_params(misspelled=1)
except ValueError:
    pass
else:
    raise AssertionError("unknown parameter accepted")
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


def test_options_guide_examples_execute():
    from pathlib import Path

    guide = Path(__file__).resolve().parents[1] / "docs/api/options.md"
    namespace = {}
    for block in guide.read_text().split("```python\n")[1:]:
        exec(compile(block.split("```", 1)[0], str(guide), "exec"), namespace)  # noqa: S102 - execute repository-owned examples
    np.testing.assert_allclose(
        namespace["result"]["A"], namespace["legacy_result"]["A"]
    )


@pytest.mark.parametrize(
    ("values", "explicit", "requested"),
    [
        ({"NUM_CPU": 4, "num_cpu": None}, False, None),
        ({"num_cpu": 4, "NUM_CPU": None}, False, None),
        ({"NUM_CPU": None, "num_cpu": 4}, True, 4),
        ({"num_cpu": None, "NUM_CPU": 4}, True, 4),
    ],
)
def test_case_normalized_thread_request_uses_last_value(values, explicit, requested):
    config = PCAOptions.from_kwargs(**values)
    assert config.num_cpu == requested
    state = _solver_options_from_raw(values)
    assert state.num_cpu_user_set is explicit
    if explicit:
        assert state.config.num_cpu == requested


def test_public_configuration_hints_resolve_at_runtime():
    from typing import get_type_hints

    for target in (PCAOptions, PCAOptions.__init__, VBPCA.__init__):
        hints = get_type_hints(target)
        assert hints["num_cpu"] == int | None
        if target is VBPCA.__init__:
            assert "xprobe" not in hints
        else:
            assert "xprobe" in hints
