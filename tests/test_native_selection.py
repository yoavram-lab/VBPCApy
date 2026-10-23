"""Native selection boundaries preserve the qualified cellwise numerical target."""

from __future__ import annotations

import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.sparse as sp
from sklearn import config_context
from sklearn.base import clone

from vbpca_py import (
    VBPCA,
    AutoEncoder,
    CVConfig,
    SelectionConfig,
    cross_validate_components,
    legacy,
    prepare_raw_cv_folds,
    select_n_components,
)
from vbpca_py.model_selection import _native_capacities

OPTIONS = {
    "maxiters": 8,
    "niter_broadprior": 0,
    "random_state": 39,
    "runtime_tuning": "off",
    "num_cpu": 1,
    "verbose": 0,
    "minangle": 0,
    "rmsstop": 0,
    "cfstop_rel": 0,
}


def _data():
    x = np.random.default_rng(39).normal(size=(18, 5))
    x[0, 0] = 0.0
    x[1, 1] = np.nan
    return x


def _probe(x):
    probe = np.full_like(x, np.nan)
    probe[3, 0] = x[3, 0]
    probe[5, 1] = x[5, 1]
    return probe


def _selection(x, **options):
    return select_n_components(
        x,
        components=[1, 2],
        config=SelectionConfig(metric="rms", return_best_model=True),
        **{**OPTIONS, **options},
    )


def _assert_records(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_records(actual[key], expected[key])
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            _assert_records(value, reference)
    elif isinstance(expected, np.ndarray | float | np.floating):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
    else:
        assert actual == expected


@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
@pytest.mark.parametrize("mode", ["modern", "strict_legacy"])
def test_native_selection_matches_explicit_feature_first_reference(algorithm, mode):
    x = _data()
    probe = _probe(x)
    actual = _selection(x, algorithm=algorithm, compat_mode=mode, xprobe=probe)
    expected = legacy.select_n_components(
        x.T,
        components=[1, 2],
        config=SelectionConfig(metric="rms", return_best_model=True),
        algorithm=algorithm,
        compat_mode=mode,
        xprobe=probe.T,
        **OPTIONS,
    )
    _assert_records(actual[:3], expected[:3])
    model, reference = actual[-1], expected[-1]
    assert type(model) is VBPCA
    assert type(reference) is legacy.VBPCA
    np.testing.assert_array_equal(model.components_, reference.components_.T)
    np.testing.assert_array_equal(model.scores_, reference.scores_.T)
    np.testing.assert_array_equal(model.mean_, reference.mean_.ravel())
    for field in ["reconstruction_", "variance_", "predictive_variance_"]:
        expected_value = getattr(reference, field)
        if expected_value is None:
            assert getattr(model, field) is None
        else:
            np.testing.assert_allclose(getattr(model, field), expected_value.T)
    assert model.n_features_in_ == x.shape[1]
    assert model.scores_.shape == (len(x), model.components_.shape[0])
    assert model.compat_mode == mode
    native_fit = VBPCA(**model.get_params()).fit(x, xprobe=probe)
    new = x[:3].copy()
    new[0] = np.nan
    np.testing.assert_allclose(model.transform(new), native_fit.transform(new))
    posterior = model.transform_posterior(new)
    np.testing.assert_allclose(posterior.mean, native_fit.transform(new))
    assert np.isfinite(model.score(x))
    assert model.inverse_transform(model.transform(x)).shape == x.shape
    assert clone(model).get_params() == model.get_params()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.transform(new), model.transform(new))


@pytest.mark.parametrize("kind", ["numpy", "pandas", "polars"])
def test_native_default_is_modern_and_preserves_named_projection(kind):
    x = _data()
    names = [f"measure_{j}" for j in range(x.shape[1])]
    table = x
    if kind == "pandas":
        table = pd.DataFrame(x, columns=names, index=np.arange(len(x)) + 30)
    elif kind == "polars":
        table = pl.DataFrame(x, schema=names)
    actual = _selection(table)
    explicit = _selection(table, compat_mode="modern")
    _assert_records(actual[:3], explicit[:3])
    model = actual[-1]
    assert model.compat_mode == "modern"
    if kind != "numpy":
        np.testing.assert_array_equal(model.feature_names_in_, names)
        changed = table.iloc[:, ::-1] if kind == "pandas" else table.select(names[::-1])
        with pytest.raises(ValueError, match="feature names"):
            model.transform(changed)
    model.set_output(transform="pandas")
    projected = model.transform(table)
    assert list(projected.columns) == model.get_feature_names_out().tolist()
    if kind == "pandas":
        np.testing.assert_array_equal(projected.index, table.index)
    model.set_output(transform="polars")
    np.testing.assert_array_equal(
        model.transform(table).to_numpy(), projected.to_numpy()
    )


@pytest.mark.parametrize("kind", [sp.csr_matrix, sp.csc_matrix, sp.csr_array])
def test_sparse_masks_preserve_requested_zero_observations_without_densifying(kind):
    x = _data()
    mask = np.isfinite(x)
    mask[3, 2] = False
    # A missing stored value requested by the mask is an observed zero.
    values = np.nan_to_num(x)
    values[3, 2] = np.inf
    values[0, 0] = 0.0
    sparse = kind(values)
    # Disable dense endpoint outputs; their allocation contract is tested below.
    result = select_n_components(
        sparse,
        mask=kind(mask),
        components=[1],
        config=SelectionConfig(
            metric="rms", compute_explained_variance=False, return_best_model=True
        ),
        max_dense_bytes=0,
        **OPTIONS,
    )
    canonical = np.where(mask, np.nan_to_num(x), np.nan)
    reference = select_n_components(
        canonical,
        components=[1],
        config=SelectionConfig(
            metric="rms", compute_explained_variance=False, return_best_model=True
        ),
        max_dense_bytes=0,
        **OPTIONS,
    )
    np.testing.assert_allclose(
        result[-1].components_, reference[-1].components_, rtol=3e-12, atol=3e-12
    )
    assert result[-1].reconstruction_ is None


def test_native_masked_inputs_and_probes_match_canonical_missingness():
    x = _data()
    mask = np.isfinite(x)
    mask[4, 0] = False
    placeholders = np.nan_to_num(x)
    placeholders[4, 0] = np.inf
    probe = _probe(x)
    actual = _selection(placeholders, mask=mask, xprobe=probe)
    expected = _selection(np.where(mask, x, np.nan), xprobe=probe)
    _assert_records(actual[:3], expected[:3])
    np.testing.assert_array_equal(actual[-1].scores_, expected[-1].scores_)


@pytest.mark.parametrize("metric", ["prms", "brier", "log_score"])
def test_cellwise_cv_transposes_feature_groups_and_preserves_metric_records(metric):
    raw = np.tile(np.array([0, 1, 2, 1, 0, 2]), 5)
    other = np.roll(raw, 1)
    x = np.column_stack(
        [raw == j for j in range(3)] + [other == j for j in range(3)]
    ).astype(float)
    groups = [0, 0, 0, 1, 1, 1]
    cfg = CVConfig(n_splits=2, seed=39, metric=metric, feature_groups=groups)
    actual = cross_validate_components(x, components=[0, 1], config=cfg, **OPTIONS)
    expected = legacy.cross_validate_components(
        x.T, components=[0, 1], config=cfg, compat_mode="modern", **OPTIONS
    )
    _assert_records(actual, expected)


@pytest.mark.parametrize("components", [[], [-1], [1.5], [True], [np.nan], [6]])
@pytest.mark.parametrize("helper", [select_n_components, cross_validate_components])
def test_native_capacities_are_never_silently_dropped_or_truncated(helper, components):
    with pytest.raises(ValueError, match=r"components|nonnegative integer"):
        helper(_data(), components=components, **OPTIONS)


def test_native_zero_selection_retains_no_model_contract_and_legacy_defaults():
    rank, metrics, trace, model = select_n_components(
        _data(),
        components=[0],
        config=SelectionConfig(metric="rms", return_best_model=True),
        **OPTIONS,
    )
    assert rank == metrics["k"] == 0
    assert len(trace) == 1
    assert model is None
    assert legacy.VBPCA(1).compat_mode == "strict_legacy"
    legacy_model = legacy.VBPCA(1, **OPTIONS)
    old = legacy_model.select_n_components(
        _data().T,
        components=[1],
        config=SelectionConfig(metric="rms", return_best_model=True),
    )[-1]
    assert type(old) is legacy.VBPCA
    assert old.compat_mode == "strict_legacy"
    assert old.transform().shape == old.scores_.shape


@pytest.mark.parametrize("helper", [select_n_components, cross_validate_components])
def test_native_validation_refuses_invalid_data_and_does_not_mutate_inputs(helper):
    x = _data()
    before = x.copy()
    x.flags.writeable = False
    helper(x, components=[1], **OPTIONS)
    np.testing.assert_array_equal(x, before)
    for bad in [
        np.ones(10),
        np.full(x.shape, np.nan),
        np.zeros((0, 5)),
        np.column_stack([x, np.full(len(x), np.nan)]),
    ]:
        with pytest.raises(ValueError):
            helper(bad, components=[1], **OPTIONS)
    with pytest.raises(ValueError, match="shape"):
        helper(x, mask=np.ones((1, 1), dtype=bool), components=[1], **OPTIONS)


@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_native_probe_names_and_representation_are_checked(kind):
    names = list("abcde")
    if kind == "pandas":
        x = pd.DataFrame(_data(), columns=names)
        probe = pd.DataFrame(_probe(_data()), columns=names)
        reordered, shorter = probe.iloc[:, ::-1], probe.iloc[:3]
    else:
        x = pl.DataFrame(_data(), schema=names)
        probe = pl.DataFrame(_probe(_data()), schema=names)
        reordered, shorter = probe.select(names[::-1]), probe.head(3)
    _selection(x, xprobe=probe)
    VBPCA(1, **OPTIONS).fit(x, xprobe=probe)
    for bad in [reordered, shorter, sp.csr_matrix(probe.to_numpy())]:
        with pytest.raises(ValueError, match=r"feature names|shape|representation"):
            _selection(x, xprobe=bad)
        with pytest.raises(ValueError, match=r"feature names|shape|representation"):
            VBPCA(1, **OPTIONS).fit(x, xprobe=bad)
    for kwargs in [{"xprobe": None}, {"xprobe_fraction": 0.1}]:
        with pytest.raises(ValueError, match="constructs its own probes"):
            cross_validate_components(x, components=[1], **OPTIONS, **kwargs)


def test_native_selection_annotations_resolve_to_native_model_and_configuration():
    assert get_type_hints(select_n_components)["return"].__args__[-1] == VBPCA | None
    assert _native_capacities(None, (4, 5)) is None
    for value in [CVConfig(), SelectionConfig()]:
        assert pickle.loads(pickle.dumps(value)) == value
    with config_context(transform_output="polars"):
        model = _selection(_data())[-1]
        assert isinstance(model.transform(_data()), pl.DataFrame)


def test_prepared_raw_folds_have_native_axes_and_versioned_serialization():
    raw = np.arange(60, dtype=float).reshape(20, 3)
    fold = prepare_raw_cv_folds(
        raw,
        encoder_factory=lambda: AutoEncoder(
            column_types=["continuous"] * 3, handle_unknown="raise"
        ),
        config=CVConfig(n_splits=2, seed=39),
    )[0]
    assert fold.training.shape == raw.shape
    restored = pickle.loads(pickle.dumps(fold))
    np.testing.assert_array_equal(restored.training, fold.training)
    _assert_records(
        restored.numeric_scores(np.zeros_like(raw)),
        fold.numeric_scores(np.zeros_like(raw)),
    )
    with pytest.raises(ValueError, match="samples-by-features"):
        replace(fold, training=fold.training.T)
    with pytest.raises(ValueError, match="prediction must match"):
        fold.numeric_scores(np.zeros_like(raw.T))
    for old_state in [
        (None, {"training": fold.training.T}),
        {"training": fold.training.T},
    ]:
        empty = object.__new__(type(fold))
        with pytest.raises(ValueError, match=r"rebuild pre-0\.5"):
            empty.__setstate__(old_state)


def test_public_selection_example_executes_from_the_installed_package():
    script = (
        Path(__file__).resolve().parents[1] / "scripts/example_component_selection.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert "Probe selection:" in completed.stdout
    assert "Cellwise CV:" in completed.stdout
