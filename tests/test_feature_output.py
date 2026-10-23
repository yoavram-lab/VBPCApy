"""Names, output tables and row alignment follow fitted transformation plans."""

import copy
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.sparse as sp
from sklearn import config_context
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from vbpca_py import VBPCA, AutoEncoder, ObservationMasker
from vbpca_py import preprocessing as p

KINDS = [
    "standard",
    "minmax",
    "log",
    "power",
    "winsor",
    "ordinal",
    "masker",
    "onehot",
    "sparse_onehot",
    "auto",
    "native0",
    "native1",
]


def _make(kind):
    if kind.startswith("native"):
        return VBPCA(
            int(kind[-1]),
            maxiters=3,
            niter_broadprior=0,
            verbose=0,
            num_cpu=1,
            runtime_tuning="off",
            random_state=23,
        )
    factories = {
        "standard": p.MissingAwareStandardScaler,
        "minmax": p.MissingAwareMinMaxScaler,
        "log": p.MissingAwareLogTransformer,
        "power": p.MissingAwarePowerTransformer,
        "winsor": p.MissingAwareWinsorizer,
        "ordinal": p.MissingAwareOrdinalEncoder,
        "masker": ObservationMasker,
        "onehot": p.MissingAwareOneHotEncoder,
        "sparse_onehot": p.MissingAwareSparseOneHotEncoder,
        "auto": lambda: AutoEncoder(column_types=["continuous", "continuous"]),
    }
    return factories[kind]()


def _data():
    return pd.DataFrame(
        [[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]],
        columns=["length", "mass"],
        index=[5, 2, 5, 9],
    )


def _expected(kind, inputs):
    if kind.startswith("native"):
        return ["vbpca0"] if kind == "native1" else []
    if kind in {"onehot", "sparse_onehot"}:
        return [f"{inputs[0]}_{v}.0" for v in range(1, 5)] + [
            f"{inputs[1]}_{v}.0" for v in range(5, 9)
        ]
    return list(inputs)


@pytest.mark.parametrize("kind", KINDS)
def test_default_names_match_fitted_order_and_custom_names_for_unnamed_inputs(kind):
    x = _data()
    model = _make(kind)
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()
    values = model.fit_transform(x)
    expected = _expected(kind, x.columns)
    np.testing.assert_array_equal(model.get_feature_names_out(), expected)
    assert model.get_feature_names_out().dtype == object
    assert len(expected) == values.shape[1]
    assert (
        sp.issparse(values)
        if kind == "sparse_onehot"
        else isinstance(values, np.ndarray)
    )
    model.fit(x.to_numpy())
    np.testing.assert_array_equal(
        model.get_feature_names_out(), _expected(kind, ["x0", "x1"])
    )
    np.testing.assert_array_equal(
        model.get_feature_names_out(["a", "b"]), _expected(kind, ["a", "b"])
    )


@pytest.mark.parametrize("kind", KINDS)
def test_names_are_owned_and_fitted_display_metadata_remains_consistent(kind):
    model = _make(kind).fit(_data())
    names = model.get_feature_names_out()
    before = names.copy()
    if len(names):
        names[0] = "mutated outside estimator"
    np.testing.assert_array_equal(model.get_feature_names_out(), before)
    np.testing.assert_array_equal(model.feature_names_in_, _data().columns)
    if hasattr(model, "feature_names_out_"):
        assert model.feature_names_out_ == before.tolist()
    model.transform(_data())
    if hasattr(model, "feature_names_out_"):
        assert model.feature_names_out_ == before.tolist()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("output", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["local", "global"])
def test_native_output_wrapping_preserves_values_names_and_rows(kind, output, mode):
    x = _data()
    reference = _make(kind).fit_transform(x)
    transformer = _make(kind)
    with config_context(transform_output=output if mode == "global" else "default"):
        if mode == "local":
            assert transformer.set_output(transform=output) is transformer
        if kind == "sparse_onehot":
            with pytest.raises(ValueError, match="sparse matrix"):
                transformer.fit_transform(x)
            return
        result = transformer.fit_transform(x)
        repeated = transformer.transform(x)
    expected = _expected(kind, x.columns)
    if output == "pandas":
        assert isinstance(result, pd.DataFrame)
        assert result.index.equals(x.index)
        assert repeated.index.equals(x.index)
        assert result.columns.tolist() == expected
    else:
        assert isinstance(result, pl.DataFrame)
        assert result.columns == expected
    assert result.shape == (len(x), len(expected))
    np.testing.assert_allclose(result.to_numpy(), reference)
    np.testing.assert_allclose(repeated.to_numpy(), reference)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("bad", [["mass", "length"], ["length"], ["length", "new"]])
def test_supplied_names_validate_fitted_input_order_and_count(kind, bad):
    with pytest.raises(ValueError, match="input_features"):
        _make(kind).fit(_data()).get_feature_names_out(bad)


@pytest.mark.parametrize("bad", [[1, 2], [["a", "b"]], ["a", "a"], "ab"])
def test_supplied_names_require_unique_one_dimensional_strings(bad):
    model = _make("standard").fit(_data().to_numpy())
    with pytest.raises(ValueError, match="unique strings"):
        model.get_feature_names_out(bad)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("labels", [[1, 2], [("a", 1), ("b", 2)]])
def test_non_string_table_columns_use_positional_names(kind, labels):
    x = _data()
    x.columns = labels
    model = _make(kind).fit(x)
    assert not hasattr(model, "feature_names_in_")
    np.testing.assert_array_equal(
        model.get_feature_names_out(), _expected(kind, ["x0", "x1"])
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("labels", [["a", "a"], [1, 1]])
def test_duplicate_table_columns_are_refused_and_failed_refits_clear_names(
    kind, labels
):
    model = _make(kind).fit(_data())
    x = _data()
    x.columns = labels
    with pytest.raises(
        ValueError, match=r"duplicate feature names|unique column names"
    ):
        model.fit(x)
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()


@pytest.mark.parametrize("kind", KINDS)
def test_mixed_string_and_non_string_table_labels_are_refused(kind):
    x = _data()
    x.columns = ["length", 1]
    with pytest.raises(TypeError, match="Feature names are only supported"):
        _make(kind).fit(x)


@pytest.mark.parametrize(
    "options",
    [
        {"binary": "single"},
        {"binary": "both"},
        {"drop": "first", "mean_center": True, "block_weighting": "equal_variance"},
    ],
)
def test_categorical_names_follow_fitted_reference_and_training_category_order(options):
    x = pd.DataFrame({"area": ["z", "a", "q", "z"], "flag": ["no", "yes", "yes", "no"]})
    model = p.MissingAwareOneHotEncoder(**options).set_output(transform="pandas")
    result = model.fit_transform(x)
    first = (
        ["area_a", "area_q"] if options.get("drop") else ["area_z", "area_a", "area_q"]
    )
    second = (
        ["flag_no", "flag_yes"] if options.get("binary") == "both" else ["flag_yes"]
    )
    assert result.columns.tolist() == first + second
    assert model.feature_names_out_ == result.columns.tolist()
    # Names depend on fitted widths, even if later parameter changes need refit.
    before = model.get_feature_names_out().copy()
    model.set_params(drop="first" if options.get("drop") is None else None)
    np.testing.assert_array_equal(model.get_feature_names_out(), before)


@pytest.mark.parametrize("output", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["local", "global"])
def test_mixed_autoencoder_uses_global_variable_names_and_keeps_nested_arrays(
    output, mode
):
    x = pd.DataFrame(
        {
            "zone": ["north", "south", "north", "east"],
            "flag": ["no", "yes", "no", "yes"],
            "size": [10.0, 20.0, 30.0, 40.0],
            "grade": ["mid", "low", "high", "mid"],
        },
        index=[9, 4, 7, 1],
    )
    kwargs = {
        "column_types": ["categorical", "categorical", "continuous", "ordinal"],
        "column_levels": [None, None, None, ["low", "mid", "high"]],
        "binary": "both",
    }
    reference = AutoEncoder(**kwargs).fit_transform(x)
    encoder = AutoEncoder(**kwargs)
    with config_context(transform_output=output if mode == "global" else "default"):
        if mode == "local":
            encoder.set_output(transform=output)
        result = encoder.fit_transform(x)
        for plan in encoder._plan:
            assert plan.encoder._sklearn_output_config["transform"] == "default"
            if plan.kind == "ordinal":
                assert (
                    plan.encoder._scaler._sklearn_output_config["transform"]
                    == "default"
                )
    assert list(result.columns) == [
        "zone_north",
        "zone_south",
        "zone_east",
        "flag_no",
        "flag_yes",
        "size",
        "grade",
    ]
    np.testing.assert_allclose(result.to_numpy(), reference)
    assert encoder.feature_names_out_ == list(result.columns)
    if output == "pandas":
        assert result.index.equals(x.index)


@pytest.mark.parametrize("kind", ["onehot", "sparse_onehot", "auto"])
def test_wholly_missing_categorical_variables_produce_zero_names(kind):
    model = (
        AutoEncoder(column_types=["categorical", "categorical"])
        if kind == "auto"
        else _make(kind)
    )
    x = np.full((4, 2), np.nan)
    result = model.fit_transform(x)
    assert result.shape == (4, 0)
    assert model.get_feature_names_out().shape == (0,)
    if kind != "sparse_onehot":
        for output in ["pandas", "polars"]:
            converted = model.set_output(transform=output).transform(x)
            assert converted.shape == (4, 0)
            inverse = model.inverse_transform(converted)
            assert inverse.shape == (4, 2)
            assert np.isnan(inverse.astype(float)).all()


def test_ambiguous_category_strings_are_disambiguated_deterministically():
    x = np.array([[1], ["1"], ["1__1"], [1]], dtype=object)
    model = p.MissingAwareOneHotEncoder(binary="both").set_output(transform="polars")
    result = model.fit_transform(x)
    assert result.columns == ["x0_1", "x0_1__1", "x0_1__1__2"]
    assert len(set(result.columns)) == result.width


@pytest.mark.parametrize("kind", KINDS)
def test_output_configuration_survives_clone_pickle_and_can_be_reset(kind):
    model = _make(kind).set_output(transform="pandas")
    copied = clone(model)
    if kind == "sparse_onehot":
        with pytest.raises(ValueError, match="sparse matrix"):
            copied.fit_transform(_data())
        return
    result = copied.fit_transform(_data())
    for restored in [copy.deepcopy(copied), pickle.loads(pickle.dumps(copied))]:
        pd.testing.assert_frame_equal(restored.transform(_data()), result)
        assert isinstance(
            restored.set_output(transform="default").transform(_data()), np.ndarray
        )


@pytest.mark.parametrize(
    "kind", ["standard", "minmax", "auto", "masker", "sparse_onehot"]
)
@pytest.mark.parametrize("output", ["pandas", "polars"])
def test_sparse_outputs_are_refused_without_densification(kind, output, monkeypatch):
    x = sp.csr_matrix(_data().to_numpy())
    model = _make(kind).set_output(transform=output)

    def forbidden(*args, **kwargs):
        pytest.fail("container wrapping must not densify sparse output")

    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(sp.csr_matrix, "todense", forbidden)
    with pytest.raises(ValueError, match="sparse matrix"):
        model.fit_transform(x)


@pytest.mark.parametrize("output", ["pandas", "polars"])
def test_masked_pipeline_and_gridsearch_keep_names_row_order_and_numerical_scores(
    output,
):
    x = pd.DataFrame(
        {"group": np.tile(["a", "b", "c"], 8), "value": np.arange(24.0)},
        index=np.arange(24)[::-1],
    )
    mask = np.ones(x.shape, dtype=bool)
    mask[::4, 0] = False
    mask[1::4, 1] = False
    data = x.copy()
    data.iloc[::4, 0] = "hidden"
    data.iloc[1::4, 1] = 10000.0

    def pipeline():
        return Pipeline([
            ("observe", ObservationMasker()),
            (
                "encode",
                AutoEncoder(column_types=["categorical", "continuous"], binary="both"),
            ),
            ("model", _make("native1")),
        ])

    with config_context(enable_metadata_routing=True):
        reference = GridSearchCV(
            pipeline(), {"model__n_components": [0, 1]}, cv=3, error_score="raise"
        ).fit(data, observation_mask=mask)
        configured = pipeline().set_output(transform=output)
        if output == "polars":
            configured.named_steps["observe"].set_output(transform="pandas")
        actual = GridSearchCV(
            configured, {"model__n_components": [0, 1]}, cv=3, error_score="raise"
        ).fit(data, observation_mask=mask)
        np.testing.assert_allclose(
            actual.cv_results_["mean_test_score"],
            reference.cv_results_["mean_test_score"],
        )
        projected = actual.best_estimator_.transform(
            data.iloc[:5], observation_mask=mask[:5]
        )
    q = actual.best_params_["model__n_components"]
    assert projected.shape == (5, q)
    assert list(projected.columns) == [f"vbpca{i}" for i in range(q)]
    if output == "pandas":
        assert projected.index.equals(x.index[:5])
    assert actual.best_estimator_.get_feature_names_out().tolist() == list(
        projected.columns
    )


@pytest.mark.parametrize("mode", ["local", "global"])
def test_polars_raw_object_output_is_explicitly_refused(mode):
    x = pd.DataFrame({"area": ["north", np.nan, "south"]})
    model = ObservationMasker()
    with config_context(transform_output="polars" if mode == "global" else "default"):
        if mode == "local":
            model.set_output(transform="polars")
        with pytest.raises(ValueError, match="raw object observations"):
            model.fit_transform(x)


@pytest.mark.parametrize("dtype", ["U8", "S8"])
@pytest.mark.parametrize("mode", ["local", "global"])
def test_masked_string_arrays_are_refused_before_polars_object_conversion(dtype, mode):
    raw = np.array([["a", "b"], ["c", "d"]], dtype=dtype)
    mask = np.array([[True, False], [True, True]])
    model = ObservationMasker()
    with config_context(transform_output="polars" if mode == "global" else "default"):
        if mode == "local":
            model.set_output(transform="polars")
        with pytest.raises(ValueError, match="raw object observations"):
            model.fit_transform(raw, mask=mask)
        # Unmasked fixed-width strings do not need object conversion.
        output = model.fit_transform(raw)
        np.testing.assert_array_equal(output.to_numpy(), raw)


@pytest.mark.parametrize("kind", ["masker", "onehot", "ordinal", "auto"])
def test_polars_null_categories_are_missing_and_not_a_fitted_none_level(kind):
    x = pl.DataFrame({"area": ["low", None, "high", "low"]})
    model = AutoEncoder(column_types=["categorical"]) if kind == "auto" else _make(kind)
    reference = pd.DataFrame({"area": ["low", np.nan, "high", "low"]})
    output = model.fit_transform(x)
    expected = (
        AutoEncoder(column_types=["categorical"]) if kind == "auto" else _make(kind)
    ).fit_transform(reference)
    assert np.isnan(np.asarray(output)[1].astype(float)).all()
    if kind != "masker":
        np.testing.assert_allclose(output, expected)
    else:
        np.testing.assert_array_equal(output[[0, 2, 3]], expected[[0, 2, 3]])
    # The input table's null remains unchanged.
    assert x["area"].null_count() == 1


@pytest.mark.parametrize("output", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["native0", "native1"])
def test_configured_projection_leaves_inverse_and_posterior_arrays_unchanged(
    output, kind
):
    model = _make(kind).set_output(transform=output).fit(_data())
    projected = model.transform(_data())
    inverse = model.inverse_transform(projected)
    assert isinstance(inverse, np.ndarray)
    assert inverse.shape == _data().shape
    posterior = model.transform_posterior(_data())
    assert isinstance(posterior.mean, np.ndarray)
    np.testing.assert_allclose(posterior.mean, projected.to_numpy())


def test_compatible_single_column_sparse_metadata_can_still_supply_names():
    single = p.MissingAwareSparseOneHotEncoder().fit(np.array([[0.0], [1.0]]))
    single.feature_groups_ = None  # compatible older single-column artifacts
    assert single.get_feature_names_out().tolist() == ["x0_0.0", "x0_1.0"]
    multiple = p.MissingAwareSparseOneHotEncoder().fit(_data())
    multiple.feature_groups_ = None
    with pytest.raises(ValueError, match="multi-column sparse encoding"):
        multiple.get_feature_names_out()


def test_documented_feature_output_example_executes():
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts/example_feature_output.py"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "preserve values and row order" in completed.stdout


@pytest.mark.parametrize("output", ["pandas", "polars"])
def test_complex_indicator_storage_roundtrips_through_table_adapters(output):
    raw = np.array([[0.0, 1.0], [1.0, np.nan], [2.0, 0.0], [0.0, 1.0]])
    model = p.MissingAwareOneHotEncoder(
        dtype=np.complex64, binary="both", mean_center=True
    ).set_output(transform=output)
    encoded = model.fit_transform(raw)
    decoded = model.inverse_transform(encoded).astype(float)
    np.testing.assert_allclose(decoded, raw, equal_nan=True)
