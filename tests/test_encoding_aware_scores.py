"""Compose dense encoders, categorical decoding and component CV (#266)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vbpca_py import (
    AutoEncoder,
    CVConfig,
    EncodedVariable,
    EncodingSchema,
    MissingAwareOneHotEncoder,
    cross_validate_components,
)
from vbpca_py._categorical_scores import categorical_scores

ENCODINGS = [
    {"binary": "both"},
    {"binary": "single"},
    {"binary": "both", "drop": "first"},
    {"binary": "both", "block_weighting": "equal_variance"},
    {"binary": "both", "mean_center": True},
    {"drop": "first", "block_weighting": "equal_variance", "mean_center": True},
]


def _data():
    rng = np.random.default_rng(42)
    x = np.column_stack([
        np.tile([2, 0, 1], 16),
        np.tile([1, 0], 24),
        rng.normal(size=48),
        np.tile([0, 2, 1, 3], 12),
    ])
    x[9, 0] = np.nan
    x[18, 1] = np.nan
    return x


@pytest.mark.parametrize("options", ENCODINGS)
def test_equivalent_categorical_predictions_have_equivalent_decoded_scores(options):
    x = _data()[:, :2]
    encoder = MissingAwareOneHotEncoder(**options)
    truth = encoder.fit_transform(x).T
    schema = encoder.encoding_schema_
    assert schema is not None
    schema.validate(truth.shape[0])
    assert schema.feature_groups == tuple(encoder.feature_groups_)
    prediction = np.empty_like(truth)
    for variable in schema.variables:
        probabilities = np.full((len(variable.categories), len(x)), 0.2)
        probabilities[0] = 1.0 - 0.2 * (len(variable.categories) - 1)
        kept = (
            np.delete(probabilities, variable.reference, axis=0)
            if variable.reference is not None
            else probabilities
        )
        prediction[list(variable.columns)] = (
            kept - np.asarray(variable.offset)[:, None]
        ) * variable.weight
        np.testing.assert_allclose(
            variable.decode(prediction[list(variable.columns)]), probabilities
        )
    scores = categorical_scores(prediction, truth, schema.feature_groups, schema=schema)
    # Compare against the identical original-level predictions in full coding.
    full_encoder = MissingAwareOneHotEncoder(binary="both")
    full_truth = full_encoder.fit_transform(x).T
    full_prediction = np.vstack([
        np.tile([[0.6], [0.2], [0.2]], (1, len(x))),
        np.tile([[0.8], [0.2]], (1, len(x))),
    ])
    expected = categorical_scores(
        full_prediction, full_truth, full_encoder.feature_groups_
    )
    assert scores == pytest.approx(expected)
    perfect = categorical_scores(truth, truth, schema.feature_groups, schema=schema)
    assert perfect["brier"] == pytest.approx(0, abs=1e-9)
    assert perfect["log_score"] == pytest.approx(0, abs=1e-5)
    assert perfect["accuracy"] == 1
    np.testing.assert_array_equal(
        encoder.inverse_transform(truth.T)[np.isfinite(x)].astype(float),
        x[np.isfinite(x)],
    )


@pytest.mark.parametrize("options", ENCODINGS)
def test_public_cv_scores_rank_zero_and_positive_candidates_with_mixed_schema(options):
    options = dict(options)
    options["mean_center_ohe"] = options.pop("mean_center", False)
    encoder = AutoEncoder(
        column_types=["categorical", "categorical", "continuous", "ordinal"],
        **options,
    )
    z = encoder.fit_transform(_data())
    schema = encoder.encoding_schema_
    assert schema is not None
    assert [v.kind for v in schema.variables] == encoder.feature_kinds_
    best, results = cross_validate_components(
        z.T,
        components=[0, 1],
        config=CVConfig(n_splits=2, encoding_schema=schema, metric="brier", seed=19),
        maxiters=20,
        niter_broadprior=0,
        verbose=0,
    )
    assert best in {0, 1}
    for entry in results:
        assert 0 <= entry["mean_brier"] <= 2
        assert np.isfinite(entry["mean_log_score"])
        assert 0 <= entry["mean_accuracy"] <= 1
    # Affine encodings must not change the mean-only model's decoded scores.
    baseline = AutoEncoder(column_types=encoder.column_types, binary="both")
    full = baseline.fit_transform(_data())
    _, expected = cross_validate_components(
        full.T,
        components=[0],
        config=CVConfig(n_splits=2, encoding_schema=baseline.encoding_schema_, seed=19),
    )
    for name in ("brier", "log_score", "accuracy"):
        assert results[0][f"mean_{name}"] == pytest.approx(expected[0][f"mean_{name}"])


def test_numeric_and_ordinal_predictions_do_not_enter_nominal_scores():
    encoder = AutoEncoder(
        column_types=["categorical", "categorical", "continuous", "ordinal"]
    )
    truth = encoder.fit_transform(_data()).T
    prediction = truth.copy()
    schema = encoder.encoding_schema_
    for variable in schema.variables:
        if variable.kind != "categorical":
            prediction[list(variable.columns)] = 12345
    scores = categorical_scores(prediction, truth, schema.feature_groups, schema=schema)
    assert scores["brier"] == pytest.approx(0, abs=1e-9)


def test_transform_keeps_fit_means_weights_and_schema_for_new_data():
    encoder = MissingAwareOneHotEncoder(
        mean_center=True, block_weighting="equal_variance"
    )
    encoder.fit(np.array([[0], [0], [0], [1]]))
    schema = encoder.encoding_schema_
    means = encoder.column_means_.copy()
    new = np.array([[1], [1], [0]])
    encoded = encoder.transform(new)
    np.testing.assert_allclose(
        encoded[:, 0], (new[:, 0] - 0.25) * schema.variables[0].weight
    )
    np.testing.assert_array_equal(encoder.column_means_, means)
    assert encoder.encoding_schema_ is schema
    np.testing.assert_array_equal(encoder.inverse_transform(encoded).astype(float), new)
    encoder.fit(np.array([[0], [1], [1], [1]]))
    assert schema.variables[0].offset == (0.25,)
    assert encoder.encoding_schema_.variables[0].offset == (0.75,)


def test_schema_and_targets_are_validated_before_any_fit(monkeypatch):
    import vbpca_py.model_selection as selection

    def unexpected_fit(*args, **kwargs):
        pytest.fail("invalid encoding reached model fitting")

    monkeypatch.setattr(selection, "_fit_candidate", unexpected_fit)
    encoder = MissingAwareOneHotEncoder(binary="both")
    z = encoder.fit_transform(_data()[:, :2]).T
    schema = encoder.encoding_schema_
    with pytest.raises(ValueError, match="must match"):
        cross_validate_components(
            z, config=CVConfig(encoding_schema=schema, feature_groups=[0] * len(z))
        )
    malformed = z.copy()
    malformed[0, 0] = np.nan
    with pytest.raises(ValueError, match="every encoded indicator"):
        cross_validate_components(malformed, config=CVConfig(encoding_schema=schema))
    malformed = z.copy()
    malformed[:3, 0] = 0
    with pytest.raises(ValueError, match="known categorical levels"):
        cross_validate_components(malformed, config=CVConfig(encoding_schema=schema))
    with pytest.raises(ValueError, match="cover every feature"):
        cross_validate_components(z[:-1], config=CVConfig(encoding_schema=schema))
    numeric = EncodingSchema((EncodedVariable("continuous", (0,)),))
    with pytest.raises(ValueError, match="observed categorical cells"):
        cross_validate_components(
            np.ones((1, 10)), config=CVConfig(encoding_schema=numeric, metric="brier")
        )


@pytest.mark.parametrize(
    "change",
    [
        {"reference": 5},
        {"offset": ()},
        {"weight": 0},
        {"weight": np.nan},
        {"kind": "unsupported"},
    ],
)
def test_invalid_affine_and_category_metadata_are_rejected(change):
    variable = EncodedVariable(
        "categorical", (0,), ("no", "yes"), reference=0, offset=(0.0,)
    )
    with pytest.raises(ValueError):
        EncodingSchema((replace(variable, **change),)).validate(1)


def test_nonfinite_predictions_and_non_categorical_decoding_are_rejected():
    encoder = MissingAwareOneHotEncoder()
    truth = encoder.fit_transform(np.array([[0], [1], [0]])).T
    schema = encoder.encoding_schema_
    with pytest.raises(ValueError, match="must be finite"):
        categorical_scores(truth * np.nan, truth, schema.feature_groups, schema=schema)
    with pytest.raises(ValueError, match="categorical block"):
        EncodedVariable("ordinal", (0,)).decode(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="categorical block"):
        schema.variables[0].decode(np.zeros((2, 3)))


@pytest.mark.parametrize("options", ENCODINGS[1:])
def test_transformed_or_singleton_categorical_selection_requires_schema(options):
    encoder = MissingAwareOneHotEncoder(**options)
    z = encoder.fit_transform(_data()[:, :2]).T
    with pytest.raises(ValueError, match="encoding_schema"):
        cross_validate_components(
            z,
            components=[0],
            config=CVConfig(feature_groups=encoder.feature_groups_, metric="brier"),
        )
