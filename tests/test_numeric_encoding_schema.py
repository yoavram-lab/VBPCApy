"""Mixed-schema validation must establish a usable numeric inverse map."""

from dataclasses import replace

import numpy as np
import pytest

from vbpca_py import (
    AutoEncoder,
    CVConfig,
    EncodedVariable,
    EncodingSchema,
    cross_validate_raw_components,
)


@pytest.mark.parametrize("kind", ["continuous", "ordinal"])
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"columns": (), "offset": ()}, "exactly one column"),
        ({"columns": (0, 1), "offset": (0.0, 1.0)}, "exactly one column"),
        ({"offset": ()}, "one offset"),
        ({"offset": (0.0, 1.0)}, "one offset"),
        ({"offset": (np.nan,)}, "finite fitted offset"),
        ({"offset": (np.inf,)}, "finite fitted offset"),
        ({"offset": (-np.inf,)}, "finite fitted offset"),
        ({"weight": 0.0}, "positive finite weight"),
        ({"weight": -1.0}, "positive finite weight"),
        ({"weight": np.nan}, "positive finite weight"),
        ({"weight": np.inf}, "positive finite weight"),
        ({"weight": -np.inf}, "positive finite weight"),
    ],
)
def test_strict_validation_and_decoding_refuse_invalid_numeric_maps(
    kind, change, message
):
    variable = replace(EncodedVariable(kind, (0,), offset=(2.0,)), **change)
    schema = EncodingSchema((variable,))
    width = len(variable.columns)
    # Categorical-only scoring intentionally does not require numeric inverses.
    schema.validate(width)
    with pytest.raises(ValueError, match=message):
        schema.validate(width, strict_numeric=True)
    with pytest.raises(ValueError, match=message):
        variable.decode_numeric(np.zeros((width, 3)))


@pytest.mark.parametrize("scaler", ["standard", "minmax"])
@pytest.mark.parametrize("support", ["varying", "constant", "tiny"])
@pytest.mark.parametrize(
    "options",
    [
        {"binary": "both"},
        {"binary": "single"},
        {"drop": "first", "mean_center_ohe": True},
        {"mean_center_ohe": True, "block_weighting": "equal_variance"},
    ],
)
def test_fitted_mixed_maps_restore_input_units_and_declared_ordinal_indices(
    scaler, support, options
):
    continuous = np.array([2.0, 4.0, 8.0, np.nan, 16.0, 32.0])
    if support == "constant":
        continuous[np.isfinite(continuous)] = 7.0
    elif support == "tiny":
        continuous *= 1e-18
    raw = np.empty((6, 3), dtype=object)
    raw[:, 0] = ["no", "yes", "yes", "no", "yes", "no"]
    raw[:, 1] = continuous
    raw[:, 2] = ["low", "medium", "high", "low", np.nan, "high"]
    encoder = AutoEncoder(
        column_types=["categorical", "continuous", "ordinal"],
        column_levels=[None, None, ["high", "low", "medium"]],
        continuous_scaler=scaler,
        **options,
    )
    encoded = encoder.fit_transform(raw).T
    schema = encoder.encoding_schema_
    schema.validate(encoded.shape[0], strict_numeric=True)
    numeric, ordinal = schema.variables[1:]
    restored = numeric.decode_numeric(encoded[list(numeric.columns)])[0]
    np.testing.assert_allclose(restored, continuous, rtol=1e-14, atol=0, equal_nan=True)
    np.testing.assert_allclose(
        ordinal.decode_numeric(encoded[list(ordinal.columns)])[0],
        [1.0, 2.0, 0.0, 1.0, np.nan, 0.0],
        atol=1e-14,
    )
    # Even an empty prediction must establish the same affine contract.
    assert numeric.decode_numeric(np.empty((1, 0))).shape == (1, 0)
    for variable in (numeric, ordinal):
        with pytest.raises(ValueError, match="matching columns"):
            variable.decode_numeric(np.zeros((2, 3)))
        with pytest.raises(ValueError, match="matching columns"):
            variable.decode_numeric(np.zeros(3))


@pytest.mark.parametrize("kind", ["continuous", "ordinal"])
@pytest.mark.parametrize("scaler", ["standard", "minmax"])
@pytest.mark.parametrize("missing", ["nan", "mask"])
def test_wholly_missing_numeric_support_cannot_certify_a_fitted_inverse(
    kind, scaler, missing
):
    raw = np.array([[0.0, 10.0], [1.0, 20.0], [0.0, 30.0]])
    mask = np.ones(raw.shape, dtype=bool)
    if missing == "nan":
        raw[:, 1] = np.nan
    else:
        mask[:, 1] = False
    encoder = AutoEncoder(
        column_types=["categorical", kind],
        column_levels=[None, [10.0, 20.0, 30.0] if kind == "ordinal" else None],
        continuous_scaler=scaler,
    )
    encoded = encoder.fit_transform(raw, mask=mask)
    assert np.isnan(encoded[:, -1]).all()
    schema = encoder.encoding_schema_
    schema.validate(encoded.shape[1])
    with pytest.raises(ValueError, match="wholly missing training support"):
        schema.validate(encoded.shape[1], strict_numeric=True)
    with pytest.raises(ValueError, match="wholly missing training support"):
        schema.variables[1].decode_numeric(np.zeros((1, 2)))


@pytest.mark.parametrize("strict", [False, True])
def test_strict_mode_keeps_categorical_validation_and_feature_coverage(strict):
    categorical = EncodedVariable("categorical", (0,), ("no", "yes"), reference=0)
    numeric = EncodedVariable("continuous", (1,), offset=(0.0,))
    with pytest.raises(ValueError, match="categorical encoding"):
        EncodingSchema((categorical, numeric)).validate(2, strict_numeric=strict)
    with pytest.raises(ValueError, match="cover every feature"):
        EncodingSchema((numeric,)).validate(2, strict_numeric=strict)


@pytest.mark.parametrize("change", [{"weight": 0.0}, {"offset": (np.nan,)}])
def test_raw_cv_refuses_invalid_numeric_metadata_before_model_fitting(
    monkeypatch, change
):
    import vbpca_py.model_selection as selection

    def unexpected_fit(*args, **kwargs):
        pytest.fail("invalid numeric metadata reached model fitting")

    monkeypatch.setattr(selection, "_fit_candidate", unexpected_fit)

    class InvalidSchemaEncoder(AutoEncoder):
        def fit(self, x, y=None, *, mask=None):
            super().fit(x, y, mask=mask)
            categorical, numeric = self.encoding_schema_.variables
            self.encoding_schema_ = EncodingSchema((
                categorical,
                replace(numeric, **change),
            ))
            return self

    raw = np.column_stack([np.tile([0.0, 1.0], 12), np.arange(24.0)])
    with pytest.raises(ValueError, match="numeric encoding"):
        cross_validate_raw_components(
            raw,
            encoder_factory=lambda: InvalidSchemaEncoder(
                column_types=["categorical", "continuous"], handle_unknown="raise"
            ),
            components=[0],
            config=CVConfig(n_splits=2, seed=17),
        )
