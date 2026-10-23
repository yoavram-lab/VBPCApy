"""Raw-cell holdouts precede every learned preprocessing operation (#267)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py import (
    AutoEncoder,
    CVConfig,
    EncodedVariable,
    MissingAwareOneHotEncoder,
    MissingAwareOrdinalEncoder,
    cross_validate_raw_components,
    prepare_raw_cv_folds,
)

KINDS = ["categorical", "categorical", "continuous", "ordinal"]
LEVELS = [["west", "east", "north"], ["no", "yes"], None, ["low", "medium", "high"]]
MODES = [
    {"binary": "both"},
    {"binary": "single"},
    {"drop": "first"},
    {"drop": "first", "mean_center_ohe": True, "block_weighting": "equal_variance"},
]


def _data():
    rng = np.random.default_rng(29)
    x = np.column_stack([
        np.tile(LEVELS[0], 16),
        np.tile(LEVELS[1], 24),
        rng.normal(100, 15, 48),
        np.tile(LEVELS[3], 16),
    ]).astype(object)
    x[:, 2] = x[:, 2].astype(float)
    x[rng.random(x.shape) < 0.08] = np.nan
    return x


def _encoder(**options):
    return AutoEncoder(
        column_types=KINDS, column_levels=LEVELS, handle_unknown="raise", **options
    )


@pytest.mark.parametrize("scaler", ["standard", "minmax"])
def test_changing_only_holdouts_cannot_change_fitted_preprocessing(scaler):
    x = _data()
    config = CVConfig(n_splits=3, seed=11)

    def factory():
        return _encoder(continuous_scaler=scaler, **MODES[-1])

    before = prepare_raw_cv_folds(x, encoder_factory=factory, config=config)[0]
    changed = x.copy()
    for j in (0, 1, 3):
        for row in np.flatnonzero(before.heldout_mask[:, j]):
            levels = LEVELS[j]
            changed[row, j] = levels[(levels.index(x[row, j]) + 1) % len(levels)]
    changed[before.heldout_mask[:, 2], 2] += 10000
    after = prepare_raw_cv_folds(changed, encoder_factory=factory, config=config)[0]
    np.testing.assert_array_equal(after.training_mask, before.training_mask)
    np.testing.assert_array_equal(after.heldout_mask, before.heldout_mask)
    np.testing.assert_array_equal(after.training, before.training)
    assert after.schema == before.schema
    assert not np.array_equal(after.held_out, before.held_out, equal_nan=True)
    training_values = x[before.training_mask[:, 2], 2].astype(float)
    expected_offset = (
        training_values.mean() if scaler == "standard" else training_values.min()
    )
    assert before.schema.variables[2].offset[0] == pytest.approx(expected_offset)
    expected_scale = (
        training_values.std() if scaler == "standard" else np.ptp(training_values)
    )
    assert before.schema.variables[2].weight == pytest.approx(1 / expected_scale)
    for j in (0, 1):
        variable = before.schema.variables[j]
        training = x[before.training_mask[:, j], j]
        kept = variable.categories[1:]
        probabilities = np.array([(training == level).mean() for level in kept])
        assert variable.offset == pytest.approx(probabilities)
        assert variable.weight == pytest.approx(
            1 / np.sqrt(np.sum(probabilities * (1 - probabilities)))
        )


@pytest.mark.parametrize("options", MODES)
@pytest.mark.parametrize("early_stop", [False, True])
def test_candidate_sweeps_fit_one_encoder_per_fold_and_score_original_variables(
    options, early_stop
):
    fitted = []

    class RecordingEncoder(AutoEncoder):
        def fit(self, x, mask=None):
            assert np.isnan(x[~mask].astype(float)).all()
            fitted.append(self)
            return super().fit(x, mask=mask)

    def factory():
        return RecordingEncoder(
            column_types=KINDS, column_levels=LEVELS, handle_unknown="raise", **options
        )

    best, results, folds = cross_validate_raw_components(
        _data(),
        encoder_factory=factory,
        components=range(4),
        config=CVConfig(
            n_splits=3,
            seed=11,
            metric="brier",
            selection_rule="first_minimum",
            early_stop=early_stop,
        ),
        maxiters=20,
        niter_broadprior=0,
    )
    assert best in range(4)
    assert len(fitted) == len(folds) == 3
    assert len({id(encoder) for encoder in fitted}) == 3
    for fold in folds:
        assert not np.any(fold.training_mask & fold.heldout_mask)
        assert fold.training_mask.any(axis=0).all()
        assert fold.training_mask.any(axis=1).all()
        scores = fold.numeric_scores(np.zeros_like(fold.training))
        for j, variable in enumerate(fold.schema.variables):
            np.testing.assert_array_equal(
                np.isfinite(fold.training[:, list(variable.columns)]),
                np.broadcast_to(
                    fold.training_mask[:, j, None],
                    (len(fold.raw), len(variable.columns)),
                ),
            )
            np.testing.assert_array_equal(
                np.isfinite(fold.held_out[:, list(variable.columns)]),
                np.broadcast_to(
                    fold.heldout_mask[:, j, None],
                    (len(fold.raw), len(variable.columns)),
                ),
            )
            if j < 2:
                continue
            truth = fold.raw[fold.heldout_mask[:, j], j]
            truth = (
                truth.astype(float)
                if j == 2
                else np.array([LEVELS[j].index(v) for v in truth])
            )
            expected = np.sqrt(np.mean((truth - variable.offset[0]) ** 2))
            assert scores[f"rmse_variable_{j}"] == pytest.approx(expected)
        assert fold.schema.variables[3].categories == tuple(LEVELS[3])
    for result in results:
        assert np.isfinite(result["mean_log_score"])
        assert 0 <= result["mean_brier"] <= 2
        assert result["n_folds_rmse_variable_2"] == 3
        assert result["n_folds_rmse_variable_3"] == 3


def test_rank_zero_decoded_scores_are_equivalent_across_encodings():
    records = [
        cross_validate_raw_components(
            _data(),
            encoder_factory=lambda options=options: _encoder(**options),
            components=[0],
            config=CVConfig(n_splits=3, seed=11, metric="log_score"),
        )[1][0]
        for options in MODES
    ]
    for name in (
        "brier",
        "log_score",
        "accuracy",
        "rmse_variable_2",
        "rmse_variable_3",
    ):
        for record in records[1:]:
            assert record[f"mean_{name}"] == pytest.approx(records[0][f"mean_{name}"])


def test_held_only_level_requires_external_vocabulary_even_with_reference_drop():
    x = _data()
    config = CVConfig(n_splits=3, seed=11)
    first = prepare_raw_cv_folds(x, encoder_factory=_encoder, config=config)[0]
    row = np.flatnonzero(first.heldout_mask[:, 0])[0]
    x[row, 0] = "novel"
    with pytest.raises(ValueError, match=r"fold 0.*column_levels.*Unknown"):
        prepare_raw_cv_folds(
            x,
            config=config,
            encoder_factory=lambda: AutoEncoder(
                column_types=KINDS, drop="first", handle_unknown="raise"
            ),
        )
    levels = [LEVELS[0] + ["novel"], *LEVELS[1:]]
    declared = prepare_raw_cv_folds(
        x,
        config=config,
        encoder_factory=lambda: AutoEncoder(
            column_types=KINDS,
            column_levels=levels,
            drop="first",
            mean_center_ohe=True,
            handle_unknown="raise",
        ),
    )[0]
    variable = declared.schema.variables[0]
    assert variable.categories[-1] == "novel"
    assert variable.offset[-1] == 0
    assert declared.held_out[row, variable.columns[-1]] == 1


def test_numeric_variable_without_eligible_holdouts_is_reported_explicitly():
    x = np.arange(60, dtype=float).reshape(20, 3)
    x[1:, 2] = np.nan
    _, records, _ = cross_validate_raw_components(
        x,
        components=[0],
        config=CVConfig(n_splits=2),
        encoder_factory=lambda: AutoEncoder(
            column_types=["continuous"] * 3, handle_unknown="raise"
        ),
    )
    assert records[0]["n_folds_rmse_variable_2"] == 0
    assert np.isnan(records[0]["mean_rmse_variable_2"])


def test_categorical_objective_requires_categorical_holdouts_in_every_fold():
    x = np.arange(90, dtype=float).reshape(30, 3)
    x[1:, 2] = np.nan
    with pytest.raises(ValueError, match="observed categorical cells"):
        prepare_raw_cv_folds(
            x,
            config=CVConfig(n_splits=2, metric="brier"),
            encoder_factory=lambda: AutoEncoder(
                column_types=["continuous", "continuous", "categorical"],
                handle_unknown="raise",
            ),
        )


def test_ordinal_hold_only_level_requires_declared_order():
    x = _data()
    first = prepare_raw_cv_folds(x, encoder_factory=_encoder)[0]
    row = np.flatnonzero(first.heldout_mask[:, 3])[0]
    x[row, 3] = "new ordinal level"
    with pytest.raises(ValueError, match="column_levels"):
        prepare_raw_cv_folds(
            x,
            encoder_factory=lambda: AutoEncoder(
                column_types=KINDS,
                column_levels=[*LEVELS[:3], None],
                handle_unknown="raise",
            ),
        )


def test_explicit_masks_and_single_level_categories_preserve_original_cells():
    x = np.column_stack([np.full(30, "only"), np.arange(30), np.arange(30)]).astype(
        object
    )
    mask = np.ones(x.shape, dtype=bool)
    mask[0, 0] = False
    _, records, folds = cross_validate_raw_components(
        x,
        mask=mask,
        components=[0],
        encoder_factory=lambda: AutoEncoder(
            column_types=["categorical", "continuous", "continuous"],
            drop="first",
            handle_unknown="raise",
        ),
    )
    for fold in folds:
        np.testing.assert_array_equal(fold.training_mask | fold.heldout_mask, mask)
        assert fold.schema.variables[0].categories == ("only",)
        assert len(fold.schema.variables[0].columns) == 1
    assert records[0]["mean_brier"] == 0


def test_nonfinite_observed_numeric_values_cannot_silently_leave_the_holdout():
    x = _data()
    first = prepare_raw_cv_folds(x, encoder_factory=_encoder)[0]
    x[np.flatnonzero(first.heldout_mask[:, 2])[0], 2] = np.inf
    with pytest.raises(ValueError, match="infinity in observed entries"):
        prepare_raw_cv_folds(x, encoder_factory=_encoder)


@pytest.mark.parametrize(
    "config",
    [
        CVConfig(n_splits=1),
        CVConfig(metric="cost"),
        CVConfig(feature_groups=[0, 1, 2, 3]),
    ],
)
def test_raw_cv_rejects_encoded_configuration(config):
    with pytest.raises(ValueError, match="raw CV needs"):
        prepare_raw_cv_folds(_data(), encoder_factory=_encoder, config=config)


def test_invalid_raw_input_factory_and_capacity_fail_clearly():
    for x in (np.ones(20), sp.csr_matrix(np.ones((20, 4)))):
        with pytest.raises(ValueError, match="dense samples-by-variables"):
            prepare_raw_cv_folds(x, encoder_factory=_encoder)
    with pytest.raises(ValueError, match="fresh AutoEncoder"):
        prepare_raw_cv_folds(_data(), encoder_factory=AutoEncoder)
    reused = _encoder()
    with pytest.raises(ValueError, match="fresh AutoEncoder"):
        prepare_raw_cv_folds(_data(), encoder_factory=lambda: reused)
    x = _data()
    x[:, 2] = np.nan
    with pytest.raises(ValueError, match="every variable and sample"):
        prepare_raw_cv_folds(x, encoder_factory=_encoder)
    with pytest.raises(ValueError, match=r"min\(n_samples, n_features\)"):
        cross_validate_raw_components(
            _data(), encoder_factory=_encoder, components=[100]
        )
    with pytest.raises(ValueError, match="do not supply xprobe"):
        cross_validate_raw_components(_data(), encoder_factory=_encoder, xprobe=_data())


@pytest.mark.parametrize("levels", [[], [0, 0], [0, np.nan]])
def test_declared_vocabularies_must_be_nonempty_unique_and_observed(levels):
    for encoder in (
        MissingAwareOneHotEncoder(categories=[levels]),
        MissingAwareOrdinalEncoder(levels=[levels]),
    ):
        with pytest.raises(ValueError, match="declared levels"):
            encoder.fit(np.array([[0], [1]]))


def test_nonfinite_numeric_predictions_and_missing_metadata_are_rejected():
    fold = prepare_raw_cv_folds(_data(), encoder_factory=_encoder)[0]
    with pytest.raises(ValueError, match="numeric predictions must be finite"):
        fold.numeric_scores(np.full_like(fold.training, np.nan))
    with pytest.raises(ValueError, match="fitted affine metadata"):
        EncodedVariable("continuous", (0,)).decode_numeric(np.zeros((1, 2)))
    fold.encoder.encoding_schema_ = None
    with pytest.raises(ValueError, match="no fitted dense encoding schema"):
        _ = fold.schema
