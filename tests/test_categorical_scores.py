"""Tests for categorical held-out scores (#250)."""

from __future__ import annotations

import numpy as np
import pytest

from vbpca_py._categorical_scores import categorical_scores

GROUPS = np.array([0, 0, 0, 1])


def test_perfect_predictions_score_zero_error() -> None:
    truth = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.3, 0.7]])

    scores = categorical_scores(truth, truth, GROUPS)

    assert scores["brier"] == pytest.approx(0.0, abs=1e-9)
    assert scores["log_score"] == pytest.approx(0.0, abs=1e-5)
    assert scores["accuracy"] == pytest.approx(1.0)


def test_scores_match_hand_calculation() -> None:
    prediction = np.array([[0.6, 0.2], [0.3, 0.5], [0.1, 0.3], [0.0, 0.0]])
    held = np.array([[1.0, np.nan], [0.0, np.nan], [0.0, np.nan], [np.nan, np.nan]])

    scores = categorical_scores(prediction, held, GROUPS)

    assert scores["brier"] == pytest.approx(0.4**2 + 0.3**2 + 0.1**2)
    assert scores["log_score"] == pytest.approx(-np.log(0.6))
    assert scores["accuracy"] == pytest.approx(1.0)


def test_negative_predictions_are_clipped_and_renormalized() -> None:
    prediction = np.array([[0.9], [0.3], [-0.2], [0.0]])
    held = np.array([[0.0], [1.0], [0.0], [np.nan]])

    scores = categorical_scores(prediction, held, GROUPS)

    assert np.isfinite(scores["log_score"])
    assert scores["log_score"] == pytest.approx(-np.log(0.3 / (1.2 + 1e-6)), rel=1e-4)
    assert scores["accuracy"] == pytest.approx(0.0)


def test_no_held_out_categorical_cell_gives_nan() -> None:
    held = np.full((4, 2), np.nan)
    held[3] = 1.0

    scores = categorical_scores(np.zeros((4, 2)), held, GROUPS)

    assert all(np.isnan(value) for value in scores.values())


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="same features"):
        categorical_scores(np.zeros((4, 2)), np.zeros((4, 3)), GROUPS)


def _structured_one_hot() -> tuple[np.ndarray, np.ndarray]:
    from vbpca_py.preprocessing import MissingAwareOneHotEncoder

    rng = np.random.default_rng(3)
    profiles = rng.integers(0, 3, size=(3, 6))
    member = rng.integers(0, 3, 120)
    codes = np.where(
        rng.random((120, 6)) < 0.7, profiles[member], rng.integers(0, 3, (120, 6))
    ).astype(float)
    codes[rng.random(codes.shape) < 0.05] = np.nan
    encoder = MissingAwareOneHotEncoder(binary="both")
    z = encoder.fit_transform(codes, mask=~np.isnan(codes))
    assert encoder.feature_groups_ is not None
    return z.T, encoder.feature_groups_


def test_cross_validation_reports_categorical_scores() -> None:
    from vbpca_py import CVConfig, cross_validate_components

    x, groups = _structured_one_hot()

    _, results = cross_validate_components(
        x,
        components=range(4),
        config=CVConfig(feature_groups=groups, seed=1),
        maxiters=200,
        verbose=0,
    )

    for entry in results:
        assert 0.0 <= entry["mean_accuracy"] <= 1.0
        assert entry["mean_log_score"] > 0.0
        assert "se_brier" in entry
    assert results[2]["mean_accuracy"] > results[0]["mean_accuracy"]


def test_brier_selection_recovers_the_planted_rank() -> None:
    from vbpca_py import CVConfig, cross_validate_components

    x, groups = _structured_one_hot()

    best_k, _ = cross_validate_components(
        x,
        components=range(8),
        config=CVConfig(
            feature_groups=groups, metric="brier", selection_rule="first_minimum"
        ),
        maxiters=200,
        verbose=0,
    )

    assert best_k == 2


def test_log_score_selection_runs_and_is_finite() -> None:
    from vbpca_py import CVConfig, cross_validate_components

    x, groups = _structured_one_hot()

    best_k, results = cross_validate_components(
        x,
        components=range(4),
        config=CVConfig(feature_groups=groups, metric="log_score"),
        maxiters=200,
        verbose=0,
    )

    assert 0 <= best_k <= 3
    assert all(np.isfinite(entry["mean_log_score"]) for entry in results)


def test_categorical_metrics_need_feature_groups() -> None:
    from vbpca_py import CVConfig, cross_validate_components

    x, _ = _structured_one_hot()

    with pytest.raises(ValueError, match=r"needs CVConfig\.feature_groups"):
        cross_validate_components(x, config=CVConfig(metric="brier"))
