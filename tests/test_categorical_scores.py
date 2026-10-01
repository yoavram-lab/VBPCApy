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
