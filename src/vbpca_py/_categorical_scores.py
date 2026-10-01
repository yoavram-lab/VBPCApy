"""Held-out scores for one-hot encoded categorical variables (#250).

Per-indicator error measures how well each 0/1 entry is reproduced. For a
categorical variable the prediction of interest is the level of a held-out
cell, so these scores treat each held-out (variable, sample) cell as one
categorical prediction: the reconstructed block, clipped below at a small
floor and renormalized, is read as level probabilities.

A Gaussian low-rank reconstruction is not a calibrated categorical model, so
the log score is dominated by cells whose observed level receives almost no
predicted mass (each costs about ``-log(floor)``). The Brier score is bounded
and usually the more useful selection metric; the log score is reported for
comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

CATEGORICAL_METRICS: tuple[str, ...] = ("brier", "log_score", "accuracy")
_FLOOR = 1e-6


def categorical_scores(
    prediction: np.ndarray, held_out: np.ndarray, groups: Sequence[object]
) -> dict[str, float]:
    """Score held-out cells of categorical variables.

    Args:
        prediction: Features-by-samples predicted means, including the bias.
        held_out: Features-by-samples held-out values, ``NaN`` elsewhere.
        groups: Variable label of every feature.

    Returns:
        Mean Brier score (sum of squared probability errors per cell), mean
        log score (negative log probability of the observed level) and
        accuracy of the most probable level, over held-out cells of variables
        with at least two indicators. ``NaN`` when no such cell is held out.

    Raises:
        ValueError: If shapes or labels do not match.
    """
    pred = np.asarray(prediction, dtype=float)
    held = np.asarray(held_out, dtype=float)
    labels = np.asarray(groups)
    if pred.shape != held.shape or labels.shape != (pred.shape[0],):
        msg = "prediction, held_out and groups must describe the same features"
        raise ValueError(msg)
    brier: list[np.ndarray] = []
    log_score: list[np.ndarray] = []
    correct: list[np.ndarray] = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        if len(rows) < 2:
            continue
        observed = held[rows]
        cells = np.flatnonzero(np.all(np.isfinite(observed), axis=0))
        if cells.size == 0:
            continue
        truth = observed[:, cells]
        probabilities = np.maximum(pred[np.ix_(rows, cells)], _FLOOR)
        probabilities /= probabilities.sum(axis=0, keepdims=True)
        level = np.argmax(truth, axis=0)
        columns = np.arange(cells.size)
        brier.append(np.sum((probabilities - truth) ** 2, axis=0))
        log_score.append(-np.log(probabilities[level, columns]))
        correct.append(np.argmax(probabilities, axis=0) == level)
    if not brier:
        return dict.fromkeys(CATEGORICAL_METRICS, float("nan"))
    return {
        "brier": float(np.mean(np.concatenate(brier))),
        "log_score": float(np.mean(np.concatenate(log_score))),
        "accuracy": float(np.mean(np.concatenate(correct))),
    }
