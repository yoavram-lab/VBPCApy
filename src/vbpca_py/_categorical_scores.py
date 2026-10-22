"""Held-out scores for one-hot encoded categorical variables.

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

from ._types import BoolArray, Float64Array, InputArray  # noqa: TC001 - runtime annotation resolution
from .encoding import EncodedVariable, EncodingSchema

if TYPE_CHECKING:
    from collections.abc import Sequence

CATEGORICAL_METRICS: tuple[str, ...] = ("brier", "log_score", "accuracy")
_FLOOR = 1e-6


def categorical_scores(
    prediction: Float64Array,
    held_out: Float64Array,
    groups: Sequence[object],
    *,
    schema: EncodingSchema | None = None,
) -> dict[str, float]:
    """Score held-out cells of categorical variables.

    Args:
        prediction: Features-by-samples predicted means, including the bias.
        held_out: Features-by-samples held-out values, ``NaN`` elsewhere.
        groups: Variable label of every feature.
        schema: Fitted dense encoding metadata. Restores weights, centering,
            reference categories and single-indicator binaries. Without it,
            only full unweighted indicator blocks with at least two features
            are interpreted as categorical; singleton groups are skipped.

    Returns:
        Mean Brier score (sum of squared probability errors per cell), mean
        log score (negative log probability of the observed level) and
        accuracy of the most probable level, over held-out cells of variables
        described by the schema (or legacy full blocks). ``NaN`` when no such
        cell is held out. Continuous and ordinal variables are excluded.

    Raises:
        ValueError: If shapes or labels do not match, schema targets are not
            known one-hot levels, or decoded predictions are nonfinite.
    """
    pred = np.asarray(prediction, dtype=float)
    held = np.asarray(held_out, dtype=float)
    labels = np.asarray(groups)
    if pred.ndim != 2 or pred.shape != held.shape or labels.shape != (pred.shape[0],):
        msg = "prediction, held_out and groups must describe the same features"
        raise ValueError(msg)
    brier: list[Float64Array] = []
    log_score: list[Float64Array] = []
    correct: list[BoolArray] = []
    for variable in _categorical_variables(labels, pred.shape[0], schema):
        scores = _cell_scores(pred, held, variable, validate=schema is not None)
        if scores is not None:
            brier.append(scores[0])
            log_score.append(scores[1])
            correct.append(scores[2])
    if not brier:
        return dict.fromkeys(CATEGORICAL_METRICS, float("nan"))
    return {
        "brier": float(np.mean(np.concatenate(brier))),
        "log_score": float(np.mean(np.concatenate(log_score))),
        "accuracy": float(np.mean(np.concatenate(correct))),
    }


def _categorical_variables(
    labels: InputArray, n_features: int, schema: EncodingSchema | None
) -> list[EncodedVariable]:
    if schema is not None:
        schema.validate(n_features)
        if not np.array_equal(labels, schema.feature_groups):
            msg = "feature_groups must match the fitted encoding schema"
            raise ValueError(msg)
        return [v for v in schema.variables if v.kind == "categorical"]
    variables = []
    for label in np.unique(labels):
        rows = tuple(int(r) for r in np.flatnonzero(labels == label))
        if len(rows) >= 2:
            variables.append(
                EncodedVariable(
                    "categorical",
                    rows,
                    tuple(range(len(rows))),
                    offset=(0.0,) * len(rows),
                )
            )
    return variables


def validate_legacy_categorical_input(
    x: Float64Array, groups: Sequence[object]
) -> None:
    """Require unambiguous full one-hot targets for schema-free selection.

    Raises:
        ValueError: If groups include singleton variables or targets use a
            transformed, dropped or otherwise non-one-hot representation.
    """
    variables = _categorical_variables(np.asarray(groups), x.shape[0], None)
    if sum(len(v.columns) for v in variables) != x.shape[0]:
        msg = (
            "categorical selection with singleton groups needs encoding_schema "
            "to distinguish binary, ordinal and continuous variables"
        )
        raise ValueError(msg)
    schema = EncodingSchema(tuple(variables))
    try:
        categorical_scores(x, x, schema.feature_groups, schema=schema)
    except ValueError as error:
        msg = (
            "schema-free categorical selection requires full unweighted one-hot "
            "targets; pass the fitted encoder's encoding_schema_"
        )
        raise ValueError(msg) from error


def _cell_scores(
    prediction: Float64Array,
    held: Float64Array,
    variable: EncodedVariable,
    *,
    validate: bool,
) -> tuple[Float64Array, Float64Array, BoolArray] | None:
    rows = np.asarray(variable.columns, dtype=int)
    if rows.size == 0:
        return None
    observed = held[rows]
    finite = np.isfinite(observed)
    if validate and np.any(np.any(finite, axis=0) != np.all(finite, axis=0)):
        msg = "categorical held-out cells must include every encoded indicator"
        raise ValueError(msg)
    cells = np.flatnonzero(np.all(finite, axis=0))
    if cells.size == 0:
        return None
    truth = variable.decode(observed[:, cells])
    decoded = variable.decode(prediction[np.ix_(rows, cells)])
    if validate:
        _validate_decoded(truth, decoded)
    probabilities = np.maximum(decoded, _FLOOR)
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    level = np.argmax(truth, axis=0)
    return (
        np.sum((probabilities - truth) ** 2, axis=0),
        -np.log(probabilities[level, np.arange(cells.size)]),
        np.argmax(probabilities, axis=0) == level,
    )


def _validate_decoded(truth: Float64Array, prediction: Float64Array) -> None:
    if not (
        np.all(np.isclose(truth, 0.0) | np.isclose(truth, 1.0))
        and np.all(np.isclose(truth.sum(axis=0), 1.0))
    ):
        msg = (
            "held-out targets must decode to known categorical levels; "
            "check the fitted schema and unknown categories"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(prediction)):
        msg = "categorical predictions must be finite in held-out cells"
        raise ValueError(msg)
