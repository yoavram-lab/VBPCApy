#!/usr/bin/env python
"""Select capacity with training-only preprocessing of mixed raw variable cells."""

from __future__ import annotations

import logging

import numpy as np

from vbpca_py import AutoEncoder, CVConfig, cross_validate_raw_components

log = logging.getLogger(__name__)


def make_encoder() -> AutoEncoder:
    """Declare variable kinds and external codebook order, before observing data.

    Returns:
        A fresh encoder for one CV training fold.
    """
    return AutoEncoder(
        column_types=["categorical", "categorical", "continuous", "ordinal"],
        column_levels=[
            ["west", "east", "north"],
            ["no", "yes"],
            None,
            ["low", "medium", "high"],
        ],
        binary="single",
        drop="first",
        mean_center_ohe=True,
        block_weighting="equal_variance",
        handle_unknown="raise",
    )


def main() -> None:
    """Run reproducible variable-cell reconstruction CV."""
    rng = np.random.default_rng(29)
    x = np.empty((60, 4), dtype=object)  # samples x original variables
    x[:, 0] = rng.choice(["west", "east", "north"], 60)
    x[:, 1] = rng.choice(["no", "yes"], 60)
    x[:, 2] = rng.normal(100, 15, 60)
    x[:, 3] = rng.choice(["low", "medium", "high"], 60)
    x[rng.random(x.shape) < 0.1] = np.nan
    best, results, folds = cross_validate_raw_components(
        x,
        encoder_factory=make_encoder,
        components=range(4),
        config=CVConfig(
            n_splits=3,
            metric="brier",
            seed=11,
            selection_rule="first_minimum",
            early_stop=True,
        ),
        maxiters=120,
        niter_broadprior=20,
        runtime_tuning="off",
    )
    record = next(row for row in results if row["k"] == best)
    log.info("Selected capacity: %d; Brier score: %.4f", best, record["mean_brier"])
    log.info("Continuous RMSE in input units: %.4f", record["mean_rmse_variable_2"])
    log.info("Ordinal RMSE in level-index units: %.4f", record["mean_rmse_variable_3"])
    for index, fold in enumerate(folds):
        variable = fold.schema.variables[2]
        log.info(
            "Fold %d training mean: %.4f; scale: %.4f",
            index,
            variable.offset[0],
            1.0 / variable.weight,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
