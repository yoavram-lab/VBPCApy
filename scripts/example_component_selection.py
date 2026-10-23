"""Run native component selection and conditional projection from public APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vbpca_py import (
    CVConfig,
    SelectionConfig,
    cross_validate_components,
    select_n_components,
)


def main() -> None:
    rng = np.random.default_rng(39)
    values = rng.normal(size=(30, 4))
    values[0, 0] = 0.0
    values[1, 1] = np.nan
    observations = pd.DataFrame(values, columns=["north", "south", "east", "west"])
    probe = observations.copy()
    probe.iloc[:, :] = np.nan
    for row, column in [(3, 0), (5, 1), (7, 2)]:
        probe.iloc[row, column] = observations.iloc[row, column]
    options = {
        "maxiters": 8,
        "niter_broadprior": 0,
        "num_cpu": 1,
        "runtime_tuning": "off",
        "random_state": 39,
        "verbose": 0,
    }
    chosen, metrics, trace, model = select_n_components(
        observations,
        xprobe=probe,
        components=[1, 2],
        config=SelectionConfig(metric="prms", return_best_model=True),
        **options,
    )
    if model is None or model.components_.shape != (chosen, observations.shape[1]):
        msg = "positive-rank selection must return a native fitted model"
        raise RuntimeError(msg)
    model.set_output(transform="pandas")
    projected = model.transform(observations.iloc[:3])
    if list(projected.columns) != model.get_feature_names_out().tolist():
        msg = "projection columns must match the fitted latent names"
        raise RuntimeError(msg)
    if model.inverse_transform(projected).shape != (3, observations.shape[1]):
        msg = "reconstruction must retain the native sample/feature axes"
        raise RuntimeError(msg)
    print("Probe selection:", chosen, metrics["prms"], [entry["k"] for entry in trace])
    cv_chosen, records = cross_validate_components(
        observations, components=[0, 1], config=CVConfig(n_splits=2, seed=39), **options
    )
    print("Cellwise CV:", cv_chosen, [entry["mean_prms"] for entry in records])


if __name__ == "__main__":
    main()
