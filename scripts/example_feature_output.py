"""Keep names and row labels through a masked mixed-data Pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from vbpca_py import VBPCA, AutoEncoder, ObservationMasker


def main() -> None:
    data = pd.DataFrame(
        {
            "region": np.tile(["west", "east", "north"], 8),
            "size": np.arange(24.0) + 100,
        },
        index=[f"sample-{i}" for i in range(24)],
    )
    mask = np.ones(data.shape, dtype=bool)
    mask[::4, 0] = False
    mask[1::4, 1] = False
    pipeline = Pipeline([
        ("observe", ObservationMasker()),
        ("encode", AutoEncoder(column_types=["categorical", "continuous"])),
        (
            "model",
            VBPCA(
                1,
                maxiters=4,
                niter_broadprior=0,
                verbose=0,
                random_state=23,
                num_cpu=1,
                runtime_tuning="off",
            ),
        ),
    ]).set_output(transform="pandas")
    with config_context(enable_metadata_routing=True):
        pandas_scores = pipeline.fit_transform(data, observation_mask=mask)
        if not pandas_scores.index.equals(data.index):
            raise RuntimeError("The Pipeline changed row labels")
        if pandas_scores.columns.tolist() != ["vbpca0"]:
            raise RuntimeError("The Pipeline component names are incorrect")
        polars_pipeline = clone(pipeline).set_output(transform="polars")
        # Keep the raw object/NaN step in pandas; numerical outputs use Polars.
        polars_pipeline.named_steps["observe"].set_output(transform="pandas")
        polars_scores = polars_pipeline.fit_transform(data, observation_mask=mask)
        np.testing.assert_allclose(polars_scores.to_numpy(), pandas_scores.to_numpy())
    print("Named pandas/Polars Pipeline outputs preserve values and row order")


if __name__ == "__main__":
    main()
