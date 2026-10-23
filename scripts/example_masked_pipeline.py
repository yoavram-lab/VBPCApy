"""Route raw observation masks through feature expansion and row-wise CV."""

from __future__ import annotations

import numpy as np
from sklearn import config_context
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from vbpca_py import VBPCA, AutoEncoder, ObservationMasker


def main() -> None:
    rng = np.random.default_rng(23)
    raw = np.empty((36, 2), dtype=object)
    raw[:, 0] = np.tile(["west", "east", "north"], 12)
    raw[:, 1] = rng.normal(100, 5, len(raw))
    mask = rng.random(raw.shape) > 0.1
    data = np.where(mask, raw, "unobserved placeholder")
    pipeline = Pipeline([
        ("observe", ObservationMasker()),
        (
            "encode",
            AutoEncoder(column_types=["categorical", "continuous"], binary="both"),
        ),
        (
            "model",
            VBPCA(
                1,
                maxiters=8,
                niter_broadprior=0,
                verbose=0,
                random_state=23,
                num_cpu=1,
                runtime_tuning="off",
            ),
        ),
    ])
    with config_context(enable_metadata_routing=True):
        result = cross_validate(
            pipeline,
            data,
            cv=3,
            params={"observation_mask": mask},
            error_score="raise",
        )
        search = GridSearchCV(
            pipeline, {"model__n_components": [0, 1]}, cv=3, error_score="raise"
        ).fit(data, observation_mask=mask)
        projected = search.best_estimator_.transform(
            data[:4], observation_mask=mask[:4]
        )
    if not np.isfinite(result["test_score"]).all():
        raise RuntimeError("The synthetic CV example returned a nonfinite score")
    if projected.shape != (4, search.best_params_["model__n_components"]):
        raise RuntimeError("The synthetic projection has the wrong shape")
    print("Masked Pipeline, row CV, GridSearchCV and new-row projection passed")


if __name__ == "__main__":
    main()
