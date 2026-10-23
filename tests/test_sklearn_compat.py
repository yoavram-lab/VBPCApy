import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn import config_context
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils.metadata_routing import MetadataRequest

from vbpca_py import ObservationMasker
from vbpca_py.estimators import VBPCA
from vbpca_py.preprocessing import (
    AutoEncoder,
    MissingAwareMinMaxScaler,
    MissingAwareOneHotEncoder,
    MissingAwareSparseOneHotEncoder,
    MissingAwareStandardScaler,
)


def test_sklearn_pipeline_roundtrip():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 5))

    scaler = MissingAwareStandardScaler()
    X_scaled = scaler.fit_transform(X)

    pipe = Pipeline([("scaler", MissingAwareStandardScaler())])
    pipe_scaled = pipe.fit_transform(X)
    np.testing.assert_allclose(X_scaled, pipe_scaled)


def test_sklearn_clone():
    estimators = [
        VBPCA(n_components=2, bias=False, maxiters=10),
        MissingAwareStandardScaler(),
        MissingAwareMinMaxScaler(),
        MissingAwareOneHotEncoder(handle_unknown="ignore"),
        MissingAwareSparseOneHotEncoder(),
        AutoEncoder(cardinality_threshold=10),
    ]
    for est in estimators:
        cloned = clone(est)
        assert est.get_params() == cloned.get_params()


class _RowScore:
    """A public-routing consumer that requests raw masks and scoring weights."""

    def __init__(self, *, weighted):
        self.weighted = weighted

    def get_metadata_routing(self):
        request = MetadataRequest(owner=type(self).__name__)
        request.score.add_request(param="observation_mask", alias=True)
        request.score.add_request(param="sample_weight", alias=self.weighted)
        return request

    def __call__(
        self, estimator, X, y=None, *, observation_mask=None, sample_weight=None
    ):
        del y
        assert observation_mask.shape == X.shape
        if self.weighted:
            assert sample_weight.shape == (len(X),)
        return estimator.score(
            X, observation_mask=observation_mask, sample_weight=sample_weight
        )


@pytest.mark.parametrize("output", ["default", "pandas", "polars"])
def test_grouped_cv_and_multimetric_grid_route_masks_and_scoring_weights(output):
    import pandas as pd

    numeric = np.random.default_rng(39).normal(size=48)
    raw = pd.DataFrame(
        {"category": np.tile(["a", "b", "c"], 16), "value": numeric},
        index=np.arange(48) + 100,
    )
    mask = np.ones(raw.shape, dtype=bool)
    mask[2, 0] = False
    mask[8, 1] = False
    raw.iloc[2, 0] = "masked_only"
    raw.iloc[8, 1] = np.inf
    weights = np.linspace(0.1, 2.0, len(raw))
    groups = np.repeat(np.arange(6), 8)
    splitter = GroupKFold(3)
    options = {
        "maxiters": 5,
        "niter_broadprior": 0,
        "runtime_tuning": "off",
        "num_cpu": 1,
        "random_state": 39,
        "return_diagnostics": False,
        "verbose": 0,
    }
    with config_context(enable_metadata_routing=True):
        pipe = Pipeline([
            (
                "mask",
                ObservationMasker()
                .set_fit_request(mask="observation_mask")
                .set_transform_request(mask="observation_mask"),
            ),
            (
                "encode",
                AutoEncoder(
                    column_types=["categorical", "continuous"],
                    column_levels=[["a", "b", "c"], None],
                    handle_unknown="raise",
                ),
            ),
            ("pca", VBPCA(1, **options).set_score_request(sample_weight=True)),
        ]).set_output(transform=output)
        if output == "polars":
            pipe.named_steps["mask"].set_output(transform="pandas")
        scoring = {
            "weighted": _RowScore(weighted=True),
            "plain": _RowScore(weighted=False),
        }
        metadata = {
            "observation_mask": mask,
            "sample_weight": weights,
            "groups": groups,
        }
        result = cross_validate(
            pipe,
            raw,
            cv=splitter,
            scoring=scoring,
            params=metadata,
            return_estimator=True,
            error_score="raise",
        )
        expected = {"weighted": [], "plain": []}
        for fitted, (train, test) in zip(
            result["estimator"], splitter.split(raw, groups=groups), strict=True
        ):
            encoder = fitted.named_steps["encode"]
            expected_mean = numeric[train][mask[train, 1]].mean()
            assert encoder.encoding_schema_.variables[1].offset[0] == pytest.approx(
                expected_mean
            )
            canonical = raw.iloc[test].to_numpy(copy=True)
            canonical[~mask[test]] = np.nan
            if hasattr(encoder, "feature_names_in_"):
                canonical = pd.DataFrame(canonical, columns=raw.columns)
            encoded = encoder.transform(canonical)
            values = fitted.named_steps["pca"].score_samples(encoded)
            expected["weighted"].append(np.average(values, weights=weights[test]))
            expected["plain"].append(values.mean())
        for name, reference in expected.items():
            np.testing.assert_allclose(
                result[f"test_{name}"], reference, rtol=3e-12, atol=3e-12
            )
        search = GridSearchCV(
            pipe,
            {"pca__n_components": [0, 1]},
            cv=splitter,
            scoring=scoring,
            refit="weighted",
            error_score="raise",
        ).fit(raw, **metadata)
        np.testing.assert_allclose(
            search.cv_results_["mean_test_weighted"][1], np.mean(expected["weighted"])
        )
        with pytest.raises(TypeError, match="unexpected keyword"):
            search.transform(raw, observation_mask=mask)
        transformed = search.best_estimator_.transform(raw, observation_mask=mask)
        assert transformed.shape == (
            len(raw),
            search.best_estimator_.named_steps["pca"].components_.shape[0],
        )
        assert np.isfinite(
            search.score(raw, observation_mask=mask, sample_weight=weights)
        )
