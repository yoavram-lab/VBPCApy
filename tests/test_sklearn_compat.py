import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.pipeline import Pipeline

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
