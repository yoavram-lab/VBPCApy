import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from vbpca_py.estimators import VBPCA
from vbpca_py.preprocessing import (
    AutoEncoder,
    MissingAwareStandardScaler,
    MissingAwareMinMaxScaler,
    MissingAwareOneHotEncoder,
    MissingAwareSparseOneHotEncoder,
)

def test_sklearn_pipeline_roundtrip():
    np.random.seed(42)
    X = np.random.randn(10, 5)
    
    scaler = MissingAwareStandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pipe = Pipeline([
        ("scaler", MissingAwareStandardScaler())
    ])
    pipe_scaled = pipe.fit_transform(X)
    np.testing.assert_allclose(X_scaled, pipe_scaled)

def test_sklearn_clone():
    estimators = [
        VBPCA(n_components=2, bias=False, maxiters=10),
        MissingAwareStandardScaler(),
        MissingAwareMinMaxScaler(),
        MissingAwareOneHotEncoder(handle_unknown="ignore"),
        MissingAwareSparseOneHotEncoder(),
        AutoEncoder(cardinality_threshold=10)
    ]
    for est in estimators:
        cloned = clone(est)
        assert est.get_params() == cloned.get_params()
