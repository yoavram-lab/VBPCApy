"""Core package for the variational Bayesian PCA implementation."""

from vbpca_py.estimators import VBPCA
from vbpca_py.model_selection import SelectionConfig, select_n_components
from vbpca_py.preprocessing import (
    AutoEncoder,
    MissingAwareMinMaxScaler,
    MissingAwareOneHotEncoder,
    MissingAwareSparseOneHotEncoder,
    MissingAwareStandardScaler,
)

__all__ = [
    "VBPCA",
    "AutoEncoder",
    "MissingAwareMinMaxScaler",
    "MissingAwareOneHotEncoder",
    "MissingAwareSparseOneHotEncoder",
    "MissingAwareStandardScaler",
    "SelectionConfig",
    "select_n_components",
]

__version__ = "0.1.0"
