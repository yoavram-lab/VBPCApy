"""Core package for the variational Bayesian PCA implementation."""

from vbpca_py.estimators import VBPCA
from vbpca_py.model_selection import SelectionConfig, select_n_components
from vbpca_py.preprocessing import (
    AutoEncoder,
    MissingAwareMinMaxScaler,
    MissingAwareOneHotEncoder,
    MissingAwareStandardScaler,
)

__all__ = [
    "VBPCA",
    "select_n_components",
    "SelectionConfig",
    "AutoEncoder",
    "MissingAwareOneHotEncoder",
    "MissingAwareStandardScaler",
    "MissingAwareMinMaxScaler",
]

__version__ = "0.1.0"
