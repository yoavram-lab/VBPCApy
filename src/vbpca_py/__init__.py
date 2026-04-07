"""Core package for the variational Bayesian PCA implementation."""

from importlib.metadata import version

from vbpca_py.estimators import VBPCA
from vbpca_py.model_selection import (
    CVConfig,
    SelectionConfig,
    cross_validate_components,
    select_n_components,
)
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
    "CVConfig",
    "MissingAwareMinMaxScaler",
    "MissingAwareOneHotEncoder",
    "MissingAwareSparseOneHotEncoder",
    "MissingAwareStandardScaler",
    "SelectionConfig",
    "cross_validate_components",
    "select_n_components",
]

__version__ = version("vbpca_py")
