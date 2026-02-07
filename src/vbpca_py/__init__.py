"""Core package for the variational Bayesian PCA implementation."""

from vbpca_py.estimators import VBPCA
from vbpca_py.model_selection import select_n_components

__all__ = [
    "VBPCA",
    "select_n_components",
]
