"""
vpbcapy.

Python translation of Illin & Raiko (2010) Variational Bayesian PCA,
with support for missing data, unique-pattern covariance structures,
and C++-accelerated sparse mean subtraction.
"""

from .pca_full import pca_full

__all__ = [
    "pca_full",
]
