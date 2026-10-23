"""Conditional row projection and plug-in Gaussian scoring of fixed factors."""

from __future__ import annotations

from collections.abc import Iterator  # noqa: TC003 - runtime annotation resolution
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve

from ._memory import check_dense_output_budget
from ._types import Float64Array, IndexArray, Matrix  # noqa: TC001 - runtime annotation resolution


@dataclass(frozen=True, slots=True)
class ProjectionPosterior:
    """Conditional score means (n,q) and covariance banks (n,q,q).

    The projection owns fresh arrays, independent of the fitted state. Frozen
    fields prevent rebinding; array payloads remain writable. Shapes are checked
    at construction. Direct record construction borrows supplied arrays; the
    projection methods allocate owned outputs. Pickle/deepcopy use ordinary
    NumPy serialization.
    """

    mean: Float64Array
    covariance: Float64Array

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mean, np.ndarray)
            or not isinstance(self.covariance, np.ndarray)
            or self.mean.dtype != np.float64
            or self.covariance.dtype != np.float64
        ):
            msg = "projection means/covariances must be float64 arrays"
            raise TypeError(msg)
        if self.mean.ndim != 2 or self.covariance.shape != (
            self.mean.shape[0],
            self.mean.shape[1],
            self.mean.shape[1],
        ):
            msg = "projection means/covariances must have shapes (n,q)/(n,q,q)"
            raise ValueError(msg)


def observed_rows(data: Matrix) -> Iterator[tuple[IndexArray, Float64Array]]:
    """Yield each row's observed feature indices and values, retaining zeros."""
    if sp.issparse(data):
        matrix = sp.csr_matrix(data)
        for row in range(matrix.shape[0]):
            start, stop = matrix.indptr[row : row + 2]
            yield (
                np.asarray(matrix.indices[start:stop], dtype=np.intp),
                matrix.data[start:stop],
            )
    else:
        for values in data:
            indices = np.flatnonzero(~np.isnan(values))
            yield indices, values[indices]


def project_rows(  # noqa: PLR0913 - fixed posterior parameters and output controls
    data: Matrix,
    loadings: Float64Array,
    mean: Float64Array,
    noise: float,
    loading_covariances: list[Float64Array] | None,
    *,
    covariance: bool,
    max_dense_bytes: int | None,
) -> tuple[Float64Array, Float64Array | None]:
    """Condition independent unit-normal latent scores on observed features.

    Full VB loading second moments enter the precision. Ordinary transform
    allocates only its score means, without an n x q x q covariance output.

    Returns:
        Score means and the optional covariance bank.
    """
    n, q = data.shape[0], loadings.shape[1]
    check_dense_output_budget(
        (n, q + q * q if covariance else q),
        max_dense_bytes,
        operation="Conditional projection outputs",
    )
    means = np.zeros((n, q), dtype=np.float64)
    covariances = np.empty((n, q, q), dtype=np.float64) if covariance else None
    identity = np.eye(q)
    for row, (indices, values) in enumerate(observed_rows(data)):
        precision = identity.copy()
        selected = loadings[indices]
        if indices.size and q:
            precision += selected.T @ selected / noise
            if loading_covariances is not None:
                for feature in indices:
                    precision += loading_covariances[int(feature)] / noise
            factor = cho_factor(precision, lower=True, check_finite=False)
            rhs = selected.T @ (values - mean[indices]) / noise
            means[row] = cho_solve(factor, rhs, check_finite=False)
            if covariances is not None:
                covariances[row] = cho_solve(factor, identity, check_finite=False)
        elif covariances is not None:
            covariances[row] = identity
    return means, covariances


def row_log_likelihood(
    data: Matrix, loadings: Float64Array, mean: Float64Array, noise: float
) -> tuple[Float64Array, int]:
    """Compute plug-in row log densities using q x q solves, without p x p arrays.

    Empty rows contribute zero. This integrates unit-normal latent scores under
    point loading means; it is not the VB objective or a fully uncertainty-
    integrated posterior predictive likelihood.

    Returns:
        Per-row log densities and the total observed-entry count.
    """
    scores = np.zeros(data.shape[0], dtype=np.float64)
    total_observed = 0
    q = loadings.shape[1]
    for row, (indices, values) in enumerate(observed_rows(data)):
        count = indices.size
        if not count:
            continue
        total_observed += int(count)
        selected = loadings[indices]
        residual = values - mean[indices]
        logdet = count * np.log(noise)
        if q:
            precision = np.eye(q) + selected.T @ selected / noise
            factor = cho_factor(precision, lower=True, check_finite=False)
            latent = cho_solve(
                factor, selected.T @ residual / noise, check_finite=False
            )
            unexplained = residual - selected @ latent
            quadratic = float(unexplained @ unexplained / noise + latent @ latent)
            logdet += 2.0 * np.log(np.diag(factor[0])).sum()
        else:
            quadratic = float(residual @ residual / noise)
        scores[row] = -0.5 * (count * np.log(2.0 * np.pi) + logdet + quadratic)
    return scores, total_observed
