"""Blocked, full-covariance posterior uncertainty contractions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._full_update import CovarianceStore
    from ._pca_full import FinalState


def _covariance_block(
    covariances: CovarianceStore,
    start: int,
    stop: int,
    patterns: np.ndarray | None = None,
) -> np.ndarray:
    if patterns is None:
        return np.asarray(covariances[start:stop], dtype=float)
    return np.asarray([covariances[int(j)] for j in patterns[start:stop]], dtype=float)


def marginal_variance(  # noqa: PLR0914  # Named blocks mirror the covariance equation.
    final: FinalState,
    *,
    block_size: int = 128,
) -> np.ndarray:
    """Evaluate posterior variance using two GEMMs per rectangular block.

    Off-diagonal covariances and mean uncertainty are retained. Observation
    noise belongs to predictive variance and is added by the caller. Only the
    output spans both complete data axes; covariance packing and outer products
    use at most ``block_size`` features or samples at a time. Scratch storage
    is O(block_size * rank**2 + block_size**2), excluding BLAS workspace.

    Returns:
        Full feature-by-sample posterior variance.

    Raises:
        ValueError: If ``block_size`` is not positive.
    """
    if block_size <= 0:
        msg = "block_size must be positive"
        raise ValueError(msg)
    loadings = np.asarray(final.a, dtype=float)
    scores = np.asarray(final.s, dtype=float)
    av, sv, patterns = final.av, final.sv, final.pattern_index
    features, rank = loadings.shape
    samples = scores.shape[1]
    result = np.empty((features, samples), dtype=float)
    mean_variance = np.asarray(final.muv, dtype=float).reshape(-1)
    for i in range(0, features, block_size):
        end_i = min(features, i + block_size)
        a = loadings[i:end_i]
        aa = np.einsum("ik,il->ikl", a, a).reshape(end_i - i, rank**2)
        av_block = (
            _covariance_block(av, i, end_i).reshape(end_i - i, rank**2)
            if len(av)
            else None
        )
        for j in range(0, samples, block_size):
            end_j = min(samples, j + block_size)
            tile = result[i:end_i, j:end_j]
            sv_block = (
                _covariance_block(sv, j, end_j, patterns).reshape(end_j - j, rank**2)
                if len(sv)
                else None
            )
            if sv_block is None:
                tile.fill(0.0)
            else:
                np.matmul(aa, sv_block.T, out=tile)
            if av_block is not None:
                s = scores[:, j:end_j]
                second_moment = np.einsum("kj,lj->jkl", s, s).reshape(
                    end_j - j, rank**2
                )
                if sv_block is not None:
                    second_moment += sv_block
                tile += av_block @ second_moment.T
            if mean_variance.size:
                tile += mean_variance[i:end_i, None]
    return result
