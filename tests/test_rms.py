"""Unit tests for ``vbpca_py._rms`` branch/error behavior."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py._rms import RmsConfig, compute_rms


def test_compute_rms_dense_requires_mask() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    loadings = np.eye(2, dtype=float)
    scores = np.eye(2, dtype=float)

    with pytest.raises(ValueError, match="mask must be provided"):
        compute_rms(data, loadings, scores, mask=None, config=RmsConfig())


def test_compute_rms_dense_n_observed_mismatch_raises() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    loadings = np.eye(2, dtype=float)
    scores = np.eye(2, dtype=float)
    mask = np.ones_like(data, dtype=float)

    with pytest.raises(ValueError, match="n_observed mismatch"):
        compute_rms(
            data,
            loadings,
            scores,
            mask=mask,
            config=RmsConfig(n_observed=1),
        )


def test_compute_rms_sparse_mask_structure_mismatch_sparse_mask_raises() -> None:
    data = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float))
    # Different structure than `data`.
    wrong_mask = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float))
    loadings = np.eye(2, dtype=float)
    scores = np.eye(2, dtype=float)

    with pytest.raises(ValueError, match="sparsity pattern"):
        compute_rms(
            data,
            loadings,
            scores,
            mask=wrong_mask,
            config=RmsConfig(validate_sparse_mask=True),
        )


def test_compute_rms_sparse_mask_structure_mismatch_dense_mask_raises() -> None:
    data = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float))
    wrong_dense_mask = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=float)
    loadings = np.eye(2, dtype=float)
    scores = np.eye(2, dtype=float)

    with pytest.raises(ValueError, match=r"mask must be sparse when data is sparse"):
        compute_rms(
            data,
            loadings,
            scores,
            mask=wrong_dense_mask,
            config=RmsConfig(validate_sparse_mask=True),
        )


def test_compute_rms_sparse_n_observed_mismatch_raises() -> None:
    data = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float))
    loadings = np.eye(2, dtype=float)
    scores = np.eye(2, dtype=float)

    with pytest.raises(ValueError, match="n_observed mismatch"):
        compute_rms(
            data,
            loadings,
            scores,
            mask=None,
            config=RmsConfig(n_observed=1, validate_sparse_mask=False),
        )
