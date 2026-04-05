"""Runtime robustness tests for compiled helpers and low-level paths."""

from __future__ import annotations

import numpy as np
import pytest

from vbpca_py.subtract_mu_from_sparse import subtract_mu_from_sparse


def test_subtract_mu_from_sparse_smoke() -> None:
    # CSR for [[1, 0], [0, 2]]
    data = np.array([1.0, 2.0], dtype=np.float64)
    indices = np.array([0, 1], dtype=np.int32)
    indptr = np.array([0, 1, 2], dtype=np.int32)
    shape = (2, 2)
    mu = np.array([1.0, 2.0], dtype=np.float64)

    out = subtract_mu_from_sparse(data, indices, indptr, shape, mu)
    assert out.shape == data.shape
    assert np.all(np.isfinite(out))
    assert np.all(out != 0.0)  # exact-by-construction


def test_subtract_mu_from_sparse_rejects_too_short_mu() -> None:
    # CSR for [[1], [2]]
    data = np.array([1.0, 2.0], dtype=np.float64)
    indices = np.array([0, 0], dtype=np.int32)
    indptr = np.array([0, 1, 2], dtype=np.int32)
    shape = (2, 1)
    mu = np.array([1.0], dtype=np.float64)  # too short for n_rows=2

    with pytest.raises(Exception):
        subtract_mu_from_sparse(data, indices, indptr, shape, mu)


def test_subtract_mu_from_sparse_allows_longer_mu() -> None:
    data = np.array([1.0], dtype=np.float64)
    indices = np.array([0], dtype=np.int32)
    indptr = np.array([0, 1], dtype=np.int32)
    shape = (1, 1)
    mu = np.array([1.0, 2.0], dtype=np.float64)  # valid: len(mu) >= n_rows

    out = subtract_mu_from_sparse(data, indices, indptr, shape, mu)
    assert out.shape == data.shape
