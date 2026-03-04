"""Regression tests for covariance writeback modes (python vs kernel/bulk)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from vbpca_py._full_update import (
    LoadingsUpdateState,
    ScoreState,
    _loadings_update_dense_masked_ext,
    _loadings_update_sparse_ext_apply,
    _score_update_dense_masked_ext,
    _score_update_sparse_ext_apply,
)


def _dense_score_state(cov_mode: str) -> ScoreState:
    x = np.array([[1.0, 0.0], [0.5, 2.0]])
    mask = np.ones_like(x, dtype=float)
    return ScoreState(
        x_data=x,
        mask=mask,
        loadings=np.array([[1.0], [1.0]]),
        scores=np.zeros((1, 2)),
        loading_covariances=[],
        score_covariances=[np.eye(1), np.eye(1)],
        pattern_index=None,
        obs_patterns=[],
        noise_var=0.1,
        eye_components=np.eye(1),
        verbose=0,
        pattern_batch_size=0,
        x_csr=None,
        x_csc=None,
        sparse_num_cpu=0,
        dense_num_cpu=1,
        cov_writeback_mode=cov_mode,
        log_progress_stride=1,
        accessor_mode="legacy",
    )


def _sparse_score_state(cov_mode: str) -> ScoreState:
    x_csr = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    mask_csr = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    return ScoreState(
        x_data=x_csr,
        mask=mask_csr,
        loadings=np.array([[1.0], [1.0]]),
        scores=np.zeros((1, 2)),
        loading_covariances=[],
        score_covariances=[np.eye(1), np.eye(1)],
        pattern_index=None,
        obs_patterns=[],
        noise_var=0.1,
        eye_components=np.eye(1),
        verbose=0,
        pattern_batch_size=0,
        x_csr=x_csr,
        x_csc=None,
        sparse_num_cpu=1,
        dense_num_cpu=0,
        cov_writeback_mode=cov_mode,
        log_progress_stride=1,
        accessor_mode="buffered",
    )


def _dense_loading_state(cov_mode: str) -> LoadingsUpdateState:
    x = np.array([[1.0, 0.0], [0.5, 2.0]])
    mask = np.ones_like(x, dtype=float)
    scores = np.array([[1.0, 1.0]])
    return LoadingsUpdateState(
        x_data=x,
        mask=mask,
        scores=scores,
        loading_covariances=[np.eye(1), np.eye(1)],
        score_covariances=[np.eye(1), np.eye(1)],
        pattern_index=None,
        va=np.array([1.0]),
        noise_var=0.1,
        verbose=0,
        x_csr=None,
        x_csc=None,
        sparse_num_cpu=0,
        dense_num_cpu=1,
        cov_writeback_mode=cov_mode,
        log_progress_stride=1,
    )


def _sparse_loading_state(cov_mode: str) -> LoadingsUpdateState:
    x_csr = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    mask_csr = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    scores = np.array([[1.0, 1.0]])
    return LoadingsUpdateState(
        x_data=x_csr,
        mask=mask_csr,
        scores=scores,
        loading_covariances=[np.eye(1), np.eye(1)],
        score_covariances=[np.eye(1), np.eye(1)],
        pattern_index=None,
        va=np.array([1.0]),
        noise_var=0.1,
        verbose=0,
        x_csr=x_csr,
        x_csc=x_csr.tocsc(),
        sparse_num_cpu=1,
        dense_num_cpu=0,
        cov_writeback_mode=cov_mode,
        log_progress_stride=1,
    )


def test_dense_score_covariances_python_vs_kernel() -> None:
    state_py = _dense_score_state("python")
    res_py = _score_update_dense_masked_ext(
        x_data=np.asarray(state_py.x_data, dtype=np.float64),
        mask=np.asarray(state_py.mask, dtype=np.float64),
        loadings=np.asarray(state_py.loadings, dtype=np.float64),
        loading_covariances=None,
        noise_var=float(state_py.noise_var),
        return_covariances=True,
        num_cpu=int(state_py.dense_num_cpu),
    )
    covs_arr = np.asarray(res_py["score_covariances"], dtype=float)
    covs_py = [covs_arr[j, :, :] for j in range(covs_arr.shape[0])]
    assert isinstance(covs_py, list)
    assert covs_py[0] is not covs_py[1]

    state_kernel = _dense_score_state("kernel")
    res = _score_update_dense_masked_ext(
        x_data=np.asarray(state_kernel.x_data, dtype=np.float64),
        mask=np.asarray(state_kernel.mask, dtype=np.float64),
        loadings=np.asarray(state_kernel.loadings, dtype=np.float64),
        loading_covariances=None,
        noise_var=float(state_kernel.noise_var),
        return_covariances=True,
        num_cpu=int(state_kernel.dense_num_cpu),
    )
    state_kernel.score_covariances = np.asarray(res["score_covariances"], dtype=float)
    covs_kernel = state_kernel.score_covariances
    assert isinstance(covs_kernel, np.ndarray)
    assert covs_kernel.shape[0] == len(covs_py)
    assert np.allclose(covs_kernel[0], covs_py[0])


def test_sparse_score_covariances_python_vs_kernel() -> None:
    state_py = _sparse_score_state("python")
    _score_update_sparse_ext_apply(state_py, state_py.x_csr.tocsc())
    covs_py = state_py.score_covariances
    assert isinstance(covs_py, list)
    assert covs_py[0] is not covs_py[1]

    state_kernel = _sparse_score_state("kernel")
    _score_update_sparse_ext_apply(state_kernel, state_kernel.x_csr.tocsc())
    covs_kernel = state_kernel.score_covariances
    assert isinstance(covs_kernel, np.ndarray)
    assert covs_kernel.shape[0] == len(covs_py)
    assert np.allclose(covs_kernel[0], covs_py[0])


def test_dense_loading_covariances_python_vs_kernel() -> None:
    state_py = _dense_loading_state("python")
    res_py = _loadings_update_dense_masked_ext(
        x_data=np.asarray(state_py.x_data, dtype=np.float64),
        mask=np.asarray(state_py.mask, dtype=np.float64),
        scores=np.asarray(state_py.scores, dtype=np.float64),
        score_covariances=np.stack(state_py.score_covariances),
        prior_prec=np.eye(1),
        noise_var=float(state_py.noise_var),
        return_covariances=True,
        num_cpu=int(state_py.dense_num_cpu),
    )
    state_py.loading_covariances = [
        res_py["loading_covariances"][i, :, :] for i in range(2)
    ]
    covs_py = state_py.loading_covariances
    assert covs_py[0] is not covs_py[1]

    state_kernel = _dense_loading_state("kernel")
    res_kernel = _loadings_update_dense_masked_ext(
        x_data=np.asarray(state_kernel.x_data, dtype=np.float64),
        mask=np.asarray(state_kernel.mask, dtype=np.float64),
        scores=np.asarray(state_kernel.scores, dtype=np.float64),
        score_covariances=np.stack(state_kernel.score_covariances),
        prior_prec=np.eye(1),
        noise_var=float(state_kernel.noise_var),
        return_covariances=True,
        num_cpu=int(state_kernel.dense_num_cpu),
    )
    state_kernel.loading_covariances = np.asarray(
        res_kernel["loading_covariances"], dtype=float
    )
    covs_kernel = state_kernel.loading_covariances
    assert isinstance(covs_kernel, np.ndarray)
    assert np.allclose(covs_kernel[0], covs_py[0])


def test_sparse_loading_covariances_python_vs_kernel() -> None:
    state_py = _sparse_loading_state("python")
    _loadings_update_sparse_ext_apply(state_py, state_py.x_csr)
    covs_py = state_py.loading_covariances
    assert isinstance(covs_py, list)
    assert covs_py[0] is not covs_py[1]

    state_kernel = _sparse_loading_state("kernel")
    _loadings_update_sparse_ext_apply(state_kernel, state_kernel.x_csr)
    covs_kernel = state_kernel.loading_covariances
    assert isinstance(covs_kernel, np.ndarray)
    assert np.allclose(covs_kernel[0], covs_py[0])
