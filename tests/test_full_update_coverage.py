"""Targeted coverage for helper branches in _full_update."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import cho_solve

from vbpca_py._full_update import (
    LoadingsUpdateState,
    ScoreState,
    _build_masks_and_counts,
    _covariances_list,
    _covariances_stack,
    _has_covariances,
    _loadings_accessors,
    _loadings_update_fast_dense_ext,
    _loadings_update_general,
    _loadings_update_sparse_no_patterns,
    _missing_patterns_info,
    _normalize_sparse_data,
    _observed_indices_with_mode,
    _patterns_from_csc,
    _prepare_dense_with_mask_override,
    _prepare_sparse_with_optional_mask,
    _progress_should_log,
    _safe_cholesky,
    _score_accessors,
    _score_update_fast_dense_no_av,
    _score_update_general_dense_ext,
    _update_loadings,
)


def test_covariance_helpers_error_and_progress() -> None:
    with pytest.raises(ValueError):
        _covariances_stack(np.ones((2, 2)))
    with pytest.raises(ValueError):
        _covariances_list(np.ones((2, 2)))
    assert not _progress_should_log(0, 1, 1, 2)
    assert _progress_should_log(1, 2, 2, 3)


def test_prepare_dense_and_sparse_mask_overrides() -> None:
    x_dense = np.array([[0.0, np.nan], [1.0, 2.0]])
    mask_override_sparse = sp.csr_matrix(np.ones_like(x_dense))
    with pytest.raises(ValueError):
        _prepare_dense_with_mask_override(
            x_dense, None, mask_override_sparse, "strict_legacy"
        )

    x_sparse = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    with pytest.raises(ValueError):
        _prepare_sparse_with_optional_mask(
            x_sparse, np.ones((2, 2)), None, "strict_legacy"
        )

    mask_override_dense = np.array([[1, 0], [1, 1]], dtype=bool)
    x_probe = np.array([[0.0, 1.0], [np.nan, 0.0]])
    x_out, x_probe_out, mask_out, mask_probe = _prepare_dense_with_mask_override(
        x_dense, x_probe, mask_override_dense, "strict_legacy"
    )
    assert mask_probe is None
    assert np.allclose(mask_out, mask_override_dense)
    assert np.any(x_out != 0.0)
    assert x_probe_out is not None


def test_observed_indices_and_patterns_modern_dense() -> None:
    x = np.array([[0.0, 5.0], [0.0, 0.0]])
    mask = np.array([[0, 1], [1, 0]], dtype=bool)
    i_idx, j_idx = _observed_indices_with_mode(x, mask, "modern")
    assert set(zip(i_idx.tolist(), j_idx.tolist())) == {(0, 1), (1, 0)}

    opts = {"auto_pattern_masked": True, "compat_mode": "modern"}
    n_patterns, obs_patterns, pattern_index = _missing_patterns_info(
        mask, opts, n_samples=2
    )
    assert n_patterns == 2
    assert pattern_index is not None
    assert len(obs_patterns) == 2

    opts_single = {"auto_pattern_masked": True, "uniquesv": False}
    mask_all = np.ones((2, 2), dtype=bool)
    n_patterns2, obs_patterns2, pattern_index2 = _missing_patterns_info(
        mask_all, opts_single, n_samples=2
    )
    assert n_patterns2 == 2
    assert obs_patterns2 == []
    assert pattern_index2 is None


def test_sparse_accessors_and_updates_cover_branches() -> None:
    x_csr = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    mask_csr = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    state = LoadingsUpdateState(
        x_data=x_csr,
        mask=mask_csr,
        scores=np.array([[1.0, 2.0]]),
        loading_covariances=[np.eye(1), np.eye(1)],
        score_covariances=[np.eye(1), np.eye(1)],
        pattern_index=None,
        va=np.array([1.0]),
        noise_var=0.1,
        verbose=0,
        x_csr=x_csr,
        x_csc=x_csr.tocsc(),
        sparse_num_cpu=1,
        cov_writeback_mode="bulk",
    )
    loadings, covs = _loadings_update_sparse_no_patterns(state)
    assert loadings.shape == (2, 1)
    assert isinstance(covs, np.ndarray)

    # Exercise _update_loadings sparse path with stacking branch.
    state2 = LoadingsUpdateState(
        x_data=x_csr,
        mask=mask_csr,
        scores=np.array([[1.0, 1.0]]),
        loading_covariances=[np.eye(1), np.eye(1)],
        score_covariances=[np.eye(1), np.eye(1)],
        pattern_index=None,
        va=np.array([1.0]),
        noise_var=0.1,
        verbose=0,
        x_csr=x_csr,
        x_csc=x_csr.tocsc(),
        sparse_num_cpu=1,
        cov_writeback_mode="bulk",
    )
    _, covs2 = _update_loadings(state2)
    assert isinstance(covs2, np.ndarray)


def test_dense_fast_paths_with_bulk_writeback() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.ones_like(x)

    score_state = ScoreState(
        x_data=x,
        mask=mask,
        loadings=np.array([[1.0], [0.5]]),
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
        cov_writeback_mode="bulk",
        log_progress_stride=1,
        accessor_mode="legacy",
    )
    score_state = _score_update_general_dense_ext(score_state)
    assert _has_covariances(score_state.score_covariances)

    load_state = LoadingsUpdateState(
        x_data=x,
        mask=mask,
        scores=np.array([[1.0, 1.0]]),
        loading_covariances=[np.eye(1), np.eye(1)],
        score_covariances=[np.eye(1), np.eye(1)],
        pattern_index=None,
        va=np.array([1.0]),
        noise_var=0.1,
        verbose=0,
        x_csr=None,
        x_csc=None,
        dense_num_cpu=1,
        cov_writeback_mode="bulk",
        log_progress_stride=1,
    )
    loadings, covs = _loadings_update_fast_dense_ext(
        state=load_state,
        x_arr=np.asarray(x, dtype=float),
        prior_prec=np.eye(1),
        n_features=2,
    )
    assert loadings.shape == (2, 1)
    assert _has_covariances(covs)


def test_build_masks_and_counts_with_probe_mask() -> None:
    x = np.array([[1.0, np.nan], [0.0, 2.0]])
    x_probe = np.array([[np.nan, 1.0], [1.0, np.nan]])
    mask_override = np.array([[1, 0], [1, 1]], dtype=bool)
    x_data, x_probe_out, mask, mask_probe, n_obs_row, n_data, n_probe = (
        _build_masks_and_counts(
            x,
            x_probe,
            {"init": "random", "compat_mode": "modern"},
            mask_override=mask_override,
        )
    )
    assert mask_probe is None or n_probe >= 0
    assert n_data > 0
    assert n_obs_row.shape[0] == x.shape[0]
    assert x_data.shape == x.shape
    assert x_probe_out is None or x_probe_out.shape == x_probe.shape


def test_sparse_normalization_mask_override_and_patterns() -> None:
    data = np.array([0.0, np.nan, 2.0])
    rows = np.array([0, 0, 1])
    cols = np.array([0, 1, 1])
    x = sp.csr_matrix((data, (rows, cols)), shape=(2, 2))
    normalized = _normalize_sparse_data(x, "strict_legacy")
    assert not np.isnan(normalized.data).any()
    assert np.isclose(normalized.data.min(), np.finfo(float).eps)

    mask_override = sp.csr_matrix([[1, 0], [1, 1]], dtype=float)
    x_out, x_probe_out, mask_out, mask_probe = _prepare_sparse_with_optional_mask(
        x, None, mask_override, "strict_legacy"
    )
    assert mask_probe is None
    assert x_probe_out is None
    assert mask_out.shape == x.shape

    n_patterns, obs_patterns, pattern_index = _patterns_from_csc(mask_out.tocsc())
    assert n_patterns == len(obs_patterns)
    assert pattern_index.shape[0] == mask_out.shape[1]


def test_buffered_accessors_cover_sparse_and_dense() -> None:
    x_csr = sp.csr_matrix([[1.0, 0.0], [0.5, 2.0]])
    mask_csr = sp.csr_matrix([[1.0, 0.0], [1.0, 1.0]])

    score_state = ScoreState(
        x_data=x_csr,
        mask=mask_csr,
        loadings=np.ones((2, 1)),
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
        cov_writeback_mode="python",
        log_progress_stride=1,
        accessor_mode="buffered",
    )
    x_col, mask_col, n_samples = _score_accessors(score_state)
    assert n_samples == 2
    assert x_col(0).shape[0] == 2
    assert mask_col(1).sum() > 0

    load_state = LoadingsUpdateState(
        x_data=x_csr,
        mask=mask_csr,
        scores=np.ones((1, 2)),
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
        cov_writeback_mode="python",
        log_progress_stride=1,
        accessor_mode="buffered",
    )
    x_row, mask_row, n_features = _loadings_accessors(load_state)
    assert n_features == 2
    assert x_row(0).shape[0] == 2
    assert mask_row(1).sum() > 0


def test_safe_cholesky_fallback_uses_jitter() -> None:
    mat = np.zeros((2, 2))
    cho = _safe_cholesky(mat, np.eye(2))
    solved = cho_solve(cho, np.eye(2), check_finite=False)
    assert np.isfinite(solved).all()


def test_score_update_fast_dense_no_av_bulk_writeback() -> None:
    x = np.array([[1.0, 0.0], [0.0, 2.0]])
    mask = np.ones_like(x)
    state = ScoreState(
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
        cov_writeback_mode="bulk",
        log_progress_stride=1,
        accessor_mode="legacy",
    )
    updated = _score_update_fast_dense_no_av(state)
    assert isinstance(updated.score_covariances, list)
    assert len(updated.score_covariances) == x.shape[1]


def test_loadings_update_sparse_bulk_covariances() -> None:
    x_csr = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    mask_csr = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    state = LoadingsUpdateState(
        x_data=x_csr,
        mask=mask_csr,
        scores=np.array([[1.0, 1.0]]),
        loading_covariances=[np.eye(1), np.eye(1)],
        score_covariances=[np.eye(1), np.eye(1)],
        pattern_index=None,
        va=np.array([1.0]),
        noise_var=0.1,
        verbose=0,
        x_csr=x_csr,
        x_csc=x_csr.tocsc(),
        sparse_num_cpu=1,
        cov_writeback_mode="bulk",
    )
    _, covs = _loadings_update_sparse_no_patterns(state)
    assert isinstance(covs, np.ndarray)


def test_loadings_general_caches_mask_pattern() -> None:
    x = np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 1.0]])
    mask = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    scores = np.array([[1.0, 2.0, 3.0]])
    load_cov = [np.eye(1), np.eye(1)]
    score_cov = [np.eye(1), np.eye(1), np.eye(1)]
    state = LoadingsUpdateState(
        x_data=x,
        mask=mask,
        scores=scores,
        loading_covariances=load_cov,
        score_covariances=score_cov,
        pattern_index=None,
        va=np.array([1.0]),
        noise_var=0.1,
        verbose=0,
        x_csr=None,
        x_csc=None,
        sparse_num_cpu=0,
        dense_num_cpu=1,
        cov_writeback_mode="python",
        log_progress_stride=2,
        accessor_mode="buffered",
    )
    loadings, covs = _loadings_update_general(state)
    assert loadings.shape == (2, 1)
    assert isinstance(covs, list)
    assert covs[0] is not covs[1]
