"""Tests for cov_writeback_mode autotuning integration."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from vbpca_py._pca_full import (
    _build_options,
    _initialize_model,
    _prepare_problem,
    _resolve_runtime_threads_for_training,
    _select_algorithm,
)


def _prepare_state(x, *, runtime_tuning: str = "safe", sparse: bool = False):
    opts = _build_options({
        "maxiters": 0,
        "runtime_tuning": runtime_tuning,
        "cov_writeback_mode": "auto",
        "num_cpu": 1,
    })
    use_prior, use_postvar = _select_algorithm(opts)
    prepared = _prepare_problem(x if not sparse else sp.csr_matrix(x), opts)
    training = _initialize_model(
        prepared=prepared,
        n_components=1,
        use_prior=use_prior,
        use_postvar=use_postvar,
        opts=opts,
    )
    return prepared, training, opts


def test_cov_writeback_autotune_dense_sets_mode_and_source() -> None:
    x = np.array([[1.0, 2.0, 3.0], [0.5, -0.1, 0.7]])
    prepared, training, opts = _prepare_state(x, runtime_tuning="safe", sparse=False)

    _, report = _resolve_runtime_threads_for_training(
        prepared=prepared,
        training=training,
        opts=opts,
    )

    assert opts["cov_writeback_mode"] in {"python", "bulk", "kernel"}
    assert report["cov_writeback_mode"] == opts["cov_writeback_mode"]
    assert report["behavior_sources"]["cov_writeback_mode"] == "autotune_measure"
    assert "cov_writeback_autotune" in report


def test_cov_writeback_autotune_sparse_sets_mode_and_source() -> None:
    x = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    prepared, training, opts = _prepare_state(x, runtime_tuning="safe", sparse=True)

    _, report = _resolve_runtime_threads_for_training(
        prepared=prepared,
        training=training,
        opts=opts,
    )

    assert opts["cov_writeback_mode"] in {"python", "bulk", "kernel"}
    assert report["cov_writeback_mode"] == opts["cov_writeback_mode"]
    assert report["behavior_sources"]["cov_writeback_mode"] == "autotune_measure"
    assert "cov_writeback_autotune" in report
