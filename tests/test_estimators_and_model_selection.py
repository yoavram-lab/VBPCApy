"""Coverage-focused tests for estimator and model selection helpers."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import vbpca_py.model_selection as ms
from vbpca_py.estimators import VBPCA
from vbpca_py.model_selection import SelectionConfig, select_n_components


def test_vbpca_fit_error_paths() -> None:
    est = VBPCA(1)
    dense = np.ones((2, 2))
    sparse_mask = sp.csr_matrix(np.ones((2, 2)))
    with pytest.raises(ValueError):
        est.fit(dense, mask=sparse_mask)

    sparse_x = sp.csr_matrix(np.ones((2, 2)))
    dense_mask = np.ones((2, 2))
    est_small_budget = VBPCA(1, max_dense_bytes=1)
    with pytest.raises(ValueError):
        est_small_budget.fit(sparse_x, mask=dense_mask)


def test_vbpca_transform_and_inverse_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    est = VBPCA(1)
    with pytest.raises(RuntimeError):
        est.transform()
    with pytest.raises(RuntimeError):
        est.inverse_transform()

    def stub_pca_full(x, n_components, mask=None, **opts):  # type: ignore[override]
        n_features, n_samples = x.shape
        return {
            "A": np.ones((n_features, n_components)),
            "S": np.ones((n_components, n_samples)),
            "Mu": np.zeros((n_features, 1)),
            "RMS": 0.1,
            "PRMS": 0.2,
            "V": 0.3,
            "Cost": 0.4,
            "Xrec": np.ones((n_features, n_samples)),
            "Vr": np.ones((n_features, n_samples)),
            "ExplainedVar": np.ones(n_components),
            "ExplainedVarRatio": np.ones(n_components),
        }

    monkeypatch.setattr("vbpca_py.estimators.pca_full", stub_pca_full)
    x = np.ones((2, 3))
    est_fitted = VBPCA(1, bias=False)
    est_fitted.fit(x)

    with pytest.raises(NotImplementedError):
        est_fitted.transform(x)

    recon = est_fitted.inverse_transform()
    assert recon.shape == x.shape


def test_vbpca_get_options_merges_overrides() -> None:
    est = VBPCA(2, bias=False, maxiters=5, tol=1e-3, verbose=2, foo="bar")
    opts = est.get_options()
    assert opts["bias"] is False
    assert opts["maxiters"] == 5
    assert opts["verbose"] == 2
    assert opts.get("foo") == "bar"


def test_select_n_components_invalid_metric_and_components() -> None:
    x = np.ones((2, 2))
    cfg = SelectionConfig(metric="bad")
    with pytest.raises(ValueError):
        select_n_components(x, config=cfg)

    with pytest.raises(ValueError):
        ms._normalize_components([], 2, 2)

    assert ms._normalize_components(None, 2, 3) == [1, 2]
    assert np.isnan(ms._to_float({}))
    assert ms._verbose_enabled({"a": 1}) is True


def test_select_n_components_mask_budget_overflow() -> None:
    x_sparse = sp.csr_matrix(np.ones((2, 2)))
    dense_mask = np.ones((2, 2))
    cfg = SelectionConfig()
    with pytest.raises(ValueError):
        select_n_components(
            x_sparse,
            mask=dense_mask,
            config=cfg,
            max_dense_bytes=1,
        )

    opts: dict[str, object] = {}
    mask_small = np.ones((2, 2))
    out_mask = ms._normalize_mask_for_selection(
        x_sparse,
        mask_small,
        max_dense_bytes=10_000,
        opts=opts,
    )
    assert isinstance(out_mask, np.ndarray)
    assert opts["_runtime_preflight"]

    sparse_mask = sp.csr_matrix(mask_small)
    out_sparse_mask = ms._normalize_mask_for_selection(
        x_sparse,
        sparse_mask,
        max_dense_bytes=10_000,
        opts={},
    )
    assert sp.isspmatrix_csr(out_sparse_mask)


def test_select_n_components_metric_reversal_and_patience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.ones((2, 2))

    def fake_fit_candidate(k, x_arr, mask, cfg, opts):  # type: ignore[override]
        val = float(k)
        dummy = VBPCA(int(k))
        dummy.components_ = np.ones((x_arr.shape[0], 1))
        dummy.scores_ = np.ones((1, x_arr.shape[1]))
        dummy.mean_ = np.zeros((x_arr.shape[0], 1))
        dummy.reconstruction_ = None
        dummy.explained_variance_ = None
        dummy.explained_variance_ratio_ = None
        return {
            "k": int(k),
            "rms": val,
            "prms": val,
            "cost": val,
            "evr": None,
        }, dummy

    monkeypatch.setattr(ms, "_fit_candidate", fake_fit_candidate)

    evr = np.array([0.5, 0.5])
    monkeypatch.setattr(ms, "_compute_evr_for_best", lambda est, **kwargs: evr)

    cfg = SelectionConfig(
        metric="rms",
        stop_on_metric_reversal=True,
        patience=0,
        return_best_model=True,
        compute_explained_variance=True,
    )

    best_k, best_metrics, trace, best_model = select_n_components(
        x,
        components=[1, 2, 3],
        config=cfg,
        selection_verbose=1,
    )

    # Metric reversal stops at previous k=1; evr injected into metrics and trace.
    assert best_k == 1
    assert best_metrics["evr"] is not None
    assert any(entry["evr"] is not None for entry in trace)
    assert best_model is not None


def test_compute_evr_for_best_returns_none_and_patience_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_est = VBPCA(1)
    assert ms._compute_evr_for_best(empty_est) is None

    est = VBPCA(1)
    est.components_ = np.ones((2, 1))
    est.scores_ = np.ones((1, 2))
    est.mean_ = np.zeros((2, 1))
    assert ms._compute_evr_for_best(est) is not None

    x = np.ones((1, 3))

    def fixed_fit_candidate(k, x_arr, mask, cfg, opts):  # type: ignore[override]
        entry = {
            "k": int(k),
            "rms": float(k),
            "prms": float(k),
            "cost": float(k),
            "evr": None,
        }
        est = VBPCA(int(k))
        est.components_ = np.ones((x_arr.shape[0], 1))
        est.scores_ = np.ones((1, x_arr.shape[1]))
        est.mean_ = np.zeros((x_arr.shape[0], 1))
        return entry, est

    monkeypatch.setattr(ms, "_fit_candidate", fixed_fit_candidate)
    cfg = SelectionConfig(
        metric="rms",
        patience=0,
        stop_on_metric_reversal=False,
        return_best_model=False,
        compute_explained_variance=False,
    )
    best_k, _, trace, _ = select_n_components(
        x,
        components=[1, 2],
        config=cfg,
        selection_verbose=1,
    )
    assert best_k in {1, 2}
    assert trace
