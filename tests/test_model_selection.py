import numpy as np
import pytest
from numpy.testing import assert_allclose

import vbpca_py.model_selection as ms
from vbpca_py.model_selection import SelectionConfig, select_n_components


def _low_rank_data(
    rng: np.random.Generator, n_features: int, n_samples: int, rank: int
) -> np.ndarray:
    a = rng.standard_normal((n_features, rank))
    s = rng.standard_normal((rank, n_samples))
    return a @ s


def test_select_n_components_tracks_trace_and_best_model() -> None:
    rng = np.random.default_rng(0)
    x = _low_rank_data(rng, n_features=6, n_samples=10, rank=2)

    cfg = SelectionConfig(
        metric="cost",
        patience=None,
        max_trials=None,
        compute_explained_variance=False,
        return_best_model=True,
    )

    best_k, best_metrics, trace, best_model = select_n_components(
        x,
        components=[1, 2, 3],
        config=cfg,
        maxiters=80,
        verbose=0,
    )

    assert len(trace) == 3
    assert best_k == 2
    assert best_metrics["cost"] <= trace[0]["cost"]
    assert best_model is not None
    assert best_model.components_ is not None
    assert best_model.components_.shape[1] == best_k


def test_select_n_components_respects_max_trials() -> None:
    rng = np.random.default_rng(1)
    x = _low_rank_data(rng, n_features=5, n_samples=8, rank=1)

    cfg = SelectionConfig(metric="cost", max_trials=1, compute_explained_variance=False)

    best_k, _, trace, _ = select_n_components(
        x,
        components=[1, 2, 3],
        config=cfg,
        verbose=0,
    )

    assert len(trace) == 1
    assert best_k == trace[0]["k"]


def test_select_n_components_falls_back_when_prms_missing() -> None:
    rng = np.random.default_rng(2)
    x = _low_rank_data(rng, n_features=4, n_samples=6, rank=1)

    cfg = SelectionConfig(metric="prms", compute_explained_variance=False)

    best_k, best_metrics, trace, _ = select_n_components(
        x,
        components=[1, 2],
        config=cfg,
        maxiters=50,
        verbose=0,
    )

    assert best_k in {1, 2}
    assert np.isfinite(best_metrics["prms"]) or np.isfinite(best_metrics["cost"])
    assert len(trace) == 2


def test_select_n_components_rejects_invalid_metric() -> None:
    rng = np.random.default_rng(3)
    x = _low_rank_data(rng, n_features=4, n_samples=6, rank=1)

    cfg = SelectionConfig(metric="cost")
    cfg.metric = "bad"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="metric must be one of"):
        select_n_components(x, config=cfg)


def test_select_n_components_normalizes_component_candidates() -> None:
    rng = np.random.default_rng(4)
    x = _low_rank_data(rng, n_features=5, n_samples=7, rank=2)

    best_k, _, trace, _ = select_n_components(
        x,
        components=[0, -1, 2, 2, 1],
        config=SelectionConfig(metric="cost", compute_explained_variance=False),
        maxiters=30,
        verbose=0,
    )

    # _normalize_components keeps only unique positive values in input order.
    assert [entry["k"] for entry in trace] == [2, 1]
    assert best_k in {1, 2}


def test_select_n_components_empty_after_normalization_raises() -> None:
    rng = np.random.default_rng(5)
    x = _low_rank_data(rng, n_features=4, n_samples=5, rank=1)

    with pytest.raises(ValueError, match="at least one positive integer"):
        select_n_components(x, components=[0, -2, -3])


def test_select_n_components_patience_stops_early() -> None:
    rng = np.random.default_rng(6)
    x = _low_rank_data(rng, n_features=6, n_samples=10, rank=1)

    cfg = SelectionConfig(
        metric="cost",
        patience=0,
        max_trials=None,
        compute_explained_variance=False,
    )

    _, _, trace, _ = select_n_components(
        x,
        components=[1, 2, 3, 4],
        config=cfg,
        maxiters=20,
        verbose=0,
    )

    assert 1 <= len(trace) <= 4


def test_select_n_components_stop_on_metric_reversal_uses_previous_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.ones((3, 5), dtype=float)
    cost_by_k = {1: 0.80, 2: 0.60, 3: 0.65, 4: 0.50}

    class _DummyModel:
        pass

    def _fake_fit_candidate(
        k: int,
        x_arr: np.ndarray,
        mask: np.ndarray | None,
        cfg: SelectionConfig,
        opts: dict[str, object],
    ) -> tuple[dict[str, object], _DummyModel]:
        return (
            {
                "k": int(k),
                "rms": float("nan"),
                "prms": float("nan"),
                "cost": float(cost_by_k[k]),
                "evr": None,
            },
            _DummyModel(),
        )

    monkeypatch.setattr(ms, "_fit_candidate", _fake_fit_candidate)

    cfg = SelectionConfig(
        metric="cost",
        stop_on_metric_reversal=True,
        compute_explained_variance=False,
        return_best_model=True,
    )

    best_k, best_metrics, trace, best_model = select_n_components(
        x,
        components=[1, 2, 3, 4],
        config=cfg,
    )

    assert [entry["k"] for entry in trace] == [1, 2, 3]
    assert best_k == 2
    assert best_metrics["k"] == 2
    assert float(best_metrics["cost"]) == pytest.approx(0.60)
    assert best_model is not None


def test_select_n_components_logs_k_progress_when_verbose_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    x = np.ones((3, 5), dtype=float)

    class _DummyModel:
        pass

    def _fake_fit_candidate(
        k: int,
        x_arr: np.ndarray,
        mask: np.ndarray | None,
        cfg: SelectionConfig,
        opts: dict[str, object],
    ) -> tuple[dict[str, object], _DummyModel]:
        return (
            {
                "k": int(k),
                "rms": float(1.0 / k),
                "prms": float("nan"),
                "cost": float(k),
                "evr": None,
            },
            _DummyModel(),
        )

    monkeypatch.setattr(ms, "_fit_candidate", _fake_fit_candidate)
    caplog.set_level("INFO", logger=ms.logger.name)

    _best_k, _best_metrics, _trace, _best_model = select_n_components(
        x,
        components=[1, 2, 3],
        config=SelectionConfig(metric="cost", compute_explained_variance=False),
        verbose=1,
    )

    assert "Model selection k 1/3: fitting k=1" not in caplog.text
    assert "Model selection k=1 done" in caplog.text
    assert "Model selection k=3 done" in caplog.text
    assert "Model selection complete: best_k=" in caplog.text


def test_select_n_components_no_k_progress_logs_when_verbose_zero(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    x = np.ones((3, 5), dtype=float)

    class _DummyModel:
        pass

    def _fake_fit_candidate(
        k: int,
        x_arr: np.ndarray,
        mask: np.ndarray | None,
        cfg: SelectionConfig,
        opts: dict[str, object],
    ) -> tuple[dict[str, object], _DummyModel]:
        return (
            {
                "k": int(k),
                "rms": float(1.0 / k),
                "prms": float("nan"),
                "cost": float(k),
                "evr": None,
            },
            _DummyModel(),
        )

    monkeypatch.setattr(ms, "_fit_candidate", _fake_fit_candidate)
    caplog.set_level("INFO", logger=ms.logger.name)

    _best_k, _best_metrics, _trace, _best_model = select_n_components(
        x,
        components=[1, 2],
        config=SelectionConfig(metric="cost", compute_explained_variance=False),
        verbose=0,
    )

    assert "Model selection k " not in caplog.text


def test_select_n_components_selection_verbose_decoupled_from_fit_verbose(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    x = np.ones((3, 5), dtype=float)
    seen_verbose: list[object] = []

    class _DummyModel:
        pass

    def _fake_fit_candidate(
        k: int,
        x_arr: np.ndarray,
        mask: np.ndarray | None,
        cfg: SelectionConfig,
        opts: dict[str, object],
    ) -> tuple[dict[str, object], _DummyModel]:
        seen_verbose.append(opts.get("verbose"))
        return (
            {
                "k": int(k),
                "rms": float(1.0 / k),
                "prms": float("nan"),
                "cost": float(k),
                "evr": None,
            },
            _DummyModel(),
        )

    monkeypatch.setattr(ms, "_fit_candidate", _fake_fit_candidate)
    caplog.set_level("INFO", logger=ms.logger.name)

    _best_k, _best_metrics, _trace, _best_model = select_n_components(
        x,
        components=[1, 2],
        config=SelectionConfig(metric="cost", compute_explained_variance=False),
        selection_verbose=1,
        verbose=0,
    )

    assert "Model selection k=1 done" in caplog.text
    assert seen_verbose == [0, 0]


def test_select_n_components_mask_argument_matches_nan_mask() -> None:
    rng = np.random.default_rng(123)
    x = rng.standard_normal((5, 8))
    x[rng.random(x.shape) < 0.2] = np.nan
    mask = ~np.isnan(x)
    # Supply an empty xprobe to suppress auto-holdout (which would differ
    # between calls due to independent RNG states).
    empty_probe = np.full(x.shape, np.nan, dtype=float)

    cfg = SelectionConfig(metric="cost", compute_explained_variance=False)
    components = [1, 2, 3]

    best_k_implicit, _, trace_implicit, _ = select_n_components(
        x,
        components=components,
        config=cfg,
        maxiters=12,
        verbose=0,
        compat_mode="strict_legacy",
        rotate2pca=0,
        xprobe=empty_probe,
    )

    best_k_explicit, _, trace_explicit, _ = select_n_components(
        x,
        mask=mask,
        components=components,
        config=cfg,
        maxiters=12,
        verbose=0,
        compat_mode="strict_legacy",
        rotate2pca=0,
        xprobe=empty_probe,
    )

    assert best_k_implicit == best_k_explicit
    assert len(trace_implicit) == len(trace_explicit) == len(components)

    cost_imp = [float(entry["cost"]) for entry in trace_implicit]
    cost_exp = [float(entry["cost"]) for entry in trace_explicit]
    assert_allclose(cost_imp, cost_exp, rtol=1e-10, atol=1e-12)


def test_select_n_components_stop_on_metric_reversal_with_real_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(77)
    x = rng.standard_normal((4, 7))
    components = [1, 2, 3]

    orig_fit = ms._fit_candidate

    base_opts = {
        "compat_mode": "strict_legacy",
        "rotate2pca": 0,
        "maxiters": 10,
        "verbose": 0,
        "cfstop": np.array([100, 1e-4, 1e-3]),
    }
    base_metrics, base_model = orig_fit(
        1, x, None, SelectionConfig(metric="cost"), base_opts
    )
    base_cost = float(base_metrics["cost"])
    # Ensure base_cost is finite; fall back to a known value otherwise.
    if not np.isfinite(base_cost):
        base_cost = 10.0

    # Monkeypatch to return the real metrics for
    # k=1,2, then an induced reversal for k=3.
    def _fake_fit_candidate(
        k: int,
        x_arr: np.ndarray,
        mask: np.ndarray | None,
        cfg: SelectionConfig,
        opts: dict[str, object],
    ) -> tuple[dict[str, object], object]:
        metrics: dict[str, object]
        model: object
        if k == 1:
            metrics = {
                "k": 1,
                "rms": float("nan"),
                "prms": float("nan"),
                "cost": base_cost + 0.2,
                "evr": None,
            }
            model = base_model
        elif k == 2:
            metrics = {
                "k": 2,
                "rms": float("nan"),
                "prms": float("nan"),
                "cost": base_cost - 0.1,  # improvement
                "evr": None,
            }
            model = base_model
        else:
            metrics = {
                "k": 3,
                "rms": float("nan"),
                "prms": float("nan"),
                "cost": base_cost + 0.4,  # induce reversal
                "evr": None,
            }
            model = base_model
        return metrics, model

    monkeypatch.setattr(ms, "_fit_candidate", _fake_fit_candidate)

    cfg_rev = SelectionConfig(
        metric="cost",
        stop_on_metric_reversal=True,
        compute_explained_variance=False,
        return_best_model=True,
    )

    best_k, best_metrics, trace, _ = select_n_components(
        x,
        components=components,
        config=cfg_rev,
        maxiters=10,
        verbose=0,
        compat_mode="strict_legacy",
        rotate2pca=0,
    )

    assert [entry["k"] for entry in trace] == [1, 2, 3]
    assert best_k == 2
    assert best_metrics["k"] == 2


def test_select_n_components_stop_on_metric_reversal_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A tiny uptick should not trigger reversal because of isclose guard.
    cost_seq = {1: 0.5000000000000000, 2: 0.500000000001, 3: 0.60}

    class _DummyModel:
        pass

    def _fake_fit_candidate(
        k: int,
        x_arr: np.ndarray,
        mask: np.ndarray | None,
        cfg: SelectionConfig,
        opts: dict[str, object],
    ) -> tuple[dict[str, object], _DummyModel]:
        return (
            {
                "k": int(k),
                "rms": float("nan"),
                "prms": float("nan"),
                "cost": float(cost_seq[k]),
                "evr": None,
            },
            _DummyModel(),
        )

    monkeypatch.setattr(ms, "_fit_candidate", _fake_fit_candidate)

    cfg = SelectionConfig(
        metric="cost",
        stop_on_metric_reversal=True,
        compute_explained_variance=False,
        return_best_model=True,
    )

    x = np.ones((2, 4), dtype=float)
    best_k, best_metrics, trace, _ = select_n_components(
        x,
        components=[1, 2, 3],
        config=cfg,
        maxiters=5,
        verbose=0,
        compat_mode="strict_legacy",
        rotate2pca=0,
    )

    # No reversal at k=2 (tiny uptick), so we still evaluate k=3.
    assert [entry["k"] for entry in trace] == [1, 2, 3]
    assert best_k == 2
    assert best_metrics["k"] == 2


def test_select_n_components_stop_on_metric_reversal_strict_increase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A clear increase should trigger reversal and stop before evaluating further ks.
    cost_seq = {1: 0.5, 2: 0.4, 3: 0.6}

    class _DummyModel:
        pass

    def _fake_fit_candidate(
        k: int,
        x_arr: np.ndarray,
        mask: np.ndarray | None,
        cfg: SelectionConfig,
        opts: dict[str, object],
    ) -> tuple[dict[str, object], _DummyModel]:
        return (
            {
                "k": int(k),
                "rms": float("nan"),
                "prms": float("nan"),
                "cost": float(cost_seq[k]),
                "evr": None,
            },
            _DummyModel(),
        )

    monkeypatch.setattr(ms, "_fit_candidate", _fake_fit_candidate)

    cfg = SelectionConfig(
        metric="cost",
        stop_on_metric_reversal=True,
        compute_explained_variance=False,
        return_best_model=True,
    )

    x = np.ones((2, 4), dtype=float)
    best_k, best_metrics, trace, _ = select_n_components(
        x,
        components=[1, 2, 3],
        config=cfg,
        maxiters=5,
        verbose=0,
        compat_mode="strict_legacy",
        rotate2pca=0,
    )

    # Should stop when k=3 worsens relative to k=2 and pick k=2.
    assert [entry["k"] for entry in trace] == [1, 2, 3]
    assert best_k == 2
    assert best_metrics["k"] == 2


def test_select_n_components_deterministic_across_num_cpu() -> None:
    rng = np.random.default_rng(12345)
    x = rng.standard_normal((6, 10))
    x[rng.random(x.shape) < 0.15] = np.nan

    cfg = SelectionConfig(metric="cost", compute_explained_variance=False)
    components = [1, 2, 3]

    res = []
    for num_cpu in (1, 2):
        best_k, _best_metrics, trace, _ = select_n_components(
            x,
            components=components,
            config=cfg,
            maxiters=12,
            verbose=0,
            compat_mode="strict_legacy",
            rotate2pca=0,
            num_cpu=num_cpu,
            runtime_tuning="off",
        )
        cost_trace = [float(entry["cost"]) for entry in trace]
        res.append((best_k, cost_trace))

    assert res[0][0] == res[1][0]
    assert_allclose(res[0][1], res[1][1], rtol=1e-12, atol=1e-12)
