import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

import vbpca_py.model_selection as ms
from vbpca_py import VBPCA
from vbpca_py.model_selection import (
    CVConfig,
    SelectionConfig,
    cross_validate_components,
    select_n_components,
)


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
        random_state=0,
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


def test_select_n_components_generates_requested_prms() -> None:
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
    assert np.isfinite(best_metrics["prms"])
    assert all(np.isfinite(entry["prms"]) for entry in trace)
    assert len(trace) == 2


def test_select_n_components_rejects_unavailable_requested_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.ones((3, 5), dtype=float)

    def _fake_fit_candidate(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, object], object]:
        return (
            {
                "k": 1,
                "rms": 0.5,
                "prms": float("nan"),
                "cost": 0.1,
                "evr": None,
            },
            object(),
        )

    monkeypatch.setattr(ms, "_fit_candidate", _fake_fit_candidate)

    with pytest.raises(ValueError, match="metric 'prms' is unavailable"):
        select_n_components(
            x,
            components=[1],
            config=SelectionConfig(
                metric="prms",
                compute_explained_variance=False,
            ),
        )


def test_select_n_components_rejects_invalid_metric() -> None:
    rng = np.random.default_rng(3)
    x = _low_rank_data(rng, n_features=4, n_samples=6, rank=1)

    cfg = SelectionConfig(metric="cost")
    cfg.metric = "bad"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="metric must be one of"):
        select_n_components(x, config=cfg)


def test_select_n_components_accepts_rms_metric() -> None:
    """The public RMS metric works through the complete selection path (#124)."""
    rng = np.random.default_rng(124)
    x = _low_rank_data(rng, n_features=5, n_samples=8, rank=2)

    best_k, best_metrics, trace, _ = select_n_components(
        x,
        components=[1, 2],
        config=SelectionConfig(
            metric="rms",
            max_trials=2,
            compute_explained_variance=False,
        ),
        maxiters=10,
        random_state=124,
        verbose=0,
    )

    assert len(trace) == 2
    assert best_k in {1, 2}
    assert np.isfinite(best_metrics["rms"])
    assert best_metrics["rms"] == min(entry["rms"] for entry in trace)


@pytest.mark.parametrize("metric", ["prms", "cost", "rms"])
def test_ensure_metric_opts_respects_configured_xprobe_fraction(
    metric: str,
) -> None:
    """An explicit probe fraction is respected independently of the metric."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 80))
    fit_opts: dict[str, object] = {"xprobe_fraction": 0.02}

    cfg = SelectionConfig(metric="prms")
    cfg.metric = metric  # type: ignore[assignment]
    ms._ensure_metric_opts(fit_opts, x.copy(), None, cfg)

    xprobe = np.asarray(fit_opts["xprobe"])
    n_probe = int(np.sum(~np.isnan(xprobe)))
    assert n_probe == pytest.approx(x.size * 0.02, rel=0.1)


@pytest.mark.parametrize("metric", ["cost", "rms"])
def test_ensure_metric_opts_does_not_invent_probe_for_non_probe_metric(
    metric: str,
) -> None:
    """Cost/RMS sweeps retain all observations unless a probe was requested."""
    rng = np.random.default_rng(125)
    x = rng.standard_normal((20, 10))
    original = x.copy()
    fit_opts: dict[str, object] = {}
    cfg = SelectionConfig(metric="cost")
    cfg.metric = metric  # type: ignore[assignment]

    ms._ensure_metric_opts(fit_opts, x, None, cfg)

    assert "xprobe" not in fit_opts
    assert_allclose(x, original)
    assert "cfstop" not in fit_opts
    assert fit_opts["record_cost"] is True


def test_ensure_metric_opts_records_cost_without_enabling_cost_stop() -> None:
    x = np.ones((4, 6), dtype=float)
    fit_opts: dict[str, object] = {"cfstop": np.array([])}

    ms._ensure_metric_opts(fit_opts, x, None, SelectionConfig(metric="cost"))

    assert fit_opts["record_cost"] is True
    assert np.size(fit_opts["cfstop"]) == 0


def test_ensure_metric_opts_falls_back_to_default_probe_fraction() -> None:
    """No xprobe_fraction configured -> the historical 10% default applies (#122)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 80))
    fit_opts: dict[str, object] = {}

    ms._ensure_metric_opts(fit_opts, x.copy(), None, SelectionConfig(metric="prms"))

    xprobe = np.asarray(fit_opts["xprobe"])
    n_probe = int(np.sum(~np.isnan(xprobe)))
    assert n_probe == pytest.approx(x.size * ms._PROBE_FRACTION, rel=0.1)


def test_single_candidate_selection_matches_direct_fit_with_probe() -> None:
    """A prepared selection probe is reused exactly once with the caller's seed."""
    x = np.random.default_rng(9).normal(size=(8, 12))
    opts = {
        "maxiters": 5,
        "random_state": 42,
        "xprobe_fraction": 0.1,
        "rotate2pca": 0,
        "record_cost": True,
    }
    config = SelectionConfig(
        metric="cost",
        compute_explained_variance=False,
        return_best_model=True,
    )

    best_k, _, _, selected = select_n_components(
        x,
        components=[1],
        config=config,
        **opts,
    )
    direct = VBPCA(1, **opts).fit(x)

    assert best_k == 1
    assert selected is not None
    assert_allclose(selected.components_, direct.components_)
    assert_allclose(selected.scores_, direct.scores_)
    assert selected.rms_ == pytest.approx(direct.rms_)
    assert selected.noise_variance_ == pytest.approx(direct.noise_variance_)


def test_prms_selection_with_explicit_mask_uses_probe_metric() -> None:
    rng = np.random.default_rng(145)
    x = _low_rank_data(rng, n_features=8, n_samples=16, rank=2)
    mask = rng.random(x.shape) > 0.15

    _, best_metrics, trace, _ = select_n_components(
        x,
        mask=mask,
        components=[1, 2],
        config=SelectionConfig(metric="prms", compute_explained_variance=False),
        maxiters=5,
        niter_broadprior=0,
        random_state=145,
        rotate2pca=0,
        verbose=0,
    )

    assert np.isfinite(best_metrics["prms"])
    assert all(np.isfinite(entry["prms"]) for entry in trace)


def test_select_n_components_normalizes_component_candidates() -> None:
    rng = np.random.default_rng(4)
    x = _low_rank_data(rng, n_features=5, n_samples=7, rank=2)

    best_k, _, trace, _ = select_n_components(
        x,
        components=[-1, 2, 2, 1],
        config=SelectionConfig(metric="cost", compute_explained_variance=False),
        maxiters=30,
        verbose=0,
    )

    # Negative values are dropped; unique values keep their input order.
    assert [entry["k"] for entry in trace] == [2, 1]
    assert best_k in {1, 2}


def test_select_n_components_empty_after_normalization_raises() -> None:
    rng = np.random.default_rng(5)
    x = _low_rank_data(rng, n_features=4, n_samples=5, rank=1)

    with pytest.raises(ValueError, match="at least one non-negative integer"):
        select_n_components(x, components=[-2, -3])


def test_select_n_components_rejects_negative_patience() -> None:
    x = np.ones((3, 5), dtype=float)

    with pytest.raises(ValueError, match="patience must be a non-negative integer"):
        select_n_components(x, config=SelectionConfig(patience=-1))


def test_select_n_components_patience_stops_after_exact_streak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.ones((3, 5), dtype=float)
    cost_by_k = {1: 0.5, 2: 0.6, 3: 0.7, 4: 0.4}

    def _fake_fit_candidate(
        k: int,
        *args: object,
        **kwargs: object,
    ) -> tuple[dict[str, object], object]:
        return (
            {
                "k": k,
                "rms": 1.0,
                "prms": 1.0,
                "cost": cost_by_k[k],
                "evr": None,
            },
            object(),
        )

    monkeypatch.setattr(ms, "_fit_candidate", _fake_fit_candidate)

    cfg = SelectionConfig(
        metric="cost",
        patience=2,
        max_trials=None,
        compute_explained_variance=False,
    )

    _, _, trace, _ = select_n_components(
        x,
        components=[1, 2, 3, 4],
        config=cfg,
    )

    assert [entry["k"] for entry in trace] == [1, 2, 3]


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
    # Supply an empty xprobe to suppress auto-holdout, and a fixed
    # random_state, so both calls share the same init/holdout draws and
    # only the mask-argument form under test differs.
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
        random_state=0,
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
        random_state=0,
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
            random_state=0,
        )
        cost_trace = [float(entry["cost"]) for entry in trace]
        res.append((best_k, cost_trace))

    assert res[0][0] == res[1][0]
    assert_allclose(res[0][1], res[1][1], rtol=1e-12, atol=1e-12)


def test_runtime_policy_cache_reuses_only_compatible_rank_regimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, dict[str, object]]] = []

    class _DummyModel:
        def __init__(self, k: int) -> None:
            self.runtime_report_ = {
                "kernel_values": {
                    "score_update_dense": 2,
                    "loadings_update_dense": 3,
                    "noise_sxv_sum": 2,
                    "rms": 1,
                },
                "workload": {"is_sparse": False},
                "cov_writeback_mode": "bulk",
                "log_progress_stride": 0,
                "accessor_mode": "buffered",
                "autotune_dense_masked": {"elapsed_sec": 0.05 * k},
            }

    def _fake_fit_candidate(k, x_arr, mask, cfg, opts):
        calls.append((int(k), dict(opts)))
        return (
            {
                "k": int(k),
                "rms": float(k),
                "prms": float(k),
                "cost": float(k),
                "evr": None,
                "n_iter": 1,
                "convergence_reason": "maxiters",
                "converged": False,
                "candidate_type": "low_rank",
            },
            _DummyModel(int(k)),
        )

    monkeypatch.setattr(ms, "_fit_candidate", _fake_fit_candidate)
    cfg = SelectionConfig(
        metric="cost",
        compute_explained_variance=False,
        reuse_runtime_policy=True,
    )
    _, _, trace, _ = select_n_components(
        np.ones((8, 12)),
        components=[2, 3, 4, 5],
        config=cfg,
        runtime_tuning="safe",
        verbose=0,
    )

    assert [entry["runtime_policy_source"] for entry in trace] == [
        "measured",
        "selection_cache",
        "measured",
        "selection_cache",
    ]
    assert [entry["runtime_policy_regime"] for entry in trace] == [
        "2-3",
        "2-3",
        "4-7",
        "4-7",
    ]
    assert [opts["runtime_tuning"] for _, opts in calls] == [
        "safe",
        "off",
        "safe",
        "off",
    ]
    assert calls[1][1]["num_cpu_score_update"] == 2
    assert calls[1][1]["num_cpu_loadings_update"] == 3


def test_runtime_policy_key_invalidates_on_workload_or_option_change() -> None:
    x = np.arange(24.0).reshape(4, 6)
    opts = {"runtime_tuning": "safe", "num_cpu": 4}
    base = ms._runtime_policy_key(k=2, x=x, mask=None, opts=opts)

    assert ms._runtime_policy_key(k=3, x=x, mask=None, opts=opts) == base
    assert ms._runtime_policy_key(k=4, x=x, mask=None, opts=opts) != base
    assert (
        ms._runtime_policy_key(
            k=2,
            x=x,
            mask=np.ones_like(x, dtype=bool),
            opts=opts,
        )
        != base
    )
    assert ms._runtime_policy_key(k=2, x=sp.csr_matrix(x), mask=None, opts=opts) != base
    assert (
        ms._runtime_policy_key(
            k=2,
            x=x,
            mask=None,
            opts={"runtime_tuning": "safe", "num_cpu": 2},
        )
        != base
    )


def test_reused_and_independently_tuned_sweeps_are_equivalent() -> None:
    rng = np.random.default_rng(211)
    x = _low_rank_data(rng, n_features=7, n_samples=12, rank=2)
    x[rng.random(x.shape) < 0.15] = np.nan
    common = {
        "maxiters": 6,
        "niter_broadprior": 0,
        "verbose": 0,
        "rotate2pca": 0,
        "runtime_tuning": "safe",
        "random_state": 211,
    }

    best_reused, _, trace_reused, _ = select_n_components(
        x,
        components=[2, 3],
        config=SelectionConfig(
            metric="cost",
            compute_explained_variance=False,
            reuse_runtime_policy=True,
        ),
        **common,
    )
    best_independent, _, trace_independent, _ = select_n_components(
        x,
        components=[2, 3],
        config=SelectionConfig(
            metric="cost",
            compute_explained_variance=False,
            reuse_runtime_policy=False,
        ),
        **common,
    )

    assert best_reused == best_independent
    assert trace_reused[1]["runtime_policy_source"] == "selection_cache"
    assert_allclose(
        [entry["cost"] for entry in trace_reused],
        [entry["cost"] for entry in trace_independent],
        rtol=1e-10,
        atol=1e-12,
    )


# ============================================================
# cross_validate_components tests
# ============================================================


def test_cross_validate_components_basic() -> None:
    """Smoke test: cv returns valid best_k and cv_results with all metrics."""
    rng = np.random.default_rng(42)
    x = _low_rank_data(rng, n_features=6, n_samples=30, rank=2)

    cv_cfg = CVConfig(metric="prms", n_splits=3, one_se_rule=True, seed=0)

    best_k, cv_results = cross_validate_components(
        x,
        components=[1, 2, 3],
        config=cv_cfg,
        maxiters=30,
        verbose=0,
    )

    assert best_k in {1, 2, 3}
    assert len(cv_results) == 3

    # Every entry should have aggregated stats for all tracked metrics.
    for entry in cv_results:
        assert "k" in entry
        for m in ("rms", "prms", "cost"):
            assert f"mean_{m}" in entry
            assert f"std_{m}" in entry
            assert f"se_{m}" in entry
            for fold_i in range(3):
                assert f"{m}_fold_{fold_i + 1}" in entry
        assert 0.0 <= entry["convergence_rate"] <= 1.0
        assert 0 < entry["mean_n_iter"] <= entry["max_n_iter"] <= 30
        assert sum(entry["convergence_reason_counts"].values()) == 3
        for fold_i in range(3):
            assert f"n_iter_fold_{fold_i + 1}" in entry
            assert f"converged_fold_{fold_i + 1}" in entry
            assert f"convergence_reason_fold_{fold_i + 1}" in entry


def test_cv_aggregation_preserves_mixed_convergence_diagnostics() -> None:
    """Candidate summaries retain every fold's convergence outcome."""
    folds = [
        {
            1: {
                "rms": 1.0,
                "prms": 1.5,
                "cost": 2.0,
                "n_iter": 10,
                "converged": True,
                "convergence_reason": "angle",
            }
        },
        {
            1: {
                "rms": 1.2,
                "prms": 1.7,
                "cost": 2.2,
                "n_iter": 20,
                "converged": False,
                "convergence_reason": "maxiters",
            }
        },
    ]

    best_k, results = ms._aggregate_cv_results([1], folds, "prms")

    assert best_k == 1
    assert results[0]["mean_n_iter"] == pytest.approx(15.0)
    assert results[0]["max_n_iter"] == 20
    assert results[0]["convergence_rate"] == pytest.approx(0.5)
    assert results[0]["convergence_reason_counts"] == {"angle": 1, "maxiters": 1}
    assert results[0]["n_iter_fold_1"] == 10
    assert results[0]["converged_fold_2"] is False
    assert results[0]["convergence_reason_fold_2"] == "maxiters"


def test_cross_validate_components_rejects_training_cost_metric() -> None:
    """Training cost is not mislabeled as a held-out CV metric."""
    rng = np.random.default_rng(7)
    x = _low_rank_data(rng, n_features=5, n_samples=20, rank=1)
    cv_cfg = CVConfig(n_splits=2, one_se_rule=False, seed=1)
    cv_cfg.metric = "cost"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="cost is a training objective"):
        cross_validate_components(x, components=[1, 2], config=cv_cfg)


def test_cross_validate_components_all_metrics_recorded() -> None:
    """Even when selecting by prms, rms and cost are still recorded."""
    rng = np.random.default_rng(99)
    x = _low_rank_data(rng, n_features=6, n_samples=20, rank=2)

    cv_cfg = CVConfig(metric="prms", n_splits=2, seed=0)

    _best_k, cv_results = cross_validate_components(
        x,
        components=[1, 2, 3],
        config=cv_cfg,
        maxiters=30,
        verbose=0,
    )

    for entry in cv_results:
        # All three metrics are present regardless of selection metric.
        assert np.isfinite(entry["mean_rms"])
        assert np.isfinite(entry["mean_prms"])
        assert np.isfinite(entry["mean_cost"])


def test_cross_validate_components_invalid_splits() -> None:
    """n_splits < 2 raises ValueError."""
    rng = np.random.default_rng(0)
    x = _low_rank_data(rng, n_features=4, n_samples=10, rank=1)

    with pytest.raises(ValueError, match="n_splits must be >= 2"):
        cross_validate_components(
            x,
            components=[1, 2],
            config=CVConfig(n_splits=1),
            maxiters=10,
        )


def test_cross_validate_components_rejects_more_folds_than_observations() -> None:
    x = np.arange(4, dtype=float).reshape(2, 2)

    with pytest.raises(ValueError, match="exceeds the 2 validation-eligible"):
        cross_validate_components(
            x,
            components=[1],
            config=CVConfig(n_splits=5),
        )


def test_cross_validate_components_rejects_sparse_input() -> None:
    x = sp.csr_matrix(np.ones((4, 6), dtype=float))

    with pytest.raises(ValueError, match="supports dense input only"):
        cross_validate_components(x, components=[1], config=CVConfig(n_splits=2))


def test_cross_validate_components_rejects_no_validation_eligible_entries() -> None:
    x = np.array([[1.0, np.nan], [np.nan, 2.0]])

    with pytest.raises(ValueError, match="0 validation-eligible"):
        cross_validate_components(
            x,
            components=[1],
            config=CVConfig(n_splits=2),
        )


def test_element_folds_keep_singleton_support_in_every_training_fold() -> None:
    x = np.array([
        [1.0, 2.0, np.nan],
        [3.0, 4.0, np.nan],
        [np.nan, np.nan, 5.0],
    ])
    obs_rows, obs_cols = np.nonzero(~np.isnan(x))
    singleton_index = int(np.flatnonzero((obs_rows == 2) & (obs_cols == 2))[0])

    folds = ms._make_element_folds(x, 2, np.random.default_rng(8))

    probes = np.concatenate([probe for probe, _train in folds])
    assert len(probes) == 2
    assert len(np.unique(probes)) == len(probes)
    for probe, training in folds:
        assert singleton_index not in probe
        assert singleton_index in training
        masked = x.copy()
        masked[obs_rows[probe], obs_cols[probe]] = np.nan
        assert np.all(np.sum(~np.isnan(masked), axis=1) > 0)
        assert np.all(np.sum(~np.isnan(masked), axis=0) > 0)


def test_element_folds_handle_many_degree_two_columns_without_retries() -> None:
    n_samples = 1000
    x = np.full((3, n_samples), np.nan)
    for column in range(n_samples):
        x[column % 3, column] = float(column)
        x[(column + 1) % 3, column] = float(column) + 0.5
    obs_rows, obs_cols = np.nonzero(~np.isnan(x))

    folds = ms._make_element_folds(x, 3, np.random.default_rng(9))

    probes = np.concatenate([probe for probe, _train in folds])
    assert len(probes) == n_samples
    assert len(np.unique(probes)) == len(probes)
    for probe, _training in folds:
        masked = x.copy()
        masked[obs_rows[probe], obs_cols[probe]] = np.nan
        assert np.all(np.sum(~np.isnan(masked), axis=1) > 0)
        assert np.all(np.sum(~np.isnan(masked), axis=0) > 0)


def test_element_folds_preserve_training_row_and_column_coverage() -> None:
    x = np.arange(30, dtype=float).reshape(5, 6)
    x[[0, 1, 2], [0, 2, 4]] = np.nan
    obs_rows, obs_cols = np.nonzero(~np.isnan(x))

    folds = ms._make_element_folds(x, 3, np.random.default_rng(144))

    for probe_sel, _train_sel in folds:
        train = x.copy()
        train[obs_rows[probe_sel], obs_cols[probe_sel]] = np.nan
        assert np.all(np.sum(~np.isnan(train), axis=1) > 0)
        assert np.all(np.sum(~np.isnan(train), axis=0) > 0)


def test_cross_validate_components_seed_reproduces_folds_and_fits() -> None:
    x = _low_rank_data(np.random.default_rng(144), n_features=5, n_samples=12, rank=2)
    cfg = CVConfig(n_splits=2, seed=144)
    kwargs = {
        "components": [1, 2],
        "config": cfg,
        "maxiters": 5,
        "verbose": 0,
        "rotate2pca": 0,
    }

    first = cross_validate_components(x, **kwargs)
    second = cross_validate_components(x, **kwargs)

    assert first == second


def test_cross_validate_components_one_se_vs_global_min() -> None:
    """1-SE rule picks k <= global-minimum k."""
    rng = np.random.default_rng(5)
    x = _low_rank_data(rng, n_features=6, n_samples=40, rank=2)

    cv_1se = CVConfig(metric="prms", n_splits=3, one_se_rule=True, seed=0)
    cv_min = CVConfig(metric="prms", n_splits=3, one_se_rule=False, seed=0)

    best_k_1se, _ = cross_validate_components(
        x, components=[1, 2, 3, 4], config=cv_1se, maxiters=40, verbose=0
    )
    best_k_min, _ = cross_validate_components(
        x, components=[1, 2, 3, 4], config=cv_min, maxiters=40, verbose=0
    )

    # 1-SE rule should pick k <= global min k (more parsimonious).
    assert best_k_1se <= best_k_min


# ---------------------------------------------------------------------------
# Marginal variance on best model (#72)
# ---------------------------------------------------------------------------


def test_select_n_components_best_model_has_variance() -> None:
    """variance_ is populated on best_model when compute_explained_variance=True."""
    rng = np.random.default_rng(0)
    x = _low_rank_data(rng, n_features=6, n_samples=10, rank=2)

    cfg = SelectionConfig(
        metric="cost",
        return_best_model=True,
        compute_explained_variance=True,
    )
    _, _, _, best_model = select_n_components(
        x, components=[1, 2, 3], config=cfg, maxiters=80, verbose=0
    )

    assert best_model is not None
    assert best_model.variance_ is not None
    assert best_model.variance_.shape == x.shape
    assert np.all(np.isfinite(best_model.variance_))
    assert np.all(best_model.variance_ >= 0)


def test_select_n_components_best_model_variance_with_missing() -> None:
    """variance_ is populated when data contains NaN entries."""
    rng = np.random.default_rng(1)
    x = _low_rank_data(rng, n_features=6, n_samples=20, rank=2)
    x[rng.random(x.shape) < 0.15] = np.nan

    cfg = SelectionConfig(
        metric="prms",
        return_best_model=True,
        compute_explained_variance=True,
    )
    _, _, _, best_model = select_n_components(
        x, components=[1, 2], config=cfg, maxiters=80, verbose=0
    )

    assert best_model is not None
    assert best_model.variance_ is not None
    assert best_model.variance_.shape == x.shape
    assert np.all(np.isfinite(best_model.variance_))


def test_select_n_components_no_variance_without_diagnostics() -> None:
    """variance_ stays None when compute_explained_variance=False."""
    rng = np.random.default_rng(2)
    x = _low_rank_data(rng, n_features=6, n_samples=10, rank=2)

    cfg = SelectionConfig(
        metric="cost",
        return_best_model=True,
        compute_explained_variance=False,
    )
    _, _, _, best_model = select_n_components(
        x, components=[1, 2], config=cfg, maxiters=80, verbose=0
    )

    assert best_model is not None
    assert best_model.variance_ is None


# ── Convergence diagnostics in trace (issue #99) ────────────────


def test_trace_contains_convergence_diagnostics() -> None:
    """Each trace entry should have n_iter and convergence_reason."""
    rng = np.random.default_rng(42)
    x = _low_rank_data(rng, n_features=6, n_samples=10, rank=2)

    _, _, trace, _ = select_n_components(
        x,
        components=[1, 2, 3],
        maxiters=20,
        verbose=0,
    )

    assert len(trace) >= 3
    for entry in trace:
        assert "n_iter" in entry, f"trace entry for k={entry['k']} missing n_iter"
        assert "convergence_reason" in entry, (
            f"trace entry for k={entry['k']} missing convergence_reason"
        )
        assert "converged" in entry
        assert isinstance(entry["n_iter"], int)
        assert entry["n_iter"] > 0
        assert isinstance(entry["convergence_reason"], str)
        assert isinstance(entry["converged"], bool)
        assert entry["convergence_reason"] != ""
