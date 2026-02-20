import numpy as np
import pytest

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
        metric="rms",
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
    assert best_metrics["rms"] <= trace[0]["rms"]
    assert best_model is not None
    assert best_model.components_ is not None
    assert best_model.components_.shape[1] == best_k


def test_select_n_components_respects_max_trials() -> None:
    rng = np.random.default_rng(1)
    x = _low_rank_data(rng, n_features=5, n_samples=8, rank=1)

    cfg = SelectionConfig(metric="rms", max_trials=1, compute_explained_variance=False)

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

    assert best_k in (1, 2)
    assert np.isfinite(best_metrics["rms"]) or np.isfinite(best_metrics["prms"])
    assert len(trace) == 2


def test_select_n_components_rejects_invalid_metric() -> None:
    rng = np.random.default_rng(3)
    x = _low_rank_data(rng, n_features=4, n_samples=6, rank=1)

    cfg = SelectionConfig(metric="rms")
    cfg.metric = "bad"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="metric must be one of"):
        select_n_components(x, config=cfg)


def test_select_n_components_normalizes_component_candidates() -> None:
    rng = np.random.default_rng(4)
    x = _low_rank_data(rng, n_features=5, n_samples=7, rank=2)

    best_k, _, trace, _ = select_n_components(
        x,
        components=[0, -1, 2, 2, 1],
        config=SelectionConfig(metric="rms", compute_explained_variance=False),
        maxiters=30,
        verbose=0,
    )

    # _normalize_components keeps only unique positive values in input order.
    assert [entry["k"] for entry in trace] == [2, 1]
    assert best_k in (1, 2)


def test_select_n_components_empty_after_normalization_raises() -> None:
    rng = np.random.default_rng(5)
    x = _low_rank_data(rng, n_features=4, n_samples=5, rank=1)

    with pytest.raises(ValueError, match="at least one positive integer"):
        select_n_components(x, components=[0, -2, -3])


def test_select_n_components_patience_stops_early() -> None:
    rng = np.random.default_rng(6)
    x = _low_rank_data(rng, n_features=6, n_samples=10, rank=1)

    cfg = SelectionConfig(
        metric="rms",
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
    rms_by_k = {1: 0.80, 2: 0.60, 3: 0.65, 4: 0.50}

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
                "rms": float(rms_by_k[k]),
                "prms": float("nan"),
                "cost": float("nan"),
                "evr": None,
            },
            _DummyModel(),
        )

    monkeypatch.setattr(ms, "_fit_candidate", _fake_fit_candidate)

    cfg = SelectionConfig(
        metric="rms",
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
    assert float(best_metrics["rms"]) == pytest.approx(0.60)
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
        config=SelectionConfig(metric="rms", compute_explained_variance=False),
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
        config=SelectionConfig(metric="rms", compute_explained_variance=False),
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
        seen_verbose.append(opts.get("verbose", None))
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
        config=SelectionConfig(metric="rms", compute_explained_variance=False),
        selection_verbose=1,
        verbose=0,
    )

    assert "Model selection k=1 done" in caplog.text
    assert seen_verbose == [0, 0]
