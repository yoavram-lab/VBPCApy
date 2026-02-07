import numpy as np

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
