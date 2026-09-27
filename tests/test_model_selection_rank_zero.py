import numpy as np
import pytest
import scipy.sparse as sp

import vbpca_py.model_selection as ms
from vbpca_py.model_selection import (
    CVConfig,
    SelectionConfig,
    cross_validate_components,
    select_n_components,
)


def test_normalize_components_retains_explicit_zero() -> None:
    assert ms._normalize_components([2, 0, -1, 0, 1], 4, 5) == [2, 0, 1]
    assert ms._normalize_components(None, 2, 3) == [1, 2]

    with pytest.raises(ValueError, match="at least one non-negative integer"):
        ms._normalize_components([-1, -2], 4, 5)


def test_mean_only_candidate_uses_training_rows_and_probe() -> None:
    x = np.array([[1.0, 2.0, 100.0], [3.0, 5.0, 7.0]])
    xprobe = np.full_like(x, np.nan)
    xprobe[0, 2] = 5.0

    metrics, model = ms._fit_mean_only_candidate(x, None, xprobe, bias=True)

    assert model is None
    assert metrics["k"] == 0
    assert metrics["candidate_type"] == "mean_only"
    assert metrics["rms"] == pytest.approx(np.sqrt(1.7))
    assert metrics["prms"] == pytest.approx(3.5)
    assert np.isnan(float(metrics["cost"]))
    assert metrics["n_iter"] == 0
    assert metrics["converged"] is True
    assert metrics["convergence_reason"] == "closed_form_mean_only"


def test_mean_only_candidate_respects_mask_and_disabled_bias() -> None:
    x = np.array([[1.0, 3.0], [2.0, 4.0]])
    mask = np.array([[True, False], [True, True]])

    metrics, _ = ms._fit_mean_only_candidate(x, mask, None, bias=False)

    assert metrics["rms"] == pytest.approx(np.sqrt((1.0 + 4.0 + 16.0) / 3.0))


def test_mean_only_candidate_requires_training_support_for_each_row() -> None:
    x = np.ones((2, 3), dtype=float)
    xprobe = np.full_like(x, np.nan)
    xprobe[0, :] = x[0, :]

    with pytest.raises(ValueError, match="training observation in each row"):
        ms._fit_mean_only_candidate(x, None, xprobe, bias=True)


def test_select_rank_zero_returns_no_fitted_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.ones((3, 5), dtype=float)

    class _DummyModel:
        pass

    def _fake_fit_candidate(
        k: int,
        *args: object,
        **kwargs: object,
    ) -> tuple[dict[str, object], object | None]:
        model = None if k == 0 else _DummyModel()
        metric = 0.5 if k == 0 else 0.8
        return (
            {
                "k": k,
                "rms": metric,
                "prms": metric,
                "cost": float("nan") if k == 0 else 1.0,
                "evr": None,
                "n_iter": 0 if k == 0 else 3,
                "convergence_reason": "closed_form_mean_only" if k == 0 else "rms",
                "converged": True,
            },
            model,
        )

    monkeypatch.setattr(ms, "_fit_candidate", _fake_fit_candidate)

    best_k, best_metrics, trace, best_model = select_n_components(
        x,
        components=[0, 1],
        config=SelectionConfig(
            metric="prms",
            compute_explained_variance=True,
            return_best_model=True,
        ),
        random_state=0,
    )

    assert best_k == 0
    assert best_metrics["k"] == 0
    assert [entry["k"] for entry in trace] == [0, 1]
    assert best_model is None


def test_rank_zero_rejects_variational_cost_selection() -> None:
    with pytest.raises(ValueError, match="rank zero has no variational cost"):
        select_n_components(
            np.ones((3, 5)),
            components=[0, 1],
            config=SelectionConfig(metric="cost"),
        )


@pytest.mark.parametrize(
    ("x", "expected_rank"),
    [
        (np.repeat(np.arange(1.0, 6.0)[:, None], 10, axis=1), 0),
        (
            np.outer(
                np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
                np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]),
            ),
            1,
        ),
    ],
)
def test_rank_zero_competes_with_positive_rank(
    x: np.ndarray, expected_rank: int
) -> None:
    best_k, _, trace, _ = select_n_components(
        x,
        components=[1, 0],
        config=SelectionConfig(metric="prms", compute_explained_variance=False),
        algorithm="ppca",
        maxiters=150,
        niter_broadprior=0,
        random_state=199,
        rotate2pca=0,
        xprobe_fraction=0.2,
        verbose=0,
    )

    assert best_k == expected_rank
    assert [entry["k"] for entry in trace] == [1, 0]


def test_rank_zero_rejects_sparse_input() -> None:
    with pytest.raises(
        ValueError, match="rank-zero candidate currently supports dense"
    ):
        select_n_components(
            sp.csr_matrix(np.ones((3, 5))),
            components=[0],
            config=SelectionConfig(metric="rms", compute_explained_variance=False),
        )


def test_cv_one_se_rule_chooses_smallest_eligible_rank() -> None:
    fold_metrics = [
        {
            k: {
                "rms": 1.0,
                "prms": 1.0,
                "cost": float("nan") if k == 0 else 2.0,
                "n_iter": 0 if k == 0 else 2,
                "converged": True,
                "convergence_reason": "closed_form_mean_only" if k == 0 else "rms",
            }
            for k in [2, 0, 1]
        }
        for _ in range(3)
    ]

    best_k, _ = ms._aggregate_cv_results([2, 0, 1], fold_metrics, "prms")

    assert best_k == 0


def test_cross_validate_rank_zero_end_to_end() -> None:
    x = np.random.default_rng(199).normal(size=(5, 7))

    best_k, results = cross_validate_components(
        x,
        components=[0],
        config=CVConfig(n_splits=2, seed=199),
    )

    assert best_k == 0
    assert len(results) == 1
    assert np.isfinite(float(results[0]["mean_prms"]))
    assert results[0]["mean_n_iter"] == pytest.approx(0.0)
    assert results[0]["convergence_rate"] == pytest.approx(1.0)
    assert results[0]["convergence_reason_counts"] == {"closed_form_mean_only": 2}
