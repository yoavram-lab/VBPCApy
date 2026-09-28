"""Equation-level tests for the dense masked VB-PCA updates."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from vbpca_py import dense_update_kernels as kernels


def _score_reference(
    x: np.ndarray,
    mask: np.ndarray,
    loadings: np.ndarray,
    loading_covariances: np.ndarray,
    noise_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the score posterior directly from the model equations."""
    n_components = loadings.shape[1]
    n_samples = x.shape[1]
    scores = np.zeros((n_components, n_samples))
    covariances = np.zeros((n_samples, n_components, n_components))

    for sample in range(n_samples):
        observed = np.flatnonzero(mask[:, sample])
        precision = noise_var * np.eye(n_components)
        rhs = np.zeros(n_components)
        for feature in observed:
            loading = loadings[feature]
            precision += np.outer(loading, loading)
            precision += loading_covariances[feature]
            rhs += loading * x[feature, sample]
        scores[:, sample] = np.linalg.solve(precision, rhs)
        covariances[sample] = noise_var * np.linalg.solve(
            precision,
            np.eye(n_components),
        )

    return scores, covariances


def _loading_reference(
    x: np.ndarray,
    mask: np.ndarray,
    scores: np.ndarray,
    score_covariances: np.ndarray,
    prior_precision: np.ndarray,
    noise_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the loading posterior directly from the model equations."""
    n_features = x.shape[0]
    n_components = scores.shape[0]
    loadings = np.zeros((n_features, n_components))
    covariances = np.zeros((n_features, n_components, n_components))

    for feature in range(n_features):
        observed = np.flatnonzero(mask[feature])
        precision = prior_precision.copy()
        rhs = np.zeros(n_components)
        for sample in observed:
            score = scores[:, sample]
            precision += np.outer(score, score)
            precision += score_covariances[sample]
            rhs += score * x[feature, sample]
        loadings[feature] = np.linalg.solve(precision, rhs)
        covariances[feature] = noise_var * np.linalg.solve(
            precision,
            np.eye(n_components),
        )

    return loadings, covariances


@pytest.fixture
def update_problem() -> dict[str, np.ndarray | float]:
    """Return a deterministic, nonsymmetric masked update problem."""
    rng = np.random.default_rng(207)
    n_features, n_samples, n_components = 7, 6, 3
    x = rng.normal(size=(n_features, n_samples))
    mask = rng.random(x.shape) > 0.3
    mask[0, :] = True
    mask[:, 0] = True
    loadings = rng.normal(scale=0.4, size=(n_features, n_components))
    scores = rng.normal(scale=0.5, size=(n_components, n_samples))

    loading_covariances = np.empty((n_features, n_components, n_components))
    for feature in range(n_features):
        root = rng.normal(scale=0.05, size=(n_components, n_components))
        loading_covariances[feature] = root @ root.T + 0.01 * np.eye(n_components)

    score_covariances = np.empty((n_samples, n_components, n_components))
    for sample in range(n_samples):
        root = rng.normal(scale=0.04, size=(n_components, n_components))
        score_covariances[sample] = root @ root.T + 0.02 * np.eye(n_components)

    return {
        "x": x,
        "mask": mask,
        "loadings": loadings,
        "scores": scores,
        "loading_covariances": loading_covariances,
        "score_covariances": score_covariances,
        "prior_precision": np.diag([0.11, 0.17, 0.23]),
        "noise_var": 0.37,
    }


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("num_cpu", [1, 2, 4])
def test_score_update_matches_model_equations(
    update_problem: dict[str, np.ndarray | float],
    order: str,
    num_cpu: int,
) -> None:
    """Native score updates should match the independent VB equations."""
    x = np.array(update_problem["x"], order=order)
    mask = np.array(update_problem["mask"], dtype=float, order=order)
    loadings = np.asarray(update_problem["loadings"])
    loading_covariances = np.asarray(update_problem["loading_covariances"])
    noise_var = float(update_problem["noise_var"])
    expected_scores, expected_covariances = _score_reference(
        x,
        mask,
        loadings,
        loading_covariances,
        noise_var,
    )

    result = kernels.score_update_dense_masked_nopattern(
        x_data=x,
        mask=mask,
        loadings=loadings,
        loading_covariances=loading_covariances,
        noise_var=noise_var,
        return_covariances=True,
        num_cpu=num_cpu,
    )

    assert_allclose(result["scores"], expected_scores, rtol=1e-12, atol=1e-13)
    assert_allclose(
        result["score_covariances"],
        expected_covariances,
        rtol=1e-12,
        atol=1e-13,
    )


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("num_cpu", [1, 2, 4])
def test_loading_update_matches_model_equations(
    update_problem: dict[str, np.ndarray | float],
    order: str,
    num_cpu: int,
) -> None:
    """Native loading updates should match the independent VB equations."""
    x = np.array(update_problem["x"], order=order)
    mask = np.array(update_problem["mask"], dtype=float, order=order)
    scores = np.asarray(update_problem["scores"])
    score_covariances = np.asarray(update_problem["score_covariances"])
    prior_precision = np.asarray(update_problem["prior_precision"])
    noise_var = float(update_problem["noise_var"])
    expected_loadings, expected_covariances = _loading_reference(
        x,
        mask,
        scores,
        score_covariances,
        prior_precision,
        noise_var,
    )

    result = kernels.loadings_update_dense_masked_nopattern(
        x_data=x,
        mask=mask,
        scores=scores,
        score_covariances=score_covariances,
        prior_prec=prior_precision,
        noise_var=noise_var,
        return_covariances=True,
        num_cpu=num_cpu,
    )

    assert_allclose(result["loadings"], expected_loadings, rtol=1e-12, atol=1e-13)
    assert_allclose(
        result["loading_covariances"],
        expected_covariances,
        rtol=1e-12,
        atol=1e-13,
    )
