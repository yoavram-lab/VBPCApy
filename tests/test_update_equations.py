"""Equation-level tests for the dense masked VB-PCA updates."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from vbpca_py import dense_update_kernels as kernels
from vbpca_py import noise_update_kernels, sparse_update_kernels


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


@pytest.mark.parametrize("use_complement", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("num_cpu", [1, 2, 4])
def test_score_update_matches_model_equations(
    update_problem: dict[str, np.ndarray | float],
    use_complement: bool,
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
        use_complement=use_complement,
        num_cpu=num_cpu,
    )

    assert_allclose(result["scores"], expected_scores, rtol=1e-12, atol=1e-13)
    assert_allclose(
        result["score_covariances"],
        expected_covariances,
        rtol=1e-12,
        atol=1e-13,
    )


@pytest.mark.parametrize("use_complement", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("num_cpu", [1, 2, 4])
def test_loading_update_matches_model_equations(
    update_problem: dict[str, np.ndarray | float],
    use_complement: bool,
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
        use_complement=use_complement,
        num_cpu=num_cpu,
    )

    assert_allclose(result["loadings"], expected_loadings, rtol=1e-12, atol=1e-13)
    assert_allclose(
        result["loading_covariances"],
        expected_covariances,
        rtol=1e-12,
        atol=1e-13,
    )


@pytest.mark.parametrize("num_cpu", [1, 4])
def test_sparse_updates_match_model_equations(update_problem, num_cpu) -> None:
    """CSR/CSC updates retain an explicitly stored observed zero."""
    x = np.array(update_problem["x"], copy=True)
    x[0, 0] = 0.0
    mask = np.asarray(update_problem["mask"])
    rows, columns = np.where(mask)
    sparse = sp.coo_matrix((x[rows, columns], (rows, columns)), shape=x.shape)
    loadings = np.asarray(update_problem["loadings"])
    scores = np.asarray(update_problem["scores"])
    av = np.asarray(update_problem["loading_covariances"])
    sv = np.asarray(update_problem["score_covariances"])
    precision = np.asarray(update_problem["prior_precision"])
    noise = float(update_problem["noise_var"])
    csc, csr = sparse.tocsc(), sparse.tocsr()
    expected_s, expected_sv = _score_reference(x, mask, loadings, av, noise)
    result_s = sparse_update_kernels.score_update_sparse_nopattern(
        csc.data,
        csc.indices,
        csc.indptr,
        loadings,
        av,
        noise,
        return_covariances=True,
        num_cpu=num_cpu,
    )
    expected_a, expected_av = _loading_reference(x, mask, scores, sv, precision, noise)
    result_a = sparse_update_kernels.loadings_update_sparse_nopattern(
        csr.data,
        csr.indices,
        csr.indptr,
        scores,
        sv,
        precision,
        noise,
        return_covariances=True,
        num_cpu=num_cpu,
    )
    for actual, expected in [
        (result_s["scores"], expected_s),
        (result_s["score_covariances"], expected_sv),
        (result_a["loadings"], expected_a),
        (result_a["loading_covariances"], expected_av),
    ]:
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("num_cpu", [1, 4])
@pytest.mark.parametrize("with_loading_covariance", [False, True])
def test_noise_kernel_matches_second_moments(
    update_problem, num_cpu, with_loading_covariance
) -> None:
    """Independent factor second moments determine predictive uncertainty."""
    loadings = np.asarray(update_problem["loadings"])
    scores = np.asarray(update_problem["scores"])
    av = np.asarray(update_problem["loading_covariances"])
    sv = np.asarray(update_problem["score_covariances"])
    rows, columns = np.where(update_problem["mask"])
    expected = 0.0
    for i, j in zip(rows, columns, strict=True):
        a, s = loadings[i], scores[:, j]
        second_a = np.outer(a, a) + (av[i] if with_loading_covariance else 0.0)
        second_s = np.outer(s, s) + sv[j]
        expected += float(np.trace(second_a @ second_s) - (a @ s) ** 2)
    actual = noise_update_kernels.noise_sxv_sum(
        rows.astype(np.int32),
        columns.astype(np.int32),
        loadings,
        scores,
        sv,
        av if with_loading_covariance else None,
        num_cpu,
    )
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
