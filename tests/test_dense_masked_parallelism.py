import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from vbpca_py import dense_update_kernels as duk


def _fixture():
    rng = np.random.default_rng(42)
    n_features = 5
    n_samples = 4
    n_components = 3

    x_data = rng.normal(size=(n_features, n_samples))
    mask = (rng.random(size=(n_features, n_samples)) > 0.25).astype(float)
    loadings = rng.normal(scale=0.5, size=(n_features, n_components))
    scores = rng.normal(scale=0.5, size=(n_components, n_samples))

    av = np.stack([np.eye(n_components) * 0.05 for _ in range(n_features)])
    sv = np.stack([np.eye(n_components) * 0.02 for _ in range(n_samples)])
    prior_prec = np.eye(n_components) * 0.1
    return x_data, mask, loadings, scores, av, sv, prior_prec


def test_score_update_dense_masked_parallel_parity() -> None:
    x_data, mask, loadings, _scores, av, _sv, _prior_prec = _fixture()

    base = duk.score_update_dense_masked_nopattern(
        x_data=x_data,
        mask=mask,
        loadings=loadings,
        loading_covariances=av,
        noise_var=0.5,
        return_covariances=True,
        num_cpu=0,
    )

    threaded = duk.score_update_dense_masked_nopattern(
        x_data=x_data,
        mask=mask,
        loadings=loadings,
        loading_covariances=av,
        noise_var=0.5,
        return_covariances=True,
        num_cpu=2,
    )

    assert np.allclose(base["scores"], threaded["scores"])
    assert np.allclose(base["score_covariances"], threaded["score_covariances"])


def test_loadings_update_dense_masked_parallel_parity() -> None:
    x_data, mask, _loadings, scores, _av, sv, prior_prec = _fixture()

    base = duk.loadings_update_dense_masked_nopattern(
        x_data=x_data,
        mask=mask,
        scores=scores,
        score_covariances=sv,
        prior_prec=prior_prec,
        noise_var=0.5,
        return_covariances=True,
        num_cpu=0,
    )

    threaded = duk.loadings_update_dense_masked_nopattern(
        x_data=x_data,
        mask=mask,
        scores=scores,
        score_covariances=sv,
        prior_prec=prior_prec,
        noise_var=0.5,
        return_covariances=True,
        num_cpu=2,
    )

    assert np.allclose(base["loadings"], threaded["loadings"])
    assert np.allclose(base["loading_covariances"], threaded["loading_covariances"])


@pytest.mark.perf
@pytest.mark.parametrize("num_cpu", [0, 4], ids=["serial", "num_cpu_4"])
def test_benchmark_score_update_dense_masked_parallel(
    benchmark: BenchmarkFixture,
    num_cpu: int,
) -> None:
    """Benchmark score update with num_cpu=4 to validate speedups."""
    rng = np.random.default_rng(123)
    n_features, n_samples, n_components = 200, 200, 10
    x_data = rng.standard_normal((n_features, n_samples))
    mask = (rng.random((n_features, n_samples)) > 0.1).astype(float)
    loadings = rng.standard_normal((n_features, n_components))
    av = np.stack([np.eye(n_components) * 0.05 for _ in range(n_features)])

    def run_once() -> dict[str, np.ndarray]:
        return duk.score_update_dense_masked_nopattern(
            x_data=x_data,
            mask=mask,
            loadings=loadings,
            loading_covariances=av,
            noise_var=0.5,
            return_covariances=True,
            num_cpu=num_cpu,
        )

    result = benchmark(run_once)
    assert np.asarray(result["scores"]).shape == (n_components, n_samples)


@pytest.mark.perf
@pytest.mark.parametrize("num_cpu", [0, 4], ids=["serial", "num_cpu_4"])
def test_benchmark_loadings_update_dense_masked_parallel(
    benchmark: BenchmarkFixture,
    num_cpu: int,
) -> None:
    """Benchmark loadings update with num_cpu=4 to validate speedups."""
    rng = np.random.default_rng(456)
    n_features, n_samples, n_components = 200, 200, 10
    x_data = rng.standard_normal((n_features, n_samples))
    mask = (rng.random((n_features, n_samples)) > 0.1).astype(float)
    scores = rng.standard_normal((n_components, n_samples))
    sv = np.stack([np.eye(n_components) * 0.02 for _ in range(n_samples)])
    prior_prec = np.eye(n_components) * 0.1

    def run_once() -> dict[str, np.ndarray]:
        return duk.loadings_update_dense_masked_nopattern(
            x_data=x_data,
            mask=mask,
            scores=scores,
            score_covariances=sv,
            prior_prec=prior_prec,
            noise_var=0.5,
            return_covariances=True,
            num_cpu=num_cpu,
        )

    result = benchmark(run_once)
    assert np.asarray(result["loadings"]).shape == (n_features, n_components)
