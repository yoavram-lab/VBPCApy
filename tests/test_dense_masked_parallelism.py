import numpy as np
import pytest
import scipy.sparse as sp
from pytest_benchmark.fixture import BenchmarkFixture

from vbpca_py import dense_update_kernels as duk
from vbpca_py._full_update import (
    LoadingsUpdateState,
    ScoreState,
    _covariances_stack,
    _loadings_update_general,
    _score_update_fast_dense_no_av,
    _score_update_general_no_patterns,
)


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


def test_score_update_fast_dense_bulk_writeback_equivalence() -> None:
    rng = np.random.default_rng(7)
    n_features, n_samples, n_components = 6, 5, 3
    x_data = rng.standard_normal((n_features, n_samples))
    loadings = rng.standard_normal((n_features, n_components))
    mask = np.ones_like(x_data)

    def _make_state(mode: str, log_stride: int) -> ScoreState:
        scores = np.zeros((n_components, n_samples), dtype=float)
        sv = [np.eye(n_components) for _ in range(n_samples)]
        return ScoreState(
            x_data=x_data,
            mask=mask,
            loadings=loadings,
            scores=scores,
            loading_covariances=[],
            score_covariances=list(sv),
            pattern_index=None,
            obs_patterns=[],
            noise_var=0.3,
            eye_components=np.eye(n_components),
            verbose=0,
            cov_writeback_mode=mode,
            log_progress_stride=log_stride,
        )

    base = _score_update_fast_dense_no_av(_make_state("python", 1))
    bulk = _score_update_fast_dense_no_av(_make_state("bulk", 0))

    assert np.allclose(base.scores, bulk.scores)
    assert len(bulk.score_covariances) == n_samples
    assert all(
        np.allclose(cov, base.score_covariances[0]) for cov in bulk.score_covariances
    )


def test_sparse_score_accessors_buffered_equivalence() -> None:
    rng = np.random.default_rng(9)
    n_features, n_samples, n_components = 4, 3, 2

    data = rng.standard_normal((n_features, n_samples))
    mask_dense = np.ones_like(data)
    x_sparse = sp.csc_matrix(data)
    mask_sparse = sp.csc_matrix(mask_dense)

    loadings = rng.standard_normal((n_features, n_components))
    scores = np.zeros((n_components, n_samples), dtype=float)

    def _state(mode: str) -> ScoreState:
        return ScoreState(
            x_data=x_sparse,
            mask=mask_sparse,
            loadings=loadings,
            scores=scores.copy(),
            loading_covariances=[],
            score_covariances=[np.eye(n_components) for _ in range(n_samples)],
            pattern_index=None,
            obs_patterns=[],
            noise_var=0.4,
            eye_components=np.eye(n_components),
            verbose=0,
            pattern_batch_size=0,
            accessor_mode=mode,
            use_python_scores=True,
        )

    legacy = _score_update_general_no_patterns(_state("legacy"))
    buffered = _score_update_general_no_patterns(_state("buffered"))

    assert np.allclose(legacy.scores, buffered.scores)
    for l_cov, b_cov in zip(
        legacy.score_covariances,
        buffered.score_covariances,
        strict=False,
    ):
        assert np.allclose(l_cov, b_cov)


def test_sparse_loadings_accessors_buffered_equivalence() -> None:
    rng = np.random.default_rng(10)
    n_features, n_samples, n_components = 5, 4, 3

    data = rng.standard_normal((n_features, n_samples))
    mask_dense = np.ones_like(data)
    x_sparse = sp.csr_matrix(data)
    mask_sparse = sp.csr_matrix(mask_dense)

    scores = rng.standard_normal((n_components, n_samples))
    loadings_cov = [np.eye(n_components) for _ in range(n_features)]

    def _state(mode: str) -> LoadingsUpdateState:
        return LoadingsUpdateState(
            x_data=x_sparse,
            mask=mask_sparse,
            scores=scores,
            loading_covariances=loadings_cov.copy(),
            score_covariances=[],
            pattern_index=None,
            va=np.ones(n_components),
            noise_var=0.5,
            verbose=0,
            accessor_mode=mode,
        )

    legacy_loadings, _ = _loadings_update_general(_state("legacy"))
    buffered_loadings, _ = _loadings_update_general(_state("buffered"))

    assert np.allclose(legacy_loadings, buffered_loadings)


def test_dense_score_bulk_cov_writeback_parity_and_type() -> None:
    x_data, mask, loadings, _scores, av, _sv, _prior_prec = _fixture()
    n_components = loadings.shape[1]
    n_samples = x_data.shape[1]

    def _state(mode: str) -> ScoreState:
        return ScoreState(
            x_data=x_data,
            mask=mask,
            loadings=loadings,
            scores=np.zeros((n_components, n_samples), dtype=float),
            loading_covariances=list(av),
            score_covariances=[np.eye(n_components) for _ in range(n_samples)],
            pattern_index=None,
            obs_patterns=[],
            noise_var=0.5,
            eye_components=np.eye(n_components),
            verbose=0,
            dense_num_cpu=0,
            cov_writeback_mode=mode,
        )

    py_state = _score_update_general_no_patterns(_state("python"))
    bulk_state = _score_update_general_no_patterns(_state("bulk"))
    kernel_state = _score_update_general_no_patterns(_state("kernel"))

    assert np.allclose(py_state.scores, bulk_state.scores)
    assert np.allclose(
        _covariances_stack(py_state.score_covariances),
        _covariances_stack(bulk_state.score_covariances),
    )
    assert isinstance(bulk_state.score_covariances, np.ndarray)
    assert np.allclose(bulk_state.scores, kernel_state.scores)
    assert np.allclose(
        _covariances_stack(bulk_state.score_covariances),
        _covariances_stack(kernel_state.score_covariances),
    )
    assert isinstance(kernel_state.score_covariances, np.ndarray)


def test_sparse_score_bulk_cov_writeback_parity_and_type() -> None:
    rng = np.random.default_rng(1234)
    n_features, n_samples, n_components = 6, 5, 3

    dense = rng.standard_normal((n_features, n_samples))
    mask_dense = np.ones_like(dense)
    x_csc = sp.csc_matrix(dense)
    mask_csc = sp.csc_matrix(mask_dense)
    loadings = rng.standard_normal((n_features, n_components))

    def _state(mode: str) -> ScoreState:
        return ScoreState(
            x_data=x_csc,
            mask=mask_csc,
            loadings=loadings,
            scores=np.zeros((n_components, n_samples), dtype=float),
            loading_covariances=[np.eye(n_components) for _ in range(n_features)],
            score_covariances=[np.eye(n_components) for _ in range(n_samples)],
            pattern_index=None,
            obs_patterns=[],
            noise_var=0.2,
            eye_components=np.eye(n_components),
            verbose=0,
            sparse_num_cpu=0,
            cov_writeback_mode=mode,
        )

    py_state = _score_update_general_no_patterns(_state("python"))
    bulk_state = _score_update_general_no_patterns(_state("bulk"))
    kernel_state = _score_update_general_no_patterns(_state("kernel"))

    assert np.allclose(py_state.scores, bulk_state.scores)
    assert np.allclose(
        _covariances_stack(py_state.score_covariances),
        _covariances_stack(bulk_state.score_covariances),
    )
    assert isinstance(bulk_state.score_covariances, np.ndarray)
    assert np.allclose(bulk_state.scores, kernel_state.scores)
    assert np.allclose(
        _covariances_stack(bulk_state.score_covariances),
        _covariances_stack(kernel_state.score_covariances),
    )
    assert isinstance(kernel_state.score_covariances, np.ndarray)


def test_dense_loadings_bulk_cov_writeback_parity_and_type() -> None:
    x_data, mask, _loadings, scores, _av, sv, _prior_prec = _fixture()
    n_components = scores.shape[0]
    n_features = x_data.shape[0]

    def _state(mode: str) -> LoadingsUpdateState:
        return LoadingsUpdateState(
            x_data=x_data,
            mask=mask,
            scores=scores,
            loading_covariances=[np.eye(n_components) for _ in range(n_features)],
            score_covariances=list(sv),
            pattern_index=None,
            va=np.ones(n_components),
            noise_var=0.5,
            verbose=0,
            cov_writeback_mode=mode,
            log_progress_stride=0,
        )

    py_loadings, py_cov = _loadings_update_general(_state("python"))
    bulk_loadings, bulk_cov = _loadings_update_general(_state("bulk"))
    kernel_loadings, kernel_cov = _loadings_update_general(_state("kernel"))

    assert np.allclose(py_loadings, bulk_loadings)
    assert np.allclose(
        _covariances_stack(py_cov),
        _covariances_stack(bulk_cov),
    )
    assert isinstance(bulk_cov, np.ndarray)
    assert np.allclose(bulk_loadings, kernel_loadings)
    assert np.allclose(
        _covariances_stack(bulk_cov),
        _covariances_stack(kernel_cov),
    )
    assert isinstance(kernel_cov, np.ndarray)


def test_sparse_loadings_bulk_cov_writeback_parity_and_type() -> None:
    rng = np.random.default_rng(456)
    n_features, n_samples, n_components = 5, 4, 3
    dense = rng.standard_normal((n_features, n_samples))
    mask_dense = np.ones_like(dense)
    x_csr = sp.csr_matrix(dense)
    mask_csr = sp.csr_matrix(mask_dense)
    scores = rng.standard_normal((n_components, n_samples))

    def _state(mode: str) -> LoadingsUpdateState:
        return LoadingsUpdateState(
            x_data=x_csr,
            mask=mask_csr,
            scores=scores,
            loading_covariances=[np.eye(n_components) for _ in range(n_features)],
            score_covariances=[np.eye(n_components) for _ in range(n_samples)],
            pattern_index=None,
            va=np.ones(n_components),
            noise_var=0.5,
            verbose=0,
            sparse_num_cpu=0,
            cov_writeback_mode=mode,
        )

    py_loadings, py_cov = _loadings_update_general(_state("python"))
    bulk_loadings, bulk_cov = _loadings_update_general(_state("bulk"))
    kernel_loadings, kernel_cov = _loadings_update_general(_state("kernel"))

    assert np.allclose(py_loadings, bulk_loadings)
    assert np.allclose(
        _covariances_stack(py_cov),
        _covariances_stack(bulk_cov),
    )
    assert isinstance(bulk_cov, np.ndarray)
    assert np.allclose(bulk_loadings, kernel_loadings)
    assert np.allclose(
        _covariances_stack(bulk_cov),
        _covariances_stack(kernel_cov),
    )
    assert isinstance(kernel_cov, np.ndarray)


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
