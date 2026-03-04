"""Performance benchmarks for compat_mode comparisons.

These tests are marked ``perf`` and excluded from default test/CI runs.
Run explicitly with:

    pytest -q -m perf --benchmark-only
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py._pca_full import pca_full


def _make_dense_with_missing(
    n_features: int,
    n_samples: int,
    *,
    seed: int,
    missing_rate: float = 0.1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_features, n_samples))
    missing = rng.random(x.shape) < missing_rate
    x[missing] = np.nan
    return x


def _make_sparse(
    n_features: int,
    n_samples: int,
    *,
    seed: int,
    zero_rate: float = 0.7,
) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((n_features, n_samples))
    dense[rng.random(dense.shape) < zero_rate] = 0.0
    return sp.csr_matrix(dense)


def _make_dense_high_masked(
    n_features: int,
    n_samples: int,
    *,
    seed: int,
    missing_rate: float = 0.98,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_features, n_samples), dtype=np.float32)
    missing = rng.random(x.shape) < missing_rate
    x = x.astype(np.float64, copy=False)
    x[missing] = np.nan
    return x


def _make_sparse_high(
    n_features: int,
    n_samples: int,
    *,
    seed: int,
    zero_rate: float = 0.995,
) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((n_features, n_samples), dtype=np.float32)
    dense[rng.random(dense.shape) < zero_rate] = 0.0
    return sp.csr_matrix(dense)


def _octave_perf_available() -> bool:
    has_octave = shutil.which("octave") is not None
    has_oct2py = importlib.util.find_spec("oct2py") is not None
    return has_octave and has_oct2py


@pytest.mark.perf
@pytest.mark.parametrize("compat_mode", ["strict_legacy", "modern"])
def test_benchmark_pca_full_dense_modes(
    benchmark: pytest.BenchmarkFixture,
    compat_mode: str,
) -> None:
    """Benchmark dense pca_full runtime across compatibility modes."""
    x = _make_dense_with_missing(30, 60, seed=123)

    def run_once() -> dict[str, object]:
        return pca_full(
            x,
            n_components=6,
            maxiters=6,
            algorithm="vb",
            autosave=0,
            display=0,
            verbose=0,
            rotate2pca=0,
            compat_mode=compat_mode,
        )

    result = benchmark(run_once)
    assert np.asarray(result["A"]).shape == (30, 6)


@pytest.mark.perf
@pytest.mark.parametrize("compat_mode", ["strict_legacy", "modern"])
def test_benchmark_pca_full_sparse_modes(
    benchmark: pytest.BenchmarkFixture,
    compat_mode: str,
) -> None:
    """Benchmark sparse pca_full runtime across compatibility modes."""
    x_sparse = _make_sparse(40, 80, seed=456)

    def run_once() -> dict[str, object]:
        return pca_full(
            x_sparse,
            n_components=5,
            maxiters=5,
            algorithm="vb",
            autosave=0,
            display=0,
            verbose=0,
            rotate2pca=0,
            compat_mode=compat_mode,
        )

    result = benchmark(run_once)
    assert np.asarray(result["A"]).shape == (40, 5)


@pytest.mark.perf
@pytest.mark.parametrize(
    ("n_features", "n_samples", "n_components"),
    [(20, 40, 4), (40, 80, 8), (80, 160, 12)],
)
def test_benchmark_scaling_dense_modern(
    benchmark: pytest.BenchmarkFixture,
    n_features: int,
    n_samples: int,
    n_components: int,
) -> None:
    """Benchmark scaling for dense data as matrix size increases."""
    x = _make_dense_with_missing(n_features, n_samples, seed=777)

    def run_once() -> dict[str, object]:
        return pca_full(
            x,
            n_components=n_components,
            maxiters=4,
            algorithm="vb",
            autosave=0,
            display=0,
            verbose=0,
            rotate2pca=0,
            compat_mode="modern",
        )

    result = benchmark(run_once)
    assert np.asarray(result["A"]).shape == (n_features, n_components)


@pytest.mark.perf
@pytest.mark.parametrize(
    ("n_features", "n_samples", "n_components"),
    [(30, 60, 4), (60, 120, 8), (120, 240, 12)],
)
def test_benchmark_scaling_sparse_modern(
    benchmark: pytest.BenchmarkFixture,
    n_features: int,
    n_samples: int,
    n_components: int,
) -> None:
    """Benchmark scaling for sparse data as matrix size increases."""
    x_sparse = _make_sparse(n_features, n_samples, seed=888)

    def run_once() -> dict[str, object]:
        return pca_full(
            x_sparse,
            n_components=n_components,
            maxiters=4,
            algorithm="vb",
            autosave=0,
            display=0,
            verbose=0,
            rotate2pca=0,
            compat_mode="modern",
        )

    result = benchmark(run_once)
    assert np.asarray(result["A"]).shape == (n_features, n_components)


@pytest.mark.perf
@pytest.mark.parametrize("compat_mode", ["modern", "strict_legacy"])
def test_benchmark_high_feature_masked_dense(
    benchmark: pytest.BenchmarkFixture,
    compat_mode: str,
) -> None:
    """Smoke benchmark for high-feature masked dense path (~100k x 1k)."""
    n_features = 100_000
    n_samples = 1_000
    x = _make_dense_high_masked(n_features, n_samples, seed=2025)

    def run_once() -> dict[str, object]:
        return pca_full(
            x,
            n_components=32,
            maxiters=2,
            algorithm="vb",
            autosave=0,
            display=0,
            verbose=0,
            rotate2pca=0,
            compat_mode=compat_mode,
            auto_pattern_masked=1,
        )

    result = benchmark(run_once)
    assert np.asarray(result["A"]).shape == (n_features, 32)


@pytest.mark.perf
@pytest.mark.parametrize("compat_mode", ["modern", "strict_legacy"])
def test_benchmark_high_feature_sparse(
    benchmark: pytest.BenchmarkFixture,
    compat_mode: str,
) -> None:
    """Smoke benchmark for high-feature sparse path (~100k x 1k)."""
    n_features = 100_000
    n_samples = 1_000
    x_sparse = _make_sparse_high(n_features, n_samples, seed=2026)

    def run_once() -> dict[str, object]:
        return pca_full(
            x_sparse,
            n_components=32,
            maxiters=2,
            algorithm="vb",
            autosave=0,
            display=0,
            verbose=0,
            rotate2pca=0,
            compat_mode=compat_mode,
            auto_pattern_masked=1,
        )

    result = benchmark(run_once)
    assert np.asarray(result["A"]).shape == (n_features, 32)


@pytest.mark.perf
@pytest.mark.skipif(
    not _octave_perf_available(),
    reason="Octave and oct2py are required for legacy pca_full.m benchmarking.",
)
def test_benchmark_octave_compare_dense(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark legacy Octave tools/pca_full.m on a dense problem."""
    from oct2py import octave  # type: ignore[import-untyped]

    x = _make_dense_with_missing(20, 40, seed=999)
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    octave.addpath(str(tools_dir))

    def run_once() -> object:
        return octave.feval(
            "pca_full",
            x,
            4,
            "maxiters",
            4,
            "algorithm",
            "vb",
            "autosave",
            0,
            "display",
            0,
            "verbose",
            0,
            "rotate2pca",
            0,
            nout=1,
        )

    result = benchmark(run_once)
    assert result is not None


@pytest.mark.perf
def test_benchmark_python_compare_dense(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark Python pca_full counterpart for direct Octave comparison."""
    x = _make_dense_with_missing(20, 40, seed=999)

    def run_once() -> dict[str, object]:
        return pca_full(
            x,
            n_components=4,
            maxiters=4,
            algorithm="vb",
            autosave=0,
            display=0,
            verbose=0,
            rotate2pca=0,
            compat_mode="strict_legacy",
        )

    result = benchmark(run_once)
    assert np.asarray(result["A"]).shape == (20, 4)
