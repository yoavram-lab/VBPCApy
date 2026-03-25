"""Property-based tests for VB-PCA invariants.

Uses hypothesis to verify mathematical properties that must hold
regardless of input data:
- ELBO (cost) is monotonically non-increasing after warmup
- Posterior variances are always positive
- Reconstruction error is finite and bounded
- Model selection returns k within the candidate range
- Explained variance ratios sum to at most 1
- Bias recovery approximates column means of observed data
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vbpca_py import VBPCA, SelectionConfig, select_n_components
from vbpca_py._pca_full import pca_full

# -- strategies for generating test data --


@st.composite
def low_rank_data(
    draw: st.DrawFn,
    *,
    min_features: int = 8,
    max_features: int = 40,
    min_samples: int = 15,
    max_samples: int = 60,
) -> tuple[np.ndarray, int]:
    """Draw a (features × samples) low-rank matrix plus noise."""
    n = draw(st.integers(min_value=min_features, max_value=max_features))
    m = draw(st.integers(min_value=min_samples, max_value=max_samples))
    rank = draw(st.integers(min_value=1, max_value=min(n, m) // 2))
    noise = draw(st.floats(min_value=0.1, max_value=2.0))

    rng = np.random.default_rng(draw(st.integers(0, 2**31)))
    w = rng.standard_normal((n, rank))
    s = rng.standard_normal((rank, m))
    x = w @ s + noise * rng.standard_normal((n, m))
    return x, rank


# -- property tests --


@given(data=low_rank_data())
@settings(max_examples=15, deadline=30_000)
def test_elbo_monotonicity(data: tuple[np.ndarray, int]) -> None:
    """After the first few warmup iterations, ELBO should not increase."""
    x, rank = data
    k = min(rank + 2, min(x.shape) - 1)

    result = pca_full(x, k, bias=True, maxiters=50, verbose=0)
    costs = np.asarray(result.get("Costs", []), dtype=float)

    # Skip if fewer than 5 cost values recorded
    if costs.size < 5:
        return

    # After warmup (first 3 iterations), cost should be non-increasing
    # Allow small numerical noise (1e-6 relative)
    post_warmup = costs[3:]
    diffs = np.diff(post_warmup)
    scale = np.abs(post_warmup[:-1]) + 1e-12
    relative_increases = diffs / scale

    # No relative increase > 1e-6
    assert np.all(relative_increases < 1e-6), (
        "ELBO increased after warmup: "
        f"max relative increase = {relative_increases.max():.2e}"
    )


@given(data=low_rank_data())
@settings(max_examples=15, deadline=30_000)
def test_reconstruction_is_finite(data: tuple[np.ndarray, int]) -> None:
    """Reconstruction from fitted model should be finite."""
    x, rank = data
    k = min(rank + 2, min(x.shape) - 1)

    model = VBPCA(n_components=k, maxiters=50)
    model.fit(x)
    recon = model.inverse_transform()

    assert np.all(np.isfinite(recon)), "Reconstruction contains non-finite values"

    # Relative reconstruction error should be bounded
    rel_err = np.linalg.norm(recon - x) / (np.linalg.norm(x) + 1e-12)
    assert rel_err < 10.0, f"Reconstruction error unreasonably large: {rel_err:.2f}"


@given(data=low_rank_data())
@settings(max_examples=15, deadline=30_000)
def test_explained_variance_ratio_sums_to_at_most_one(
    data: tuple[np.ndarray, int],
) -> None:
    """Sum of explained variance ratios must be <= 1 + epsilon."""
    x, rank = data
    k = min(rank + 2, min(x.shape) - 1)

    model = VBPCA(n_components=k, maxiters=50)
    model.fit(x)

    evr = model.explained_variance_ratio_
    if evr is None:
        pytest.skip("No explained_variance_ratio_ available")

    total = float(np.sum(evr))
    assert total <= 1.0 + 1e-6, f"EVR sums to {total:.6f}, expected <= 1.0"
    assert np.all(np.asarray(evr) >= 0), "Negative explained variance ratio"


@given(data=low_rank_data())
@settings(max_examples=10, deadline=60_000)
def test_model_selection_returns_valid_k(data: tuple[np.ndarray, int]) -> None:
    """select_n_components must return k within the candidate range."""
    x, rank = data
    max_k = min(rank + 4, min(x.shape) - 1)
    candidates = list(range(1, max_k + 1))

    cfg = SelectionConfig(metric="cost", patience=1, max_trials=len(candidates))
    best_k, metrics, trace, _ = select_n_components(
        x, components=candidates, config=cfg, maxiters=30
    )

    assert best_k in candidates, f"selected k={best_k} not in candidates {candidates}"
    assert isinstance(metrics, dict)
    assert len(trace) > 0


@given(data=low_rank_data())
@settings(max_examples=15, deadline=30_000)
def test_bias_recovery(data: tuple[np.ndarray, int]) -> None:
    """With bias=True, the estimated mean should approximate the column means."""
    x, rank = data
    k = min(rank + 2, min(x.shape) - 1)

    model = VBPCA(n_components=k, bias=True, maxiters=80)
    model.fit(x)

    assert model.mean_ is not None
    col_means = x.mean(axis=1)

    # The mean should be in the same ballpark (within 2x the data std per feature)
    feature_stds = x.std(axis=1) + 1e-12
    relative_error = np.abs(model.mean_.ravel() - col_means) / feature_stds
    median_rel_err = float(np.median(relative_error))

    assert median_rel_err < 2.0, (
        f"Median relative bias error = {median_rel_err:.2f}, expected < 2.0"
    )


@given(
    n_components=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31),
)
@settings(max_examples=10, deadline=30_000)
def test_noise_variance_positive(n_components: int, seed: int) -> None:
    """Estimated noise variance should always be positive."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((20, 50))

    model = VBPCA(n_components=n_components, maxiters=30)
    model.fit(x)

    assert model.noise_variance_ is not None
    assert model.noise_variance_ > 0, (
        f"Noise variance = {model.noise_variance_}, expected > 0"
    )
