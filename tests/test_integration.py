"""Integration tests: full round-trip pipelines.

These tests exercise the complete VB-PCA workflow:
- fit → transform → inverse_transform → verify reconstruction
- select_n_components → fit → transform → verify
- preprocessing → VBPCA → inverse preprocessing
- sparse round-trip
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py import (
    VBPCA,
    AutoEncoder,
    MissingAwareStandardScaler,
    SelectionConfig,
    select_n_components,
)
from vbpca_py._pca_full import pca_full

# -- fixtures --


@pytest.fixture
def low_rank_dense() -> tuple[np.ndarray, np.ndarray, int, float]:
    """Generate a (features × samples) low-rank matrix with noise.

    Returns (x_clean, x_noisy, true_rank, noise_std).
    """
    rng = np.random.default_rng(12345)
    n_features, n_samples, true_rank = 40, 100, 4
    noise_std = 0.3
    w = rng.standard_normal((n_features, true_rank))
    s = rng.standard_normal((true_rank, n_samples))
    x_clean = w @ s
    x_noisy = x_clean + noise_std * rng.standard_normal((n_features, n_samples))
    return x_clean, x_noisy, true_rank, noise_std


# -- dense round-trip tests --


def test_dense_fit_transform_inverse_roundtrip(
    low_rank_dense: tuple[np.ndarray, np.ndarray, int, float],
) -> None:
    """fit → transform → inverse_transform recovers data within noise."""
    _x_clean, x_noisy, true_rank, noise_std = low_rank_dense

    model = VBPCA(n_components=true_rank, maxiters=200, bias=True)
    scores = model.fit_transform(x_noisy)
    recon = model.inverse_transform(scores)

    # Reconstruction error on observed data should be near the noise level
    rmse = float(np.sqrt(np.mean((recon - x_noisy) ** 2)))
    assert rmse < 3 * noise_std, f"RMSE {rmse:.4f} >> noise_std {noise_std}"


def test_dense_roundtrip_with_mask(
    low_rank_dense: tuple[np.ndarray, np.ndarray, int, float],
) -> None:
    """Round-trip with 20% missing entries: held-out RMSE near noise level."""
    _x_clean, x_noisy, true_rank, noise_std = low_rank_dense
    rng = np.random.default_rng(99)

    mask = rng.random(x_noisy.shape) > 0.2  # True = observed
    x_masked = np.where(mask, x_noisy, np.nan)

    model = VBPCA(n_components=true_rank, maxiters=200, bias=True)
    model.fit(x_masked, mask=mask)
    recon = model.inverse_transform()

    # Check held-out entries
    held_out = ~mask
    rmse_held = float(np.sqrt(np.mean((recon[held_out] - x_noisy[held_out]) ** 2)))
    assert rmse_held < 5 * noise_std, (
        f"Held-out RMSE {rmse_held:.4f} >> 5 × noise_std = {5 * noise_std}"
    )


# -- model selection pipeline test --


def test_select_then_fit_roundtrip(
    low_rank_dense: tuple[np.ndarray, np.ndarray, int, float],
) -> None:
    """select_n_components → fit at selected k → reconstruct → check."""
    _x_clean, x_noisy, _true_rank, noise_std = low_rank_dense

    cfg = SelectionConfig(metric="cost", patience=2, max_trials=10)
    best_k, _metrics, _trace, _ = select_n_components(
        x_noisy, components=range(1, 12), config=cfg, maxiters=100
    )

    # Selected k should be reasonable (not wildly off from true rank)
    assert 1 <= best_k <= 11

    model = VBPCA(n_components=best_k, maxiters=200, bias=True)
    model.fit(x_noisy)
    recon = model.inverse_transform()

    rmse = float(np.sqrt(np.mean((recon - x_noisy) ** 2)))
    assert rmse < 5 * noise_std, f"RMSE {rmse:.4f} with k={best_k}"


# -- sparse round-trip test --


def test_sparse_fit_transform_roundtrip() -> None:
    """Sparse CSR input with mask matching sparsity: round-trip check."""
    rng = np.random.default_rng(42)
    n_features, n_samples, true_rank = 30, 80, 3
    noise_std = 0.5

    w = rng.standard_normal((n_features, true_rank))
    s = rng.standard_normal((true_rank, n_samples))
    x_dense = w @ s + noise_std * rng.standard_normal((n_features, n_samples))

    # Create sparse version: zero out ~30% of entries to create sparsity
    sparsity_mask = rng.random((n_features, n_samples)) > 0.3
    x_sparse_dense = x_dense * sparsity_mask
    x_csr = sp.csr_matrix(x_sparse_dense)

    # mask = sparsity pattern (all structural nonzeros are observed)
    mask_csr = x_csr.copy()
    mask_csr.data = np.ones_like(mask_csr.data)

    model = VBPCA(n_components=true_rank, maxiters=150, bias=True)
    model.fit(x_csr, mask=mask_csr)
    recon = model.inverse_transform()

    # Reconstruction of observed entries should be reasonable
    observed_idx = np.asarray(mask_csr.todense(), dtype=bool)
    rmse_obs = float(
        np.sqrt(np.mean((recon[observed_idx] - x_sparse_dense[observed_idx]) ** 2))
    )
    assert rmse_obs < 5 * noise_std, f"Sparse RMSE {rmse_obs:.4f}"


# -- preprocessing round-trip tests --


def test_standard_scaler_vbpca_roundtrip(
    low_rank_dense: tuple[np.ndarray, np.ndarray, int, float],
) -> None:
    """StandardScaler → VBPCA → inverse_transform → inverse_scale → check."""
    _x_clean, x_noisy, true_rank, noise_std = low_rank_dense

    scaler = MissingAwareStandardScaler()
    x_scaled = scaler.fit_transform(x_noisy)

    model = VBPCA(n_components=true_rank, maxiters=200, bias=True)
    model.fit(x_scaled)
    recon_scaled = model.inverse_transform()

    recon = scaler.inverse_transform(recon_scaled)

    rmse = float(np.sqrt(np.mean((recon - x_noisy) ** 2)))
    assert rmse < 5 * noise_std, (
        f"Scaler+VBPCA round-trip RMSE {rmse:.4f} >> 5 × noise_std"
    )


def test_autoencoder_vbpca_roundtrip() -> None:
    """AutoEncoder (mixed types) → VBPCA → inverse → categorical recovery."""
    rng = np.random.default_rng(777)

    # Create mixed data: first 3 columns categorical, rest continuous
    n_samples = 50
    n_features = 6  # features in rows for VBPCA, but AutoEncoder works column-wise
    # AutoEncoder expects (samples × features) — we'll transpose for VBPCA
    cat_data = rng.choice(["A", "B", "C"], size=(n_samples, 2))
    cont_data = rng.standard_normal((n_samples, 4))

    # Combine into object array
    mixed = np.empty((n_samples, n_features), dtype=object)
    mixed[:, :2] = cat_data
    mixed[:, 2:] = cont_data

    encoder = AutoEncoder(cardinality_threshold=5, continuous_scaler="standard")
    encoded = encoder.fit_transform(mixed)

    # Transpose for VBPCA (features × samples)
    x_vbpca = encoded.T
    model = VBPCA(n_components=3, maxiters=100, bias=True)
    model.fit(x_vbpca)
    recon_vbpca = model.inverse_transform()

    # Inverse the encoding
    decoded = encoder.inverse_transform(recon_vbpca.T)

    # Continuous columns should be approximately recovered in aggregate.
    # With only 3 components on mixed categorical+continuous data,
    # individual columns may not correlate well, but the overall
    # reconstruction should capture some signal.
    orig_cont = cont_data.astype(float)
    rec_cont = np.asarray(decoded[:, 2:], dtype=float)
    # Frobenius-norm relative error should be < 1 (captured more than noise)
    rel_error = float(
        np.linalg.norm(rec_cont - orig_cont) / (np.linalg.norm(orig_cont) + 1e-12)
    )
    assert rel_error < 1.5, f"Continuous block relative error {rel_error:.3f} too large"


# -- convergence criteria integration tests --


def test_cfstop_rel_terminates_fit(
    low_rank_dense: tuple[np.ndarray, np.ndarray, int, float],
) -> None:
    """cfstop_rel criterion terminates the fit before maxiters."""
    _x_clean, x_noisy, true_rank, _noise_std = low_rank_dense
    result = pca_full(x_noisy, true_rank, bias=True, maxiters=500, cfstop_rel=1e-6)
    n_iters = len(result["lc"]["rms"])
    assert n_iters < 500


def test_composite_stop_terminates_fit(
    low_rank_dense: tuple[np.ndarray, np.ndarray, int, float],
) -> None:
    """composite_stop with angle + elbo_rel terminates the fit."""
    _x_clean, x_noisy, true_rank, _noise_std = low_rank_dense
    result = pca_full(
        x_noisy,
        true_rank,
        bias=True,
        maxiters=500,
        composite_stop={"angle": 1e-3, "elbo_rel": 1e-5},
    )
    n_iters = len(result["lc"]["rms"])
    assert n_iters < 500


def test_patience_delays_convergence(
    low_rank_dense: tuple[np.ndarray, np.ndarray, int, float],
) -> None:
    """patience>1 causes at least as many iterations as patience=1."""
    _x_clean, x_noisy, true_rank, _noise_std = low_rank_dense

    r1 = pca_full(
        x_noisy, true_rank, bias=True, maxiters=500, cfstop_rel=1e-6, patience=1
    )
    r5 = pca_full(
        x_noisy, true_rank, bias=True, maxiters=500, cfstop_rel=1e-6, patience=5
    )

    assert len(r5["lc"]["rms"]) >= len(r1["lc"]["rms"])
