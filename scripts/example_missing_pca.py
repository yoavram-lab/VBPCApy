#!/usr/bin/env python
"""Minimal example: VB-PCA on data with missing entries.

Generates a low-rank matrix, masks a fraction of entries as missing,
fits VB-PCA, selects the number of components, and reports reconstruction
quality on the held-out entries.
"""

from __future__ import annotations

import logging

import numpy as np

from vbpca_py import VBPCA, SelectionConfig, select_n_components

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    """Run a VB-PCA missing-data demonstration."""
    rng = np.random.default_rng(42)

    # -- generate low-rank data (features × samples) --
    n_features, n_samples, true_rank = 60, 200, 5
    w_true = rng.standard_normal((n_features, true_rank))
    s_true = rng.standard_normal((true_rank, n_samples))
    noise_std = 0.5
    x_clean = w_true @ s_true + noise_std * rng.standard_normal((n_features, n_samples))

    # -- introduce 20 % missing entries --
    missing_rate = 0.20
    mask = rng.random((n_features, n_samples)) > missing_rate  # True = observed
    x_obs = np.where(mask, x_clean, np.nan)

    log.info(
        "Data: %d features × %d samples, true rank = %d, %.0f %% missing",
        n_features,
        n_samples,
        true_rank,
        100 * (1 - mask.mean()),
    )

    # -- select number of components via RMS-based model selection --
    cfg = SelectionConfig(metric="rms", patience=2, max_trials=12)
    best_k, metrics, _trace, _ = select_n_components(
        x_obs, mask=mask, components=range(1, 15), config=cfg, maxiters=200
    )
    log.info("Selected k = %d  (true rank = %d)", best_k, true_rank)
    log.info("  rms = %.4f", metrics["rms"])

    # -- fit final model at chosen rank --
    model = VBPCA(n_components=best_k, maxiters=200)
    model.fit(x_obs, mask=mask)

    # -- evaluate on held-out entries --
    x_hat = model.inverse_transform()
    held_out = ~mask
    rmse_held = float(np.sqrt(np.nanmean((x_hat[held_out] - x_clean[held_out]) ** 2)))
    rmse_obs = float(np.sqrt(np.nanmean((x_hat[mask] - x_clean[mask]) ** 2)))
    log.info(
        "Reconstruction RMSE — observed: %.4f, held-out: %.4f  (noise σ = %.2f)",
        rmse_obs,
        rmse_held,
        noise_std,
    )

    # -- quick check: does the model recover roughly the right structure? --
    ev = model.explained_variance_ratio_
    if ev is not None:
        log.info(
            "Explained variance (top %d components): %s",
            best_k,
            ", ".join(f"{v:.3f}" for v in ev[:best_k]),
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
