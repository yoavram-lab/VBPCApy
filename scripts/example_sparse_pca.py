#!/usr/bin/env python
"""Example: VB-PCA on a sparse CSR matrix.

Generates a low-rank dense signal, samples a sparse observation pattern with
``scipy.sparse.random``, fits VB-PCA with the mask inferred from that
pattern, selects the number of components with a cost metric, and compares
reconstruction error against a dense-mask baseline.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from vbpca_py import VBPCA, SelectionConfig, select_n_components

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    """Run a VB-PCA sparse CSR demonstration."""
    rng = np.random.default_rng(42)

    n_features, n_samples, true_rank = 50, 160, 4
    w_true = rng.standard_normal((n_features, true_rank))
    s_true = rng.standard_normal((true_rank, n_samples))
    noise_std = 0.3
    x_clean = w_true @ s_true + noise_std * rng.standard_normal(
        (n_features, n_samples)
    )

    # Observation pattern from scipy.sparse.random; stored entries = observed.
    density = 0.35
    pattern = sp.random(
        n_features,
        n_samples,
        density=density,
        format="csr",
        dtype=float,
        random_state=42,
    )
    rows, cols = pattern.nonzero()
    x_sparse = pattern.copy()
    x_sparse.data = x_clean[rows, cols]
    observed = np.zeros((n_features, n_samples), dtype=bool)
    observed[rows, cols] = True

    log.info(
        "Data: %d features \u00d7 %d samples, true rank = %d, nnz = %d (%.1f %% dense)",
        n_features,
        n_samples,
        true_rank,
        x_sparse.nnz,
        100.0 * x_sparse.nnz / (n_features * n_samples),
    )

    cfg = SelectionConfig(metric="cost", patience=2, max_trials=10)
    best_k, metrics, _trace, _ = select_n_components(
        x_sparse, components=range(1, 12), config=cfg, maxiters=200
    )
    log.info("Selected k = %d  (true rank = %d)", best_k, true_rank)
    if "cost" in metrics:
        log.info("  cost = %.4f", float(metrics["cost"]))
    if "rms" in metrics:
        log.info("  rms  = %.4f", float(metrics["rms"]))

    model_sparse = VBPCA(n_components=best_k, maxiters=200)
    model_sparse.fit(x_sparse)

    x_dense = np.where(observed, x_clean, np.nan)
    model_dense = VBPCA(n_components=best_k, maxiters=200)
    model_dense.fit(x_dense, mask=observed)

    x_hat_sparse = model_sparse.inverse_transform()
    x_hat_dense = model_dense.inverse_transform()

    rmse_sparse_obs = float(
        np.sqrt(np.mean((x_hat_sparse[observed] - x_clean[observed]) ** 2))
    )
    rmse_dense_obs = float(
        np.sqrt(np.mean((x_hat_dense[observed] - x_clean[observed]) ** 2))
    )
    held_out = ~observed
    rmse_sparse_held = float(
        np.sqrt(np.mean((x_hat_sparse[held_out] - x_clean[held_out]) ** 2))
    )
    rmse_dense_held = float(
        np.sqrt(np.mean((x_hat_dense[held_out] - x_clean[held_out]) ** 2))
    )

    log.info(
        "Sparse reconstruction RMSE \u2014 observed: %.4f, held-out: %.4f",
        rmse_sparse_obs,
        rmse_sparse_held,
    )
    log.info(
        "Dense  reconstruction RMSE \u2014 observed: %.4f, held-out: %.4f",
        rmse_dense_obs,
        rmse_dense_held,
    )
    log.info(
        "Observed RMSE |sparse \u2212 dense| = %.4e  (noise \u03c3 = %.2f)",
        abs(rmse_sparse_obs - rmse_dense_obs),
        noise_std,
    )

    ev = model_sparse.explained_variance_ratio_
    if ev is not None:
        log.info(
            "Explained variance (top %d components): %s",
            best_k,
            ", ".join(f"{v:.3f}" for v in ev[:best_k]),
        )

    try:
        from vbpca_py.plotting import scree_plot

        fig = scree_plot(model_sparse)
        fig.savefig("example_sparse_pca_scree.png", dpi=120, bbox_inches="tight")
        log.info("Wrote scree plot to example_sparse_pca_scree.png")
    except Exception as exc:  # noqa: BLE001 — plotting is optional
        log.info("Skipping scree plot (%s)", exc)

    log.info("Done.")


if __name__ == "__main__":
    main()
