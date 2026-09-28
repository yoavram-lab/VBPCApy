"""Scoring helpers for the modern-defaults study."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._modern_defaults_data import SimulationData


def mean_only_fit(data: SimulationData) -> tuple[np.ndarray, np.ndarray]:
    """Fit the rank-zero model and return predictive moments."""
    mask = data.train_mask
    counts = np.sum(mask, axis=1)
    means = np.divide(
        np.sum(np.where(mask, data.x_true, 0.0), axis=1),
        counts,
        out=np.zeros(data.x_true.shape[0], dtype=float),
        where=counts > 0,
    )
    reconstruction = np.broadcast_to(means[:, np.newaxis], data.x_true.shape).copy()
    residual = data.x_true[mask] - reconstruction[mask]
    noise_variance = max(float(np.mean(residual**2)), np.finfo(float).eps)
    mean_variance = noise_variance / counts
    predictive_variance = np.broadcast_to(
        (noise_variance + mean_variance)[:, np.newaxis],
        data.x_true.shape,
    ).copy()
    return reconstruction, predictive_variance


def projection_distance(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Return normalized projector distance for possibly unequal ranks."""
    true_rank = truth.shape[1]
    estimated_rank = estimate.shape[1]
    if true_rank + estimated_rank == 0:
        return 0.0
    q_true = np.linalg.qr(truth, mode="reduced")[0]
    q_estimate = np.linalg.qr(estimate, mode="reduced")[0]
    overlap = float(np.sum((q_true.T @ q_estimate) ** 2))
    squared = max(0.0, true_rank + estimated_rank - 2.0 * overlap)
    return float(np.sqrt(squared / (true_rank + estimated_rank)))


def rms_tail_diagnostics(values: list[float]) -> tuple[float, float]:
    """Measure non-monotonicity and detrended alternating tail amplitude."""
    rms = np.asarray(values, dtype=float)
    rms = rms[np.isfinite(rms)]
    if rms.size < 3:
        return float("nan"), float("nan")
    increase_fraction = float(np.mean(np.diff(rms) > 0.0))
    tail = rms[-min(20, rms.size) :]
    index = np.arange(tail.size, dtype=float)
    slope, intercept = np.polyfit(index, tail, deg=1)
    detrended = tail - (slope * index + intercept)
    even = detrended[::2]
    odd = detrended[1::2]
    if odd.size == 0:
        return increase_fraction, float("nan")
    amplitude = abs(float(np.mean(even) - np.mean(odd)))
    scale = max(float(np.mean(np.abs(tail))), np.finfo(float).eps)
    return increase_fraction, amplitude / scale


def predictive_scores(
    data: SimulationData,
    reconstruction: np.ndarray,
    predictive_variance: np.ndarray,
) -> dict[str, float]:
    """Score held-out prediction and interval calibration."""
    holdout = data.holdout_mask
    residual = data.x_true[holdout] - reconstruction[holdout]
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    std = np.sqrt(np.maximum(predictive_variance, 0.0))
    lower = reconstruction - 1.96 * std
    upper = reconstruction + 1.96 * std
    truth = data.x_true[holdout]
    lower_holdout = lower[holdout]
    upper_holdout = upper[holdout]
    coverage = float(np.mean((truth >= lower_holdout) & (truth <= upper_holdout)))
    width = upper_holdout - lower_holdout
    below = truth < lower_holdout
    above = truth > upper_holdout
    interval_score = width.copy()
    interval_score += 40.0 * (lower_holdout - truth) * below
    interval_score += 40.0 * (truth - upper_holdout) * above
    return {
        "holdout_rmse": rmse,
        "holdout_mae": mae,
        "coverage_95": coverage,
        "mean_interval_width": float(np.mean(width)),
        "interval_score": float(np.mean(interval_score)),
    }
