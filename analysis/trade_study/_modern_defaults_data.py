"""Paired data generation for the modern-defaults study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit


@dataclass(frozen=True, slots=True)
class SimulationData:
    """One generated matrix and its external evaluation split."""

    x_true: np.ndarray
    true_loadings: np.ndarray
    observation_mask: np.ndarray
    train_mask: np.ndarray
    holdout_mask: np.ndarray
    init_seed: int


def trial_seeds(regime_seed: int, rep: int) -> tuple[int, int]:
    """Return independent data and model-initialization seeds."""
    return regime_seed + 1_000_003 * rep, regime_seed + 2_000_003 * rep


def _generate_matrix(
    p: int,
    n: int,
    rank: int,
    noise_std: float,
    noise_model: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    loadings = rng.standard_normal((p, rank))
    scores = rng.standard_normal((rank, n))
    signal = loadings @ scores
    if noise_model == "gaussian":
        noise = rng.standard_normal((p, n))
    elif noise_model == "student_t3":
        # A t_3 variate has variance 3; standardize before applying noise_std.
        noise = rng.standard_t(df=3, size=(p, n)) / np.sqrt(3.0)
    else:
        msg = f"unknown noise_model {noise_model!r}"
        raise ValueError(msg)
    return signal + noise_std * noise, loadings


def _calibrated_mar_probability(
    anchor: np.ndarray,
    *,
    target: float,
    slope: float = 1.5,
) -> np.ndarray:
    centered = anchor - np.mean(anchor)
    scale = float(np.std(centered))
    standardized = centered / scale if scale > 0.0 else np.zeros_like(centered)
    lower, upper = -20.0, 20.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if float(np.mean(expit(midpoint + slope * standardized))) < target:
            lower = midpoint
        else:
            upper = midpoint
    return expit(0.5 * (lower + upper) + slope * standardized)


def _repair_support(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    repaired = np.array(mask, dtype=bool, copy=True)
    for row in np.flatnonzero(~repaired.any(axis=1)):
        repaired[row, rng.integers(repaired.shape[1])] = True
    for column in np.flatnonzero(~repaired.any(axis=0)):
        repaired[rng.integers(repaired.shape[0]), column] = True
    return repaired


def apply_missingness(
    x: np.ndarray,
    mechanism: str,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an observation mask for one registered mechanism."""
    if not 0.0 <= fraction < 1.0:
        msg = f"missing fraction must be in [0, 1), got {fraction}"
        raise ValueError(msg)
    p, n = x.shape
    if mechanism == "complete":
        mask = np.ones_like(x, dtype=bool)
    elif mechanism == "mcar":
        mask = rng.random(x.shape) >= fraction
    elif mechanism == "mar":
        mask = np.ones_like(x, dtype=bool)
        if p > 1:
            conditional_target = min(0.99, fraction * p / (p - 1))
            positive = _calibrated_mar_probability(
                x[0],
                target=conditional_target,
            )
            for row in range(1, p):
                probability = positive if row % 2 else positive[::-1]
                mask[row] = rng.random(n) >= probability
    elif mechanism == "mnar_censored":
        threshold = np.quantile(x, fraction, axis=1, keepdims=True)
        mask = x > threshold
    elif mechanism == "block":
        mask = np.ones_like(x, dtype=bool)
        side_fraction = np.sqrt(fraction)
        n_rows = max(1, min(p, int(round(p * side_fraction))))
        n_columns = max(1, min(n, int(round(n * side_fraction))))
        row_start = int(rng.integers(0, p - n_rows + 1))
        column_start = int(rng.integers(0, n - n_columns + 1))
        mask[
            row_start : row_start + n_rows,
            column_start : column_start + n_columns,
        ] = False
    else:
        msg = f"unknown missingness mechanism {mechanism!r}"
        raise ValueError(msg)
    return _repair_support(mask, rng)


def support_preserving_holdout(
    observation_mask: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split observed cells while retaining training support in every row/column."""
    observed = np.argwhere(observation_mask)
    n_holdout = max(1, int(round(fraction * len(observed))))
    chosen = rng.choice(
        len(observed), size=min(n_holdout, len(observed)), replace=False
    )
    holdout = np.zeros_like(observation_mask, dtype=bool)
    holdout[tuple(observed[chosen].T)] = True
    train = observation_mask & ~holdout

    for row in np.flatnonzero(~train.any(axis=1)):
        columns = np.flatnonzero(holdout[row])
        if columns.size:
            train[row, columns[0]] = True
            holdout[row, columns[0]] = False
    for column in np.flatnonzero(~train.any(axis=0)):
        rows = np.flatnonzero(holdout[:, column])
        if rows.size:
            train[rows[0], column] = True
            holdout[rows[0], column] = False
    return train, holdout


def generate_simulation(
    regime: dict[str, Any],
    *,
    rep: int,
    missing_fraction: float,
    holdout_fraction: float,
) -> SimulationData:
    """Generate one paired simulation shared by every candidate."""
    data_seed, init_seed = trial_seeds(int(regime["seed"]), rep)
    rng = np.random.default_rng(data_seed)
    x_true, loadings = _generate_matrix(
        int(regime["p"]),
        int(regime["n"]),
        int(regime["true_rank"]),
        float(regime["noise_std"]),
        str(regime["noise_model"]),
        rng,
    )
    observation_mask = apply_missingness(
        x_true,
        str(regime["missingness"]),
        missing_fraction,
        rng,
    )
    train_mask, holdout_mask = support_preserving_holdout(
        observation_mask,
        holdout_fraction,
        rng,
    )
    return SimulationData(
        x_true=x_true,
        true_loadings=loadings,
        observation_mask=observation_mask,
        train_mask=train_mask,
        holdout_mask=holdout_mask,
        init_seed=init_seed,
    )
