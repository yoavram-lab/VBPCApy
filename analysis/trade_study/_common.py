"""Shared definitions: factors, observables, data regimes, data generation.

This module is imported by all phase scripts.  Changes here propagate
to every phase.
"""

from __future__ import annotations

import json
from itertools import product
from typing import Any

import numpy as np
from trade_study import (
    Constraint,
    Direction,
    Factor,
    FactorConstraint,
    FactorType,
    Observable,
)

# ── Random seed ──────────────────────────────────────────────────

RNG_SEED = 42

# ── Criterion ordering and active-criteria presets (v2) ──────────
#
# These map to the ``criterion_order`` and ``convergence_criteria``
# parameters added in PR #102.  Used as categorical factor levels
# in Stage 2 of the v2 trade study.

CRITERION_ORDER_LEVELS: dict[str, list[str]] = {
    "default": [
        "angle",
        "earlystop",
        "rms_plateau",
        "cost",
        "composite",
        "slowing_down",
    ],
    "cost_first": [
        "cost",
        "angle",
        "composite",
        "rms_plateau",
        "earlystop",
        "slowing_down",
    ],
    "composite_first": [
        "composite",
        "cost",
        "angle",
        "rms_plateau",
        "earlystop",
        "slowing_down",
    ],
    "rms_first": [
        "rms_plateau",
        "angle",
        "cost",
        "composite",
        "earlystop",
        "slowing_down",
    ],
    "cost_angle": [
        "cost",
        "angle",
        "rms_plateau",
        "composite",
        "earlystop",
        "slowing_down",
    ],
    "angle_cost": [
        "angle",
        "cost",
        "composite",
        "rms_plateau",
        "earlystop",
        "slowing_down",
    ],
}

_ALL_CRITERIA = [
    "angle",
    "earlystop",
    "rms_plateau",
    "cost",
    "composite",
    "slowing_down",
]

ACTIVE_CRITERIA_PRESETS: dict[str, dict[str, bool]] = {
    "all": dict.fromkeys(_ALL_CRITERIA, True),
    "core": {
        "angle": True,
        "earlystop": False,
        "rms_plateau": True,
        "cost": True,
        "composite": False,
        "slowing_down": False,
    },
    "elbo_only": {
        "angle": False,
        "earlystop": False,
        "rms_plateau": False,
        "cost": True,
        "composite": False,
        "slowing_down": False,
    },
    "angle_elbo": {
        "angle": True,
        "earlystop": False,
        "rms_plateau": False,
        "cost": True,
        "composite": False,
        "slowing_down": False,
    },
    "practical": {
        "angle": True,
        "earlystop": False,
        "rms_plateau": True,
        "cost": False,
        "composite": True,
        "slowing_down": False,
    },
}

# ── Observables (response variables) ────────────────────────────
#
# Quality metrics carry higher weight; iteration cost is the "cost"
# axis with a lower weight so it only breaks ties among equally-good
# configurations.

# All observables are computed and reported, but only the Pareto
# objectives carry non-zero weight.  The others are tracked for
# diagnostics but excluded from Pareto ranking.

PARETO_OBSERVABLES: list[Observable] = [
    Observable("holdout_rmse", Direction.MINIMIZE, weight=1.0),
    Observable("rank_mae", Direction.MINIMIZE, weight=1.0),
    Observable("total_iters", Direction.MINIMIZE, weight=0.5),
]

TRACKED_OBSERVABLES: list[Observable] = [
    Observable("holdout_mae", Direction.MINIMIZE, weight=0.0),
    Observable("coverage_95", Direction.MAXIMIZE, weight=0.0),
    Observable("interval_score", Direction.MINIMIZE, weight=0.0),
    Observable("calibration_gap", Direction.MINIMIZE, weight=0.0),
    Observable("wall_seconds", Direction.MINIMIZE, weight=0.0),
    Observable("best_k_iters", Direction.MINIMIZE, weight=0.0),
]

observables: list[Observable] = PARETO_OBSERVABLES + TRACKED_OBSERVABLES

# Convenience: weights dict for weighted-sum filter (Pareto objectives only).
WEIGHTED_SUM_WEIGHTS: dict[str, float] = {o.name: o.weight for o in PARETO_OBSERVABLES}

# ── v2 stage-specific observables ────────────────────────────────
#
# Stage 1: 3 independent quality axes (no total_iters).
# Stage 2: 2-objective — quality_composite vs total_iters.

STAGE1_PARETO_OBSERVABLES: list[Observable] = [
    Observable("holdout_rmse", Direction.MINIMIZE, weight=1.0),
    Observable("coverage_95", Direction.MAXIMIZE, weight=1.0),
    Observable("rank_mae", Direction.MINIMIZE, weight=1.0),
]

STAGE2_PARETO_OBSERVABLES: list[Observable] = [
    Observable("quality_composite", Direction.MINIMIZE, weight=1.0),
    Observable("total_iters", Direction.MINIMIZE, weight=1.0),
]

# Standalone observable for total_iters (real, scored by VBPCAScorer).
# Used by Stage 2 scripts so the scorer actually returns the value.
TOTAL_ITERS_OBSERVABLE = Observable("total_iters", Direction.MINIMIZE, weight=0.0)

# Weights for the quality composite (Stage 2).
# RMSE is the primary quality metric; coverage and rank are secondary.
QUALITY_COMPOSITE_WEIGHTS: dict[str, float] = {
    "holdout_rmse": 0.5,
    "coverage_95": 0.25,
    "rank_mae": 0.25,
}

# ── Constraints ──────────────────────────────────────────────────

MIN_COVERAGE = Constraint(
    name="min_coverage",
    observable="coverage_95",
    op=">=",
    threshold=0.40,
)

MAX_RANK_MAE = Constraint(
    name="max_rank_mae",
    observable="rank_mae",
    op="<=",
    threshold=3.0,
)

ALL_CONSTRAINTS: list[Constraint] = [MIN_COVERAGE, MAX_RANK_MAE]

# ── Tunable VBPCA factors ───────────────────────────────────────
#
# Prior / ARD
#   hp_va, hp_vb, hp_v  – inverse-Gamma shape for ARD priors
#   va_init              – initial broad prior variance
#   niter_broadprior     – iterations under broad prior before ARD
#
# Convergence
#   maxiters             – hard iteration cap
#   minangle             – subspace-angle stopping threshold
#   patience             – consecutive passes before stop
#   cfstop_rel           – relative ELBO change threshold
#   rmsstop_window       – RMS plateau window length
#   rmsstop_atol         – RMS plateau absolute tolerance
#   rmsstop_rtol         – RMS plateau relative tolerance
#
# Structural
#   rotate2pca           – post-rotation to PCA eigenbasis
#   bias                 – estimate bias / mean term
#   xprobe_fraction      – fraction of entries auto-held-out for probe

factors_prior: list[Factor] = [
    Factor("hp_va", FactorType.CONTINUOUS, bounds=(1e-5, 1.0)),
    Factor("hp_vb", FactorType.CONTINUOUS, bounds=(1e-5, 1.0)),
    Factor("hp_v", FactorType.CONTINUOUS, bounds=(1e-5, 1.0)),
    Factor("va_init", FactorType.CONTINUOUS, bounds=(10.0, 10_000.0)),
    Factor("niter_broadprior", FactorType.DISCRETE, levels=[0, 25, 50, 100, 200]),
]

factors_convergence: list[Factor] = [
    Factor("maxiters", FactorType.DISCRETE, levels=[100, 200, 500, 1000]),
    Factor("minangle", FactorType.CONTINUOUS, bounds=(1e-10, 1e-4)),
    Factor("patience", FactorType.DISCRETE, levels=[1, 2, 3, 5]),
    Factor("cfstop_rel", FactorType.CONTINUOUS, bounds=(1e-8, 1e-3)),
    Factor("rmsstop_window", FactorType.DISCRETE, levels=[50, 100, 200]),
    Factor("rmsstop_atol", FactorType.CONTINUOUS, bounds=(1e-6, 1e-2)),
    Factor("rmsstop_rtol", FactorType.CONTINUOUS, bounds=(1e-5, 1e-2)),
]

factors_structural: list[Factor] = [
    Factor("xprobe_fraction", FactorType.CONTINUOUS, bounds=(0.0, 0.25)),
]

# rotate2pca and bias are fixed based on Phase 1 analysis.
FIXED_STRUCTURAL: dict[str, Any] = {
    "rotate2pca": True,
    "bias": True,
}

ALL_FACTORS: list[Factor] = factors_prior + factors_convergence + factors_structural

# ── v2 factor groups (stage-partitioned) ─────────────────────────
#
# Stage 1: prior + structural only (bp OFF, convergence fixed).
# Stage 2: convergence + broadprior + criterion ordering/enablement.

# Stage 1 continuous factors — used for Sobol screening (Phase 1.0).
STAGE1_FACTORS: list[Factor] = [
    Factor("hp_va", FactorType.CONTINUOUS, bounds=(1e-5, 1.0)),
    Factor("hp_vb", FactorType.CONTINUOUS, bounds=(1e-5, 1.0)),
    Factor("hp_v", FactorType.CONTINUOUS, bounds=(1e-5, 1.0)),
    Factor("va_init", FactorType.CONTINUOUS, bounds=(10.0, 10_000.0)),
    Factor("xprobe_fraction", FactorType.CONTINUOUS, bounds=(0.0, 0.25)),
]

# Fixed convergence settings for Stage 1 (generous — won't limit quality).
STAGE1_FIXED_CONVERGENCE: dict[str, Any] = {
    "niter_broadprior": 0,
    "maxiters": 1000,
    "minangle": 1e-10,
    "patience": 1,
    "cfstop_rel": 1e-8,
    "rmsstop": [200, 1e-6, 1e-5],
}

# Stage 2 discrete/categorical factors (Phase 2.0 full factorial).
STAGE2_DISCRETE_FACTORS: list[Factor] = [
    Factor(
        "niter_broadprior",
        FactorType.DISCRETE,
        levels=[0, 25, 50, 100, 200],
    ),
    Factor(
        "criterion_order",
        FactorType.CATEGORICAL,
        levels=list(CRITERION_ORDER_LEVELS.keys()),
    ),
    Factor(
        "active_criteria",
        FactorType.CATEGORICAL,
        levels=list(ACTIVE_CRITERIA_PRESETS.keys()),
    ),
    Factor("maxiters", FactorType.DISCRETE, levels=[100, 200, 500, 1000]),
    Factor("patience", FactorType.DISCRETE, levels=[1, 2, 3, 5]),
    Factor("rmsstop_window", FactorType.DISCRETE, levels=[50, 100, 200]),
]

# Stage 2 continuous threshold factors (Phase 2.1 Sobol QMC).
STAGE2_CONTINUOUS_FACTORS: list[Factor] = [
    Factor("minangle", FactorType.CONTINUOUS, bounds=(1e-10, 1e-4)),
    Factor("cfstop_rel", FactorType.CONTINUOUS, bounds=(1e-8, 1e-3)),
    Factor("rmsstop_atol", FactorType.CONTINUOUS, bounds=(1e-6, 1e-2)),
    Factor("rmsstop_rtol", FactorType.CONTINUOUS, bounds=(1e-5, 1e-2)),
]

# ── Data-regime grid (environmental, not optimised) ──────────────

REGIME_N: list[int] = [20, 50, 100, 200]
REGIME_P: list[int] = [10, 30, 100, 200]
REGIME_RANK: list[int] = [2, 5, 10]
REGIME_MISS: list[str] = ["complete", "mcar", "mnar_censored", "block"]
REGIME_NOISE: list[float] = [0.3, 0.5, 1.0]
MISS_FRACTION: float = 0.15
HOLDOUT_FRACTION: float = 0.10

# Representative regimes for Sobol screening (one per missingness type).
SCREEN_REGIMES: list[dict[str, Any]] = [
    {"n": 50, "p": 30, "true_rank": 5, "missingness": "mcar", "noise_std": 0.5},
    {"n": 100, "p": 100, "true_rank": 5, "missingness": "complete", "noise_std": 0.5},
    {
        "n": 200,
        "p": 50,
        "true_rank": 2,
        "missingness": "mnar_censored",
        "noise_std": 1.0,
    },
    {"n": 200, "p": 200, "true_rank": 10, "missingness": "block", "noise_std": 1.0},
]

# ── Regime families ──────────────────────────────────────────────
# Family A: well-specified (VBPCA assumptions hold).
# Family B: mis-specified (structured missingness violates model).

FAMILY_A_MISSINGNESS: list[str] = ["complete", "mcar"]
FAMILY_B_MISSINGNESS: list[str] = ["mnar_censored", "block"]

STUDY_REGIMES_A: list[dict[str, Any]] = [
    {"n": 50, "p": 30, "true_rank": 5, "missingness": "mcar", "noise_std": 0.5},
    {"n": 100, "p": 100, "true_rank": 5, "missingness": "complete", "noise_std": 0.5},
    {"n": 200, "p": 100, "true_rank": 5, "missingness": "complete", "noise_std": 1.0},
]

STUDY_REGIMES_B: list[dict[str, Any]] = [
    {
        "n": 100,
        "p": 50,
        "true_rank": 2,
        "missingness": "mnar_censored",
        "noise_std": 0.5,
    },
    {"n": 200, "p": 200, "true_rank": 10, "missingness": "block", "noise_std": 1.0},
    {
        "n": 100,
        "p": 100,
        "true_rank": 5,
        "missingness": "mnar_censored",
        "noise_std": 1.0,
    },
]

STUDY_REGIMES: list[dict[str, Any]] = STUDY_REGIMES_A + STUDY_REGIMES_B

# Held-out regimes for Phase 4 cross-validation (never seen during refinement).
VALIDATION_REGIMES: list[dict[str, Any]] = [
    {"n": 70, "p": 50, "true_rank": 5, "missingness": "mcar", "noise_std": 0.3},
    {"n": 100, "p": 200, "true_rank": 10, "missingness": "complete", "noise_std": 0.5},
    {
        "n": 200,
        "p": 50,
        "true_rank": 2,
        "missingness": "mnar_censored",
        "noise_std": 1.0,
    },
    {"n": 150, "p": 150, "true_rank": 5, "missingness": "block", "noise_std": 0.5},
]

# Relaxed coverage constraint for Family B (structural coverage ceiling).
MIN_COVERAGE_B = Constraint(
    name="min_coverage_b",
    observable="coverage_95",
    op=">=",
    threshold=0.30,
)
CONSTRAINTS_FAMILY_B: list[Constraint] = [MIN_COVERAGE_B, MAX_RANK_MAE]

# ── v3 regime families ───────────────────────────────────────────
#
# Three empirically-identified performance regimes (from v2 comparison):
#   small-p  : p ≤ 30  — v2-opt is uniformly worse; needs dedicated tuning.
#   trans    : p ∈ {50,70} — crossover zone; neither v2-opt nor default wins.
#   large    : p ≥ 100 AND n ≥ 100 — v2-opt wins; use existing v2 results.
#
# Gate at inference time:  p ≤ 30 → smallp config
#                          p ≤ 70 → trans config
#                          else   → v2 large config

# Relaxed coverage floor for small-p (p=10 has structural coverage ceiling).
MIN_COVERAGE_SMALLP = Constraint(
    name="min_coverage_smallp",
    observable="coverage_95",
    op=">=",
    threshold=0.25,
)
CONSTRAINTS_SMALLP: list[Constraint] = [MIN_COVERAGE_SMALLP, MAX_RANK_MAE]

# ── small-p regime definitions ───────────────────────────────────

SMALLP_STUDY_REGIMES: list[dict[str, Any]] = [
    {"n": 30, "p": 10, "true_rank": 2, "missingness": "mcar", "noise_std": 0.5},
    {"n": 50, "p": 10, "true_rank": 2, "missingness": "complete", "noise_std": 0.3},
    {"n": 70, "p": 20, "true_rank": 2, "missingness": "mcar", "noise_std": 0.5},
    {"n": 100, "p": 20, "true_rank": 5, "missingness": "complete", "noise_std": 0.5},
    {
        "n": 50,
        "p": 30,
        "true_rank": 5,
        "missingness": "mnar_censored",
        "noise_std": 0.5,
    },
    {"n": 100, "p": 30, "true_rank": 5, "missingness": "block", "noise_std": 1.0},
]

SMALLP_SCREEN_REGIMES: list[dict[str, Any]] = [
    {"n": 30, "p": 10, "true_rank": 2, "missingness": "mcar", "noise_std": 0.5},
    {"n": 50, "p": 20, "true_rank": 2, "missingness": "complete", "noise_std": 0.5},
    {
        "n": 100,
        "p": 30,
        "true_rank": 5,
        "missingness": "mnar_censored",
        "noise_std": 0.5,
    },
    {"n": 150, "p": 10, "true_rank": 2, "missingness": "block", "noise_std": 1.0},
]

SMALLP_VALIDATION_REGIMES: list[dict[str, Any]] = [
    {"n": 40, "p": 15, "true_rank": 2, "missingness": "complete", "noise_std": 0.5},
    {"n": 80, "p": 20, "true_rank": 2, "missingness": "block", "noise_std": 0.5},
    {
        "n": 150,
        "p": 30,
        "true_rank": 5,
        "missingness": "mnar_censored",
        "noise_std": 1.0,
    },
    {"n": 200, "p": 25, "true_rank": 5, "missingness": "mcar", "noise_std": 0.3},
]

# ── transition regime definitions ────────────────────────────────

TRANS_STUDY_REGIMES: list[dict[str, Any]] = [
    {"n": 50, "p": 50, "true_rank": 5, "missingness": "mcar", "noise_std": 0.5},
    {"n": 70, "p": 50, "true_rank": 5, "missingness": "complete", "noise_std": 0.5},
    {
        "n": 100,
        "p": 50,
        "true_rank": 5,
        "missingness": "mnar_censored",
        "noise_std": 1.0,
    },
    {"n": 150, "p": 50, "true_rank": 5, "missingness": "mcar", "noise_std": 0.3},
    {
        "n": 70,
        "p": 70,
        "true_rank": 5,
        "missingness": "mnar_censored",
        "noise_std": 0.5,
    },
    {"n": 100, "p": 70, "true_rank": 5, "missingness": "block", "noise_std": 0.5},
    {"n": 200, "p": 70, "true_rank": 5, "missingness": "complete", "noise_std": 1.0},
]

TRANS_SCREEN_REGIMES: list[dict[str, Any]] = [
    {"n": 70, "p": 50, "true_rank": 5, "missingness": "mcar", "noise_std": 0.5},
    {"n": 100, "p": 70, "true_rank": 5, "missingness": "complete", "noise_std": 0.5},
    {
        "n": 150,
        "p": 50,
        "true_rank": 5,
        "missingness": "mnar_censored",
        "noise_std": 1.0,
    },
    {"n": 200, "p": 70, "true_rank": 5, "missingness": "block", "noise_std": 0.5},
]

TRANS_VALIDATION_REGIMES: list[dict[str, Any]] = [
    {"n": 50, "p": 60, "true_rank": 5, "missingness": "mcar", "noise_std": 0.5},
    {"n": 100, "p": 60, "true_rank": 5, "missingness": "block", "noise_std": 0.5},
    {
        "n": 150,
        "p": 70,
        "true_rank": 5,
        "missingness": "mnar_censored",
        "noise_std": 1.0,
    },
    {"n": 200, "p": 50, "true_rank": 2, "missingness": "complete", "noise_std": 0.3},
]

# ── v4 additions: rank-diverse trans regimes ─────────────────────
#
# The original TRANS_STUDY_REGIMES used true_rank=5 for every cell,
# which biased the optimizer toward configs that collapse to k≈5.
# These rank=2 and rank=10 cells force the search to find configs
# robust across the rank span seen at evaluation time.

TRANS_STUDY_REGIMES_V4: list[dict[str, Any]] = [
    *TRANS_STUDY_REGIMES,
    {"n": 100, "p": 50, "true_rank": 2, "missingness": "mcar", "noise_std": 0.5},
    {"n": 150, "p": 70, "true_rank": 2, "missingness": "complete", "noise_std": 0.3},
    {"n": 100, "p": 50, "true_rank": 10, "missingness": "complete", "noise_std": 0.5},
    {"n": 200, "p": 70, "true_rank": 10, "missingness": "mcar", "noise_std": 0.5},
]

# ── v4: trans-regime factor constraints ──────────────────────────
#
# Applied at design-generation time via ``build_grid(constraints=…)``.
# These prune coupled factor combinations known to cause catastrophic
# rank collapse in the trans regime (see docs/IMPLEMENTATION_PLAN.md
# v4 root-cause analysis).

TRANS_FACTOR_CONSTRAINTS: list[FactorConstraint] = [
    FactorConstraint(
        predicate=lambda c: (
            not (
                c.get("active_criteria") == "elbo_only"
                and int(c.get("patience", 1)) >= 3
            )
        ),
        name="no_elbo_only_with_high_patience",
    ),
    FactorConstraint(
        predicate=lambda c: (
            not (
                float(c.get("va_init", 0.0)) > 1000.0
                and float(c.get("hp_v", 1.0)) < 0.7
            )
        ),
        name="no_broad_prior_with_strong_ard",
    ),
    FactorConstraint(
        predicate=lambda c: int(c.get("maxiters", 200)) >= 200,
        name="trans_needs_iter_budget",
    ),
]

# ── v4: asymmetric rank-error penalty ────────────────────────────
#
# Replaces ``rank_mae`` in the Stage 2 quality composite.  Penalises
# under-selection (k̂ < k*) quadratically but over-selection only
# linearly at 1/4 weight.  Computed by VBPCAScorer when present in
# the observables list; safe to leave unused.


def asymmetric_rank_penalty(selected_k: int, true_rank: int) -> float:
    """Return ``relu(k* − k̂)² + 0.25·relu(k̂ − k*)``.

    Under-selection is the dominant failure mode in the trans regime
    (ARD shrinks too aggressively), so it carries a quadratic penalty.
    Over-selection costs little in reconstruction RMSE, so it is
    penalised only linearly at 1/4 the magnitude.
    """
    under = max(0, true_rank - selected_k)
    over = max(0, selected_k - true_rank)
    return float(under * under + 0.25 * over)


# Quality-composite weights for the asymmetric variant.  Same as the
# symmetric weights but the rank component now reads
# ``asymmetric_rank_penalty`` instead of ``rank_mae``.
QUALITY_COMPOSITE_WEIGHTS_V4: dict[str, float] = {
    "holdout_rmse": 0.5,
    "coverage_95": 0.25,
    "rank_penalty": 0.25,
}

# ── v4: under-shrinkage feasibility constraint ───────────────────
MAX_K_UNDER_SHRINK = Constraint(
    name="max_k_under_shrink",
    observable="rank_under",  # = max(0, k* − k̂); written by VBPCAScorer
    op="<=",
    threshold=2.0,
)

# ── v4: regime features for the regime-conditional surrogate ─────
#
# Feed these to ``trade_study.fit_regime_surrogate`` together with
# the union of Stage 1 + Stage 2 design factors.

REGIME_FEATURES: list[Factor] = [
    Factor("n", FactorType.CONTINUOUS, bounds=(20.0, 200.0)),
    Factor("p", FactorType.CONTINUOUS, bounds=(10.0, 200.0)),
    Factor("true_rank", FactorType.DISCRETE, levels=[2, 5, 10]),
    Factor("noise_std", FactorType.CONTINUOUS, bounds=(0.3, 1.0)),
    Factor(
        "missingness",
        FactorType.CATEGORICAL,
        levels=["complete", "mcar", "mnar_censored", "block"],
    ),
]


def build_regime_grid() -> list[dict[str, Any]]:
    """Full factorial data-regime grid, skipping infeasible cells."""
    regimes: list[dict[str, Any]] = []
    for n, p, rank, miss, noise in product(
        REGIME_N,
        REGIME_P,
        REGIME_RANK,
        REGIME_MISS,
        REGIME_NOISE,
    ):
        if rank >= min(n, p):
            continue
        regimes.append({
            "n": n,
            "p": p,
            "true_rank": rank,
            "missingness": miss,
            "noise_std": noise,
        })
    return regimes


# ── Data generation helpers ──────────────────────────────────────


def generate_low_rank(
    n: int,
    p: int,
    true_rank: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a (p × n) low-rank-plus-noise matrix."""
    W = rng.standard_normal((p, true_rank))
    S = rng.standard_normal((true_rank, n))
    return W @ S + noise_std * rng.standard_normal((p, n))


def apply_missingness(
    x: np.ndarray,
    pattern: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return boolean mask (True = observed) for the given pattern."""
    p, n = x.shape
    mask = np.ones_like(x, dtype=bool)

    if pattern == "complete":
        return mask
    if pattern == "mcar":
        mask = rng.random(x.shape) > MISS_FRACTION
    elif pattern == "mnar_censored":
        threshold = np.nanquantile(x, MISS_FRACTION, axis=1, keepdims=True)
        mask = x > threshold
    elif pattern == "block":
        nc = max(1, int(n * MISS_FRACTION))
        nr = max(1, int(p * MISS_FRACTION))
        c0 = rng.integers(0, max(1, n - nc))
        r0 = rng.integers(0, max(1, p - nr))
        mask[r0 : r0 + nr, c0 : c0 + nc] = False

    # Guarantee at least one observed entry per row / column.
    for i in range(p):
        if not mask[i].any():
            mask[i, rng.integers(n)] = True
    for j in range(n):
        if not mask[:, j].any():
            mask[rng.integers(p), j] = True
    return mask


def holdout_split(
    mask: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split observed entries into train and holdout masks."""
    observed = np.argwhere(mask)
    n_hold = max(1, int(len(observed) * fraction))
    idx = rng.choice(len(observed), size=n_hold, replace=False)
    holdout = np.zeros_like(mask, dtype=bool)
    for k in idx:
        holdout[observed[k, 0], observed[k, 1]] = True
    train = mask & ~holdout

    # Ensure every row/col remains observed in train.
    p, n = mask.shape
    for i in range(p):
        if not train[i].any():
            cands = np.where(holdout[i])[0]
            if len(cands):
                train[i, cands[0]] = True
                holdout[i, cands[0]] = False
    for j in range(n):
        if not train[:, j].any():
            cands = np.where(holdout[:, j])[0]
            if len(cands):
                train[cands[0], j] = True
                holdout[cands[0], j] = False
    return train, holdout


def inject_regime(
    factor_grid: list[dict[str, Any]],
    regime: dict[str, Any],
) -> list[dict[str, Any]]:
    """Combine each factor config with a fixed data regime."""
    return [{**cfg, **regime} for cfg in factor_grid]


def round_discrete_factors(cfg: dict[str, Any]) -> dict[str, Any]:
    """Round discrete factor values in-place after QMC sampling."""
    for key in ("niter_broadprior", "maxiters", "patience", "rmsstop_window"):
        if key in cfg:
            cfg[key] = int(round(cfg[key]))
    return cfg


def aggregate_config_medians(
    config_scores: dict[str, list[dict[str, float]]],
    obs_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    """Aggregate per-config score lists into median dicts.

    Returns
    -------
    tuple
        (agg_configs, agg_medians) — parallel lists of config dicts
        and their per-observable median scores.
    """
    agg_configs: list[dict[str, Any]] = []
    agg_medians: list[dict[str, float]] = []
    for key, scores_list in config_scores.items():
        cfg = json.loads(key)
        agg_configs.append(cfg)
        medians: dict[str, float] = {}
        for name in obs_names:
            vals = [s[name] for s in scores_list if np.isfinite(s[name])]
            medians[name] = float(np.median(vals)) if vals else float("inf")
        agg_medians.append(medians)
    return agg_configs, agg_medians


def compute_quality_bounds(
    agg_medians: list[dict[str, float]],
    raw_quality_names: list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute min-max normalisation bounds for quality composite.

    Parameters
    ----------
    agg_medians
        Per-config median score dicts.
    raw_quality_names
        Observable names to bound (default: holdout_rmse, coverage_95, rank_mae).

    Returns
    -------
    dict
        ``{name: (lo, hi)}`` bounds for :func:`compute_quality_composite`.
    """
    if raw_quality_names is None:
        raw_quality_names = ["holdout_rmse", "coverage_95", "rank_mae"]
    all_vals: dict[str, list[float]] = {n: [] for n in raw_quality_names}
    for m in agg_medians:
        for n in raw_quality_names:
            v = m[n]
            if n == "coverage_95":
                all_vals[n].append(1.0 - v)
            else:
                all_vals[n].append(v)

    bounds: dict[str, tuple[float, float]] = {}
    for n in raw_quality_names:
        arr = [v for v in all_vals[n] if np.isfinite(v)]
        bounds[n] = (float(np.min(arr)), float(np.max(arr))) if arr else (0.0, 1.0)
    return bounds
