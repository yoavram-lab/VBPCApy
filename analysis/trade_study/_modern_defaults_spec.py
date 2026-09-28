"""Registered factors, regimes, and gates for the modern-defaults study."""

from __future__ import annotations

from typing import Any

DESIGN_VERSION = "v1_post_backend_modern_defaults"
MANIFEST_VERSION = "vbpca.modern-defaults.v1"
MINIMUM_VBPCA_VERSION = "0.4.2"

REGISTERED_SCREEN_REPS = 3
REGISTERED_SCREEN_SEED = 20261001
REGISTERED_CONFIRM_REPS = 12
REGISTERED_CONFIRM_SEED = 20261101
REGISTERED_SCALE_REPS = 3
REGISTERED_SCALE_SEED = 20261201

N_SOBOL_CANDIDATES = 24
MAX_FINALISTS = 4
MANDATORY_CANDIDATES = (
    "recommended_post_factor",
    "recommended_legacy",
)
ANCHOR_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "recommended_post_factor",
        "base": "recommended",
        "compat_mode": "modern",
        "bias_update_order": "post_factor",
    },
    {
        "id": "recommended_legacy",
        "base": "recommended",
        "compat_mode": "modern",
        "bias_update_order": "legacy",
    },
    {
        "id": "recommended_probe_005",
        "base": "recommended",
        "compat_mode": "modern",
        "bias_update_order": "post_factor",
        "xprobe_fraction": 0.05,
    },
    {
        "id": "recommended_probe_010",
        "base": "recommended",
        "compat_mode": "modern",
        "bias_update_order": "post_factor",
        "xprobe_fraction": 0.10,
    },
    {
        "id": "recommended_probe_020",
        "base": "recommended",
        "compat_mode": "modern",
        "bias_update_order": "post_factor",
        "xprobe_fraction": 0.20,
    },
    {
        "id": "estimator_default_post_factor",
        "base": "estimator_default",
        "compat_mode": "modern",
        "bias_update_order": "post_factor",
        "xprobe_fraction": 0.10,
    },
)

_ALL_CRITERIA = (
    "angle",
    "earlystop",
    "rms_plateau",
    "cost",
    "composite",
    "slowing_down",
)
CRITERION_POLICIES: dict[str, dict[str, bool]] = {
    "production": dict.fromkeys(_ALL_CRITERIA, True),
    "no_rms": {name: name != "rms_plateau" for name in _ALL_CRITERIA},
    "angle_cost": {name: name in {"angle", "cost"} for name in _ALL_CRITERIA},
}
BIAS_ORDERS = ("post_factor", "legacy")
NITER_LEVELS = (0, 25, 50, 100)
MAXITER_SCALES = (0.75, 1.0, 1.5)
PATIENCE_LEVELS = (1, 2, 3, 5)
RMS_WINDOWS = (50, 100, 200)

_RegimeRow = tuple[str, int, int, int, str, str, float]


def _regimes(rows: tuple[_RegimeRow, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": name,
            "n": n,
            "p": p,
            "true_rank": rank,
            "missingness": missingness,
            "noise_model": noise_model,
            "noise_std": noise_std,
        }
        for name, n, p, rank, missingness, noise_model, noise_std in rows
    )


SCREEN_REGIMES = _regimes((
    ("wide_complete", 80, 400, 5, "complete", "gaussian", 0.5),
    ("wide_mcar", 80, 400, 5, "mcar", "gaussian", 0.5),
    ("wide_mar", 80, 400, 5, "mar", "gaussian", 0.5),
    ("wide_mnar", 80, 400, 5, "mnar_censored", "gaussian", 0.5),
    ("wide_block", 80, 400, 5, "block", "gaussian", 0.5),
    ("square_mcar", 160, 160, 5, "mcar", "gaussian", 0.5),
    ("square_block_t3", 160, 160, 5, "block", "student_t3", 0.5),
    ("tall_complete", 500, 50, 5, "complete", "gaussian", 0.5),
    ("tall_mar", 500, 50, 5, "mar", "gaussian", 0.5),
    ("tall_mnar_t3", 500, 50, 5, "mnar_censored", "student_t3", 0.5),
    ("large_mcar", 500, 500, 10, "mcar", "gaussian", 0.5),
    ("genomics_surrogate_mar", 320, 744, 10, "mar", "gaussian", 0.5),
    ("wide_null_mcar", 80, 400, 0, "mcar", "gaussian", 1.0),
    ("square_null_mar", 160, 160, 0, "mar", "gaussian", 1.0),
))

CONFIRM_REGIMES = _regimes((
    ("wide_complete_holdout", 120, 600, 8, "complete", "gaussian", 0.7),
    ("wide_mar_holdout", 120, 600, 8, "mar", "gaussian", 0.7),
    ("wide_mnar_t3_holdout", 120, 600, 8, "mnar_censored", "student_t3", 0.7),
    ("square_mcar_holdout", 220, 220, 5, "mcar", "gaussian", 0.3),
    ("square_block_t3_holdout", 220, 220, 5, "block", "student_t3", 0.7),
    ("tall_complete_holdout", 750, 60, 5, "complete", "gaussian", 0.3),
    ("tall_mnar_holdout", 750, 60, 5, "mnar_censored", "gaussian", 0.7),
    ("large_mar_holdout", 700, 700, 10, "mar", "gaussian", 0.5),
    ("wide_null_block_holdout", 120, 600, 0, "block", "gaussian", 1.0),
))

SCALE_REGIMES = _regimes((
    ("genomics_complete", 2504, 5846, 10, "complete", "gaussian", 0.5),
    ("genomics_mar", 2504, 5846, 10, "mar", "gaussian", 0.5),
    ("genomics_mnar", 2504, 5846, 10, "mnar_censored", "gaussian", 0.5),
))

SMOKE_REGIMES = _regimes((
    ("wide_smoke", 30, 60, 2, "mar", "gaussian", 0.5),
    ("null_smoke", 30, 30, 0, "mcar", "student_t3", 1.0),
))

PROFILE_REGIMES = {
    "smoke": SMOKE_REGIMES,
    "screen": SCREEN_REGIMES,
    "confirm": CONFIRM_REGIMES,
    "scale": SCALE_REGIMES,
}

SELECTION_DESIGN: dict[str, Any] = {
    "metric": "prms",
    "include_rank_zero": True,
    "component_margin": 5,
    "selection_patience": 2,
    "external_holdout_fraction": 0.10,
    "missing_fraction": 0.15,
}

SCREEN_GATES: dict[str, float] = {
    "holdout_rmse_relative_upper": 0.01,
    "interval_score_relative_upper": 0.02,
    "coverage_difference_lower": -0.02,
    "rank_mae_difference_upper": 0.10,
    "null_selection_rate_difference_upper": 0.05,
}
CONFIRM_GATES: dict[str, float] = {
    **SCREEN_GATES,
    "material_rank_mae_improvement": 0.10,
    "material_iteration_reduction": 0.20,
}
BOOTSTRAP_DESIGN: dict[str, Any] = {
    "method": "paired_regime_stratified",
    "confidence": 0.95,
    "n_resamples": 5000,
    "seed": 20261301,
}
