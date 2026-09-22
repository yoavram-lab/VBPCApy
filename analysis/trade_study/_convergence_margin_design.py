"""Immutable design for validating post-warmup convergence margins."""

from __future__ import annotations

import copy
from typing import Any

from vbpca_py import defaults as vbpca_defaults
from vbpca_py import recommend_config

DESIGN_VERSION = "v1_post_warmup_margin"
MANIFEST_VERSION = "vbpca.convergence-margin.v1"
REFERENCE_CONDITION = "cap800"

CONDITIONS = (
    "shipped",
    "cap400",
    "cap800",
    "warmup50_cap400",
    "no_warmup_cap400",
    "forced800",
)

_ALL_CRITERIA_FALSE = {
    "angle": False,
    "earlystop": False,
    "rms_plateau": False,
    "cost": False,
    "composite": False,
    "slowing_down": False,
}

# Smoke shapes are smaller representatives that still exercise all affected
# routing buckets. Screen uses the exact regimes that produced the shipped
# configs. Confirm adds held-out shapes and missingness mechanisms without
# crossing into a neighboring bucket.
REGIME_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "smoke": {
        "wide_moderate_smoke": {
            "n": 20,
            "p": 100,
            "true_rank": 2,
            "missingness": "mcar",
            "noise_std": 0.5,
        },
        "tall_moderate_smoke": {
            "n": 320,
            "p": 20,
            "true_rank": 2,
            "missingness": "complete",
            "noise_std": 0.5,
        },
        "tall_extreme_smoke": {
            "n": 1020,
            "p": 20,
            "true_rank": 2,
            "missingness": "complete",
            "noise_std": 0.5,
        },
    },
    "screen": {
        "microbiome": {
            "n": 50,
            "p": 300,
            "true_rank": 2,
            "missingness": "mcar",
            "noise_std": 0.5,
        },
        "cultural": {
            "n": 1000,
            "p": 50,
            "true_rank": 5,
            "missingness": "complete",
            "noise_std": 0.5,
        },
        "ecological": {
            "n": 3000,
            "p": 30,
            "true_rank": 3,
            "missingness": "complete",
            "noise_std": 0.5,
        },
    },
    "confirm": {
        "microbiome": {
            "n": 50,
            "p": 300,
            "true_rank": 2,
            "missingness": "mcar",
            "noise_std": 0.5,
        },
        "wide_complete": {
            "n": 40,
            "p": 200,
            "true_rank": 5,
            "missingness": "complete",
            "noise_std": 0.3,
        },
        "wide_mnar": {
            "n": 100,
            "p": 300,
            "true_rank": 10,
            "missingness": "mnar_censored",
            "noise_std": 0.5,
        },
        "cultural": {
            "n": 1000,
            "p": 50,
            "true_rank": 5,
            "missingness": "complete",
            "noise_std": 0.5,
        },
        "tall_moderate_mcar": {
            "n": 640,
            "p": 40,
            "true_rank": 2,
            "missingness": "mcar",
            "noise_std": 0.5,
        },
        "tall_moderate_block": {
            "n": 1600,
            "p": 80,
            "true_rank": 10,
            "missingness": "block",
            "noise_std": 0.5,
        },
        "ecological": {
            "n": 3000,
            "p": 30,
            "true_rank": 3,
            "missingness": "complete",
            "noise_std": 0.5,
        },
        "tall_extreme_mcar": {
            "n": 1530,
            "p": 30,
            "true_rank": 5,
            "missingness": "mcar",
            "noise_std": 0.5,
        },
        "tall_extreme_mnar": {
            "n": 1020,
            "p": 20,
            "true_rank": 3,
            "missingness": "mnar_censored",
            "noise_std": 1.0,
        },
    },
}


def condition_config(n: int, p: int, condition: str) -> dict[str, Any]:
    """Return an estimator-ready condition derived from the shipped config.

    Returns:
        A fresh VBPCA keyword dictionary.

    Raises:
        ValueError: If *condition* is unknown.
    """
    if condition not in CONDITIONS:
        msg = f"unknown convergence-margin condition {condition!r}"
        raise ValueError(msg)
    config = recommend_config(n=n, p=p)
    if condition == "shipped":
        return config
    if condition == "cap400":
        config["maxiters"] = 400
    elif condition == "cap800":
        config["maxiters"] = 800
    elif condition == "warmup50_cap400":
        config["niter_broadprior"] = 50
        config["maxiters"] = 400
    elif condition == "no_warmup_cap400":
        config["niter_broadprior"] = 0
        config["maxiters"] = 400
    elif condition == "forced800":
        config["maxiters"] = 800
        config["convergence_criteria"] = copy.deepcopy(_ALL_CRITERIA_FALSE)
    return config


def build_manifest(
    profile: str,
    *,
    n_reps: int,
    seed: int,
) -> dict[str, Any]:
    """Build a JSON-compatible immutable convergence-margin manifest.

    Returns:
        Validated manifest mapping.

    Raises:
        ValueError: If the profile or replicate count is invalid, or if a
            regime does not route to an affected bucket.
    """
    if profile not in REGIME_PROFILES:
        msg = f"unknown profile {profile!r}; choose from {tuple(REGIME_PROFILES)}"
        raise ValueError(msg)
    if n_reps < 1:
        msg = f"n_reps must be positive, got {n_reps}"
        raise ValueError(msg)

    regimes: list[dict[str, Any]] = []
    affected = {"wide_moderate", "tall_moderate", "tall_extreme"}
    for index, (name, raw) in enumerate(REGIME_PROFILES[profile].items()):
        regime = {"name": name, **copy.deepcopy(raw)}
        bucket = vbpca_defaults._bucket(regime["n"], regime["p"])  # noqa: SLF001
        if bucket not in affected:
            msg = f"regime {name!r} routes to {bucket!r}, not an affected bucket"
            raise ValueError(msg)
        regime["bucket"] = bucket
        regime["seed"] = seed + index
        regimes.append(regime)

    return {
        "manifest_version": MANIFEST_VERSION,
        "design_version": DESIGN_VERSION,
        "profile": profile,
        "n_reps": n_reps,
        "seed": seed,
        "reference_condition": REFERENCE_CONDITION,
        "conditions": list(CONDITIONS),
        "regimes": regimes,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the schema and immutable design fields of a manifest.

    Raises:
        ValueError: If any required design field differs from this code.
    """
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        msg = f"unsupported manifest_version {manifest.get('manifest_version')!r}"
        raise ValueError(msg)
    if manifest.get("design_version") != DESIGN_VERSION:
        msg = f"unsupported design_version {manifest.get('design_version')!r}"
        raise ValueError(msg)
    if tuple(manifest.get("conditions", ())) != CONDITIONS:
        msg = "manifest conditions differ from the immutable design"
        raise ValueError(msg)
    if manifest.get("reference_condition") != REFERENCE_CONDITION:
        msg = "manifest reference condition differs from the immutable design"
        raise ValueError(msg)
    if int(manifest.get("n_reps", 0)) < 1:
        msg = "manifest n_reps must be positive"
        raise ValueError(msg)
    regimes = manifest.get("regimes")
    if not isinstance(regimes, list) or not regimes:
        msg = "manifest regimes must be a non-empty list"
        raise ValueError(msg)
