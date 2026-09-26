"""Immutable design for the released-default residual-cap follow-up (#174)."""

from __future__ import annotations

import copy
from typing import Any

from vbpca_py import defaults as vbpca_defaults
from vbpca_py import recommend_config

DESIGN_VERSION = "v1_vbpca_0_4_residual_caps"
MANIFEST_VERSION = "vbpca.residual-convergence.v1"
VBPCA_RELEASE = "0.4.0"
REFERENCE_CONDITION = "released"
CONDITIONS = ("released", "double_cap", "forced_long")
REGISTERED_N_REPS = 16
REGISTERED_SEED = 20260926

_AFFECTED_BUCKETS = {"wide_moderate", "tall_extreme"}
_ALL_CRITERIA_FALSE = {
    "angle": False,
    "earlystop": False,
    "rms_plateau": False,
    "cost": False,
    "composite": False,
    "slowing_down": False,
}

# The confirmation profile retains the three regimes with residual cap hits in
# #166 and adds independently shaped complete/MNAR cases around both routing
# buckets. The seed schedule and replicate count are frozen before inspection.
REGIME_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "smoke": {
        "wide_complete_smoke": {
            "n": 20,
            "p": 100,
            "true_rank": 2,
            "missingness": "complete",
            "noise_std": 0.5,
        },
        "tall_extreme_mnar_smoke": {
            "n": 255,
            "p": 5,
            "true_rank": 2,
            "missingness": "mnar_censored",
            "noise_std": 0.75,
        },
    },
    "confirm": {
        "wide_complete_anchor": {
            "n": 40,
            "p": 200,
            "true_rank": 5,
            "missingness": "complete",
            "noise_std": 0.3,
        },
        "wide_mcar_anchor": {
            "n": 50,
            "p": 300,
            "true_rank": 2,
            "missingness": "mcar",
            "noise_std": 0.5,
        },
        "wide_complete_holdout": {
            "n": 64,
            "p": 320,
            "true_rank": 5,
            "missingness": "complete",
            "noise_std": 0.3,
        },
        "wide_mnar_holdout": {
            "n": 80,
            "p": 400,
            "true_rank": 8,
            "missingness": "mnar_censored",
            "noise_std": 0.7,
        },
        "tall_extreme_mnar_anchor": {
            "n": 1020,
            "p": 20,
            "true_rank": 3,
            "missingness": "mnar_censored",
            "noise_std": 1.0,
        },
        "tall_extreme_mnar_holdout": {
            "n": 1275,
            "p": 25,
            "true_rank": 4,
            "missingness": "mnar_censored",
            "noise_std": 1.0,
        },
        "tall_extreme_complete_holdout": {
            "n": 1800,
            "p": 30,
            "true_rank": 3,
            "missingness": "complete",
            "noise_std": 0.5,
        },
    },
}


def condition_config(n: int, p: int, condition: str) -> dict[str, Any]:
    """Return a fresh current-release configuration for one condition.

    Returns:
        Estimator-ready VBPCA keyword arguments.

    Raises:
        ValueError: If the condition or routing bucket is outside the
            preregistered follow-up.
    """
    if condition not in CONDITIONS:
        msg = f"unknown residual-convergence condition {condition!r}"
        raise ValueError(msg)
    bucket = vbpca_defaults._bucket(n, p)  # noqa: SLF001
    if bucket not in _AFFECTED_BUCKETS:
        msg = f"residual follow-up does not cover bucket {bucket!r}"
        raise ValueError(msg)

    config = copy.deepcopy(recommend_config(n=n, p=p))
    if condition == "released":
        return config

    config["maxiters"] = 2 * int(config["maxiters"])
    if condition == "forced_long":
        config["convergence_criteria"] = copy.deepcopy(_ALL_CRITERIA_FALSE)
    return config


def build_manifest(
    profile: str,
    *,
    n_reps: int,
    seed: int,
    conditions: tuple[str, ...] = CONDITIONS,
    reference_condition: str = REFERENCE_CONDITION,
) -> dict[str, Any]:
    """Build a JSON-compatible residual-convergence manifest.

    Returns:
        Validated manifest mapping.

    Raises:
        ValueError: If the requested design is invalid.
    """
    if profile not in REGIME_PROFILES:
        msg = f"unknown profile {profile!r}; choose from {tuple(REGIME_PROFILES)}"
        raise ValueError(msg)
    if n_reps < 1:
        msg = f"n_reps must be positive, got {n_reps}"
        raise ValueError(msg)
    if not conditions:
        msg = "conditions must contain at least one registered condition"
        raise ValueError(msg)
    if len(set(conditions)) != len(conditions):
        msg = "conditions must not contain duplicates"
        raise ValueError(msg)
    unknown_conditions = set(conditions).difference(CONDITIONS)
    if unknown_conditions:
        msg = f"unknown residual-convergence conditions: {sorted(unknown_conditions)}"
        raise ValueError(msg)
    if reference_condition not in conditions:
        msg = f"reference condition {reference_condition!r} is absent from conditions"
        raise ValueError(msg)

    regimes: list[dict[str, Any]] = []
    for index, (name, raw) in enumerate(REGIME_PROFILES[profile].items()):
        regime = {"name": name, **copy.deepcopy(raw)}
        bucket = vbpca_defaults._bucket(regime["n"], regime["p"])  # noqa: SLF001
        if bucket not in _AFFECTED_BUCKETS:
            msg = f"regime {name!r} routes to unregistered bucket {bucket!r}"
            raise ValueError(msg)
        regime["bucket"] = bucket
        regime["seed"] = seed + index
        regimes.append(regime)

    return {
        "manifest_version": MANIFEST_VERSION,
        "design_version": DESIGN_VERSION,
        "vbpca_release": VBPCA_RELEASE,
        "profile": profile,
        "n_reps": n_reps,
        "seed": seed,
        "reference_condition": reference_condition,
        "conditions": list(conditions),
        "regimes": regimes,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate schema and reconstruct every immutable design field.

    Raises:
        ValueError: If the manifest differs from the registered design.
    """
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        msg = f"unsupported manifest_version {manifest.get('manifest_version')!r}"
        raise ValueError(msg)
    if manifest.get("design_version") != DESIGN_VERSION:
        msg = f"unsupported design_version {manifest.get('design_version')!r}"
        raise ValueError(msg)
    if manifest.get("vbpca_release") != VBPCA_RELEASE:
        msg = f"unsupported vbpca_release {manifest.get('vbpca_release')!r}"
        raise ValueError(msg)

    conditions = manifest.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        msg = "manifest conditions must be a non-empty list"
        raise ValueError(msg)
    reference = manifest.get("reference_condition")
    expected = build_manifest(
        str(manifest.get("profile")),
        n_reps=int(manifest.get("n_reps", 0)),
        seed=int(manifest.get("seed", 0)),
        conditions=tuple(str(condition) for condition in conditions),
        reference_condition=str(reference),
    )
    if manifest != expected:
        msg = "manifest fields differ from the code-registered design"
        raise ValueError(msg)


def registered_manifest() -> dict[str, Any]:
    """Return the preregistered confirmation manifest.

    Returns:
        The frozen seven-regime, three-condition confirmation design.
    """
    return build_manifest(
        "confirm",
        n_reps=REGISTERED_N_REPS,
        seed=REGISTERED_SEED,
    )
