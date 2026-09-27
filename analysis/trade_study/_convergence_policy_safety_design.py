"""Frozen design for held-out convergence-policy safety validation (#195)."""

from __future__ import annotations

import copy
import math
from itertools import product
from typing import Any

from ._convergence_detector_design import SCENARIOS

MANIFEST_VERSION = "vbpca.convergence-policy-safety.v1"
DESIGN_VERSION = "v1_heldout_production_vs_no_rms"
RESULT_SCHEMA_VERSION = 1
CONDITIONS = ("production", "without_rms_plateau")
REFERENCE_CONDITION = "production"
CANDIDATE_CONDITION = "without_rms_plateau"
MISSING_FRACTION = 0.30
HOLDOUT_FRACTION = 0.10
MAX_COMPONENTS = 8
PP_REPS = 200
PA_QUANTILE = 0.745
SEQ_ALPHA = 0.055

PROFILE_SETTINGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "shapes": {
            "wide_smoke": (20, 30),
            "square_smoke": (24, 24),
        },
        "scenarios": ("null_gaussian", "strong_gaussian"),
        "missingness": ("complete", "mcar"),
    },
    "confirm": {
        "shapes": {
            "wide": (60, 180),
            "square": (120, 120),
            "tall": (240, 60),
        },
        "scenarios": (
            "null_gaussian",
            "strong_gaussian",
            "weak_gaussian",
            "heteroskedastic",
            "heavy_tailed",
        ),
        "missingness": ("complete", "mcar", "mar", "mnar", "block"),
    },
}

NONINFERIORITY_MARGINS = {
    "relative_holdout_rmse": 0.01,
    "relative_interval_score": 0.01,
    "coverage_difference": -0.02,
    "exact_recovery_difference": -0.02,
    "mae_gain": -0.10,
    "null_positive_rate_difference": 0.02,
    "capacity_mean_increase": 0.25,
}

POSTERIOR_DRIFT_REFERENCE_MARGINS = {
    "reconstruction_relative_frobenius": 0.01,
    "predictive_variance_relative_frobenius": 0.02,
    "noise_variance_relative_change": 0.02,
    "loading_subspace_max_angle_radians": 0.08726646259971647,
}


def build_manifest(profile: str, *, n_reps: int, seed: int) -> dict[str, Any]:
    """Build and validate one JSON-compatible immutable manifest."""
    if profile not in PROFILE_SETTINGS:
        msg = f"unknown profile {profile!r}; choose from {tuple(PROFILE_SETTINGS)}"
        raise ValueError(msg)
    if n_reps < 1:
        msg = f"n_reps must be positive, got {n_reps}"
        raise ValueError(msg)
    settings = PROFILE_SETTINGS[profile]
    cells: list[dict[str, Any]] = []
    for shape, scenario, missingness in product(
        settings["shapes"], settings["scenarios"], settings["missingness"]
    ):
        n, p = settings["shapes"][shape]
        cells.append({
            "cell_id": f"{shape}__{scenario}__{missingness}",
            "shape": shape,
            "n": n,
            "p": p,
            "scenario": scenario,
            "missingness": missingness,
            **SCENARIOS[scenario],
        })
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "design_version": DESIGN_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "profile": profile,
        "seed": int(seed),
        "n_reps": int(n_reps),
        "conditions": list(CONDITIONS),
        "reference_condition": REFERENCE_CONDITION,
        "candidate_condition": CANDIDATE_CONDITION,
        "cells": cells,
        "missing_fraction": MISSING_FRACTION,
        "holdout_fraction": HOLDOUT_FRACTION,
        "max_components": MAX_COMPONENTS,
        "pp_reps": PP_REPS,
        "pa_quantile": PA_QUANTILE,
        "seq_alpha": SEQ_ALPHA,
        "noninferiority_margins": dict(NONINFERIORITY_MARGINS),
        "posterior_drift_reference_margins": dict(POSTERIOR_DRIFT_REFERENCE_MARGINS),
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:  # noqa: PLR0912
    """Reject malformed or altered safety-study manifests."""
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        msg = "unexpected convergence-policy safety manifest version"
        raise ValueError(msg)
    if manifest.get("design_version") != DESIGN_VERSION:
        msg = "unexpected convergence-policy safety design version"
        raise ValueError(msg)
    if manifest.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        msg = "unexpected convergence-policy safety result schema"
        raise ValueError(msg)
    if tuple(manifest.get("conditions", ())) != CONDITIONS:
        msg = "safety-study conditions differ from the frozen comparison"
        raise ValueError(msg)
    if manifest.get("reference_condition") != REFERENCE_CONDITION:
        msg = "unexpected reference condition"
        raise ValueError(msg)
    if manifest.get("candidate_condition") != CANDIDATE_CONDITION:
        msg = "unexpected candidate condition"
        raise ValueError(msg)
    if int(manifest.get("n_reps", 0)) < 1:
        msg = "manifest n_reps must be positive"
        raise ValueError(msg)
    cells = manifest.get("cells", [])
    if not cells or len({cell["cell_id"] for cell in cells}) != len(cells):
        msg = "manifest cells must be non-empty with unique identifiers"
        raise ValueError(msg)
    if not math.isclose(
        float(manifest.get("missing_fraction", -1.0)), MISSING_FRACTION
    ):
        msg = "unexpected missing fraction"
        raise ValueError(msg)
    if not math.isclose(
        float(manifest.get("holdout_fraction", -1.0)), HOLDOUT_FRACTION
    ):
        msg = "unexpected holdout fraction"
        raise ValueError(msg)
    if int(manifest.get("max_components", 0)) != MAX_COMPONENTS:
        msg = "unexpected component-search cap"
        raise ValueError(msg)
    if int(manifest.get("pp_reps", 0)) != PP_REPS:
        msg = "unexpected posterior-predictive replicate count"
        raise ValueError(msg)
    if not math.isclose(float(manifest.get("pa_quantile", -1.0)), PA_QUANTILE):
        msg = "unexpected PA quantile"
        raise ValueError(msg)
    if not math.isclose(float(manifest.get("seq_alpha", -1.0)), SEQ_ALPHA):
        msg = "unexpected Seq alpha"
        raise ValueError(msg)
    if manifest.get("noninferiority_margins") != NONINFERIORITY_MARGINS:
        msg = "noninferiority margins differ from the frozen design"
        raise ValueError(msg)
    if (
        manifest.get("posterior_drift_reference_margins")
        != POSTERIOR_DRIFT_REFERENCE_MARGINS
    ):
        msg = "posterior-drift reference margins differ from the frozen design"
        raise ValueError(msg)


def condition_options(base_options: dict[str, Any], condition: str) -> dict[str, Any]:
    """Return isolated fit options for one registered condition."""
    if condition not in CONDITIONS:
        msg = f"unknown safety-study condition {condition!r}"
        raise ValueError(msg)
    options = copy.deepcopy(base_options)
    options.pop("xprobe_fraction", None)
    criteria = dict(options.get("convergence_criteria") or {})
    if condition == CANDIDATE_CONDITION:
        criteria["rms_plateau"] = False
    options.update({
        "convergence_criteria": criteria,
        "runtime_tuning": "off",
        "num_cpu": 1,
        "verbose": 0,
    })
    return options
