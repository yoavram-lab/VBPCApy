# ruff: noqa: FURB113, PERF401 - explicit loops keep the factor grid readable
"""Immutable design for posterior-stability convergence validation (#186)."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
from typing import Any

from .convergence_detector import DetectorPolicy

DESIGN_VERSION = "v1_posterior_stability_detector"
MANIFEST_VERSION = "vbpca.convergence-detector.v1"
MISSING_FRACTION = 0.30
HOLDOUT_FRACTION = 0.10

FIDELITY_MARGINS = {
    "reconstruction_relative_frobenius": 0.01,
    "predictive_variance_relative_frobenius": 0.02,
    "noise_variance_relative_change": 0.02,
    "loading_subspace_max_angle_radians": 0.08726646259971647,
    "holdout_rmse_relative_change": 0.01,
    "active_components_must_match": True,
}

PROFILE_SETTINGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "shapes": {
            "wide_smoke": (20, 30),
            "tall_smoke": (30, 20),
        },
        "scenarios": ("null_gaussian", "strong_gaussian"),
        "missingness": ("complete", "mcar"),
        "checkpoints": (5, 10, 20),
        "tail_points": 1,
        "late_margin": 5,
    },
    "screen": {
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
        "checkpoints": (50, 100, 200, 400, 800, 1200, 1600),
        "tail_points": 2,
        "late_margin": 100,
    },
}

SCENARIOS: dict[str, dict[str, Any]] = {
    "null_gaussian": {
        "true_rank": 0,
        "signal_std": 0.0,
        "noise_std": 1.0,
        "noise_model": "gaussian",
    },
    "strong_gaussian": {
        "true_rank": 3,
        "signal_std": 1.5,
        "noise_std": 0.5,
        "noise_model": "gaussian",
    },
    "weak_gaussian": {
        "true_rank": 3,
        "signal_std": 0.45,
        "noise_std": 1.0,
        "noise_model": "gaussian",
    },
    "heteroskedastic": {
        "true_rank": 3,
        "signal_std": 1.0,
        "noise_std": 0.7,
        "noise_model": "heteroskedastic_gaussian",
    },
    "heavy_tailed": {
        "true_rank": 3,
        "signal_std": 1.0,
        "noise_std": 0.7,
        "noise_model": "student_t3",
    },
}


def candidate_policies(*, warmup: int) -> list[DetectorPolicy]:
    """Build the prespecified detector grid, excluding criterion order.

    Returns:
        Candidate policies before stop-time-equivalence collapse.
    """
    policies: list[DetectorPolicy] = []
    for threshold, patience in product((1e-4, 1e-6, 1e-8), (1, 2, 3)):
        policies.append(
            DetectorPolicy(
                name=f"angle_{threshold:.0e}_p{patience}_w{warmup}",
                enabled=("angle",),
                minangle=threshold,
                patience=patience,
                warmup=warmup,
            )
        )
    for window, relative, patience in product(
        (25, 50, 100), (1e-2, 1e-3, 1e-4), (1, 3)
    ):
        policies.append(
            DetectorPolicy(
                name=f"rms_w{window}_r{relative:.0e}_p{patience}_w{warmup}",
                enabled=("rms_plateau",),
                rmsstop=(window, 1e-6, relative),
                patience=patience,
                warmup=warmup,
            )
        )
    for threshold, patience in product((1e-3, 1e-5, 1e-7), (1, 3)):
        policies.append(
            DetectorPolicy(
                name=f"cost_rel_{threshold:.0e}_p{patience}_w{warmup}",
                enabled=("cost",),
                cfstop_rel=threshold,
                patience=patience,
                warmup=warmup,
            )
        )
    for angle, relative, patience in product((1e-4, 1e-6), (1e-2, 1e-3), (1, 3)):
        suffix = f"a{angle:.0e}_r{relative:.0e}_p{patience}_w{warmup}"
        policies.append(
            DetectorPolicy(
                name=f"angle_or_rms_{suffix}",
                enabled=("angle", "rms_plateau"),
                minangle=angle,
                rmsstop=(50, 1e-6, relative),
                patience=patience,
                warmup=warmup,
            )
        )
        policies.append(
            DetectorPolicy(
                name=f"composite_{suffix}",
                enabled=("composite",),
                composite_stop=(
                    ("angle", angle),
                    ("rms", relative),
                    ("elbo_rel", 1e-5),
                ),
                patience=patience,
                warmup=warmup,
            )
        )
    for patience in (1, 2, 3):
        policies.append(
            DetectorPolicy(
                name=f"probe_deterioration_p{patience}_w{warmup}",
                enabled=("earlystop",),
                earlystop=True,
                patience=patience,
                warmup=warmup,
            )
        )
    return policies


def _policy_json(policy: DetectorPolicy) -> dict[str, Any]:
    payload = asdict(policy)
    payload["enabled"] = list(policy.enabled)
    payload["criterion_order"] = list(policy.criterion_order)
    payload["composite_stop"] = [list(item) for item in policy.composite_stop]
    payload["rmsstop"] = list(policy.rmsstop) if policy.rmsstop is not None else None
    payload["cfstop"] = list(policy.cfstop) if policy.cfstop is not None else None
    return payload


def policy_from_json(payload: dict[str, Any]) -> DetectorPolicy:
    """Reconstruct one immutable policy from a manifest mapping."""
    values = dict(payload)
    values["enabled"] = tuple(values["enabled"])
    values["criterion_order"] = tuple(values["criterion_order"])
    values["composite_stop"] = tuple(
        (str(name), float(threshold)) for name, threshold in values["composite_stop"]
    )
    if values["rmsstop"] is not None:
        values["rmsstop"] = tuple(values["rmsstop"])
    if values["cfstop"] is not None:
        values["cfstop"] = tuple(values["cfstop"])
    return DetectorPolicy(**values)


def build_manifest(profile: str, *, n_reps: int, seed: int) -> dict[str, Any]:
    """Build a JSON-compatible immutable Stage 1 manifest.

    Returns:
        Validated manifest mapping.
    """
    if profile not in PROFILE_SETTINGS:
        msg = f"unknown profile {profile!r}; choose from {tuple(PROFILE_SETTINGS)}"
        raise ValueError(msg)
    if n_reps < 1:
        msg = f"n_reps must be positive, got {n_reps}"
        raise ValueError(msg)
    settings = PROFILE_SETTINGS[profile]
    cells: list[dict[str, Any]] = []
    for shape_name, scenario_name, missingness in product(
        settings["shapes"], settings["scenarios"], settings["missingness"]
    ):
        n, p = settings["shapes"][shape_name]
        scenario = SCENARIOS[scenario_name]
        cells.append({
            "cell_id": f"{shape_name}__{scenario_name}__{missingness}",
            "shape": shape_name,
            "n": n,
            "p": p,
            "scenario": scenario_name,
            "missingness": missingness,
            **scenario,
        })
    policies = [
        policy
        for warmup in (0, 50, 100, 200)
        for policy in candidate_policies(warmup=warmup)
    ]
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "design_version": DESIGN_VERSION,
        "profile": profile,
        "seed": int(seed),
        "n_reps": int(n_reps),
        "cells": cells,
        "checkpoints": list(settings["checkpoints"]),
        "tail_points": int(settings["tail_points"]),
        "late_margin": int(settings["late_margin"]),
        "missing_fraction": MISSING_FRACTION,
        "holdout_fraction": HOLDOUT_FRACTION,
        "fidelity_margins": dict(FIDELITY_MARGINS),
        "policies": [_policy_json(policy) for policy in policies],
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject malformed or internally inconsistent manifests."""
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        msg = f"unexpected manifest version: {manifest.get('manifest_version')!r}"
        raise ValueError(msg)
    if manifest.get("design_version") != DESIGN_VERSION:
        msg = f"unexpected design version: {manifest.get('design_version')!r}"
        raise ValueError(msg)
    if int(manifest.get("n_reps", 0)) < 1:
        msg = "manifest n_reps must be positive"
        raise ValueError(msg)
    cells = manifest.get("cells", [])
    if not cells or len({cell["cell_id"] for cell in cells}) != len(cells):
        msg = "manifest cells must be non-empty with unique cell_id values"
        raise ValueError(msg)
    checkpoints = [int(value) for value in manifest.get("checkpoints", [])]
    if (
        len(checkpoints) < 3
        or checkpoints != sorted(set(checkpoints))
        or checkpoints[0] < 1
    ):
        msg = "manifest checkpoints must be at least three unique positive values"
        raise ValueError(msg)
    tail_points = int(manifest.get("tail_points", 0))
    if not 1 <= tail_points < len(checkpoints):
        msg = "tail_points must index at least one pre-endpoint checkpoint"
        raise ValueError(msg)
    if manifest.get("fidelity_margins") != FIDELITY_MARGINS:
        msg = "manifest fidelity margins differ from the registered design"
        raise ValueError(msg)
    policies = [policy_from_json(payload) for payload in manifest.get("policies", [])]
    if not policies or len({policy.name for policy in policies}) != len(policies):
        msg = "manifest policies must be non-empty with unique names"
        raise ValueError(msg)
