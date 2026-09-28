"""Frozen design for the post-optimization VBPCA defaults study (#214/#225)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.stats import qmc

from ._modern_defaults_spec import (
    ANCHOR_CANDIDATES,
    BIAS_ORDERS,
    BOOTSTRAP_DESIGN,
    CONFIRM_GATES,
    CRITERION_POLICIES,
    DESIGN_VERSION,
    MANDATORY_CANDIDATES,
    MANIFEST_VERSION,
    MAX_FINALISTS,
    MAXITER_SCALES,
    MINIMUM_VBPCA_VERSION,
    N_SOBOL_CANDIDATES,
    NITER_LEVELS,
    PATIENCE_LEVELS,
    PROFILE_REGIMES,
    REGISTERED_SCREEN_REPS,
    REGISTERED_SCREEN_SEED,
    RMS_WINDOWS,
    SCREEN_GATES,
    SELECTION_DESIGN,
)


def _level(values: tuple[Any, ...], u: float) -> Any:
    return values[min(int(u * len(values)), len(values) - 1)]


def _log_value(lower: float, upper: float, u: float) -> float:
    return float(10.0 ** (np.log10(lower) + u * (np.log10(upper) - np.log10(lower))))


def _sobol_candidates() -> list[dict[str, Any]]:
    points = qmc.Sobol(d=15, scramble=True, seed=REGISTERED_SCREEN_SEED).random_base2(5)
    candidates: list[dict[str, Any]] = []
    for index, point in enumerate(points[:N_SOBOL_CANDIDATES]):
        candidate = {
            "id": f"screen_{index:03d}",
            "base": "recommended",
            "hp_va_scale": round(_log_value(0.35, 2.0, point[0]), 12),
            "hp_vb_scale": round(_log_value(0.35, 2.0, point[1]), 12),
            "hp_v_scale": round(_log_value(0.35, 2.0, point[2]), 12),
            "va_init_scale": round(_log_value(0.5, 2.0, point[3]), 12),
            "xprobe_fraction": round(0.03 + 0.17 * point[4], 12),
            "niter_broadprior": _level(NITER_LEVELS, point[5]),
            "maxiters_scale": _level(MAXITER_SCALES, point[6]),
            "patience": _level(PATIENCE_LEVELS, point[7]),
            "rmsstop_window": _level(RMS_WINDOWS, point[8]),
            "rmsstop_atol": round(_log_value(1e-6, 1e-2, point[9]), 14),
            "rmsstop_rtol": round(_log_value(1e-5, 1e-2, point[10]), 14),
            "minangle": round(_log_value(1e-10, 1e-4, point[11]), 16),
            "cfstop_rel": round(_log_value(1e-8, 1e-3, point[12]), 16),
            "criterion_policy": _level(tuple(CRITERION_POLICIES), point[13]),
            "compat_mode": "modern",
            "bias_update_order": _level(BIAS_ORDERS, point[14]),
        }
        candidates.append(candidate)
    return candidates


def all_screen_candidates() -> list[dict[str, Any]]:
    """Return independent copies of every preregistered screen candidate."""
    return [copy.deepcopy(item) for item in (*ANCHOR_CANDIDATES, *_sobol_candidates())]


def _candidates_by_id() -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in all_screen_candidates()}


def _selected_candidates(
    profile: str, candidate_ids: tuple[str, ...] | None
) -> list[dict[str, Any]]:
    available = _candidates_by_id()
    if candidate_ids is not None and len(set(candidate_ids)) != len(candidate_ids):
        msg = "candidate_ids must not contain duplicates"
        raise ValueError(msg)
    if profile == "screen":
        expected = tuple(available)
        if candidate_ids is not None and candidate_ids != expected:
            msg = "screen candidate_ids are frozen by the registered design"
            raise ValueError(msg)
        return list(available.values())
    if profile == "smoke":
        ids = MANDATORY_CANDIDATES if candidate_ids is None else candidate_ids
    else:
        if candidate_ids is None:
            msg = f"{profile} manifests require explicit screen finalist ids"
            raise ValueError(msg)
        ids = candidate_ids
        if any(required not in ids for required in MANDATORY_CANDIDATES):
            msg = f"{profile} candidates must include {MANDATORY_CANDIDATES}"
            raise ValueError(msg)
        optional = set(ids).difference(MANDATORY_CANDIDATES)
        if len(optional) > MAX_FINALISTS:
            msg = f"{profile} accepts at most {MAX_FINALISTS} screen finalists"
            raise ValueError(msg)
    unknown = set(ids).difference(available)
    if unknown:
        msg = f"unknown candidate ids: {sorted(unknown)}"
        raise ValueError(msg)
    return [copy.deepcopy(available[item]) for item in ids]


def build_manifest(
    profile: str,
    *,
    n_reps: int,
    seed: int,
    candidate_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build one deterministic, JSON-compatible study manifest."""
    if profile not in PROFILE_REGIMES:
        msg = f"unknown profile {profile!r}; choose from {tuple(PROFILE_REGIMES)}"
        raise ValueError(msg)
    if n_reps < 1:
        msg = f"n_reps must be positive, got {n_reps}"
        raise ValueError(msg)
    candidates = _selected_candidates(profile, candidate_ids)
    regimes = []
    for index, raw in enumerate(PROFILE_REGIMES[profile]):
        regime = copy.deepcopy(raw)
        regime["seed"] = seed + index
        regimes.append(regime)
    return {
        "manifest_version": MANIFEST_VERSION,
        "design_version": DESIGN_VERSION,
        "minimum_vbpca_version": MINIMUM_VBPCA_VERSION,
        "profile": profile,
        "n_reps": n_reps,
        "seed": seed,
        "candidates": candidates,
        "regimes": regimes,
        "selection": copy.deepcopy(SELECTION_DESIGN),
        "screen_gates": copy.deepcopy(SCREEN_GATES),
        "confirm_gates": copy.deepcopy(CONFIRM_GATES),
        "bootstrap": copy.deepcopy(BOOTSTRAP_DESIGN),
        "wall_time_is_quality_objective": False,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject any manifest that differs from the code-registered design."""
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        msg = f"unsupported manifest_version {manifest.get('manifest_version')!r}"
        raise ValueError(msg)
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        msg = "manifest candidates must be a non-empty list"
        raise ValueError(msg)
    candidate_ids = tuple(str(item.get("id")) for item in candidates)
    expected = build_manifest(
        str(manifest.get("profile")),
        n_reps=int(manifest.get("n_reps", 0)),
        seed=int(manifest.get("seed", 0)),
        candidate_ids=candidate_ids,
    )
    if manifest != expected:
        msg = "manifest fields differ from the code-registered design"
        raise ValueError(msg)


def registered_screen_manifest() -> dict[str, Any]:
    """Return the frozen screen manifest."""
    return build_manifest(
        "screen",
        n_reps=REGISTERED_SCREEN_REPS,
        seed=REGISTERED_SCREEN_SEED,
    )
