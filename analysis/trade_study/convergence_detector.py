"""Replay VBPCA stopping policies against a completed learning curve.

The replay contract mirrors the training loop: criterion-specific patience is
updated on every iteration, but hits during broad-prior warmup cannot stop a
fit and do not contribute patience credit after warmup. Criterion ordering is
retained for reason attribution and excluded from policy equivalence because
it cannot change the first iteration on which any rule becomes eligible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from vbpca_py._converge import (
    DEFAULT_CRITERION_ORDER,
    convergence_check,
)

VALID_CRITERIA = frozenset(DEFAULT_CRITERION_ORDER)


@dataclass(frozen=True)
class DetectorPolicy:
    """Immutable stopping policy for offline replay."""

    name: str
    enabled: tuple[str, ...]
    minangle: float | None = None
    earlystop: bool = False
    rmsstop: tuple[int, float, float] | None = None
    cfstop: tuple[int, float, float] | None = None
    cfstop_rel: float | None = None
    cfstop_curv: float | None = None
    composite_stop: tuple[tuple[str, float], ...] = ()
    patience: int = 1
    warmup: int = 0
    criterion_order: tuple[str, ...] = tuple(DEFAULT_CRITERION_ORDER)

    def __post_init__(self) -> None:
        """Validate names and internally coupled settings."""
        if not self.name:
            msg = "policy name must be non-empty"
            raise ValueError(msg)
        if len(set(self.enabled)) != len(self.enabled):
            msg = f"enabled criteria contain duplicates: {self.enabled}"
            raise ValueError(msg)
        unknown = set(self.enabled).difference(VALID_CRITERIA)
        if unknown:
            msg = f"unknown enabled criteria: {sorted(unknown)}"
            raise ValueError(msg)
        if set(self.criterion_order) != VALID_CRITERIA:
            msg = "criterion_order must contain each registered criterion once"
            raise ValueError(msg)
        if self.patience < 1:
            msg = f"patience must be positive, got {self.patience}"
            raise ValueError(msg)
        if self.warmup < 0:
            msg = f"warmup must be non-negative, got {self.warmup}"
            raise ValueError(msg)
        if "angle" in self.enabled and self.minangle is None:
            msg = "angle is enabled but minangle is unset"
            raise ValueError(msg)
        if "earlystop" in self.enabled and not self.earlystop:
            msg = "earlystop is enabled but its detector flag is false"
            raise ValueError(msg)
        if "rms_plateau" in self.enabled and self.rmsstop is None:
            msg = "rms_plateau is enabled but rmsstop is unset"
            raise ValueError(msg)
        if "cost" in self.enabled and all(
            value is None for value in (self.cfstop, self.cfstop_rel, self.cfstop_curv)
        ):
            msg = "cost is enabled but no cost detector is configured"
            raise ValueError(msg)
        if "composite" in self.enabled and not self.composite_stop:
            msg = "composite is enabled but composite_stop is empty"
            raise ValueError(msg)

    def options(self) -> dict[str, object]:
        """Return options consumed by :func:`convergence_check`."""
        return {
            "criterion_order": list(self.criterion_order),
            "convergence_criteria": {
                criterion: criterion in self.enabled
                for criterion in DEFAULT_CRITERION_ORDER
            },
            "minangle": self.minangle,
            "earlystop": self.earlystop,
            "rmsstop": self.rmsstop,
            "cfstop": self.cfstop,
            "cfstop_rel": self.cfstop_rel,
            "cfstop_curv": self.cfstop_curv,
            "composite_stop": dict(self.composite_stop),
            "patience": self.patience,
        }

    def equivalence_key(self) -> tuple[object, ...]:
        """Return the stop-time identity, excluding name and reason order."""
        return (
            tuple(sorted(self.enabled)),
            self.minangle,
            self.earlystop,
            self.rmsstop,
            self.cfstop,
            self.cfstop_rel,
            self.cfstop_curv,
            tuple(sorted(self.composite_stop)),
            self.patience,
            self.warmup,
        )


@dataclass(frozen=True)
class ReplayResult:
    """First eligible stopping event for one policy and learning curve."""

    policy: str
    stop_iteration: int | None
    reason: str
    message: str
    terminal_iteration: int


def _triple_or_none(value: object) -> tuple[int, float, float] | None:
    """Normalize one three-value convergence option."""
    if value is None:
        return None
    array = np.asarray(value, dtype=float).ravel()
    if array.size == 0:
        return None
    if array.size != 3:
        msg = f"expected a three-value convergence option, got {array.size}"
        raise ValueError(msg)
    return int(array[0]), float(array[1]), float(array[2])


def policy_from_options(
    name: str,
    options: Mapping[str, object],
    *,
    disabled: tuple[str, ...] = (),
) -> DetectorPolicy:
    """Build the effective replay policy from resolved VBPCA options.

    Criteria that are nominally enabled but lack their required configuration
    remain inactive, matching :func:`convergence_check`. The slowing-down
    criterion remains in the policy when enabled; callers reproduce current
    live behavior by supplying no slowing-down event iterations.
    """
    order_raw = options.get("criterion_order")
    order = tuple(order_raw or DEFAULT_CRITERION_ORDER)
    enabled_raw = options.get("convergence_criteria")
    if enabled_raw is not None and not isinstance(enabled_raw, Mapping):
        msg = "convergence_criteria must be a mapping"
        raise TypeError(msg)
    enabled_map = enabled_raw or {}
    disabled_set = set(disabled)
    unknown = disabled_set.difference(VALID_CRITERIA)
    if unknown:
        msg = f"unknown disabled criteria: {sorted(unknown)}"
        raise ValueError(msg)

    minangle_raw = options.get("minangle")
    minangle = float(minangle_raw) if minangle_raw is not None else None
    rmsstop = _triple_or_none(options.get("rmsstop"))
    cfstop = _triple_or_none(options.get("cfstop"))
    cfstop_rel_raw = options.get("cfstop_rel")
    cfstop_rel = float(cfstop_rel_raw) if cfstop_rel_raw is not None else None
    cfstop_curv_raw = options.get("cfstop_curv")
    cfstop_curv = float(cfstop_curv_raw) if cfstop_curv_raw is not None else None
    composite_raw = options.get("composite_stop")
    if composite_raw is None:
        composite: tuple[tuple[str, float], ...] = ()
    elif isinstance(composite_raw, Mapping):
        composite = tuple(
            sorted((str(key), float(value)) for key, value in composite_raw.items())
        )
    else:
        msg = "composite_stop must be a mapping or None"
        raise TypeError(msg)

    configured = {
        "angle": minangle is not None,
        "earlystop": bool(options.get("earlystop", False)),
        "rms_plateau": rmsstop is not None,
        "cost": any(value is not None for value in (cfstop, cfstop_rel, cfstop_curv)),
        "composite": bool(composite),
        "slowing_down": True,
    }
    enabled = tuple(
        criterion
        for criterion in order
        if bool(enabled_map.get(criterion, True))
        and configured[criterion]
        and criterion not in disabled_set
    )
    return DetectorPolicy(
        name=name,
        enabled=enabled,
        minangle=minangle if "angle" in enabled else None,
        earlystop="earlystop" in enabled,
        rmsstop=rmsstop if "rms_plateau" in enabled else None,
        cfstop=cfstop if "cost" in enabled else None,
        cfstop_rel=cfstop_rel if "cost" in enabled else None,
        cfstop_curv=cfstop_curv if "cost" in enabled else None,
        composite_stop=composite if "composite" in enabled else (),
        patience=int(options.get("patience") or 1),
        warmup=int(options.get("niter_broadprior") or 0),
        criterion_order=order,
    )


def collapse_equivalent_policies(
    policies: tuple[DetectorPolicy, ...] | list[DetectorPolicy],
) -> tuple[list[DetectorPolicy], dict[str, str]]:
    """Collapse policies with identical stop behavior.

    Returns:
        Unique policies in input order and a mapping from every policy name to
        its representative name.
    """
    representatives: list[DetectorPolicy] = []
    aliases: dict[str, str] = {}
    keys: dict[tuple[object, ...], str] = {}
    for policy in policies:
        if policy.name in aliases:
            msg = f"duplicate policy name: {policy.name!r}"
            raise ValueError(msg)
        key = policy.equivalence_key()
        representative = keys.get(key)
        if representative is None:
            representative = policy.name
            keys[key] = representative
            representatives.append(policy)
        aliases[policy.name] = representative
    return representatives, aliases


def _numeric_series(
    learning_curve: dict[str, Any],
    name: str,
    *,
    length: int,
    required: bool = False,
) -> np.ndarray:
    values = learning_curve.get(name)
    if values is None:
        if required:
            msg = f"learning curve is missing required {name!r} series"
            raise ValueError(msg)
        return np.full(length, np.nan, dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != length:
        msg = f"learning-curve {name!r} must be one-dimensional with length {length}"
        raise ValueError(msg)
    return array


def replay_policy(
    learning_curve: dict[str, Any],
    policy: DetectorPolicy,
    *,
    slowing_down_iterations: set[int] | frozenset[int] = frozenset(),
) -> ReplayResult:
    """Replay one policy against a forced-long VBPCA learning curve.

    Iteration zero is the initialized state and is never eligible to stop.

    Returns:
        The first accepted stop and its reason, or a ``maxiters`` result.
    """
    rms_values = np.asarray(learning_curve.get("rms", []), dtype=float)
    if rms_values.ndim != 1 or len(rms_values) < 2:
        msg = "learning curve must contain initialization plus at least one RMS point"
        raise ValueError(msg)
    length = len(rms_values)
    prms_values = _numeric_series(learning_curve, "prms", length=length)
    cost_values = _numeric_series(learning_curve, "cost", length=length)
    angle_values = _numeric_series(
        learning_curve, "angle", length=length, required=True
    )

    replay_curve: dict[str, Any] = {
        "rms": [float(rms_values[0])],
        "prms": [float(prms_values[0])],
        "cost": [float(cost_values[0])],
    }
    options = policy.options()
    for iteration in range(1, length):
        replay_curve["rms"].append(float(rms_values[iteration]))
        replay_curve["prms"].append(float(prms_values[iteration]))
        replay_curve["cost"].append(float(cost_values[iteration]))
        sd_iter = 40 if iteration in slowing_down_iterations else None
        message = convergence_check(
            options,
            replay_curve,
            float(angle_values[iteration]),
            sd_iter=sd_iter,
        )

        if iteration <= policy.warmup:
            replay_curve.pop("_candidate_convergence_reason", None)
            replay_curve["_criterion_patience"] = {}
            continue
        if message:
            reason = str(replay_curve.pop("_candidate_convergence_reason", ""))
            return ReplayResult(
                policy=policy.name,
                stop_iteration=iteration,
                reason=reason,
                message=message,
                terminal_iteration=length - 1,
            )

    return ReplayResult(
        policy=policy.name,
        stop_iteration=None,
        reason="maxiters",
        message="",
        terminal_iteration=length - 1,
    )


def replay_policies(
    learning_curve: dict[str, Any],
    policies: tuple[DetectorPolicy, ...] | list[DetectorPolicy],
) -> list[ReplayResult]:
    """Replay a sequence of policies in input order."""
    return [replay_policy(learning_curve, policy) for policy in policies]
