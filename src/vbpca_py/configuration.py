"""Typed solver settings and the explicit legacy keyword conversion boundary.

``PCAOptions`` is immutable configuration. It snapshots small stopping-policy
containers; data, probe and initialization payloads are borrowed until fitting,
where the solver takes its existing owned working copies. A supplied Generator
keeps NumPy's stateful semantics. Runtime tuning and numerical working state
live separately from this record.
"""

# ruff: noqa: PLC2801 - frozen dataclass normalization requires object.__setattr__

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from pathlib import Path  # noqa: TC003 - public runtime type-hint resolution
from typing import Generic, SupportsFloat, SupportsIndex, TypeVar, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp  # noqa: TC002 - public runtime type-hint resolution

from ._memory import resolve_max_dense_bytes
from .options import (
    AccessorMode,
    Algorithm,
    BiasUpdateOrder,
    CompatMode,
    CostDiagnostics,
    CovarianceWriteback,
    ExplainedVarianceSolver,
    RuntimeTuning,
    StatisticsMode,
    VarianceUpdateOrder,
)

_Value = TypeVar("_Value")


@dataclass(frozen=True, slots=True, eq=False)
class _FrozenMapping(Mapping[str, _Value], Generic[_Value]):
    """Small owned, immutable and picklable policy mapping."""

    entries: tuple[tuple[str, _Value], ...]

    def __getitem__(self, key: str) -> _Value:
        for name, value in self.entries:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _value in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def _float_value(value: object, default: float) -> float:
    """Preserve legacy numeric-string/fallback conversion at this boundary.

    Returns:
        The converted value or the supplied established fallback.
    """
    if value is None:
        return default
    if isinstance(value, bytes | bytearray):
        value = value.decode()
    try:
        return float(cast("str | SupportsFloat | SupportsIndex", value))
    except (TypeError, ValueError):
        return default


def _int_value(value: object, default: int) -> int:
    try:
        return int(cast("str | SupportsIndex", value))
    except (TypeError, ValueError, OverflowError):
        return default


def _threshold(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(item) for item in np.asarray(value, dtype=float).ravel())


@dataclass(frozen=True, slots=True)
class PCAOptions:
    """Typed configuration accepted by ``pca_full(..., options=config)``.

    Use :meth:`from_kwargs` to import legacy keyword dictionaries (including
    case normalization and runtime defaults). Direct construction exposes typed
    fields and snapshots stopping containers. Estimators store their original
    constructor arguments and assemble this record at fit time. See the options
    guide for the statistical and execution meaning of every field.

    ``init`` is the localized dynamic MAT/struct boundary: it also accepts
    third-party objects exposing ``_fieldnames``; normalization and shape checks
    run during fitting. Array shapes are runtime contracts, not type promises.
    """

    init: object = "random"
    maxiters: int = 1000
    bias: bool = True
    max_dense_bytes: int | None = 2_000_000_000
    uniquesv: bool = False
    auto_pattern_masked: bool = False
    angle_every: int = 1
    minangle: float = 1e-8
    algorithm: Algorithm = Algorithm.VB
    niter_broadprior: int = 100
    hp_va: float = 0.001
    hp_vb: float = 0.001
    hp_v: float = 0.001
    va_init: float = 1000.0
    random_state: int | np.random.Generator | None = None
    earlystop: bool = False
    rmsstop: tuple[float, ...] | None = (100.0, 1e-4, 1e-3)
    cfstop: tuple[float, ...] | None = ()
    cfstop_rel: float | None = None
    cfstop_curv: float | None = None
    record_cost: bool = False
    cost_diagnostics: CostDiagnostics = CostDiagnostics.TRACE
    record_prior_trace: bool = False
    composite_stop: Mapping[str, float] | None = None
    patience: int = 1
    criterion_order: tuple[str, ...] | None = None
    convergence_criteria: Mapping[str, bool] | None = None
    verbose: int = 1
    num_cpu: int | None = None
    num_cpu_score_update: int | None = None
    num_cpu_loadings_update: int | None = None
    num_cpu_noise_update: int | None = None
    num_cpu_rms: int | None = None
    runtime_tuning: RuntimeTuning = RuntimeTuning.SAFE
    runtime_profile: str | Path | None = None
    dense_sufficient_statistics: StatisticsMode = StatisticsMode.AUTO
    noise_sufficient_statistics: StatisticsMode = StatisticsMode.AUTO
    masked_batch_size: int = 0
    cov_writeback_mode: CovarianceWriteback | None = None
    log_progress_stride: int | None = None
    accessor_mode: AccessorMode | None = None
    xprobe: npt.ArrayLike | sp.spmatrix | None = None
    rotate2pca: bool = True
    display: bool = False
    bias_update_order: BiasUpdateOrder = BiasUpdateOrder.AUTO
    variance_update_order: VarianceUpdateOrder = VarianceUpdateOrder.AUTO
    compat_mode: CompatMode = CompatMode.STRICT_LEGACY
    return_diagnostics: bool = True
    runtime_report: bool = False
    explained_var_solver: ExplainedVarianceSolver = ExplainedVarianceSolver.AUTO
    explained_var_gram_ratio: float = 4.0

    def __post_init__(self) -> None:
        """Normalize settings and own their small policy containers."""
        self._normalize_categories()
        self._normalize_scalars()
        self._own_stopping_rules()
        self._validate_random_state()

    def _normalize_categories(self) -> None:
        """Resolve string-compatible enums without altering legacy case rules.

        Raises:
            ValueError: A categorical value is unsupported.
        """
        for name, option_type in _ENUM_FIELDS.items():
            value = getattr(self, name)
            if value is not None:
                # Algorithm historically lowercases without stripping spaces.
                normalized = (
                    str(value).lower()
                    if name == "algorithm"
                    else str(value).strip().lower()
                )
                try:
                    parsed = option_type(normalized)
                except ValueError as exc:
                    msg = f"{name} has unsupported value: {value!r}"
                    raise ValueError(msg) from exc
                object.__setattr__(self, name, parsed)

    def _normalize_scalars(self) -> None:
        """Convert scalar controls with their established fallback behavior."""
        for name in _BOOL_FIELDS:
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name, default in _INT_FIELDS.items():
            object.__setattr__(self, name, _int_value(getattr(self, name), default))
        for name, float_default in _FLOAT_FIELDS.items():
            object.__setattr__(
                self, name, _float_value(getattr(self, name), float_default)
            )
        for name in _OPTIONAL_INTS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _int_value(value, 0))
        for name in ("cfstop_rel", "cfstop_curv"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _float_value(value, 0.0))
        object.__setattr__(
            self, "max_dense_bytes", resolve_max_dense_bytes(self.max_dense_bytes)
        )
        object.__setattr__(self, "angle_every", max(1, self.angle_every))

    def _own_stopping_rules(self) -> None:
        """Snapshot small stopping containers without sharing mutable defaults.

        Raises:
            TypeError: A criterion enablement is not Boolean.
        """
        for name in ("rmsstop", "cfstop"):
            object.__setattr__(self, name, _threshold(getattr(self, name)))
        if self.criterion_order is not None:
            object.__setattr__(self, "criterion_order", tuple(self.criterion_order))
        if self.convergence_criteria is not None:
            if any(
                not isinstance(value, bool | np.bool_)
                for value in self.convergence_criteria.values()
            ):
                msg = "convergence_criteria values must be booleans"
                raise TypeError(msg)
            object.__setattr__(
                self,
                "convergence_criteria",
                _FrozenMapping(
                    tuple(
                        (str(key), bool(value))
                        for key, value in self.convergence_criteria.items()
                    )
                ),
            )
        if self.composite_stop is not None:
            object.__setattr__(
                self,
                "composite_stop",
                _FrozenMapping(
                    tuple(
                        (str(key), float(value))
                        for key, value in self.composite_stop.items()
                    )
                ),
            )

    def _validate_random_state(self) -> None:
        """Retain Generator semantics and normalize integer seeds.

        Raises:
            TypeError: A seed is outside the supported Generator/integer contract.
        """
        if self.random_state is not None and not isinstance(
            self.random_state, int | np.integer | np.random.Generator
        ):
            msg = "random_state must be an integer, Generator or None"
            raise TypeError(msg)
        if isinstance(self.random_state, np.integer):
            object.__setattr__(self, "random_state", int(self.random_state))

    @classmethod
    def from_kwargs(cls, **kwargs: object) -> PCAOptions:
        """Resolve an existing keyword dictionary into typed configuration.

        Returns:
            Validated immutable settings with the legacy defaults resolved.
        """
        from ._pca_full import _build_options  # noqa: PLC0415

        values = _build_options(kwargs)
        # Configuration records the request. Effective threads belong to a fit;
        # exporting or replacing this record must preserve inheritance.
        if not values["_num_cpu_user_set"]:
            values["num_cpu"] = None
        return _from_normalized(values)

    def to_kwargs(self) -> dict[str, object]:
        """Export fresh policy containers for legacy low-level keyword callers.

        Data, probe and initialization payloads retain their identities. Fitting
        owns their numerical working copies; exporting does not copy large data.

        Returns:
            A new canonical-name keyword mapping. Inherited threads remain None.
        """
        result = {
            item.name: cast("object", getattr(self, item.name))
            for item in fields(self)
            if item.init
        }
        for name in ("criterion_order",):
            value = result[name]
            if value is not None:
                result[name] = list(cast("tuple[str, ...]", value))
        for name in ("convergence_criteria", "composite_stop"):
            value = result[name]
            if value is not None:
                result[name] = dict(cast("Mapping[str, object]", value))
        for name in ("rmsstop", "cfstop"):
            value = result[name]
            if value is not None:
                result[name] = np.asarray(value, dtype=float).copy()
        return result


def _from_normalized(values: Mapping[str, object]) -> PCAOptions:
    """Decode the validated compatibility mapping into immutable typed fields.

    Returns:
        Configuration without retaining the input dictionary.
    """
    config = PCAOptions()
    for item in fields(PCAOptions):
        if item.init and item.name in values:
            object.__setattr__(config, item.name, values[item.name])
    config.__post_init__()
    return config


def _replace_config(config: PCAOptions, changes: Mapping[str, object]) -> PCAOptions:
    """Validate helper writes without re-resolving runtime environment defaults.

    Returns:
        A separate immutable record retaining thread-hint provenance.
    """
    values = config.to_kwargs()
    values.update(changes)
    return _from_normalized(values)


_ENUM_FIELDS: dict[
    str,
    type[
        Algorithm
        | CompatMode
        | BiasUpdateOrder
        | VarianceUpdateOrder
        | CostDiagnostics
        | StatisticsMode
        | ExplainedVarianceSolver
        | RuntimeTuning
        | CovarianceWriteback
        | AccessorMode
    ],
] = {
    "algorithm": Algorithm,
    "cost_diagnostics": CostDiagnostics,
    "runtime_tuning": RuntimeTuning,
    "dense_sufficient_statistics": StatisticsMode,
    "noise_sufficient_statistics": StatisticsMode,
    "cov_writeback_mode": CovarianceWriteback,
    "accessor_mode": AccessorMode,
    "bias_update_order": BiasUpdateOrder,
    "variance_update_order": VarianceUpdateOrder,
    "compat_mode": CompatMode,
    "explained_var_solver": ExplainedVarianceSolver,
}
_BOOL_FIELDS = (
    "bias",
    "uniquesv",
    "auto_pattern_masked",
    "earlystop",
    "record_cost",
    "record_prior_trace",
    "rotate2pca",
    "display",
    "return_diagnostics",
    "runtime_report",
)
_INT_FIELDS = {
    "maxiters": 0,
    "angle_every": 1,
    "niter_broadprior": 0,
    "patience": 1,
    "verbose": 0,
    "masked_batch_size": 0,
}
_FLOAT_FIELDS = {
    "minangle": float("inf"),
    "hp_va": 0.001,
    "hp_vb": 0.001,
    "hp_v": 0.001,
    "va_init": 1000.0,
    "explained_var_gram_ratio": 4.0,
}
_OPTIONAL_INTS = (
    "num_cpu",
    "num_cpu_score_update",
    "num_cpu_loadings_update",
    "num_cpu_noise_update",
    "num_cpu_rms",
    "log_progress_stride",
)
