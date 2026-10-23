"""String-compatible categorical options for solver configuration.

Ordinary strings remain accepted by the numerical solver. These enums provide
named values without changing the solver's numerical defaults. ``parse`` is
an explicit, strict conversion for configuration boundaries; legacy keyword
normalizers retain their documented fallback and whitespace behavior.

Initialization is a separate contract: ``init`` can also be a MAT filename or
a warm-start mapping, so it is not represented by a random/custom enum.

Examples:
    >>> from vbpca_py.options import Algorithm, CompatMode
    >>> Algorithm.parse(" VB ") is Algorithm.VB
    True
    >>> str(Algorithm.VB), Algorithm.VB == "vb"
    ('vb', True)
    >>> import json
    >>> json.dumps({"algorithm": Algorithm.VB})
    '{"algorithm": "vb"}'
    >>> CompatMode.parse("modern")
    <CompatMode.MODERN: 'modern'>
    >>> Algorithm.parse("approximate")
    Traceback (most recent call last):
    ...
    ValueError: Algorithm must be one of {'ppca', 'map', 'vb'} (got 'approximate').
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

__all__ = [
    "AccessorMode",
    "Algorithm",
    "BiasUpdateOrder",
    "CompatMode",
    "CostDiagnostics",
    "CovarianceWriteback",
    "Criterion",
    "ExplainedVarianceSolver",
    "RuntimeTuning",
    "StatisticsMode",
    "VarianceUpdateOrder",
]


class _StringOption(StrEnum):
    """Explicit conversion shared by categorical option types."""

    @classmethod
    def parse(cls, value: str) -> Self:
        """Normalize case and surrounding whitespace, then validate.

        Args:
            value: A string or member of the requested option type.

        Returns:
            The corresponding enum member.

        Raises:
            TypeError: If the input is not a string.
            ValueError: If the normalized value is unknown.
        """
        if not isinstance(value, str):
            msg = f"{cls.__name__} requires a string (got {type(value).__name__})."
            raise TypeError(msg)
        try:
            return cls(value.strip().lower())
        except ValueError:
            choices = ", ".join(repr(member.value) for member in cls)
            msg = f"{cls.__name__} must be one of {{{choices}}} (got {value!r})."
            raise ValueError(msg) from None


class Algorithm(_StringOption):
    """PPCA, maximum a posteriori PCA or full-covariance VB PCA."""

    PPCA = "ppca"
    MAP = "map"
    VB = "vb"


class CompatMode(_StringOption):
    """Historical versus modern numerical update and observation policies."""

    STRICT_LEGACY = "strict_legacy"
    MODERN = "modern"


class BiasUpdateOrder(_StringOption):
    """Requested mean-update order; auto resolves from compatibility mode."""

    AUTO = "auto"
    LEGACY = "legacy"
    POST_FACTOR = "post_factor"


class VarianceUpdateOrder(_StringOption):
    """Requested prior-variance update order; auto resolves from compatibility."""

    AUTO = "auto"
    LEGACY = "legacy"
    POST_ROTATION = "post_rotation"


class CostDiagnostics(_StringOption):
    """Full objective trace or endpoint-only objective diagnostics."""

    TRACE = "trace"
    ENDPOINT = "endpoint"


class StatisticsMode(_StringOption):
    """Requested sufficient-statistic accumulation, subject to eligibility."""

    AUTO = "auto"
    OBSERVED = "observed"
    COMPLEMENT = "complement"


class ExplainedVarianceSolver(_StringOption):
    """Factor-based explained-variance decomposition strategy."""

    AUTO = "auto"
    SVD = "svd"
    GRAM = "gram"


class RuntimeTuning(_StringOption):
    """Automatic execution-policy tuning, without changing statistical settings."""

    OFF = "off"
    SAFE = "safe"
    AGGRESSIVE = "aggressive"


class CovarianceWriteback(_StringOption):
    """Covariance return/writeback path; bulk and kernel currently share code."""

    AUTO = "auto"
    PYTHON = "python"
    BULK = "bulk"
    KERNEL = "kernel"


class AccessorMode(_StringOption):
    """Observation access strategy for sparse and masked updates."""

    AUTO = "auto"
    LEGACY = "legacy"
    BUFFERED = "buffered"


class Criterion(_StringOption):
    """Accepted stopping-rule names, including unavailable legacy slowing-down.

    Naming a criterion does not enable or qualify it. The live fitter keeps
    ``slowing_down`` disabled because it has no event counter.
    """

    ANGLE = "angle"
    EARLYSTOP = "earlystop"
    RMS_PLATEAU = "rms_plateau"
    COST = "cost"
    COMPOSITE = "composite"
    SLOWING_DOWN = "slowing_down"
