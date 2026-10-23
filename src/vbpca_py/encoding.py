"""Fitted metadata for decoding dense categorical and numeric representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._types import Float64Array  # noqa: TC001 - runtime annotation resolution


@dataclass(frozen=True, slots=True)
class EncodedVariable:
    """Describe one original variable in an encoded feature matrix.

    Attributes:
        kind: Original variable kind; only categorical variables receive
            nominal categorical scores.
        columns: Encoded feature indices, in category order with the reference
            omitted when applicable.
        categories: Fitted nominal category or ordinal level order.
        reference: Index of an omitted category, or ``None`` for full coding.
        offset: Unweighted means (or numeric minima) subtracted before scaling.
        weight: Positive multiplier applied to the block or numeric column.
    """

    kind: Literal["categorical", "continuous", "ordinal"]
    columns: tuple[int, ...]
    categories: tuple[object, ...] = ()
    reference: int | None = None
    offset: tuple[float, ...] = ()
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Own sequence membership while borrowing category label objects."""
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "categories", tuple(self.categories))
        object.__setattr__(self, "offset", tuple(self.offset))

    def decode(self, block: Float64Array) -> Float64Array:
        """Undo a categorical block's affine encoding and reference drop.

        Args:
            block: Encoded features-by-samples values for this variable.

        Returns:
            Full category-by-samples values before clipping or normalization.

        Raises:
            ValueError: If the variable is not categorical or dimensions differ.
        """
        if (
            self.kind != "categorical"
            or block.ndim != 2
            or block.shape[0] != len(self.columns)
        ):
            msg = "decode needs a categorical block matching its fitted columns"
            raise ValueError(msg)
        decoded = np.asarray(block, dtype=float) / self.weight
        decoded += np.asarray(self.offset)[:, np.newaxis]
        if self.reference is not None:
            reference = 1.0 - decoded.sum(axis=0)
            decoded = np.insert(decoded, self.reference, reference, axis=0)
        return decoded

    def decode_numeric(self, block: Float64Array) -> Float64Array:
        """Restore continuous values or unrounded ordinal level-index scores.

        Args:
            block: Encoded features-by-samples values for this numeric variable.

        Returns:
            Values in continuous input units or ordinal level-index units.

        Raises:
            ValueError: If numeric affine metadata or dimensions are unavailable.
        """
        dimensions_match = block.ndim == 2 and block.shape[0] == len(self.columns)
        if self.kind not in {"continuous", "ordinal"} or not dimensions_match:
            msg = "numeric decoding needs matching columns and fitted affine metadata"
            raise ValueError(msg)
        self._validate_numeric_affine()
        decoded: Float64Array = (
            block / self.weight + np.asarray(self.offset)[:, np.newaxis]
        )
        return decoded

    def _validate_numeric_affine(self) -> None:
        if len(self.columns) != 1 or len(self.offset) != 1:
            msg = (
                "numeric encoding needs fitted affine metadata with "
                "exactly one column and one offset"
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(self.offset)):
            msg = (
                "numeric encoding needs a finite fitted offset "
                "(unavailable for wholly missing training support)"
            )
            raise ValueError(msg)
        if not np.isfinite(self.weight) or self.weight <= 0:
            msg = "numeric encoding needs a positive finite weight"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class EncodingSchema:
    """Snapshot of dense encoding metadata, in original variable order.

    Obtain this from a fitted encoder's ``encoding_schema_``. Category order,
    centering and weighting belong to that fit and must match the encoded data
    supplied to component CV. This schema does not fit preprocessing per fold.

    Attributes:
        variables: One description per original variable, including numeric
            variables that are excluded from nominal categorical scoring.
    """

    variables: tuple[EncodedVariable, ...]

    def __post_init__(self) -> None:
        """Snapshot the ordered immutable variable descriptors."""
        object.__setattr__(self, "variables", tuple(self.variables))

    @property
    def feature_groups(self) -> tuple[int, ...]:
        """Original variable index of each encoded feature."""
        groups = [0] * sum(len(v.columns) for v in self.variables)
        for index, variable in enumerate(self.variables):
            for column in variable.columns:
                groups[column] = index
        return tuple(groups)

    def validate(self, n_features: int, *, strict_numeric: bool = False) -> None:
        """Check that metadata describes every encoded feature exactly once.

        Args:
            n_features: Number of rows of the model's features-by-samples input.
            strict_numeric: Also require continuous and ordinal variables to
                have exactly one column, one finite fitted offset and a positive
                finite weight. The default checks their feature coverage only,
                retaining compatibility with categorical-only scoring schemas.
                Wholly missing numeric training columns have no fitted offset
                and cannot pass strict validation.

        Raises:
            ValueError: If columns, kind, category dimensions or affine metadata
                are inconsistent with the supplied matrix.
        """
        columns = [c for v in self.variables for c in v.columns]
        if sorted(columns) != list(range(n_features)):
            msg = "encoding schema must cover every feature exactly once"
            raise ValueError(msg)
        for variable in self.variables:
            if variable.kind not in {"categorical", "continuous", "ordinal"}:
                msg = f"unsupported encoding kind {variable.kind!r}"
                raise ValueError(msg)
            if variable.kind != "categorical":
                if strict_numeric:
                    variable._validate_numeric_affine()  # noqa: SLF001 - shared schema contract
                continue
            count = len(variable.categories)
            dropped = variable.reference is not None
            if (
                variable.reference is not None and not 0 <= variable.reference < count
            ) or (len(variable.columns) != count - int(dropped)):
                msg = (
                    "categorical columns must match the fitted category/reference order"
                )
                raise ValueError(msg)
            if (
                len(variable.offset) != len(variable.columns)
                or not np.all(np.isfinite(variable.offset))
                or not np.isfinite(variable.weight)
                or variable.weight <= 0
            ):
                msg = "categorical encoding needs finite offsets and a positive weight"
                raise ValueError(msg)
