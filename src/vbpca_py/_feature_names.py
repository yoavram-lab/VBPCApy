"""Typed feature-name delegation and native sklearn output configuration."""

from __future__ import annotations

from typing import Literal, Self, cast

import numpy as np
from sklearn.base import OneToOneFeatureMixin, TransformerMixin  # type: ignore[import-untyped]

from ._types import InputArray, ObjectArray  # noqa: TC001 - runtime annotation resolution


class FeatureNamesMixin(OneToOneFeatureMixin, TransformerMixin):  # type: ignore[misc]
    """Validate input names through sklearn and name fitted output columns."""

    def get_feature_names_out(self, input_features: object = None) -> ObjectArray:
        """Return owned output names in transformed column order.

        Args:
            input_features: Unique input strings matching fitted names, or
                replacement names for an unnamed fit. Omission uses fitted
                string names or positional ``x0``, ``x1``, ... names.

        Returns:
            An object array of strings. Ambiguous generated names receive a
            deterministic suffix based on output position.

        Raises:
            ValueError: Supplied names are not a one-dimensional unique string
                sequence, or do not match the fitted features.
        """
        if input_features is not None:
            supplied = np.asarray(input_features, dtype=object)
            if (
                supplied.ndim != 1
                or not all(isinstance(name, str) for name in supplied)
                or len(set(supplied)) != len(supplied)
            ):
                msg = (
                    "input_features must be a one-dimensional sequence "
                    "of unique strings"
                )
                raise ValueError(msg)
        names = cast("ObjectArray", super().get_feature_names_out(input_features))
        generated = self._output_feature_names(names)
        unique: list[str] = []
        used: set[str] = set()
        for index, name in enumerate(generated):
            candidate = str(name)
            suffix = index
            while candidate in used:
                candidate = f"{name}__{suffix}"
                suffix += 1
            unique.append(candidate)
            used.add(candidate)
        return np.asarray(unique, dtype=object)

    def _output_feature_names(  # noqa: PLR6301 - overridable fitted-output hook
        self, input_features: ObjectArray
    ) -> ObjectArray:
        return input_features

    def set_output(
        self, *, transform: Literal["default", "pandas", "polars"] | None = None
    ) -> Self:
        """Select native sklearn wrapping for transform and fit_transform.

        Args:
            transform: Default NumPy/SciPy output, pandas, or optional Polars.
                ``None`` retains the existing configuration. Sparse outputs
                cannot be wrapped as dense tables.

        Returns:
            This transformer with sklearn's output configuration.
        """
        super().set_output(transform=transform)
        return self


def check_named_columns(x: object) -> None:
    """Reject duplicate table labels even when they are not captured as strings.

    Raises:
        ValueError: A named input table exposes nonunique column labels.
    """
    columns = getattr(x, "columns", None)
    if columns is not None and not getattr(columns, "is_unique", True):
        msg = "X has duplicate feature names"
        raise ValueError(msg)


def canonical_table_nulls(original: object, checked: InputArray) -> InputArray:
    """Translate Polars object-column nulls to canonical NaNs on owned storage.

    Returns:
        The original validated array, or an owned copy when nulls need replacing.
        Ordinary NumPy/Pandas object-category semantics are unchanged.
    """
    if type(original).__module__.split(".")[0] != "polars" or checked.dtype != object:
        return checked
    null_count = getattr(original, "null_count", None)
    if null_count is None:
        return checked
    counts = null_count().row(0)
    if not any(counts):
        return checked
    result = checked.copy()
    for column, count in enumerate(counts):
        if not count:
            continue
        for row in range(0, result.shape[0], 65_536):
            values = result[row : row + 65_536, column]
            missing = np.fromiter(
                (value is None for value in values), dtype=bool, count=values.size
            )
            values[missing] = np.nan
    return result


def empty_table_array(x: object) -> object:
    """Normalize empty-column tables before sklearn's dtype reduction.

    Returns:
        An equivalent zero-width array for a named table, otherwise the input.
    """
    shape = getattr(x, "shape", None)
    if (
        type(x).__module__.split(".")[0] in {"pandas", "polars"}
        and shape is not None
        and len(shape) == 2
        and shape[1] == 0
    ):
        return np.asarray(x)
    return x
