"""Typed delegation to the real, required scikit-learn parameter machinery.

Scikit-learn does not ship complete annotations. These forwarding methods only
supply return types; validation, nested updates and introspection remain its
native implementations. There are no optional imports or stand-in bases.
"""

from __future__ import annotations

from typing import Self, cast

from sklearn.base import BaseEstimator  # type: ignore[import-untyped]


class TypedEstimator(BaseEstimator):  # type: ignore[misc]
    """Native estimator with a localized third-party typing boundary."""

    def get_params(self, deep: bool = True) -> dict[str, object]:  # noqa: FBT001, FBT002
        """Return scikit-learn's unmodified parameter mapping."""
        return cast("dict[str, object]", super().get_params(deep=deep))

    def set_params(self, **params: object) -> Self:
        """Apply scikit-learn's parameter validation and nested updates.

        Returns:
            This estimator after native parameter validation.
        """
        super().set_params(**params)
        return self
