"""Optional scikit-learn base classes.

scikit-learn is an *optional* dependency.  When it is installed, :class:`VBPCA`
and the preprocessing transformers inherit the real ``BaseEstimator`` and
``TransformerMixin`` so they integrate with sklearn pipelines and ``clone``.
When scikit-learn is absent, a small parameter-introspection fallback keeps
the core library usable with only numpy and scipy installed.

At type-check time we expose concrete stand-ins so that subclasses remain
strictly typed even though scikit-learn ships incomplete type information
(which would otherwise trip ``--strict`` subclassing checks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, cast


class _ParameterEstimator(Protocol):
    """Localized structural contract for nested optional estimator parameters."""

    def get_params(self, deep: bool = True) -> dict[str, object]:  # noqa: FBT001, FBT002
        """Return nested parameters."""
        ...

    def set_params(self, **params: object) -> object:
        """Update nested parameters."""
        ...


if TYPE_CHECKING:

    class BaseEstimator:
        """Typed stand-in for ``sklearn.base.BaseEstimator``."""

        def get_params(self, deep: bool = True) -> dict[str, object]:  # noqa: FBT001, FBT002
            """Return explicit constructor parameters."""
            ...

        def set_params(self, **params: object) -> Self:
            """Set explicit constructor parameters."""
            ...

    class TransformerMixin:
        """Typed stand-in for ``sklearn.base.TransformerMixin``."""

else:
    try:
        from sklearn.base import BaseEstimator, TransformerMixin
    except ModuleNotFoundError:  # pragma: no cover - only hit without sklearn

        class BaseEstimator:
            """Parameter introspection fallback when scikit-learn is absent."""

            def get_params(self, deep: bool = True) -> dict[str, object]:  # noqa: FBT001, FBT002
                """Return explicit parameters with optional nested expansion.

                Returns:
                    Constructor parameter names and unchanged stored values.
                """
                import inspect  # noqa: PLC0415

                names = inspect.signature(type(self).__init__).parameters
                result: dict[str, object] = {}
                for name in names:
                    if name == "self":
                        continue
                    value: object = getattr(self, name)
                    result[name] = value
                    if deep and callable(getattr(value, "get_params", None)):
                        nested = cast("_ParameterEstimator", value).get_params()
                        result.update({
                            f"{name}__{key}": item for key, item in nested.items()
                        })
                return result

            def set_params(self, **params: object) -> Self:
                """Set known parameters and reject misspelled names.

                Returns:
                    This estimator.

                Raises:
                    ValueError: A parameter is not exposed by the constructor.
                """
                valid = self.get_params(deep=True)
                for name, value in params.items():
                    head, separator, tail = name.partition("__")
                    if head not in valid:
                        msg = f"Invalid parameter {head!r} for {type(self).__name__}"
                        raise ValueError(msg)
                    if separator:
                        cast("_ParameterEstimator", valid[head]).set_params(**{
                            tail: value
                        })
                    else:
                        setattr(self, head, value)
                return self

        class TransformerMixin:
            """No-op fallback used when scikit-learn is not installed."""


__all__ = ["BaseEstimator", "TransformerMixin"]
