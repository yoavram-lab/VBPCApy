"""Optional scikit-learn base classes.

scikit-learn is an *optional* dependency.  When it is installed, :class:`VBPCA`
and the preprocessing transformers inherit the real ``BaseEstimator`` and
``TransformerMixin`` so they integrate with sklearn pipelines and ``clone``.
When scikit-learn is absent, these fall back to no-op base classes so the core
library still imports and runs with only numpy and scipy installed.

At type-check time we expose concrete stand-ins so that subclasses remain
strictly typed even though scikit-learn ships incomplete type information
(which would otherwise trip ``--strict`` subclassing checks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:

    class BaseEstimator:
        """Typed stand-in for ``sklearn.base.BaseEstimator``."""

    class TransformerMixin:
        """Typed stand-in for ``sklearn.base.TransformerMixin``."""

else:
    try:
        from sklearn.base import BaseEstimator, TransformerMixin
    except ModuleNotFoundError:  # pragma: no cover - only hit without sklearn

        class BaseEstimator:
            """No-op fallback used when scikit-learn is not installed."""

        class TransformerMixin:
            """No-op fallback used when scikit-learn is not installed."""


__all__ = ["BaseEstimator", "TransformerMixin"]
