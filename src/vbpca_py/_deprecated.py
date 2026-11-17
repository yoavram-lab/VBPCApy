"""Deprecated attribute access on the package root."""

from __future__ import annotations

import warnings

# Typing aliases describe result arrays and trace keys. They belong to
# vbpca_py.results; access through the package root is deprecated.
RESULTS_TYPING_ALIASES = frozenset({
    "Float64Array",
    "IndexArray",
    "NumericTraceKey",
    "PhaseTimingKey",
})


def module_getattr(name: str) -> object:
    """Return deprecated root aliases of ``vbpca_py.results`` typing names.

    Returns:
        The alias from ``vbpca_py.results``.

    Raises:
        AttributeError: ``name`` is not a module attribute.
    """
    if name in RESULTS_TYPING_ALIASES:
        from vbpca_py import results  # noqa: PLC0415 - avoid an import cycle

        warnings.warn(
            f"vbpca_py.{name} is deprecated and will be removed in VBPCApy 0.7; "
            f"import it from vbpca_py.results instead.",
            FutureWarning,
            stacklevel=2,
        )
        return getattr(results, name)
    msg = f"module 'vbpca_py' has no attribute {name!r}"
    raise AttributeError(msg)
