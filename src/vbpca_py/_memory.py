"""Lightweight memory estimation helpers for densification guards."""

from __future__ import annotations

from math import prod
from typing import SupportsIndex, SupportsInt, cast

import numpy as np


def estimate_dense_bytes(shape: tuple[int, int], dtype: np.dtype | type) -> int:
    """Estimate bytes for a dense array of given shape and dtype.

    Returns:
        Estimated size in bytes as an integer.
    """
    dt = np.dtype(dtype)
    total = prod(int(length) for length in shape)
    return int(total * dt.itemsize)


def check_dense_output_budget(
    shape: tuple[int, int],
    max_bytes: int | None,
    *,
    arrays: int = 1,
    operation: str = "Dense output",
) -> None:
    """Check combined float64 output storage before materializing arrays.

    This counts persistent full-shape outputs in one operation. Factors,
    covariance stores, temporary arrays and peak process memory are excluded.

    Raises:
        ValueError: If the combined output size exceeds the explicit budget.

    Examples:
        >>> check_dense_output_budget((2, 3), 144, arrays=3)
        >>> check_dense_output_budget((2, 3), None, arrays=3)
    """
    required = arrays * estimate_dense_bytes(shape, np.float64)
    if max_bytes is not None and required > max_bytes:
        msg = (
            f"{operation} would exceed max_dense_bytes: "
            f"{required} bytes for {arrays} float64 array(s) of shape {shape} "
            f"> {max_bytes} bytes. Increase max_dense_bytes or set it to None "
            "to disable this output limit."
        )
        raise ValueError(msg)


def format_bytes(num_bytes: int) -> str:
    """Human-friendly byte formatter.

    Returns:
        Size formatted using 1024-based units.
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def exceeds_budget(
    shape: tuple[int, int], dtype: np.dtype | type, max_bytes: int | None
) -> tuple[bool, int]:
    """Return whether dense size would exceed ``max_bytes`` along with estimate.

    Returns:
        Tuple ``(over_budget, estimated_bytes)`` where ``over_budget`` is True
        when ``max_bytes`` is set and the estimate exceeds it.
    """
    est = estimate_dense_bytes(shape, dtype)
    if max_bytes is None:
        return False, est
    return est > max_bytes, est


def resolve_max_dense_bytes(
    val: object | None, default: int | None = None
) -> int | None:
    """Coerce ``max_dense_bytes`` style options to ``int | None``.

    Returns:
        ``None`` when unset or explicitly disabled, otherwise an integer budget.

    Raises:
        ValueError: If the value cannot be coerced to an integer.
    """
    if val is None:
        return default
    try:
        coerced = cast("SupportsInt | SupportsIndex | str | bytes | bytearray", val)
        return int(coerced)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        msg = "max_dense_bytes must be int-like or None"
        raise ValueError(msg) from exc
