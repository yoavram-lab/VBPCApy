"""Sparse/dense compatibility utilities to prevent unintended densification."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

__all__ = ["validate_mask_compatibility"]


def _record_preflight(
    preflight: list[dict[str, object]] | None,
    *,
    context: str | None,
    reason: str,
    payload: dict[str, object],
) -> None:
    if preflight is None:
        return

    entry: dict[str, object] = {
        "check": "sparsity_policy",
        "context": context,
        "reason": reason,
    }
    entry.update(payload)
    preflight.append(entry)


def _shape(obj: object) -> tuple[int, int] | None:
    shape_attr = getattr(obj, "shape", None)
    if shape_attr is not None:
        try:
            dims = tuple(int(v) for v in shape_attr)  # type: ignore[arg-type]
            if len(dims) == 2:
                return dims  # type: ignore[return-value]
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    try:
        shp = tuple(np.shape(np.asarray(cast("npt.ArrayLike", obj))))
        if len(shp) == 2:
            return int(shp[0]), int(shp[1])
    except (TypeError, ValueError, AttributeError):  # pragma: no cover - defensive
        return None
    return None


def validate_mask_compatibility(  # noqa: PLR0913
    data: object,
    mask: object,
    *,
    allow_sparse_mask_for_dense: bool = False,
    allow_dense_mask_for_sparse: bool = False,
    context: str | None = None,
    preflight: list[dict[str, object]] | None = None,
) -> None:
    """Raise when data/mask sparsity or shape is incompatible.

    Args:
        data: Data matrix (dense or sparse).
        mask: Mask matrix (dense or sparse).
        allow_sparse_mask_for_dense: Permit sparse mask with dense data.
        allow_dense_mask_for_sparse: Permit dense mask with sparse data.
        context: Optional label to include in error messages.
        preflight: Optional list to receive structured rejection entries.

    Raises:
        ValueError: When sparsity or shape is incompatible.
    """
    if mask is None or data is None:
        return

    data_is_sparse = sp.isspmatrix(data)
    mask_is_sparse = sp.isspmatrix(mask)
    data_shape = _shape(data)
    mask_shape = _shape(mask)

    label = f" in {context}" if context else ""

    if data_is_sparse and mask_is_sparse is False and not allow_dense_mask_for_sparse:
        reason = "mask must be sparse when data is sparse"
        _record_preflight(
            preflight,
            context=context,
            reason=reason,
            payload={
                "data_is_sparse": data_is_sparse,
                "mask_is_sparse": mask_is_sparse,
                "data_shape": data_shape,
                "mask_shape": mask_shape,
            },
        )
        msg = f"{reason}{label}"
        raise ValueError(msg)

    if (not data_is_sparse) and mask_is_sparse and not allow_sparse_mask_for_dense:
        reason = "mask must be dense when data is dense"
        _record_preflight(
            preflight,
            context=context,
            reason=reason,
            payload={
                "data_is_sparse": data_is_sparse,
                "mask_is_sparse": mask_is_sparse,
                "data_shape": data_shape,
                "mask_shape": mask_shape,
            },
        )
        msg = f"{reason}{label}"
        raise ValueError(msg)

    if data_shape is None or mask_shape is None or data_shape != mask_shape:
        reason = "mask must have the same shape as data"
        _record_preflight(
            preflight,
            context=context,
            reason=reason,
            payload={
                "data_is_sparse": data_is_sparse,
                "mask_is_sparse": mask_is_sparse,
                "data_shape": data_shape,
                "mask_shape": mask_shape,
            },
        )
        msg = f"{reason}{label}"
        raise ValueError(msg)
