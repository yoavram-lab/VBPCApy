"""Detect categorical encodings in an already-encoded matrix (#250).

When data are one-hot encoded before reaching VBPCApy, the variable structure
is lost: entrywise cross-validation then leaks through each block's
sum-to-one constraint. These checks recover likely one-hot blocks from the
matrix itself and flag encodings that deserve attention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

RARE_LEVEL_FRACTION = 0.05
RARE_LEVEL_COUNT = 5
MAX_ORDINAL_LEVELS = 10
_TOLERANCE = 1e-9


def _is_indicator(values: np.ndarray) -> bool:
    return values.size > 0 and bool(np.all(np.isin(values, (0.0, 1.0))))


def detect_one_hot_blocks(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Group contiguous indicator columns that form one-hot blocks.

    A block is a run of two or more 0/1 columns with identical missingness
    whose observed rows each sum to exactly one.

    Args:
        x: Samples-by-features matrix.
        observed: Boolean mask of observed entries.

    Returns:
        Group label for every column; columns outside a block get their own
        label. Labels are consecutive integers in column order.
    """
    n_features = x.shape[1]
    groups = np.empty(n_features, dtype=int)
    label = 0
    j = 0
    while j < n_features:
        end = _block_end(x, observed, j)
        groups[j:end] = label
        label += 1
        j = end
    return groups


def _block_end(x: np.ndarray, observed: np.ndarray, start: int) -> int:
    """Return one past the last column of the block starting at ``start``."""
    rows = observed[:, start]
    if not _is_indicator(x[rows, start]):
        return start + 1
    totals = x[rows, start].copy()
    for j in range(start + 1, x.shape[1]):
        if not np.array_equal(observed[:, j], rows) or not _is_indicator(x[rows, j]):
            break
        totals += x[rows, j]
        if np.any(totals > 1.0 + _TOLERANCE):
            break
        if np.allclose(totals, 1.0, rtol=0.0, atol=_TOLERANCE):
            return j + 1
    return start + 1


def encoding_warnings(
    x: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    column_names: Sequence[str] | None = None,
) -> list[str]:
    """Describe the categorical encodings found in ``x``.

    Returns:
        Messages for detected one-hot blocks, rare levels, binary variables
        coded as a single indicator, and integer codes that look ordinal.
    """
    names = (
        list(column_names)
        if column_names is not None
        else [str(j) for j in range(x.shape[1])]
    )
    messages: list[str] = []
    labels, counts = np.unique(groups, return_counts=True)
    blocks = labels[counts > 1]
    if blocks.size:
        width = int(counts[counts > 1].sum())
        messages.append(
            f"{blocks.size} one-hot block(s) detected ({width} indicator columns): "
            "each block sums to one (an exact linear dependence), so pass "
            "report.suggested_feature_groups as CVConfig(feature_groups=...) to "
            "hold out whole variables"
        )
    for j in range(x.shape[1]):
        values = x[observed[:, j], j]
        in_block = counts[np.searchsorted(labels, groups[j])] > 1
        if in_block:
            ones = int(values.sum())
            if ones < RARE_LEVEL_COUNT or ones < RARE_LEVEL_FRACTION * values.size:
                messages.append(
                    f"{names[j]}: rare level ({ones} of {values.size} observed)"
                )
        elif _is_indicator(values) and 0 < values.sum() < values.size:
            messages.append(
                f"{names[j]}: binary variable coded as one indicator; "
                "binary='both' gives it one indicator per level"
            )
        elif _looks_ordinal(values):
            messages.append(
                f"{names[j]}: {np.unique(values).size} integer codes; treat as "
                "ordinal (scaled numeric) if ordered, otherwise one-hot encode"
            )
    return messages


def _looks_ordinal(values: np.ndarray) -> bool:
    if values.size == 0 or not np.all(values == np.round(values)):
        return False
    levels = np.unique(values)
    return 2 < levels.size <= MAX_ORDINAL_LEVELS
