"""Exact full-covariance rearrangement with conservative numerical screening.

Finite-sum identities require no positivity or symmetry assumptions. The
screen below is a floating-point heuristic, not a certified error bound.
An ineligible or poorly scaled subtraction retains observed-cell accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt

from .noise_update_kernels import noise_sxv_sum
from .options import StatisticsMode

if TYPE_CHECKING:
    from ._full_update import NoiseState
    from .results import Float64Array


@dataclass(frozen=True, slots=True)
class NoiseComplement:
    """Missing coordinates derived once from the exact observed coordinate set."""

    shape: tuple[int, int]
    observed_count: int
    rows: npt.NDArray[np.int32]
    columns: npt.NDArray[np.int32]


def normalize_noise_statistics(
    value: object,
) -> Literal["auto", "observed", "complement"]:
    """Normalize the explicit execution-mode option.

    Returns:
        The supported lowercase mode.

    Raises:
        ValueError: If the value is not a supported mode.
    """
    mode = str(value).strip().lower()
    if mode not in {item.value for item in StatisticsMode}:
        msg = (
            "noise_sufficient_statistics must be one of "
            "{'auto', 'observed', 'complement'} "
            f"(got {value!r})."
        )
        raise ValueError(msg)
    return cast("Literal['auto', 'observed', 'complement']", mode)


def prepare_noise_complement(
    shape: tuple[int, int], ix: np.ndarray, jx: np.ndarray
) -> NoiseComplement | None:
    """Prepare a small complement without inferring membership from data values.

    Return no plan for low density, duplicates or unsupported coordinate
    metadata. A temporary boolean grid validates the partition; only missing
    int32 coordinates remain in the plan. Fully observed data retain their
    existing complete-data path.

    Returns:
        A read-only coordinate plan, or ``None`` for observed accumulation.
    """
    total = int(shape[0]) * int(shape[1])
    if not _supported_coordinates(shape, ix, jx):
        return None
    missing_count = total - ix.size
    if missing_count <= 0 or missing_count > total // 10:
        return None
    if (
        np.any(ix < 0)
        or np.any(ix >= shape[0])
        or np.any(jx < 0)
        or np.any(jx >= shape[1])
    ):
        return None
    missing = np.ones(shape, dtype=bool)
    missing[ix, jx] = False
    if np.count_nonzero(missing) != missing_count:
        return None
    rows_raw, columns_raw = np.nonzero(missing)
    rows = rows_raw.astype(np.int32)
    columns = columns_raw.astype(np.int32)
    rows.setflags(write=False)
    columns.setflags(write=False)
    return NoiseComplement(shape, int(ix.size), rows, columns)


def _supported_coordinates(
    shape: tuple[int, int], ix: np.ndarray, jx: np.ndarray
) -> bool:
    dimensions = min(shape) > 0 and max(shape) <= np.iinfo(np.int32).max
    same_vectors = ix.ndim == jx.ndim == 1 and ix.size == jx.size
    integer_indices = np.issubdtype(ix.dtype, np.integer) and np.issubdtype(
        jx.dtype, np.integer
    )
    return dimensions and same_vectors and integer_indices


def complement_covariance_sum(
    state: NoiseState,
    sv: Float64Array,
    av: Float64Array | None,
) -> float | None:
    """Subtract missing-cell covariance terms from complete-data contractions.

    The loading and score quadratic forms and their full covariance cross
    term are retained. Bias uncertainty, RMS, hyperpriors and variance floors
    stay in the existing noise-update caller.

    Returns:
        The screened covariance sum, or ``None`` to request the original path.
    """
    plan = state.complement
    shape = (state.loadings.shape[0], state.scores.shape[1])
    if plan is None or plan.shape != shape or plan.observed_count != state.n_data:
        return None
    # Overflow in a Gram contraction need not imply overflow in the original
    # cell-wise multiplication order. Preserve that path when screening fails.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        sv_sum = np.sum(sv, axis=0)
        terms = [float(np.sum((state.loadings.T @ state.loadings) * sv_sum))]
        if av is not None:
            av_sum = np.sum(av, axis=0)
            terms.extend((
                float(np.sum((state.scores @ state.scores.T) * av_sum)),
                float(np.sum(av_sum * sv_sum)),
            ))
    if not all(np.isfinite(term) for term in terms):
        return None
    try:
        complete = fsum(terms)
    except OverflowError:
        return None
    missing = float(
        noise_sxv_sum(
            ix=plan.rows,
            jx=plan.columns,
            loadings=np.asarray(state.loadings, dtype=np.float64),
            scores=np.asarray(state.scores, dtype=np.float64),
            sv_by_col=sv,
            loading_covariances=av,
            num_cpu=int(state.sparse_num_cpu),
        )
    )
    candidate = complete - missing
    try:
        scale = fsum([*(abs(term) for term in terms), abs(missing)])
    except OverflowError:
        return None
    accumulation_length = max(*shape, state.loadings.shape[1] ** 2, plan.rows.size)
    screen = 64 * np.finfo(np.float64).eps * accumulation_length * scale
    if (
        not np.isfinite(candidate)
        or candidate <= 0
        or not np.isfinite(screen)
        or screen > 1e-8 * candidate
    ):
        return None
    return candidate
