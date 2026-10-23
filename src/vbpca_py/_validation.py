"""Samples-first observation and capacity validation at the native boundary."""

from __future__ import annotations

from numbers import Integral
from typing import cast

import numpy as np
import scipy.sparse as sp

from ._full_update import _filter_csr_entries
from ._memory import exceeds_budget
from ._missing import make_xprobe_mask
from ._preprocessing_state import _check_observed_infinity, _check_sparse_structure
from ._sparsity import validate_mask_compatibility
from ._types import BoolArray, InputMask, InputMatrix, Matrix  # noqa: TC001 - runtime annotation resolution


def check_raw_sparse_observations(x: object, mask: object | None) -> None:
    """Check raw sparse buffers and observed infinity before sklearn conversion.

    Retain the native boundary's existing numeric-mask coercion. CSR/CSC/COO
    buffers are checked before conversions can coalesce duplicate coordinates;
    other scipy formats expose their entries through COO without aggregation.

    Raises:
        ValueError: Sparse buffers, mask dimensions or observed values are invalid.
    """
    if sp.issparse(mask):
        _checked_sparse_input(mask, numeric_mask=True)
    if not sp.issparse(x):
        return
    raw = _checked_sparse_input(x)
    if not np.any(np.isinf(raw.data)):
        return
    observed: BoolArray | sp.csr_matrix | None = None
    if mask is not None:
        observed = (
            sp.csr_matrix(cast("sp.csr_matrix", mask), dtype=bool, copy=True)
            if sp.issparse(mask)
            else np.asarray(mask, dtype=bool)
        )
        if observed.shape != raw.shape:
            msg = "mask must have the same shape as data"
            raise ValueError(msg)
    _check_observed_infinity(raw, observed)


def _checked_sparse_input(
    x: object, *, numeric_mask: bool = False
) -> sp.csr_matrix | sp.csc_matrix | sp.coo_matrix:
    """Validate coordinate buffers while preserving other sklearn sparse formats.

    Returns:
        Original CSR/CSC/COO storage, or unaggregated COO for another format.
    """
    raw = cast("sp.csr_matrix | sp.csc_matrix | sp.coo_matrix", x)
    if raw.format not in {"csr", "csc", "coo"}:
        raw = raw.tocoo(copy=False)
    _check_sparse_structure(raw, complex_indicators=numeric_mask, check_infinity=False)
    return raw


def validate_rank(rank: object, shape: tuple[int, int]) -> int:
    """Require a nonnegative integral capacity within declared dimensions.

    Returns:
        The validated Python integer.

    Raises:
        ValueError: Capacity is not integral, negative or exceeds dimensions.
    """
    if isinstance(rank, bool | np.bool_) or not isinstance(rank, Integral):
        msg = "n_components must be a nonnegative integer"
        raise ValueError(msg)  # noqa: TRY004 - capacity validation follows sklearn ValueError contract
    value = int(rank)
    if value < 0 or value > min(shape):
        msg = (
            "n_components must be between 0 and "
            f"min(n_samples, n_features)={min(shape)}"
        )
        raise ValueError(msg)
    return value


def observed_matrix(
    x: InputMatrix, mask: InputMask | None, max_dense_bytes: int | None
) -> Matrix:
    """Normalize observations without losing stored sparse zeros.

    Dense missing entries become NaN. A sparse explicit mask defines the stored
    observation set, including requested zero values absent from the input.
    NaN is missing; infinity in an observed entry is rejected.

    Returns:
        Float64 observations in dense NaN or sparse stored-entry form.

    Raises:
        ValueError: Mask alignment/budget or observed finite-value checks fail.
    """
    # Normalize sparse arrays and matrices to one classification boundary.
    if sp.issparse(x):
        x = sp.csr_matrix(x, dtype=np.float64, copy=False)
    if sp.issparse(mask):
        mask = sp.csr_matrix(mask, dtype=bool, copy=False)
    validate_mask_compatibility(x, mask, allow_dense_mask_for_sparse=True)
    if not sp.issparse(x):
        dense = np.asarray(x, dtype=np.float64)
        if mask is not None:
            dense = np.where(np.asarray(mask, dtype=bool), dense, np.nan)
        if np.any(np.isinf(dense)):
            msg = "X contains infinity in observed entries"
            raise ValueError(msg)
        return dense
    return _sparse_observations(sp.csr_matrix(x), mask, max_dense_bytes)


def _sparse_observations(
    x: sp.csr_matrix, mask: InputMask | None, max_dense_bytes: int | None
) -> sp.csr_matrix:
    """Normalize sparse structure after common shape/sparsity validation.

    Returns:
        Canonical float64 CSR observations, borrowing already valid buffers.

    Raises:
        ValueError: A dense mask exceeds its budget or observed values are infinite.
    """
    data = sp.csr_matrix(x, dtype=np.float64, copy=False)
    if not data.has_canonical_format:
        data = data.copy()
        data.sum_duplicates()
    if mask is not None:
        if (
            not sp.issparse(mask)
            and exceeds_budget(data.shape, np.bool_, max_dense_bytes)[0]
        ):
            msg = "Dense mask would exceed max_dense_bytes"
            raise ValueError(msg)
        pattern = sp.csr_matrix(mask, dtype=bool, copy=True)
        pattern.sum_duplicates()
        pattern.eliminate_zeros()
        pattern.sort_indices()
        if not pattern.nnz:
            return sp.csr_matrix(data.shape, dtype=np.float64)
        rows, columns = pattern.nonzero()
        values = np.asarray(data[rows, columns]).ravel()
        data = sp.csr_matrix(
            (values, pattern.indices.copy(), pattern.indptr.copy()), shape=data.shape
        )
    if np.any(np.isinf(data.data)):
        msg = "X contains infinity in observed entries"
        raise ValueError(msg)
    if np.any(np.isnan(data.data)):
        data = _filter_csr_entries(data, ~np.isnan(data.data))
    return data


def require_training_observations(data: Matrix, probe: Matrix | None) -> None:
    """Validate training coverage and probe rows after probe exclusion.

    Raises:
        ValueError: Training features or probe-bearing rows lack observations.
    """
    if sp.issparse(data):
        coordinates = data.tocoo(copy=False)
        keep = np.ones(data.nnz, dtype=bool)
        if probe is not None and data.nnz:
            pattern = sp.csr_matrix(probe, copy=True)
            pattern.data = np.ones(pattern.nnz, dtype=bool)
            keep = ~np.asarray(
                pattern[coordinates.row, coordinates.col]
            ).ravel().astype(bool)
        counts = np.bincount(coordinates.col[keep], minlength=data.shape[1])
        row_counts = np.bincount(coordinates.row[keep], minlength=data.shape[0])
        probe_rows = (
            np.diff(sp.csr_matrix(probe).indptr) > 0 if probe is not None else None
        )
    else:
        observed = ~np.isnan(data)
        if probe is not None:
            observed &= np.isnan(np.asarray(probe))
        counts = np.count_nonzero(observed, axis=0)
        row_counts = np.count_nonzero(observed, axis=1)
        probe_rows = (
            np.any(~np.isnan(np.asarray(probe)), axis=1) if probe is not None else None
        )
    if not np.any(counts):
        msg = "X has no training observations (all NaN/missing or held out)"
        raise ValueError(msg)
    if np.any(counts == 0):
        msg = (
            "X has entirely unobserved training features; "
            "select observed features explicitly"
        )
        raise ValueError(msg)
    if probe_rows is not None and np.any(probe_rows & (row_counts == 0)):
        msg = (
            "Each row with probe entries must retain a training observation; "
            "the solver cannot evaluate probe-only rows"
        )
        raise ValueError(msg)


def make_native_probe(
    data: Matrix, fraction: float, rng: np.random.Generator
) -> tuple[Matrix, Matrix]:
    """Partition observations while retaining every unselected stored zero.

    Returns:
        Owned training/probe matrices in the same samples-first layout.

    Raises:
        ValueError: There are no observations to split.
    """
    if not sp.issparse(data):
        return make_xprobe_mask(data, fraction=fraction, rng=rng)
    matrix = sp.csr_matrix(data)
    if not matrix.nnz:
        msg = "cannot create a probe set from data with no observed entries"
        raise ValueError(msg)
    count = min(matrix.nnz, max(1, round(matrix.nnz * fraction)))
    chosen = rng.choice(matrix.nnz, size=count, replace=False)
    held_out = np.zeros(matrix.nnz, dtype=bool)
    held_out[chosen] = True
    return _filter_csr_entries(matrix, ~held_out), _filter_csr_entries(matrix, held_out)
