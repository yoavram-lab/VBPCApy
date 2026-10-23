"""Consume original-variable masks before learned Pipeline transformations."""

# ruff: noqa: N803 - native sklearn X/y metadata signatures

from __future__ import annotations

from collections.abc import Iterator  # noqa: TC003 - runtime annotation resolution
from typing import ClassVar, Self, cast

import numpy as np
import scipy.sparse as sp
from sklearn import get_config  # type: ignore[import-untyped]
from sklearn.utils.validation import validate_data  # type: ignore[import-untyped]

from ._feature_names import (
    canonical_table_nulls,
    check_named_columns,
    named_table_input,
)
from ._preprocessing_state import (
    PreprocessorState,
    _check_sparse_structure,
    _finite_observations,
)
from ._types import BoolArray, InputArray, InputMask, InputMatrix  # noqa: TC001 - runtime annotation resolution
from ._validation import observed_matrix

_MASK_BLOCK_ENTRIES = 65_536


class ObservationMasker(PreprocessorState):
    """Normalize missingness before scaling or expanding original variables.

    Dense false mask entries become NaNs; existing NaNs stay missing. Dense
    strings become object arrays when a mask is supplied. Sparse output is CSR:
    stored entries, including zeros, are observed unless excluded by a mask or
    NaN. A sparse-data mask can also request observed zeros absent from storage.
    Inputs and masks are never modified; no training mask/data is retained.

    Direct calls use ``mask=...`` with Boolean samples-by-variables metadata,
    aligned by position. With sklearn metadata routing enabled, a Pipeline
    accepts the alias ``observation_mask=...`` for fit, transform and score.
    This alias reaches only the masker, leaving downstream transformers to
    infer missingness from their own feature layout. Leave ``transform_input``
    unset: entry masks must not undergo numerical feature transformations.

    Without an explicit mask, dense output may borrow the validated input.
    Masked dense output owns its array; integer/Boolean input becomes float64
    and floating input retains its dtype. Sparse normalization uses storage
    proportional to input/mask entries and never densifies a sparse mask.
    """

    _numeric_input = False
    _accept_sparse = True
    __metadata_request__fit: ClassVar[dict[str, str]] = {"mask": "observation_mask"}
    __metadata_request__transform: ClassVar[dict[str, str]] = {
        "mask": "observation_mask"
    }

    def __init__(self) -> None:
        """Create a masker without data or fitted state."""
        super().__init__()

    def fit(self, X: object, y: object = None, *, mask: object | None = None) -> Self:  # noqa: ARG002
        """Validate original features and observations, retaining feature metadata.

        Returns:
            This fitted masker.
        """
        with self._fitting():
            self._validated(X, mask, reset=True)
        return self

    def transform(self, X: object, *, mask: object | None = None) -> InputMatrix:
        """Consume a fresh mask for this table without reusing training metadata.

        Returns:
            Dense NaN observations or a sparse CSR observation matrix.
        """
        self._check_fitted()
        data, observed = self._validated(X, mask, reset=False)
        return self._output(data, observed)

    def fit_transform(
        self,
        X: object,
        y: object = None,  # noqa: ARG002 - unsupervised transformer protocol
        *,
        mask: object | None = None,
    ) -> InputMatrix:
        """Fit and apply the same mask once, including routed transform metadata.

        Returns:
            Dense NaN observations or a sparse CSR observation matrix.
        """
        with self._fitting():
            data, observed = self._validated(X, mask, reset=True)
            return self._output(data, observed)

    def _output(self, data: InputMatrix, observed: BoolArray | None) -> InputMatrix:
        mode = getattr(self, "_sklearn_output_config", {}).get(
            "transform", get_config()["transform_output"]
        )
        if (
            mode == "polars"
            and not sp.issparse(data)
            and (
                data.dtype.kind == "O"
                or (observed is not None and data.dtype.kind not in "biuf")
            )
        ):
            msg = (
                "Polars output does not support raw object observations safely; "
                "set ObservationMasker output to 'pandas' or 'default' and "
                "wrap downstream numerical outputs in Polars"
            )
            raise ValueError(msg)
        return _canonical_output(data, observed)

    def _validated(
        self, X: object, mask: object | None, *, reset: bool
    ) -> tuple[InputMatrix, BoolArray | None]:
        _check_sparse_structure(X, check_infinity=False)
        check_named_columns(X)
        sparse = sp.issparse(X)
        observed: BoolArray | sp.csr_matrix | None = None
        if sparse:
            raw = cast("sp.csr_matrix | sp.csc_matrix | sp.coo_matrix", X)
            observed = _observation_mask(raw.shape, mask, sparse=True)
            _check_observed_infinity(raw, observed)
        with named_table_input(self, X, reset=reset) as (validator, values):
            checked = cast(
                "InputMatrix",
                validate_data(
                    validator,
                    values,
                    reset=reset,
                    dtype=None,
                    accept_sparse=["csr"],
                    ensure_all_finite=False,
                ),
            )
        names = getattr(self, "feature_names_in_", None)
        if names is not None and np.unique(names).size != len(names):
            msg = "X has duplicate feature names"
            raise ValueError(msg)
        if sparse:
            return observed_matrix(
                checked, cast("InputMask | None", observed), None
            ), None
        dense = canonical_table_nulls(X, cast("InputArray", checked))
        dense_mask = cast(
            "BoolArray | None", _observation_mask(dense.shape, mask, sparse=False)
        )
        for block in _dense_blocks(dense.shape):
            _finite_observations(
                dense[block], None if dense_mask is None else dense_mask[block]
            )
        return dense, dense_mask


def _observation_mask(
    shape: tuple[int, ...], mask: object | None, *, sparse: bool
) -> BoolArray | sp.csr_matrix | None:
    if mask is None:
        return None
    if sp.issparse(mask):
        _check_sparse_structure(mask)
        if not sparse:
            msg = "mask must be dense for dense input"
            raise ValueError(msg)
        raw = cast("sp.csr_matrix", mask)
        if raw.dtype != np.bool_:
            msg = "observation mask must have Boolean dtype"
            raise TypeError(msg)
        if raw.shape != shape:
            msg = f"mask shape {raw.shape} does not match data shape {shape}"
            raise ValueError(msg)
        return sp.csr_matrix(raw, copy=True)
    observed = np.asarray(mask)
    if observed.dtype != np.bool_:
        msg = "observation mask must have Boolean dtype"
        raise TypeError(msg)
    if observed.shape != shape:
        msg = f"mask shape {observed.shape} does not match data shape {shape}"
        raise ValueError(msg)
    return observed


def _check_observed_infinity(
    raw: sp.csr_matrix | sp.csc_matrix | sp.coo_matrix,
    observed: BoolArray | sp.csr_matrix | None,
) -> None:
    # Check raw duplicates before a conversion can combine +inf/-inf into NaN.
    infinite = np.isinf(raw.data)
    if not np.any(infinite):
        return
    if observed is not None:
        coordinates = raw.tocoo(copy=False)
        rows, columns = coordinates.row[infinite], coordinates.col[infinite]
        if sp.issparse(observed):
            selected = np.asarray(observed[rows, columns]).ravel()
        else:
            selected = observed[rows, columns]
        if not np.any(selected):
            return
    msg = "Input contains infinity in observed entries"
    raise ValueError(msg)


def _canonical_output(data: InputMatrix, observed: BoolArray | None) -> InputMatrix:
    if observed is None:
        return data
    dense = cast("InputArray", data)
    dtype = (
        dense.dtype
        if dense.dtype.kind == "f"
        else np.dtype(float)
        if dense.dtype.kind in "biu"
        else np.dtype(object)
    )
    result = np.array(dense, dtype=dtype, copy=True)
    for block in _dense_blocks(dense.shape):
        result[block][~observed[block]] = np.nan
    return result


def _dense_blocks(shape: tuple[int, ...]) -> Iterator[tuple[slice, slice]]:
    for column in range(0, shape[1], _MASK_BLOCK_ENTRIES):
        stop = min(column + _MASK_BLOCK_ENTRIES, shape[1])
        row_step = max(1, _MASK_BLOCK_ENTRIES // (stop - column))
        for row in range(0, shape[0], row_step):
            yield slice(row, row + row_step), slice(column, stop)
