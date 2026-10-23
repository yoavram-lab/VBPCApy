"""Native fitted-state lifecycle shared by missing-aware preprocessors."""

from __future__ import annotations

from collections.abc import Generator, Sequence  # noqa: TC003 - runtime annotation resolution
from contextlib import contextmanager
from copy import copy
from typing import ClassVar, cast

import numpy as np
import scipy.sparse as sp
from sklearn.base import TransformerMixin  # type: ignore[import-untyped]
from sklearn.utils import Tags  # type: ignore[import-untyped]  # noqa: TC002 - runtime annotation resolution
from sklearn.utils.validation import check_array, check_is_fitted, validate_data  # type: ignore[import-untyped]

from ._native import TypedEstimator
from ._types import BoolArray, InputArray  # noqa: TC001 - runtime annotation resolution

_PRIVATE_FIT_STATE = frozenset({"_output_width", "_output_widths", "_plan", "_scaler"})


class PreprocessorState(TransformerMixin, TypedEstimator):  # type: ignore[misc]
    """Real sklearn transformer with transactional, fit-owned learned state."""

    n_features_in_: int | None
    _numeric_input: ClassVar[bool] = True
    _accept_sparse: ClassVar[bool] = False

    def __sklearn_tags__(self) -> Tags:  # noqa: PLW3201 - sklearn protocol
        """Declare the missing-data and input representations we validate.

        Returns:
            Native tags matching this transformer's implemented boundary.
        """
        tags = cast("Tags", super().__sklearn_tags__())
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = self._accept_sparse
        tags.input_tags.string = not self._numeric_input
        tags.input_tags.categorical = not self._numeric_input
        output_dtype = np.dtype(getattr(self, "dtype", np.float64))
        tags.transformer_tags.preserves_dtype = (
            [output_dtype.name] if output_dtype.kind == "f" else []
        )
        return tags

    def _validate_input(
        self, x: object, mask: object | None, *, reset: bool
    ) -> InputArray | sp.spmatrix:
        """Validate dimensions, feature metadata and observed input values.

        Returns:
            Validated data, borrowing existing buffers when possible.

        Raises:
            ValueError: Input feature names contain duplicates.
        """
        _check_sparse_structure(x)
        checked = cast(
            "InputArray | sp.spmatrix",
            validate_data(
                self,
                x,
                reset=reset,
                dtype="numeric" if self._numeric_input else None,
                accept_sparse=["csr", "csc"] if self._accept_sparse else False,
                ensure_all_finite=False,
            ),
        )
        names = getattr(self, "feature_names_in_", None)
        if names is not None and np.unique(names).size != len(names):
            msg = "X has duplicate feature names"
            raise ValueError(msg)
        observed = _check_mask(checked.shape, mask, sparse=sp.issparse(checked))
        return _finite_observations(checked, observed)

    def _validate_inverse(
        self,
        z: object,
        mask: object | None,
        *,
        width: int,
        mask_widths: Sequence[int] | None = None,
    ) -> InputArray | sp.spmatrix:
        """Validate encoded dimensions without replacing input feature metadata.

        Returns:
            Validated numeric encoded values.

        Raises:
            ValueError: The encoded width differs from the fitted output width.
        """
        complex_indicators = np.dtype(getattr(self, "dtype", np.float64)).kind == "c"
        _check_sparse_structure(z, complex_indicators=complex_indicators)
        if complex_indicators:
            if sp.issparse(z):
                sparse = cast("sp.csr_matrix | sp.csc_matrix | sp.coo_matrix", z)
                if np.iscomplexobj(sparse.data):
                    sparse = copy(sparse)
                    sparse.data = _real_indicators(sparse.data)
                    z = sparse
            else:
                values = np.asarray(z)
                if np.iscomplexobj(values):
                    z = _real_indicators(values)
        checked = cast(
            "InputArray | sp.spmatrix",
            check_array(
                z,
                dtype="numeric",
                accept_sparse=["csr", "csc"] if self._accept_sparse else False,
                ensure_all_finite=False,
                ensure_min_features=0,
                estimator=self,
                input_name="Z",
            ),
        )
        if checked.shape[1] != width:
            msg = (
                f"Z has {checked.shape[1]} features; "
                f"expected fitted output width {width}"
            )
            raise ValueError(msg)
        observed = _check_mask(
            (checked.shape[0], width if mask_widths is None else len(mask_widths)),
            mask,
            sparse=sp.issparse(checked),
        )
        if mask_widths is not None and observed is not None:
            # Check each variable block without allocating an expanded mask.
            dense = cast("InputArray", checked)
            start = 0
            for j, block_width in enumerate(mask_widths):
                stop = start + block_width
                _finite_observations(dense[:, start:stop], observed[:, j])
                start = stop
            return dense
        return _finite_observations(checked, observed)

    def _clear_fit_state(self) -> None:
        """Release previous or partially fitted arrays and encoding plans."""
        # Invalidate fitted status before releasing any array/encoder payload.
        if "n_features_in_" in vars(self):
            del self.n_features_in_
        for name in tuple(vars(self)):
            if name.endswith("_") or name in _PRIVATE_FIT_STATE:
                delattr(self, name)

    @contextmanager
    def _fitting(self) -> Generator[None, None, None]:
        """Clear obsolete state and discard partial state when fitting fails.

        Yields:
            Control to the existing numerical fit implementation.
        """
        try:
            self._clear_fit_state()
            yield
        except BaseException:
            self._clear_fit_state()
            raise

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201 - sklearn protocol
        """Recognize fitted feature metadata, including compatible old artifacts.

        Returns:
            Whether a feature count was learned; old unfitted None defaults
            remain distinguishable from successful fits.
        """
        return getattr(self, "n_features_in_", None) is not None

    def _check_fitted(self) -> None:
        """Require a completed fit through sklearn's validation helper."""
        check_is_fitted(self)


def _check_mask(
    shape: tuple[int, ...], mask: object | None, *, sparse: bool
) -> BoolArray | None:
    if mask is None:
        return None
    if sparse:
        msg = (
            "mask must be None for sparse preprocessing; "
            "stored entries define observations"
        )
        raise ValueError(msg)
    if sp.issparse(mask):
        msg = "mask must be dense for dense preprocessing"
        raise ValueError(msg)
    observed = np.asarray(mask, dtype=bool)
    if observed.shape != shape:
        msg = f"mask shape {observed.shape} does not match data shape {shape}"
        raise ValueError(msg)
    return observed


def _check_sparse_structure(x: object, *, complex_indicators: bool = False) -> None:
    if not sp.issparse(x):
        return
    supported = cast("sp.csr_matrix | sp.csc_matrix | sp.coo_matrix", x)
    if supported.format not in {"csr", "csc", "coo"}:
        msg = "sparse preprocessing supports CSR, CSC and COO input"
        raise TypeError(msg)
    if supported.ndim != 2:
        msg = f"Expected 2D sparse input, got {supported.ndim}D input"
        raise ValueError(msg)
    # COO conversion can coalesce duplicates before sklearn returns. Reject
    # raw infinity first so +inf/-inf cannot disappear into a missing NaN.
    if np.iscomplexobj(supported.data) and not complex_indicators:
        msg = "Complex data not supported"
        raise ValueError(msg)
    if supported.data.dtype.kind in "biufc" and np.any(np.isinf(supported.data)):
        msg = "Input contains infinity in observed entries"
        raise ValueError(msg)
    if supported.format == "coo":
        _check_coordinates(cast("sp.coo_matrix", x))
        return
    matrix = cast("sp.csr_matrix | sp.csc_matrix", x)
    major, minor = matrix.shape if matrix.format == "csr" else matrix.shape[::-1]
    pointer, indices = matrix.indptr, matrix.indices
    if (
        pointer.ndim != 1
        or indices.ndim != 1
        or matrix.data.ndim != 1
        or pointer.dtype.kind not in "iu"
        or indices.dtype.kind not in "iu"
    ):
        msg = "Invalid sparse data, indices or indptr arrays"
        raise ValueError(msg)
    if (
        pointer.size != major + 1
        or pointer[0] != 0
        or pointer[-1] != indices.size
        or np.any(pointer[1:] < pointer[:-1])
    ):
        msg = "Invalid sparse indptr array"
        raise ValueError(msg)
    if (
        indices.size != matrix.data.size
        or np.any(indices < 0)
        or np.any(indices >= minor)
    ):
        msg = "Invalid sparse data, indices or indptr arrays"
        raise ValueError(msg)


def _check_coordinates(x: sp.coo_matrix) -> None:
    if x.data.ndim != 1:
        msg = "Invalid sparse data array"
        raise ValueError(msg)
    for indices, size in zip((x.row, x.col), x.shape, strict=True):
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or indices.size != x.data.size
            or np.any(indices < 0)
            or np.any(indices >= size)
        ):
            msg = "Invalid sparse coordinate arrays"
            raise ValueError(msg)


def _real_indicators(values: InputArray) -> InputArray:
    missing = np.isnan(values.real) & np.isnan(values.imag)
    if np.any((values.imag != 0) & ~missing):
        msg = "Complex indicator storage requires finite zero imaginary components"
        raise ValueError(msg)
    return values.real


def _finite_observations(
    checked: InputArray | sp.spmatrix, observed: BoolArray | None
) -> InputArray | sp.spmatrix:
    if sp.issparse(checked):
        sparse = cast("sp.csr_matrix | sp.csc_matrix", checked)
        if not sparse.has_canonical_format:
            sparse = sparse.copy()
            sparse.sum_duplicates()
        if np.any(np.isinf(sparse.data)):
            msg = "Input contains infinity in observed entries"
            raise ValueError(msg)
        if np.any(np.isnan(sparse.data)):
            # Filter missing entries without eliminating explicitly stored zeros.
            from ._full_update import _filter_csr_entries  # noqa: PLC0415

            csr = sp.csr_matrix(sparse)
            return _filter_csr_entries(csr, ~np.isnan(csr.data))
        return sparse
    array = cast("InputArray", checked)
    values = array if observed is None else array[observed]
    if array.dtype.kind == "O":
        for value in values.flat:
            if isinstance(value, complex | np.complexfloating):
                msg = "Complex data not supported"
                raise ValueError(msg)  # noqa: TRY004 - sklearn complex-data contract
            if isinstance(value, float | np.floating) and np.isinf(value):
                msg = "Input contains infinity in observed entries"
                raise ValueError(msg)
    if values.dtype.kind in "biufc" and np.any(np.isinf(values)):
        msg = "Input contains infinity in observed entries"
        raise ValueError(msg)
    return array
