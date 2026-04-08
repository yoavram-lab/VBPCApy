"""Missing-aware preprocessing utilities with scikit-learn-like APIs."""

from __future__ import annotations

# ruff: noqa: D102, D107, DOC201, DOC501, TRY003, EM101, EM102, TC003
import warnings as _warnings_mod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import scipy.sparse as sp

from ._sparsity import validate_mask_compatibility

__all__ = [
    "AutoEncoder",
    "DataReport",
    "MissingAwareLogTransformer",
    "MissingAwareMinMaxScaler",
    "MissingAwareOneHotEncoder",
    "MissingAwarePowerTransformer",
    "MissingAwareSparseOneHotEncoder",
    "MissingAwareStandardScaler",
    "MissingAwareWinsorizer",
    "check_data",
]

Mask = np.ndarray
ArrayLike = np.ndarray | sp.spmatrix


def _to_csr(x: ArrayLike, *, copy: bool = False) -> sp.csr_matrix:
    return sp.csr_matrix(cast("Any", x), copy=copy)


def _to_csc(x: ArrayLike) -> sp.csc_matrix:
    return sp.csc_matrix(cast("Any", x))


def _is_sparse(x: object) -> bool:
    return sp.issparse(x)


def _ensure_mask(x: np.ndarray, mask: Mask | None) -> Mask:
    """Return a boolean mask where True marks observed entries."""
    if mask is None:
        if np.issubdtype(x.dtype, np.number):
            return ~np.isnan(x.astype(float))  # type: ignore[no-any-return]
        return ~np.not_equal(x, x)  # type: ignore[no-any-return]
    return np.asarray(mask, dtype=bool)


def _sparse_col_counts(mat: sp.csc_matrix) -> np.ndarray:
    """Return nonzero counts per column for a CSC matrix."""
    return np.diff(mat.indptr)


def _sparse_safe_min(mat: sp.csc_matrix, counts: np.ndarray) -> np.ndarray:
    mins = np.full(mat.shape[1], np.nan, dtype=float)
    if mat.nnz:
        for j in range(mat.shape[1]):
            if counts[j] == 0:
                continue
            col = mat.getcol(j)
            mins[j] = float(col.data.min()) if col.data.size else np.nan
    return mins


def _sparse_safe_max(mat: sp.csc_matrix, counts: np.ndarray) -> np.ndarray:
    maxs = np.full(mat.shape[1], np.nan, dtype=float)
    if mat.nnz:
        for j in range(mat.shape[1]):
            if counts[j] == 0:
                continue
            col = mat.getcol(j)
            maxs[j] = float(col.data.max()) if col.data.size else np.nan
    return maxs


def _sparse_mean_var(mat: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/var per column ignoring missing (unstored) entries.

    Returns:
        Tuple of ``(means, variances)`` per column.
    """
    counts = _sparse_col_counts(mat)
    sums = np.array(mat.sum(axis=0)).ravel()
    sumsq = np.array(mat.power(2).sum(axis=0)).ravel()
    means = np.full(mat.shape[1], np.nan, dtype=float)
    vars_ = np.full(mat.shape[1], np.nan, dtype=float)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]
    vars_[nonzero] = sumsq[nonzero] / counts[nonzero] - means[nonzero] ** 2
    return means, vars_


def _safe_unique(values: np.ndarray) -> list[Any]:
    """Return ordered unique values preserving first occurrence."""
    uniq: list[Any] = []
    seen: set[Any] = set()
    for v in values:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


class MissingAwareOneHotEncoder:
    """One-hot encode categorical columns while respecting missing values."""

    def __init__(
        self,
        *,
        handle_unknown: Literal["ignore", "raise"] = "ignore",
        mean_center: bool = False,
        dtype: type = float,
    ) -> None:
        self.handle_unknown = handle_unknown
        self.mean_center = mean_center
        self.dtype = dtype
        self.categories_: list[list[Any]] = []
        self.feature_names_out_: list[str] = []
        self.column_means_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._output_widths: list[int] = []

    def fit(self, x: np.ndarray, mask: Mask | None = None) -> MissingAwareOneHotEncoder:
        x_arr = np.asarray(x)
        obs_mask = _ensure_mask(x_arr, mask)
        n_features = x_arr.shape[1]
        self.categories_ = []
        self.feature_names_out_ = []
        self._output_widths = []
        means: list[float] = []

        for j in range(n_features):
            col_obs = obs_mask[:, j]
            observed_vals = x_arr[col_obs, j]
            cats = _safe_unique(observed_vals)
            self.categories_.append(cats)
            if len(cats) == 2:
                # Mirror legacy OHspecial_transform: collapse binary to a single
                # indicator column for the second category.
                self.feature_names_out_.append(f"col{j}_{cats[1]}")
                self._output_widths.append(1)
                means.extend([0.0])
            else:
                for c in cats:
                    self.feature_names_out_.append(f"col{j}_{c}")
                self._output_widths.append(len(cats))
                means.extend([0.0] * len(cats))

        self.column_means_ = np.array(means, dtype=self.dtype)
        self.n_features_in_ = n_features
        return self

    def _transform_single(
        self, x: np.ndarray, mask: Mask, j: int
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        cats = self.categories_[j]
        n_cats = len(cats)
        if n_cats == 0:
            empty: np.ndarray = np.zeros((x.shape[0], 0), dtype=self.dtype)
            return empty, [], np.zeros(0, dtype=self.dtype)

        col_mask = np.asarray(mask[:, j], dtype=bool)
        col_vals = x[:, j]

        if n_cats == 2:
            return self._transform_binary(col_vals, col_mask, cats, j)
        return self._transform_multicat(col_vals, col_mask, cats, j)

    def _transform_binary(
        self,
        col_vals: np.ndarray,
        col_mask: np.ndarray,
        cats: list[Any],
        j: int,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        out: np.ndarray = np.zeros((col_vals.shape[0], 1), dtype=self.dtype)
        if not np.all(col_mask):
            out[~col_mask, :] = np.nan

        if np.any(col_mask):
            cat0, cat1 = cats[0], cats[1]
            obs_rows = np.nonzero(col_mask)[0]
            observed_vals = col_vals[col_mask]
            for row_idx, val in zip(obs_rows, observed_vals, strict=False):
                if val == cat1:
                    out[row_idx, 0] = 1.0
                elif val == cat0:
                    out[row_idx, 0] = 0.0
                else:
                    if self.handle_unknown == "raise":
                        raise ValueError(
                            f"Unknown category {val!r} in binary column {j}"
                        )
                    out[row_idx, 0] = 0.0

        means: np.ndarray = np.zeros(1, dtype=self.dtype)
        if self.mean_center:
            with np.errstate(invalid="ignore"):
                means = np.nanmean(out, axis=0)
            out -= means
            out[~np.isfinite(out)] = np.nan
        names = [f"col{j}_{cats[1]}"]
        return out, names, means

    def _transform_multicat(
        self,
        col_vals: np.ndarray,
        col_mask: np.ndarray,
        cats: list[Any],
        j: int,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        n_cats = len(cats)
        out: np.ndarray = np.zeros((col_vals.shape[0], n_cats), dtype=self.dtype)
        if not np.all(col_mask):
            out[~col_mask, :] = np.nan

        if np.any(col_mask):
            cat_to_idx = {cat: idx for idx, cat in enumerate(cats)}
            observed_vals = col_vals[col_mask]
            idxs = np.fromiter(
                (cat_to_idx.get(val, -1) for val in observed_vals),
                dtype=int,
                count=observed_vals.size,
            )
            if self.handle_unknown == "raise" and np.any(idxs < 0):
                bad_val = observed_vals[idxs < 0][0]
                raise ValueError(f"Unknown category {bad_val!r} in column {j}")
            valid = idxs >= 0
            if np.any(valid):
                obs_rows = np.nonzero(col_mask)[0]
                out[obs_rows[valid], idxs[valid]] = 1.0

        means: np.ndarray = np.zeros(n_cats, dtype=self.dtype)
        if self.mean_center:
            with np.errstate(invalid="ignore"):
                means = np.nanmean(out, axis=0)
            out -= means
            out[~np.isfinite(out)] = np.nan
        names = [f"col{j}_{c}" for c in cats]
        return out, names, means

    def transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        if self.n_features_in_ is None:
            raise RuntimeError("Encoder not fitted")
        x_arr = np.asarray(x)
        obs_mask = _ensure_mask(x_arr, mask)
        parts: list[np.ndarray] = []
        means: list[float] = []
        names: list[str] = []
        for j in range(self.n_features_in_):
            block, block_names, block_means = self._transform_single(x_arr, obs_mask, j)
            parts.append(block)
            names.extend(block_names)
            means.extend(block_means.tolist())
        self.feature_names_out_ = names
        self.column_means_ = (
            np.array(means, dtype=self.dtype) if self.mean_center else None
        )
        return (
            np.concatenate(parts, axis=1)
            if parts
            else np.empty((x_arr.shape[0], 0), dtype=self.dtype)
        )

    def fit_transform(
        self, x: np.ndarray, mask: Mask | None = None
    ) -> np.ndarray | sp.csr_matrix:
        return self.fit(x, mask).transform(x, mask)

    def inverse_transform(self, z: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        if self.n_features_in_ is None:
            raise RuntimeError("Encoder not fitted")
        z_arr: np.ndarray = np.asarray(z, dtype=self.dtype)
        obs_mask = None
        if mask is not None:
            obs_mask = np.asarray(mask, dtype=bool)
        out_cols: list[np.ndarray] = []
        col_start = 0
        mean_idx = 0
        for j, cats in enumerate(self.categories_):
            n_cats = len(cats)
            width = self._output_widths[j] if j < len(self._output_widths) else n_cats
            if n_cats == 0:
                out_cols.append(np.full(z_arr.shape[0], np.nan))
                continue
            block = z_arr[:, col_start : col_start + width]
            if self.mean_center and self.column_means_ is not None:
                means = self.column_means_[mean_idx : mean_idx + width]
                block += means
            col_vals = np.full(z_arr.shape[0], np.nan, dtype=object)
            if n_cats == 2 and width == 1:
                self._decode_binary(block, col_vals, cats, obs_mask, j)
            else:
                self._decode_multicat(block, col_vals, cats, obs_mask, j)
            out_cols.append(col_vals.astype(object))
            col_start += width
            mean_idx += width
        return np.column_stack(out_cols)

    @staticmethod
    def _decode_binary(
        block: np.ndarray,
        col_vals: np.ndarray,
        cats: list[Any],
        obs_mask: np.ndarray | None,
        j: int,
    ) -> None:
        cat0, cat1 = cats[0], cats[1]
        for i in range(block.shape[0]):
            if obs_mask is not None and not obs_mask[i, j]:
                col_vals[i] = np.nan
                continue
            val = block[i, 0]
            if np.isnan(val):
                col_vals[i] = np.nan
            elif val >= 0.5:
                col_vals[i] = cat1
            else:
                col_vals[i] = cat0

    def _decode_multicat(
        self,
        block: np.ndarray,
        col_vals: np.ndarray,
        cats: list[Any],
        obs_mask: np.ndarray | None,
        j: int,
    ) -> None:
        for i in range(block.shape[0]):
            if obs_mask is not None and not obs_mask[i, j]:
                col_vals[i] = np.nan
                continue
            row = block[i, :]
            if np.all(np.isnan(row)):
                col_vals[i] = np.nan
                continue
            idx = int(np.nanargmax(row))
            if np.allclose(row, 0.0) and self.handle_unknown == "ignore":
                col_vals[i] = np.nan
            else:
                col_vals[i] = cats[idx]


class _BaseScaler:
    def __init__(self) -> None:
        self.n_features_in_: int | None = None

    def fit(self, x: ArrayLike, mask: Mask | None = None) -> _BaseScaler:
        raise NotImplementedError

    def transform(
        self, x: ArrayLike, mask: Mask | None = None
    ) -> np.ndarray | sp.csr_matrix:
        raise NotImplementedError

    def inverse_transform(
        self, z: ArrayLike, mask: Mask | None = None
    ) -> np.ndarray | sp.csr_matrix:
        raise NotImplementedError

    def fit_transform(
        self, x: ArrayLike, mask: Mask | None = None
    ) -> np.ndarray | sp.csr_matrix:
        return self.fit(x, mask).transform(x, mask)


class MissingAwareSparseOneHotEncoder:
    """Sparse one-hot encoder for categorical columns.

    Assumptions:
        - Input is sparse (CSR/CSC) with one column.
        - Observed entries are stored; missing entries are absent.
        - Categories must be numeric to round-trip through sparse matrices.
        - ``mean_center`` adjusts stored values per category without
          densifying.
    """

    def __init__(
        self,
        *,
        handle_unknown: Literal["ignore", "raise"] = "ignore",
        mean_center: bool = False,
        dtype: type = float,
    ) -> None:
        self.handle_unknown = handle_unknown
        self.mean_center = mean_center
        self.dtype = dtype
        self.categories_: list[float] = []
        self.feature_names_out_: list[str] = []
        self.column_means_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._output_width: int = 0

    def fit(
        self, x: sp.spmatrix, mask: Mask | None = None
    ) -> MissingAwareSparseOneHotEncoder:
        if mask is not None:
            msg = "mask must be None for sparse one-hot encoder"
            raise ValueError(msg)
        x_csr = _to_csr(x)
        if x_csr.shape[1] != 1:
            msg = "sparse one-hot encoder expects a single column"
            raise ValueError(msg)

        if x_csr.data.size == 0:
            self.categories_ = []
            self.feature_names_out_ = []
            self.column_means_ = np.array([], dtype=self.dtype)
            self.n_features_in_ = 1
            self._output_width = 0
            return self

        # Require numeric categories to round-trip through sparse matrices.
        cats = np.unique(np.asarray(x_csr.data, dtype=float))
        self.categories_ = [float(c) for c in cats.tolist()]
        self.feature_names_out_ = [f"col0_{c}" for c in self.categories_]
        self._output_width = len(self.categories_)

        n_rows = x_csr.shape[0]
        counts = np.zeros(len(self.categories_), dtype=float)
        cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories_)}
        for val in x_csr.data:
            counts[cat_to_idx[float(val)]] += 1.0
        means_arr: np.ndarray = (counts / float(max(n_rows, 1))).astype(
            self.dtype, copy=False
        )
        self.column_means_ = means_arr
        self.n_features_in_ = 1
        return self

    def transform(self, x: sp.spmatrix, mask: Mask | None = None) -> sp.csr_matrix:
        if self.n_features_in_ is None:
            raise RuntimeError("Encoder not fitted")
        if mask is not None:
            msg = "mask must be None for sparse one-hot encoder"
            raise ValueError(msg)
        x_csr = _to_csr(x)
        if x_csr.shape[1] != 1:
            msg = "sparse one-hot encoder expects a single column"
            raise ValueError(msg)

        n_rows = x_csr.shape[0]
        n_cats = len(self.categories_)
        if n_cats == 0:
            return sp.csr_matrix((n_rows, 0), dtype=self.dtype)
        return self._transform_multicat_sparse(x_csr, n_rows)

    def inverse_transform(
        self, z: sp.spmatrix, mask: Mask | None = None
    ) -> sp.csr_matrix:
        if self.n_features_in_ is None:
            raise RuntimeError("Encoder not fitted")
        if mask is not None:
            msg = "mask must be None for sparse one-hot encoder"
            raise ValueError(msg)
        z_csr = _to_csr(z)
        n_rows, n_cols = z_csr.shape
        expected_width = len(self.categories_)
        if n_cols != expected_width:
            msg = "Input width must match fitted categories"
            raise ValueError(msg)
        return self._inverse_multicat_sparse(z_csr, n_rows)

    def _transform_multicat_sparse(
        self, x_csr: sp.csr_matrix, n_rows: int
    ) -> sp.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories_)}
        coo = x_csr.tocoo()
        for row, val in zip(coo.row, np.asarray(coo.data, dtype=float), strict=False):
            cat_idx = cat_to_idx.get(val, -1)
            if cat_idx < 0:
                if self.handle_unknown == "raise":
                    msg = f"Unknown category {val!r} in sparse one-hot encoder"
                    raise ValueError(msg)
                continue
            rows.append(int(row))
            cols.append(cat_idx)
            data.append(1.0)

        out: sp.csr_matrix = sp.csr_matrix(
            (np.array(data, dtype=self.dtype), (rows, cols)),
            shape=(n_rows, len(self.categories_)),
        )
        if self.mean_center and self.column_means_ is not None and out.data.size > 0:
            out.data -= self.column_means_[out.indices]
        return out

    def _inverse_binary_sparse(
        self, z_csr: sp.csr_matrix, n_rows: int
    ) -> sp.csr_matrix:
        means = self.column_means_ if self.mean_center else None
        data_out: list[float] = []
        row_idx_out: list[int] = []
        for row in range(n_rows):
            start, end = z_csr.indptr[row], z_csr.indptr[row + 1]
            if end <= start:
                continue
            row_data = z_csr.data[start:end].copy()
            if means is not None:
                row_data += means[z_csr.indices[start:end]]
            cat_val = self.categories_[int(np.argmax(row_data))]
            row_idx_out.append(row)
            data_out.append(cat_val)

        if not data_out:
            return sp.csr_matrix((n_rows, 1), dtype=self.dtype)

        cols_out = np.zeros(len(data_out), dtype=int)
        return sp.csr_matrix(
            (np.array(data_out, dtype=self.dtype), (np.array(row_idx_out), cols_out)),
            shape=(n_rows, 1),
        )

    def _inverse_multicat_sparse(
        self, z_csr: sp.csr_matrix, n_rows: int
    ) -> sp.csr_matrix:
        means = self.column_means_ if self.mean_center else None
        data_out: list[float] = []
        row_idx_out: list[int] = []
        for row in range(n_rows):
            start, end = z_csr.indptr[row], z_csr.indptr[row + 1]
            if end <= start:
                continue  # missing row -> no stored entry
            row_indices = z_csr.indices[start:end]
            row_data = z_csr.data[start:end].copy()
            if means is not None:
                row_data += means[row_indices]
            max_pos = int(np.argmax(row_data))
            cat_idx = int(row_indices[max_pos])
            cat_val = self.categories_[cat_idx]
            row_idx_out.append(row)
            data_out.append(cat_val)

        if not data_out:
            return sp.csr_matrix((n_rows, 1), dtype=self.dtype)

        cols_out = np.zeros(len(data_out), dtype=int)
        return sp.csr_matrix(
            (np.array(data_out, dtype=self.dtype), (np.array(row_idx_out), cols_out)),
            shape=(n_rows, 1),
        )


class MissingAwareStandardScaler(_BaseScaler):
    """Standardize continuous columns ignoring missing entries."""

    def __init__(self) -> None:
        super().__init__()
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.var_: np.ndarray | None = None

    def fit(
        self, x: np.ndarray | sp.spmatrix, mask: Mask | None = None
    ) -> MissingAwareStandardScaler:
        if _is_sparse(x):
            x_csc = _to_csc(x)
            self.n_features_in_ = x_csc.shape[1]
            means, vars_ = _sparse_mean_var(x_csc)
        else:
            x_arr = np.asarray(x, dtype=float)
            obs_mask = _ensure_mask(x_arr, mask)
            self.n_features_in_ = x_arr.shape[1]
            means = np.zeros(self.n_features_in_)
            vars_ = np.zeros(self.n_features_in_)
            for j in range(self.n_features_in_):
                col = x_arr[:, j]
                col_obs = obs_mask[:, j]
                if not np.any(col_obs):
                    means[j] = np.nan
                    vars_[j] = np.nan
                    continue
                col_vals = col[col_obs]
                means[j] = float(np.mean(col_vals))
                vars_[j] = float(np.var(col_vals))
        scales = np.sqrt(vars_)
        scales[~np.isfinite(scales)] = 1.0
        zero_mask = np.isclose(scales, 0.0)
        scales[zero_mask] = 1.0
        self.mean_ = means
        self.var_ = vars_
        self.scale_ = scales
        return self

    def transform(
        self, x: np.ndarray | sp.spmatrix, mask: Mask | None = None
    ) -> np.ndarray | sp.csr_matrix:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fitted")
        if _is_sparse(x):
            x_csr = _to_csr(x, copy=True)
            if x_csr.shape[1] != len(self.mean_):
                raise ValueError("Input feature count mismatch")
            data = x_csr.data
            cols = x_csr.indices
            data = (data - self.mean_[cols]) / self.scale_[cols]
            x_csr.data = data
            return x_csr
        x_arr = np.asarray(x, dtype=float)
        obs_mask = _ensure_mask(x_arr, mask)
        z = (x_arr - self.mean_) / self.scale_
        z[~obs_mask] = np.nan
        return z  # type: ignore[no-any-return]

    def inverse_transform(
        self, z: np.ndarray | sp.spmatrix, mask: Mask | None = None
    ) -> np.ndarray | sp.csr_matrix:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fitted")
        if _is_sparse(z):
            z_csr = _to_csr(z, copy=True)
            data = z_csr.data
            cols = z_csr.indices
            data = data * self.scale_[cols] + self.mean_[cols]
            z_csr.data = data
            return z_csr
        z_arr = np.asarray(z, dtype=float)
        obs_mask = _ensure_mask(z_arr, mask)
        x = z_arr * self.scale_ + self.mean_
        x[~obs_mask] = np.nan
        return x  # type: ignore[no-any-return]


class MissingAwareMinMaxScaler(_BaseScaler):
    """Scale features to [0, 1] range while ignoring missing entries."""

    def __init__(self) -> None:
        super().__init__()
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.data_range_: np.ndarray | None = None

    def fit(
        self, x: np.ndarray | sp.spmatrix, mask: Mask | None = None
    ) -> MissingAwareMinMaxScaler:
        if _is_sparse(x):
            x_csc = _to_csc(x)
            self.n_features_in_ = x_csc.shape[1]
            counts = _sparse_col_counts(x_csc)
            data_min = _sparse_safe_min(x_csc, counts)
            data_max = _sparse_safe_max(x_csc, counts)
        else:
            x_arr = np.asarray(x, dtype=float)
            obs_mask = _ensure_mask(x_arr, mask)
            self.n_features_in_ = x_arr.shape[1]
            data_min = np.full(self.n_features_in_, np.nan, dtype=float)
            data_max = np.full(self.n_features_in_, np.nan, dtype=float)
            for j in range(self.n_features_in_):
                col_obs = obs_mask[:, j]
                if not np.any(col_obs):
                    continue
                col_vals = x_arr[col_obs, j]
                data_min[j] = float(np.min(col_vals))
                data_max[j] = float(np.max(col_vals))

        data_range = data_max - data_min
        data_range[~np.isfinite(data_range)] = 1.0
        zero_mask = np.isclose(data_range, 0.0)
        data_range[zero_mask] = 1.0
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(
        self, x: np.ndarray | sp.spmatrix, mask: Mask | None = None
    ) -> np.ndarray | sp.csr_matrix:
        if self.data_min_ is None or self.data_range_ is None:
            raise RuntimeError("Scaler not fitted")
        if _is_sparse(x):
            x_csr = _to_csr(x, copy=True)
            if x_csr.shape[1] != len(self.data_min_):
                raise ValueError("Input feature count mismatch")
            data = x_csr.data
            cols = x_csr.indices
            data = (data - self.data_min_[cols]) / self.data_range_[cols]
            x_csr.data = data
            return x_csr
        x_arr = np.asarray(x, dtype=float)
        obs_mask = _ensure_mask(x_arr, mask)
        z = (x_arr - self.data_min_) / self.data_range_
        z[~obs_mask] = np.nan
        return z  # type: ignore[no-any-return]

    def inverse_transform(
        self, z: np.ndarray | sp.spmatrix, mask: Mask | None = None
    ) -> np.ndarray | sp.csr_matrix:
        if self.data_min_ is None or self.data_range_ is None:
            raise RuntimeError("Scaler not fitted")
        if _is_sparse(z):
            z_csr = _to_csr(z, copy=True)
            data = z_csr.data
            cols = z_csr.indices
            data = data * self.data_range_[cols] + self.data_min_[cols]
            z_csr.data = data
            return z_csr
        z_arr = np.asarray(z, dtype=float)
        obs_mask = _ensure_mask(z_arr, mask)
        x = z_arr * self.data_range_ + self.data_min_
        x[~obs_mask] = np.nan
        return x  # type: ignore[no-any-return]


@dataclass
class _ColumnPlan:
    kind: Literal["categorical", "continuous"]
    encoder: Any
    slice_start: int
    slice_end: int


class AutoEncoder:
    """Column-wise router that applies missing-aware OHE or scaling."""

    def __init__(
        self,
        *,
        cardinality_threshold: int = 20,
        continuous_scaler: Literal["standard", "minmax"] = "standard",
        handle_unknown: Literal["ignore", "raise"] = "ignore",
        mean_center_ohe: bool = False,
        column_types: Sequence[Literal["categorical", "continuous"]] | None = None,
    ) -> None:
        self.cardinality_threshold = cardinality_threshold
        self.continuous_scaler = continuous_scaler
        self.handle_unknown = handle_unknown
        self.mean_center_ohe = mean_center_ohe
        self.column_types = column_types
        self.n_features_in_: int | None = None
        self.feature_names_out_: list[str] = []
        self._plan: list[_ColumnPlan] = []

    def _infer_kind(
        self, col: np.ndarray, mask: Mask, idx: int
    ) -> Literal["categorical", "continuous"]:
        if self.column_types is not None:
            return self.column_types[idx]
        observed = col[mask]
        if observed.size == 0:
            return "categorical"
        if observed.dtype.kind in {"U", "S", "O"}:
            return "categorical"
        if np.issubdtype(observed.dtype, np.integer):
            uniq = np.unique(observed)
            return (
                "categorical"
                if uniq.size <= self.cardinality_threshold
                else "continuous"
            )
        return "continuous"

    def _infer_kind_sparse(
        self, col: sp.spmatrix, idx: int
    ) -> Literal["categorical", "continuous"]:
        if self.column_types is not None:
            return self.column_types[idx]
        col_csr = _to_csr(col)
        data = np.asarray(col_csr.data)
        if data.size == 0:
            return "categorical"
        if np.issubdtype(data.dtype, np.integer):
            uniq = np.unique(data)
            return (
                "categorical"
                if uniq.size <= self.cardinality_threshold
                else "continuous"
            )
        return "continuous"

    def fit(self, x: np.ndarray | sp.spmatrix, mask: Mask | None = None) -> AutoEncoder:
        is_sparse = _is_sparse(x)
        if is_sparse and mask is not None:
            raise ValueError("mask must be None when fitting sparse data")
        if is_sparse:
            x_csr = _to_csr(x)
            self.n_features_in_ = x_csr.shape[1]

            self._plan = []
            self.feature_names_out_ = []
            col_start = 0
            for j in range(self.n_features_in_):
                kind = self._infer_kind_sparse(x_csr.getcol(j), j)
                encoder: Any
                if kind == "categorical":
                    encoder = MissingAwareSparseOneHotEncoder(
                        handle_unknown=self.handle_unknown,
                        mean_center=self.mean_center_ohe,
                    )
                else:
                    encoder = (
                        MissingAwareStandardScaler()
                        if self.continuous_scaler == "standard"
                        else MissingAwareMinMaxScaler()
                    )
                col_mat = _to_csr(x_csr.getcol(j))
                encoder.fit(col_mat, mask=None)
                z = encoder.transform(col_mat, mask=None)
                width = z.shape[1]
                if kind == "categorical" and hasattr(encoder, "categories_"):
                    names = [f"col{j}_{c}" for c in getattr(encoder, "categories_", [])]
                else:
                    names = [f"col{j}"] * width
                self.feature_names_out_.extend(names)
                self._plan.append(
                    _ColumnPlan(
                        kind=kind,
                        encoder=encoder,
                        slice_start=col_start,
                        slice_end=col_start + width,
                    )
                )
                col_start += width
            return self

        x_dense = np.asarray(x)
        obs_mask = _ensure_mask(x_dense, mask)
        self.n_features_in_ = x_dense.shape[1]
        self._plan = []
        self.feature_names_out_ = []
        col_start = 0
        for j in range(self.n_features_in_):
            kind = (
                self.column_types[j]
                if self.column_types is not None
                else self._infer_kind(x_dense[:, j], obs_mask[:, j], j)
            )
            encoder_dense: Any
            if kind == "categorical":
                encoder_dense = MissingAwareOneHotEncoder(
                    handle_unknown=self.handle_unknown,
                    mean_center=self.mean_center_ohe,
                )
            else:
                encoder_dense = (
                    MissingAwareStandardScaler()
                    if self.continuous_scaler == "standard"
                    else MissingAwareMinMaxScaler()
                )
            encoder_dense.fit(x_dense[:, [j]], mask=obs_mask[:, [j]])
            z = encoder_dense.transform(x_dense[:, [j]], mask=obs_mask[:, [j]])
            width = z.shape[1]
            self.feature_names_out_.extend(
                getattr(encoder_dense, "feature_names_out_", [f"col{j}"] * width)
            )
            self._plan.append(
                _ColumnPlan(
                    kind=kind,
                    encoder=encoder_dense,
                    slice_start=col_start,
                    slice_end=col_start + width,
                )
            )
            col_start += width
        return self

    def transform(
        self, x: np.ndarray | sp.spmatrix, mask: Mask | None = None
    ) -> np.ndarray | sp.spmatrix:
        if self.n_features_in_ is None:
            raise RuntimeError("AutoEncoder not fitted")
        is_sparse = _is_sparse(x)
        if is_sparse:
            if mask is not None:
                raise ValueError("mask must be None when transforming sparse data")
            x_csr = _to_csr(x)
            parts_sparse: list[sp.csr_matrix] = []
            for col_idx, plan in enumerate(self._plan):
                z_part = plan.encoder.transform(x_csr.getcol(col_idx), mask=None)
                parts_sparse.append(cast("sp.csr_matrix", z_part))
            if not parts_sparse:
                return sp.csr_matrix((x_csr.shape[0], 0))
            return sp.hstack(parts_sparse, format="csr")  # type: ignore[no-any-return]

        x_dense = np.asarray(x)
        obs_mask = _ensure_mask(x_dense, mask)
        parts_dense: list[np.ndarray] = []
        for col_idx, plan in enumerate(self._plan):
            z = plan.encoder.transform(
                x_dense[:, [col_idx]], mask=obs_mask[:, [col_idx]]
            )
            parts_dense.append(z)
        if not parts_dense:
            return np.empty((x_dense.shape[0], 0))
        return np.concatenate(tuple(parts_dense), axis=1)

    def fit_transform(
        self, x: np.ndarray | sp.spmatrix, mask: Mask | None = None
    ) -> np.ndarray | sp.spmatrix:
        return self.fit(x, mask).transform(x, mask)

    def inverse_transform(
        self, z: np.ndarray | sp.spmatrix, mask: Mask | None = None
    ) -> np.ndarray | sp.csr_matrix:
        if self.n_features_in_ is None:
            raise RuntimeError("AutoEncoder not fitted")
        if _is_sparse(z):
            if mask is not None:
                msg = "mask must be None when inverse_transform on sparse data"
                raise ValueError(msg)
            z_csr = _to_csr(z)
            col_blocks: list[sp.csr_matrix] = []
            for plan in self._plan:
                col_idx = np.arange(plan.slice_start, plan.slice_end, dtype=int)
                block_csr = z_csr[:, col_idx]
                inv = plan.encoder.inverse_transform(block_csr, mask=None)
                if not sp.isspmatrix(inv):
                    msg = (
                        "sparse inverse_transform requires encoder to return a"
                        f" sparse matrix (column kind={plan.kind})"
                    )
                    raise ValueError(msg)
                inv_csr = sp.csr_matrix(cast("Any", inv))
                if inv_csr.shape[1] != 1:
                    msg = (
                        "encoder inverse must output a single column for sparse decode"
                    )
                    raise ValueError(msg)
                col_blocks.append(inv_csr)
            if not col_blocks:
                return sp.csr_matrix((z_csr.shape[0], 0))
            return cast("sp.csr_matrix", sp.hstack(col_blocks, format="csr"))

        z_arr = np.asarray(z)
        if mask is not None:
            validate_mask_compatibility(
                z_arr,
                mask,
                allow_sparse_mask_for_dense=False,
                allow_dense_mask_for_sparse=False,
                context="autoencoder.inverse_transform",
            )
        out_cols: list[np.ndarray] = []
        for plan in self._plan:
            block = z_arr[:, plan.slice_start : plan.slice_end]
            inv = plan.encoder.inverse_transform(block, mask=None)
            out_cols.append(inv[:, 0])
        return np.column_stack(out_cols)


# ── Distributional transforms (#80) ──────────────────────────────────────


class MissingAwareLogTransformer:
    """Apply ``log1p`` (or ``log(x + offset)``) while preserving NaN entries.

    This is a *pre-scaling* transform: compress right-skewed features
    before standardisation so that the standard deviation better reflects
    the bulk of the data rather than extreme tails.

    Args:
        offset: Additive constant before taking the log.  The default
            ``1.0`` gives the standard ``log1p`` transform.
    """

    def __init__(self, *, offset: float = 1.0) -> None:
        if offset <= 0:
            msg = "offset must be positive"
            raise ValueError(msg)
        self.offset = offset
        self.n_features_in_: int | None = None

    def fit(
        self,
        x: np.ndarray,
        mask: Mask | None = None,  # noqa: ARG002
    ) -> MissingAwareLogTransformer:
        """Record input width (stateless transform)."""
        self.n_features_in_ = np.asarray(x).shape[1]
        return self

    def transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        """Apply ``log(x + offset)`` to observed entries."""
        if self.n_features_in_ is None:
            raise RuntimeError("Transformer not fitted")
        x_arr = np.asarray(x, dtype=float)
        obs = _ensure_mask(x_arr, mask)
        out = np.full_like(x_arr, np.nan)
        out[obs] = np.log(x_arr[obs] + self.offset)
        return out

    def fit_transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        return self.fit(x, mask).transform(x, mask)

    def inverse_transform(self, z: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        """Reverse via ``exp(z) - offset``."""
        if self.n_features_in_ is None:
            raise RuntimeError("Transformer not fitted")
        z_arr = np.asarray(z, dtype=float)
        obs = _ensure_mask(z_arr, mask)
        out = np.full_like(z_arr, np.nan)
        out[obs] = np.exp(z_arr[obs]) - self.offset
        return out


class MissingAwareWinsorizer:
    """Clip features at fitted percentiles while preserving NaN entries.

    Winsorization prevents outlier-driven variance inflation *before*
    scaling, so that z-scoring better equalises feature scales.

    Args:
        lower_quantile: Lower clipping quantile (default ``0.01``).
        upper_quantile: Upper clipping quantile (default ``0.99``).

    Note:
        This transform is lossy — ``inverse_transform`` is a no-op
        (returns data unchanged) because the original tail values
        cannot be recovered.
    """

    def __init__(
        self,
        *,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            msg = "quantiles must satisfy 0 <= lower < upper <= 1"
            raise ValueError(msg)
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.n_features_in_: int | None = None
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, x: np.ndarray, mask: Mask | None = None) -> MissingAwareWinsorizer:
        """Compute per-column clipping bounds from observed values."""
        x_arr = np.asarray(x, dtype=float)
        obs = _ensure_mask(x_arr, mask)
        n_features = x_arr.shape[1]
        self.n_features_in_ = n_features
        lo = np.full(n_features, np.nan)
        hi = np.full(n_features, np.nan)
        for j in range(n_features):
            col_obs = x_arr[obs[:, j], j]
            if col_obs.size == 0:
                continue
            lo[j] = float(np.quantile(col_obs, self.lower_quantile))
            hi[j] = float(np.quantile(col_obs, self.upper_quantile))
        self.lower_ = lo
        self.upper_ = hi
        return self

    def transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        """Clip observed entries to fitted bounds."""
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("Transformer not fitted")
        x_arr = np.asarray(x, dtype=float)
        obs = _ensure_mask(x_arr, mask)
        out = np.full_like(x_arr, np.nan)
        for j in range(x_arr.shape[1]):
            col_obs = obs[:, j]
            out[col_obs, j] = np.clip(x_arr[col_obs, j], self.lower_[j], self.upper_[j])
        return out

    def fit_transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        return self.fit(x, mask).transform(x, mask)

    def inverse_transform(  # noqa: PLR6301
        self,
        z: np.ndarray,
        mask: Mask | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """No-op: winsorization is lossy.

        Returns a copy of the input unchanged.
        """
        return np.asarray(z, dtype=float).copy()


class MissingAwarePowerTransformer:
    """Yeo-Johnson variance-stabilising transform preserving NaN entries.

    Supports mixed-sign data.  The transform is applied element-wise
    with a per-column ``lmbda`` parameter estimated by maximum
    likelihood on observed values.

    Args:
        standardize: If ``True`` (default), z-score the transformed
            output so each feature has zero mean and unit variance.
    """

    def __init__(self, *, standardize: bool = True) -> None:
        self.standardize = standardize
        self.n_features_in_: int | None = None
        self.lambdas_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    # ── Yeo-Johnson element-wise helpers ──

    @staticmethod
    def _yj_transform(x: np.ndarray, lmbda: float) -> np.ndarray:
        """Vectorised Yeo-Johnson forward transform for a single lambda."""
        out = np.zeros_like(x, dtype=float)
        pos = x >= 0
        neg = ~pos

        if np.abs(lmbda) > 1e-10:
            out[pos] = ((x[pos] + 1.0) ** lmbda - 1.0) / lmbda
        else:
            out[pos] = np.log1p(x[pos])

        if np.abs(lmbda - 2.0) > 1e-10:
            out[neg] = -((-x[neg] + 1.0) ** (2.0 - lmbda) - 1.0) / (2.0 - lmbda)
        else:
            out[neg] = -np.log1p(-x[neg])

        return out

    @staticmethod
    def _yj_inverse(z: np.ndarray, lmbda: float) -> np.ndarray:
        """Vectorised Yeo-Johnson inverse transform for a single lambda."""
        out = np.zeros_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos

        if np.abs(lmbda) > 1e-10:
            out[pos] = (z[pos] * lmbda + 1.0) ** (1.0 / lmbda) - 1.0
        else:
            out[pos] = np.expm1(z[pos])

        if np.abs(lmbda - 2.0) > 1e-10:
            out[neg] = 1.0 - (-(2.0 - lmbda) * z[neg] + 1.0) ** (1.0 / (2.0 - lmbda))
        else:
            out[neg] = -np.expm1(-z[neg])

        return out

    @staticmethod
    def _yj_neg_loglik(lmbda: float, x_obs: np.ndarray) -> float:
        """Negative profile log-likelihood for Yeo-Johnson lambda."""
        n = x_obs.size
        if n == 0:
            return 0.0
        y = MissingAwarePowerTransformer._yj_transform(x_obs, lmbda)
        var = float(np.var(y))
        if var <= 0:
            return 1e18
        loglik = -0.5 * n * np.log(var)
        loglik += (lmbda - 1.0) * np.sum(np.sign(x_obs) * np.log1p(np.abs(x_obs)))
        return float(-loglik)

    def fit(
        self, x: np.ndarray, mask: Mask | None = None
    ) -> MissingAwarePowerTransformer:
        """Estimate per-column Yeo-Johnson lambda by profile MLE."""
        from scipy.optimize import minimize_scalar  # noqa: PLC0415

        x_arr = np.asarray(x, dtype=float)
        obs = _ensure_mask(x_arr, mask)
        n_features = x_arr.shape[1]
        self.n_features_in_ = n_features
        lambdas = np.ones(n_features)

        for j in range(n_features):
            col_obs = x_arr[obs[:, j], j]
            if col_obs.size < 2:
                continue
            res = minimize_scalar(
                self._yj_neg_loglik,
                bounds=(-2.0, 5.0),
                args=(col_obs,),
                method="bounded",
            )
            lambdas[j] = float(res.x)

        self.lambdas_ = lambdas

        # Compute standardisation stats on the transformed data.
        transformed = self._apply_yj(x_arr, obs)
        means = np.nanmean(transformed, axis=0)
        scales = np.nanstd(transformed, axis=0)
        scales[~np.isfinite(scales) | np.isclose(scales, 0.0)] = 1.0
        self.mean_ = means
        self.scale_ = scales
        return self

    def _apply_yj(self, x: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Apply per-column Yeo-Johnson, preserving NaN."""
        if self.lambdas_ is None:
            raise RuntimeError("Transformer not fitted")
        out = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            col_obs = obs[:, j]
            out[col_obs, j] = self._yj_transform(x[col_obs, j], self.lambdas_[j])
        return out

    def transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        """Apply Yeo-Johnson transform and optional standardisation."""
        if self.lambdas_ is None:
            raise RuntimeError("Transformer not fitted")
        x_arr = np.asarray(x, dtype=float)
        obs = _ensure_mask(x_arr, mask)
        out = self._apply_yj(x_arr, obs)
        if self.standardize and self.mean_ is not None and self.scale_ is not None:
            out = (out - self.mean_) / self.scale_
            out[~obs] = np.nan
        return out

    def fit_transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        return self.fit(x, mask).transform(x, mask)

    def inverse_transform(self, z: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        """Reverse transform: un-standardise then invert Yeo-Johnson."""
        if self.lambdas_ is None:
            raise RuntimeError("Transformer not fitted")
        z_arr = np.asarray(z, dtype=float)
        obs = _ensure_mask(z_arr, mask)
        out = z_arr.copy()
        if self.standardize and self.mean_ is not None and self.scale_ is not None:
            out = out * self.scale_ + self.mean_
        result = np.full_like(out, np.nan)
        for j in range(out.shape[1]):
            col_obs = obs[:, j]
            result[col_obs, j] = self._yj_inverse(out[col_obs, j], self.lambdas_[j])
        return result


# ── Data diagnostics (#81) ───────────────────────────────────────────────


@dataclass
class DataReport:
    """Result of :func:`check_data` preflight validation.

    Attributes:
        warnings: Human-readable diagnostic messages.
        summary: Per-feature statistics dictionary.
        suggested_pretransforms: Mapping of column index (or name) to a
            suggested transform string (e.g. ``"log1p"``).
        passed: ``True`` when no warnings were raised.
    """

    warnings: list[str] = field(default_factory=list)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)
    suggested_pretransforms: dict[str | int, str] = field(default_factory=dict)
    passed: bool = True


@dataclass
class _CheckDataConfig:
    """Internal container for :func:`check_data` thresholds."""

    skewness_threshold: float
    outlier_mad_threshold: float
    near_zero_var_eps: float
    missing_fraction_warn: float
    n_samples: int
    emit_warnings: bool


def _emit(report: DataReport, msg: str, *, emit: bool) -> None:
    """Append *msg* to *report* and optionally emit a stdlib warning."""
    report.warnings.append(msg)
    if emit:
        _warnings_mod.warn(msg, UserWarning, stacklevel=3)


def _check_missing(
    obs: np.ndarray,
    name: str,
    col_summary: dict[str, float],
    cfg: _CheckDataConfig,
    report: DataReport,
) -> None:
    frac = 1.0 - obs.size / max(cfg.n_samples, 1)
    col_summary["missing_fraction"] = frac
    if frac >= cfg.missing_fraction_warn:
        _emit(
            report,
            f"{name}: {frac:.0%} missing"
            " — high missingness degrades covariance estimates",
            emit=cfg.emit_warnings,
        )


def _check_variance(
    obs: np.ndarray,
    name: str,
    col_summary: dict[str, float],
    cfg: _CheckDataConfig,
    report: DataReport,
) -> bool:
    """Return ``True`` if the column has meaningful variance."""
    variance = float(np.var(obs))
    col_summary["variance"] = variance
    if variance < cfg.near_zero_var_eps:
        _emit(
            report,
            f"{name}: near-zero variance ({variance:.2e})"
            " — feature carries no information",
            emit=cfg.emit_warnings,
        )
        return False
    return True


def _check_skewness(
    obs: np.ndarray,
    name: str,
    col_summary: dict[str, float],
    cfg: _CheckDataConfig,
    report: DataReport,
) -> None:
    std = float(np.std(obs))
    skew = float(np.mean(((obs - np.mean(obs)) / std) ** 3)) if std > 0 else 0.0
    col_summary["skewness"] = skew
    if abs(skew) > cfg.skewness_threshold:
        _emit(
            report,
            f"{name}: skewness={skew:.1f}"
            " — consider log or power transform before scaling",
            emit=cfg.emit_warnings,
        )
        if skew > 0 and float(np.min(obs)) >= 0:
            report.suggested_pretransforms[name] = "log1p"
        else:
            report.suggested_pretransforms[name] = "power"


def _check_outliers(
    obs: np.ndarray,
    name: str,
    col_summary: dict[str, float],
    cfg: _CheckDataConfig,
    report: DataReport,
) -> None:
    median = float(np.median(obs))
    mad = float(np.median(np.abs(obs - median)))
    if mad <= 0:
        return
    z_scores = np.abs(obs - median) / mad
    outlier_frac = float(np.mean(z_scores > cfg.outlier_mad_threshold))
    col_summary["outlier_fraction"] = outlier_frac
    if outlier_frac > 0.01:
        _emit(
            report,
            f"{name}: {outlier_frac:.1%} of entries are outliers"
            f" (>{cfg.outlier_mad_threshold} MAD)"
            " — consider winsorization before scaling",
            emit=cfg.emit_warnings,
        )
        if name not in report.suggested_pretransforms:
            report.suggested_pretransforms[name] = "winsorize"


def check_data(  # noqa: PLR0913
    x: np.ndarray,
    *,
    column_names: Sequence[str] | None = None,
    skewness_threshold: float = 2.0,
    outlier_mad_threshold: float = 5.0,
    near_zero_var_eps: float = 1e-10,
    missing_fraction_warn: float = 0.5,
    warn: bool = False,
) -> DataReport:
    """Run preflight diagnostics on a data matrix before VBPCA fitting.

    Checks focus on *scale comparability* — conditions that cause
    individual features to dominate the decomposition — rather than
    distributional shape.

    Args:
        x: Data matrix of shape ``(n_samples, n_features)``.
        column_names: Optional feature names for readable messages.
        skewness_threshold: Absolute skewness above which a feature is
            flagged (default ``2.0``).
        outlier_mad_threshold: Number of MADs from the median beyond
            which an entry is considered an outlier (default ``5.0``).
        near_zero_var_eps: Variance threshold below which a feature is
            flagged as near-zero-variance (default ``1e-10``).
        missing_fraction_warn: Per-feature missing fraction above which
            a warning is emitted (default ``0.5``).
        warn: If ``True``, also emit :func:`warnings.warn` for each
            issue.

    Returns:
        A :class:`DataReport` with warnings, per-feature summary, and
        suggested pre-transforms.
    """
    x_arr = np.asarray(x, dtype=float)
    n_samples, n_features = x_arr.shape
    report = DataReport()
    cfg = _CheckDataConfig(
        skewness_threshold=skewness_threshold,
        outlier_mad_threshold=outlier_mad_threshold,
        near_zero_var_eps=near_zero_var_eps,
        missing_fraction_warn=missing_fraction_warn,
        n_samples=n_samples,
        emit_warnings=warn,
    )

    for j in range(n_features):
        col = x_arr[:, j]
        name = column_names[j] if column_names is not None else str(j)
        obs = col[~np.isnan(col)]
        col_summary: dict[str, float] = {"n_obs": float(obs.size)}

        _check_missing(obs, name, col_summary, cfg, report)

        if obs.size < 2:
            col_summary["variance"] = 0.0
            report.summary[name] = col_summary
            continue

        if not _check_variance(obs, name, col_summary, cfg, report):
            report.summary[name] = col_summary
            continue

        _check_skewness(obs, name, col_summary, cfg, report)
        _check_outliers(obs, name, col_summary, cfg, report)
        report.summary[name] = col_summary

    report.passed = len(report.warnings) == 0
    return report
