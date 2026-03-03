"""Missing-aware preprocessing utilities with scikit-learn-like APIs."""

from __future__ import annotations

# ruff: noqa: D102, D107, TRY003, EM101, EM102, TC003
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import scipy.sparse as sp

from ._sparsity import validate_mask_compatibility

__all__ = [
    "AutoEncoder",
    "MissingAwareMinMaxScaler",
    "MissingAwareOneHotEncoder",
    "MissingAwareSparseOneHotEncoder",
    "MissingAwareStandardScaler",
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
            return ~np.isnan(x.astype(float))
        return ~np.not_equal(x, x)
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

    def fit(self, x: np.ndarray, mask: Mask | None = None) -> MissingAwareOneHotEncoder:
        x_arr = np.asarray(x)
        obs_mask = _ensure_mask(x_arr, mask)
        n_features = x_arr.shape[1]
        self.categories_ = []
        self.feature_names_out_ = []
        means: list[float] = []

        for j in range(n_features):
            col_obs = obs_mask[:, j]
            observed_vals = x_arr[col_obs, j]
            cats = _safe_unique(observed_vals)
            self.categories_.append(cats)
            for c in cats:
                self.feature_names_out_.append(f"col{j}_{c}")
            means.extend([0.0] * len(cats))

        self.column_means_ = np.array(means, dtype=self.dtype)
        self.n_features_in_ = n_features
        return self

    def _transform_single(
        self, x: np.ndarray, mask: Mask, j: int
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        cats = self.categories_[j]
        n_rows = x.shape[0]
        n_cats = len(cats)
        out: np.ndarray = np.zeros((n_rows, n_cats), dtype=self.dtype)
        col_mask = np.asarray(mask[:, j], dtype=bool)
        col_vals = x[:, j]

        if n_cats == 0:
            return out, [], np.zeros(0, dtype=self.dtype)

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
            if n_cats == 0:
                out_cols.append(np.full(z_arr.shape[0], np.nan))
                continue
            block = z_arr[:, col_start : col_start + n_cats]
            if self.mean_center and self.column_means_ is not None:
                means = self.column_means_[mean_idx : mean_idx + n_cats]
                block += means
            col_vals = np.full(z_arr.shape[0], np.nan, dtype=object)
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
            out_cols.append(col_vals.astype(object))
            col_start += n_cats
            mean_idx += n_cats
        return np.column_stack(out_cols)


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
            return self

        # Require numeric categories to round-trip through sparse matrices.
        cats = np.unique(np.asarray(x_csr.data, dtype=float))
        self.categories_ = [float(c) for c in cats.tolist()]
        self.feature_names_out_ = [f"col0_{c}" for c in self.categories_]

        n_rows = x_csr.shape[0]
        counts = np.zeros(len(self.categories_), dtype=float)
        cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories_)}
        for val in x_csr.data:
            counts[cat_to_idx[float(val)]] += 1.0
        means = counts / float(max(n_rows, 1))
        self.column_means_ = means.astype(self.dtype, copy=False)
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

        cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories_)}
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
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
            shape=(n_rows, n_cats),
        )
        if self.mean_center and self.column_means_ is not None and out.data.size > 0:
            out.data -= self.column_means_[out.indices]
        return out

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
        if n_cols != len(self.categories_):
            msg = "Input width must match fitted categories"
            raise ValueError(msg)

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
        return z

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
        return x


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
        return z

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
        return x


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
            return sp.hstack(parts_sparse, format="csr")

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
