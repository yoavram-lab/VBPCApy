"""Missing-aware preprocessing utilities with scikit-learn-like APIs."""

from __future__ import annotations

# ruff: noqa: D102, D107, TRY003, EM101, EM102, B904, TC003
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

__all__ = [
    "AutoEncoder",
    "MissingAwareMinMaxScaler",
    "MissingAwareOneHotEncoder",
    "MissingAwareStandardScaler",
]

Mask = np.ndarray


def _ensure_mask(x: np.ndarray, mask: Mask | None) -> Mask:
    """Return a boolean mask where True marks observed entries."""
    if mask is None:
        if np.issubdtype(x.dtype, np.number):
            return ~np.isnan(x.astype(float))
        return ~np.not_equal(x, x)
    return np.asarray(mask, dtype=bool)


def _safe_unique(values: np.ndarray) -> list[Any]:
    """Unique values preserving order for hashable scalars.

    Returns:
        Ordered list of unique values.
    """
    seen: set[Any] = set()
    uniq: list[Any] = []
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
            # Placeholder; filled after transform to keep length consistent.
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
        col_mask = mask[:, j]
        col_vals = x[:, j]

        if n_cats == 0:
            return out, [], np.zeros(0, dtype=self.dtype)

        for i in range(n_rows):
            if not col_mask[i]:
                out[i, :] = np.nan
                continue
            val = col_vals[i]
            try:
                idx = cats.index(val)
            except ValueError:
                if self.handle_unknown == "raise":
                    raise ValueError(f"Unknown category {val!r} in column {j}")
                # ignore -> leave zeros
                continue
            out[i, idx] = 1.0

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

    def fit_transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
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
            else:
                means = np.zeros(n_cats, dtype=self.dtype)
            # Missing rows -> NaN
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

    def fit(
        self, x: np.ndarray, mask: Mask | None = None
    ) -> _BaseScaler:  # pragma: no cover - abstract
        raise NotImplementedError

    def transform(
        self, x: np.ndarray, mask: Mask | None = None
    ) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def inverse_transform(
        self, z: np.ndarray, mask: Mask | None = None
    ) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit_transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        return self.fit(x, mask).transform(x, mask)


class MissingAwareStandardScaler(_BaseScaler):
    """Standardize continuous columns ignoring missing entries."""

    def __init__(self) -> None:
        super().__init__()
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.var_: np.ndarray | None = None

    def fit(
        self, x: np.ndarray, mask: Mask | None = None
    ) -> MissingAwareStandardScaler:
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
        scales[scales == 0.0] = 1.0
        self.mean_ = means
        self.var_ = vars_
        self.scale_ = scales
        return self

    def transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fitted")
        x_arr = np.asarray(x, dtype=float)
        obs_mask = _ensure_mask(x_arr, mask)
        z = (x_arr - self.mean_) / self.scale_
        z[~obs_mask] = np.nan
        return z

    def inverse_transform(self, z: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fitted")
        z_arr = np.asarray(z, dtype=float)
        obs_mask = _ensure_mask(z_arr, mask)
        x = z_arr * self.scale_ + self.mean_
        x[~obs_mask] = np.nan
        return x


class MissingAwareMinMaxScaler(_BaseScaler):
    """Min-max scale continuous columns ignoring missing entries."""

    def __init__(self) -> None:
        super().__init__()
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.data_range_: np.ndarray | None = None

    def fit(self, x: np.ndarray, mask: Mask | None = None) -> MissingAwareMinMaxScaler:
        x_arr = np.asarray(x, dtype=float)
        obs_mask = _ensure_mask(x_arr, mask)
        self.n_features_in_ = x_arr.shape[1]
        mins = np.zeros(self.n_features_in_)
        maxs = np.zeros(self.n_features_in_)
        for j in range(self.n_features_in_):
            col_obs = obs_mask[:, j]
            if not np.any(col_obs):
                mins[j] = np.nan
                maxs[j] = np.nan
                continue
            vals = x_arr[col_obs, j]
            mins[j] = float(np.min(vals))
            maxs[j] = float(np.max(vals))
        ranges = maxs - mins
        ranges[~np.isfinite(ranges)] = 1.0
        ranges[ranges == 0.0] = 1.0
        self.data_min_ = mins
        self.data_max_ = maxs
        self.data_range_ = ranges
        return self

    def transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        if self.data_min_ is None or self.data_range_ is None:
            raise RuntimeError("Scaler not fitted")
        x_arr = np.asarray(x, dtype=float)
        obs_mask = _ensure_mask(x_arr, mask)
        z = (x_arr - self.data_min_) / self.data_range_
        z[~obs_mask] = np.nan
        return z

    def inverse_transform(self, z: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        if self.data_min_ is None or self.data_range_ is None:
            raise RuntimeError("Scaler not fitted")
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

    def fit(self, x: np.ndarray, mask: Mask | None = None) -> AutoEncoder:
        x_arr = np.asarray(x)
        obs_mask = _ensure_mask(x_arr, mask)
        self.n_features_in_ = x_arr.shape[1]
        self._plan = []
        self.feature_names_out_ = []
        col_start = 0
        for j in range(self.n_features_in_):
            kind = self._infer_kind(x_arr[:, j], obs_mask[:, j], j)
            enc: MissingAwareOneHotEncoder | _BaseScaler
            if kind == "categorical":
                enc = MissingAwareOneHotEncoder(
                    handle_unknown=self.handle_unknown,
                    mean_center=self.mean_center_ohe,
                )
            else:
                enc = (
                    MissingAwareStandardScaler()
                    if self.continuous_scaler == "standard"
                    else MissingAwareMinMaxScaler()
                )
            enc.fit(x_arr[:, [j]], mask=obs_mask[:, [j]])
            # Determine block width after transform
            z = enc.transform(x_arr[:, [j]], mask=obs_mask[:, [j]])
            width = z.shape[1]
            self.feature_names_out_.extend(
                getattr(enc, "feature_names_out_", [f"col{j}"] * width)
            )
            self._plan.append(
                _ColumnPlan(
                    kind=kind,
                    encoder=enc,
                    slice_start=col_start,
                    slice_end=col_start + width,
                )
            )
            col_start += width
        return self

    def transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        if self.n_features_in_ is None:
            raise RuntimeError("AutoEncoder not fitted")
        x_arr = np.asarray(x)
        obs_mask = _ensure_mask(x_arr, mask)
        parts: list[np.ndarray] = []
        for col_idx, plan in enumerate(self._plan):
            z: np.ndarray = plan.encoder.transform(
                x_arr[:, [col_idx]], mask=obs_mask[:, [col_idx]]
            )
            parts.append(z)
        return np.concatenate(parts, axis=1) if parts else np.empty((x_arr.shape[0], 0))

    def fit_transform(self, x: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        return self.fit(x, mask).transform(x, mask)

    def inverse_transform(self, z: np.ndarray, mask: Mask | None = None) -> np.ndarray:
        del mask
        if self.n_features_in_ is None:
            raise RuntimeError("AutoEncoder not fitted")
        z_arr = np.asarray(z)
        out_cols: list[np.ndarray] = []
        for _plan_idx, plan in enumerate(self._plan):
            block = z_arr[:, plan.slice_start : plan.slice_end]
            inv = plan.encoder.inverse_transform(block, mask=None)
            out_cols.append(inv[:, 0])
        return np.column_stack(out_cols)
