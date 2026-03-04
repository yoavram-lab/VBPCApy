"""Helper functions for full VB-PCA update.

This module factors out pieces of the :func:`pca_full` implementation
to keep the main loop readable while preserving the original algorithmic
behaviour. The helpers are intentionally small and focused; shared
state is carried via simple data classes rather than long parameter
lists so that static analysis tools remain happy.
"""

from __future__ import annotations

import logging
from collections.abc import MutableSequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve

from ._mean import Matrix, ProbeMatrices, subtract_mu
from ._missing import _missing_patterns
from ._monitoring import InitResult, InitShapes, init_params, log_progress
from ._remove_empty import remove_empty_entries
from ._rms import RmsConfig, compute_rms
from ._rotate import RotateParams, rotate_to_pca
from .dense_update_kernels import (
    loadings_update_dense_masked_nopattern as _loadings_update_dense_masked_ext,
)
from .dense_update_kernels import (
    loadings_update_dense_no_sv as _loadings_update_dense_ext,
)
from .dense_update_kernels import (
    score_update_dense_masked_nopattern as _score_update_dense_masked_ext,
)
from .dense_update_kernels import score_update_dense_no_av as _score_update_dense_ext
from .noise_update_kernels import noise_sxv_sum as _noise_sxv_sum_ext
from .sparse_update_kernels import (
    loadings_update_sparse_nopattern as _loadings_update_sparse_ext,
)
from .sparse_update_kernels import (
    score_update_sparse_nopattern as _score_update_sparse_ext,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping

logger = logging.getLogger(__name__)

_EPS_VAR = 1e-15  # minimum variance for numerical safety
ChoFactor = tuple[np.ndarray, bool]
CovarianceStore = MutableSequence[np.ndarray] | np.ndarray


def _has_covariances(covariances: CovarianceStore) -> bool:
    return len(covariances) > 0


def _covariance_at(covariances: CovarianceStore, idx: int) -> np.ndarray:
    return np.asarray(covariances[idx], dtype=float)


def _covariances_stack(covariances: CovarianceStore) -> np.ndarray:
    if isinstance(covariances, np.ndarray):
        if covariances.ndim != 3:
            msg = "covariances array must be 3-D when provided as ndarray"
            raise ValueError(msg)
        return np.asarray(covariances, dtype=np.float64, copy=False)

    return np.stack(covariances, axis=0).astype(np.float64, copy=False)


def _covariances_list(covariances: CovarianceStore) -> list[np.ndarray]:
    if isinstance(covariances, np.ndarray):
        if covariances.ndim != 3:
            msg = "covariances array must be 3-D when provided as ndarray"
            raise ValueError(msg)
        return [covariances[i, :, :] for i in range(covariances.shape[0])]

    return [np.asarray(cov, dtype=float) for cov in covariances]


def _progress_should_log(verbose: int, stride: int, current: int, total: int) -> bool:
    if verbose <= 0:
        return False
    effective_stride = max(1, stride)
    return (current % effective_stride) == 0 or current >= total


def _log_progress_if_needed(
    verbose: int,
    stride: int,
    current: int,
    total: int,
    *,
    phase: str,
) -> None:
    if not _progress_should_log(verbose, stride, current, total):
        return
    log_progress(verbose, current=current, total=total, phase=phase)


# ---------------------------------------------------------------------------
# Small state / context containers
# ---------------------------------------------------------------------------


@dataclass
class BiasState:
    """State required for updating the bias/mean parameter."""

    mu: np.ndarray
    muv: np.ndarray
    noise_var: float
    vmu: float
    n_obs_row: np.ndarray


@dataclass
class CenteringState:
    """State needed when (re)centering the data and probe matrices."""

    x_data: Matrix
    x_probe: Matrix | None
    mask: Matrix
    mask_probe: Matrix | None


@dataclass
class ScoreState:
    """State required for updating scores and their covariances."""

    x_data: Matrix
    mask: Matrix
    loadings: np.ndarray  # A
    scores: np.ndarray  # S
    loading_covariances: CovarianceStore  # Av
    score_covariances: CovarianceStore  # Sv
    pattern_index: np.ndarray | None
    obs_patterns: list[list[int]]
    noise_var: float
    eye_components: np.ndarray
    verbose: int
    pattern_batch_size: int = 0
    sparse_num_cpu: int = 0
    dense_num_cpu: int = 0
    x_csr: sp.csr_matrix | None = None
    x_csc: sp.csc_matrix | None = None
    use_python_scores: bool = False
    cov_writeback_mode: str = "python"
    log_progress_stride: int = 1
    accessor_mode: str = "legacy"


@dataclass
class NoiseState:
    """State required for updating the noise variance."""

    loadings: np.ndarray
    scores: np.ndarray
    loading_covariances: CovarianceStore
    score_covariances: CovarianceStore
    mu_variances: np.ndarray
    pattern_index: np.ndarray | None
    n_data: float
    noise_var: float
    sparse_num_cpu: int = 0


@dataclass
class InitContext:
    """Context needed for initial parameter setup and centering."""

    x_data: Matrix
    x_probe: Matrix | None
    mask: Matrix
    mask_probe: Matrix | None
    shapes: InitShapes
    pattern_index: np.ndarray | None
    n_obs_row: np.ndarray
    use_prior: bool
    use_postvar: bool
    opts: MutableMapping[str, object]


@dataclass
class HyperpriorContext:
    """Context for updating hyperpriors Va and Vmu."""

    iteration: int
    use_prior: bool
    niter_broadprior: int
    bias_enabled: bool
    mu: np.ndarray
    mu_variances: np.ndarray
    loadings: np.ndarray
    loading_covariances: CovarianceStore
    n_features: int
    hp_va: float
    hp_vb: float
    va: np.ndarray
    vmu: float


@dataclass
class LoadingsUpdateState:
    """State for updating loadings and their covariances."""

    x_data: Matrix
    mask: Matrix
    scores: np.ndarray
    loading_covariances: CovarianceStore
    score_covariances: CovarianceStore
    pattern_index: np.ndarray | None
    va: np.ndarray
    noise_var: float
    verbose: int
    sparse_num_cpu: int = 0
    dense_num_cpu: int = 0
    x_csr: sp.csr_matrix | None = None
    x_csc: sp.csc_matrix | None = None
    cov_writeback_mode: str = "python"
    log_progress_stride: int = 1
    accessor_mode: str = "legacy"


@dataclass
class RmsContext:
    """Context for recomputing RMS and (optional) probe RMS."""

    x_data: Matrix
    x_probe: Matrix | None
    mask: Matrix
    mask_probe: Matrix | None
    n_data: float
    n_probe: int
    loadings: np.ndarray
    scores: np.ndarray
    num_cpu: int


@dataclass
class RotationContext:
    """Context for final PCA-style rotation."""

    loadings: np.ndarray
    loading_covariances: CovarianceStore
    scores: np.ndarray
    score_covariances: CovarianceStore
    mu: np.ndarray
    pattern_index: np.ndarray | None
    obs_patterns: list[list[int]]
    bias_enabled: bool


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _prepare_data(
    x: Matrix,
    opts: MutableMapping[str, object],
    *,
    mask_override: Matrix | None = None,
) -> tuple[Matrix, Matrix | None, Matrix | None, int, int, np.ndarray, np.ndarray]:
    """Copy input, apply probe handling, and remove empty rows/cols.

    Returns:
        Data, probe data, original shapes, and kept row/col indices.

    Raises:
        ValueError: If ``x_probe`` sparsity does not match ``x``.
    """
    x_probe_opt = opts.get("xprobe", None)

    if sp.issparse(x):
        x_data: Matrix = sp.csr_matrix(cast("Any", x))
    else:
        x_data = np.array(x, dtype=float).copy()

    x_probe: Matrix | None = None
    if x_probe_opt is not None:
        x_probe_array = np.asarray(x_probe_opt, dtype=float)
        if x_probe_array.size != 0:
            if sp.issparse(x_probe_opt):
                if not sp.issparse(x_data):
                    msg = "x_probe must be dense when x is dense"
                    raise ValueError(msg)
                x_probe = sp.csr_matrix(cast("Any", x_probe_opt))
            else:
                if sp.issparse(x_data):
                    msg = "x_probe must be sparse when x is sparse"
                    raise ValueError(msg)
                x_probe = x_probe_array.copy()

    n_features_original, n_samples_original = x_data.shape

    mask_clean: Matrix | None = None
    if mask_override is not None:
        mask_clean = (
            mask_override
            if sp.isspmatrix(mask_override)
            else np.asarray(mask_override, dtype=bool)
        )

    x_data, x_probe, row_idx, col_idx, init_opt = remove_empty_entries(
        x_data,
        x_probe,
        opts["init"],
        compat_mode=str(opts.get("compat_mode", "strict_legacy")),
        mask=mask_clean,
    )
    # Carry updated init option forward.
    opts["init"] = init_opt

    if mask_clean is not None:
        if sp.isspmatrix(mask_clean):
            mask_clean = sp.csr_matrix(mask_clean)[row_idx[:, None], col_idx]
        else:
            mask_clean = np.asarray(mask_clean)[np.ix_(row_idx, col_idx)]

    return (
        x_data,
        x_probe,
        mask_clean,
        n_features_original,
        n_samples_original,
        row_idx,
        col_idx,
    )


# -- mask helpers -------------------------------------------------------------


def _normalize_sparse_data(x_csr: sp.csr_matrix, compat_mode: str) -> sp.csr_matrix:
    """Apply strict_legacy normalization in-place on sparse data without densifying.

    Returns:
        Normalized CSR matrix with stored zeros mapped to ``eps`` (strict_legacy)
        and stored NaNs removed.
    """
    x_csr = sp.csr_matrix(x_csr, copy=True)

    data = x_csr.data
    use_eps_for_zeros = compat_mode.strip().lower() == "strict_legacy"
    if use_eps_for_zeros:
        zero_mask = np.isclose(data, 0.0)
        data[zero_mask] = np.finfo(float).eps

    nan_mask = np.isnan(data)
    if nan_mask.any():
        data[nan_mask] = 0.0
        x_csr.eliminate_zeros()
    return x_csr


def _build_masks_sparse(
    x_data: Matrix,
    x_probe: Matrix | None,
    compat_mode: str,
) -> tuple[Matrix, Matrix | None, Matrix, Matrix | None]:
    """Build masks and normalized data for sparse inputs.

    Returns:
        Tuple of data, probe, mask, and probe mask.
    """
    x_csr = _normalize_sparse_data(sp.csr_matrix(x_data), compat_mode)
    mask_csr = x_csr.copy()
    mask_csr.data[:] = 1.0
    mask: Matrix = mask_csr

    if x_probe is not None:
        if sp.issparse(x_probe):
            x_probe_csr = _normalize_sparse_data(sp.csr_matrix(x_probe), compat_mode)
            mask_probe_csr = x_probe_csr.copy()
            mask_probe_csr.data[:] = 1.0
            mask_probe = mask_probe_csr
            x_probe = x_probe_csr
        else:
            x_probe_arr = np.asarray(x_probe, dtype=float)
            x_probe = x_probe_arr
            mask_probe = (x_probe_arr != 0).astype(float)
    else:
        mask_probe = None

    return x_csr, x_probe, mask, mask_probe


def _build_masks_dense(
    x_data: Matrix,
    x_probe: Matrix | None,
    compat_mode: str,
) -> tuple[Matrix, Matrix | None, Matrix, Matrix | None]:
    """Build masks and normalized data for dense inputs.

    Returns:
        Tuple of data, probe, mask, and probe mask.
    """
    x_arr = np.asarray(x_data, dtype=float)
    mask = ~np.isnan(x_arr)

    mode = compat_mode.strip().lower()
    use_eps_for_zeros = mode == "strict_legacy"
    eps = np.finfo(float).eps
    if use_eps_for_zeros:
        zero_mask = np.isclose(x_arr, 0.0)
        x_arr[zero_mask] = eps
    x_arr[np.isnan(x_arr)] = 0.0
    x_data = x_arr

    if x_probe is not None:
        x_probe_arr = np.asarray(x_probe, dtype=float)
        x_probe = x_probe_arr
        mask_probe = ~np.isnan(x_probe_arr)
        if use_eps_for_zeros:
            zero_mask_probe = np.isclose(x_probe_arr, 0.0)
            x_probe_arr[zero_mask_probe] = eps
        x_probe_arr[np.isnan(x_probe_arr)] = 0.0
    else:
        mask_probe = None

    return x_data, x_probe, mask, mask_probe


def _prepare_sparse_with_optional_mask(
    x_data: Matrix,
    x_probe: Matrix | None,
    mask_override: Matrix | None,
    compat_mode: str,
) -> tuple[Matrix, Matrix | None, Matrix, Matrix | None]:
    if x_probe is not None and not sp.issparse(x_probe):
        msg = "x_probe must be sparse when x is sparse"
        raise ValueError(msg)

    if mask_override is not None:
        mask = sp.csr_matrix(mask_override)
        x_data_out = _normalize_sparse_data(sp.csr_matrix(x_data), compat_mode)
        x_probe_out = (
            _normalize_sparse_data(sp.csr_matrix(x_probe), compat_mode)
            if x_probe is not None and sp.issparse(x_probe)
            else x_probe
        )
        return x_data_out, x_probe_out, mask, None

    return _build_masks_sparse(x_data, x_probe, compat_mode)


def _prepare_dense_with_mask_override(
    x_data: Matrix,
    x_probe: Matrix | None,
    mask_override: Matrix,
    compat_mode: str,
) -> tuple[Matrix, Matrix | None, Matrix, Matrix | None]:
    if sp.issparse(mask_override):
        msg = "mask must be dense when x is dense"
        raise ValueError(msg)

    mask_arr = np.asarray(mask_override, dtype=bool)
    x_arr = np.asarray(x_data, dtype=float)
    eps = np.finfo(float).eps if compat_mode == "strict_legacy" else 0.0
    zero_mask = np.isclose(x_arr, 0.0)
    x_arr[zero_mask] = eps
    x_arr[np.isnan(x_arr)] = 0.0

    x_probe_out: Matrix | None = None
    mask_probe: Matrix | None = None
    if x_probe is not None:
        x_probe_arr = np.asarray(x_probe, dtype=float)
        zero_mask_probe = np.isclose(x_probe_arr, 0.0)
        x_probe_arr[zero_mask_probe] = eps
        x_probe_arr[np.isnan(x_probe_arr)] = 0.0
        x_probe_out = x_probe_arr

    return x_arr, x_probe_out, mask_arr, mask_probe


def _build_masks_and_counts(
    x_data: Matrix,
    x_probe: Matrix | None,
    opts: MutableMapping[str, object],
    *,
    mask_override: Matrix | None = None,
) -> tuple[Matrix, Matrix | None, Matrix, Matrix | None, np.ndarray, float, int]:
    """Build missingness masks, handle NaNs/zeros, and count observations.

    Returns:
        Data, probe, masks, per-row counts, total data count, probe count.
    """
    mask: Matrix
    mask_probe: Matrix | None
    compat_mode = str(opts.get("compat_mode", "strict_legacy")).strip().lower()

    if sp.issparse(x_data):
        x_data, x_probe, mask, mask_probe = _prepare_sparse_with_optional_mask(
            x_data, x_probe, mask_override, compat_mode
        )
    elif mask_override is not None:
        x_data, x_probe, mask, mask_probe = _prepare_dense_with_mask_override(
            x_data, x_probe, mask_override, compat_mode
        )
    else:
        x_data, x_probe, mask, mask_probe = _build_masks_dense(
            x_data,
            x_probe,
            str(opts.get("compat_mode", "strict_legacy")),
        )

    if sp.issparse(mask):
        n_obs_row = np.array(mask.sum(axis=1)).ravel()
    else:
        mask_arr = np.asarray(mask)
        n_obs_row = np.sum(mask_arr, axis=1)
    n_data = float(np.sum(n_obs_row))

    if mask_probe is None:
        n_probe = 0
    elif sp.issparse(mask_probe):
        mask_probe_csr = sp.csr_matrix(mask_probe)
        mask_probe = mask_probe_csr
        n_probe = int(mask_probe_csr.count_nonzero())
    else:
        mask_probe_arr = np.asarray(mask_probe)
        mask_probe = mask_probe_arr
        n_probe = int(np.count_nonzero(mask_probe_arr))

    if n_probe == 0:
        x_probe = None
        mask_probe = None
        # Turn off early stopping if there is no probe set.
        opts["earlystop"] = 0

    return x_data, x_probe, mask, mask_probe, n_obs_row, n_data, n_probe


def _observed_indices(x_data: Matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return indices of observed entries in ``x_data``."""
    if isinstance(x_data, sp.csr_matrix):
        i_idx_sparse, j_idx_sparse = x_data.nonzero()
        return np.asarray(i_idx_sparse, dtype=np.intp), np.asarray(
            j_idx_sparse, dtype=np.intp
        )

    x_arr = np.asarray(x_data)
    i_idx_dense, j_idx_dense = np.nonzero(x_arr)
    return np.asarray(i_idx_dense, dtype=np.intp), np.asarray(
        j_idx_dense, dtype=np.intp
    )


def _observed_indices_with_mode(
    x_data: Matrix,
    mask: Matrix,
    compat_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed indices according to compatibility mode.

    In ``strict_legacy`` mode this matches historical nonzero semantics.
    In ``modern`` mode for dense inputs this uses the explicit observation
    mask, decoupling observedness from numeric normalization conventions.
    """
    mode = compat_mode.strip().lower()
    if mode == "modern" and not sp.issparse(x_data):
        mask_arr = np.asarray(mask, dtype=bool)
        i_idx_dense, j_idx_dense = np.nonzero(mask_arr)
        return np.asarray(i_idx_dense, dtype=np.intp), np.asarray(
            j_idx_dense, dtype=np.intp
        )
    return _observed_indices(x_data)


def _patterns_from_csc(
    mask_csc: sp.csc_matrix,
) -> tuple[int, list[list[int]], np.ndarray]:
    """Build missingness patterns from a CSC mask without densifying.

    Returns:
        Tuple of pattern count, pattern row lists, and column-to-pattern map.
    """
    pattern_to_id: dict[tuple[int, ...], int] = {}
    pattern_index = np.empty(mask_csc.shape[1], dtype=int)

    for col in range(mask_csc.shape[1]):
        start, end = mask_csc.indptr[col : col + 2]
        rows_tuple = tuple(mask_csc.indices[start:end].tolist())
        pid = pattern_to_id.get(rows_tuple)
        if pid is None:
            pid = len(pattern_to_id)
            pattern_to_id[rows_tuple] = pid
        pattern_index[col] = pid

    obs_patterns = [list(pattern) for pattern in pattern_to_id]
    return len(obs_patterns), obs_patterns, pattern_index


def _missing_patterns_info(
    mask: Matrix,
    opts: MutableMapping[str, object],
    n_samples: int,
) -> tuple[int, list[list[int]], np.ndarray | None]:
    """Compute missingness-pattern information and Isv mapping.

    Returns:
        Number of patterns, pattern lists, and optional pattern index map.
    """
    auto_patterns = bool(opts.get("auto_pattern_masked", 0))
    if opts.get("uniquesv") or auto_patterns:
        if isinstance(mask, sp.csr_matrix):
            n_patterns, obs_patterns, pattern_index = _patterns_from_csc(mask.tocsc())
            if not (auto_patterns and n_patterns == 1):
                return n_patterns, obs_patterns, pattern_index
            auto_patterns = False
        else:
            mask_arr = np.asarray(mask, dtype=bool)
            if auto_patterns and np.all(mask_arr):
                auto_patterns = False
            if opts.get("uniquesv") or auto_patterns:
                n_patterns, obs_patterns, isv = _missing_patterns(mask_arr)
                return int(n_patterns), obs_patterns, np.array(isv, dtype=int)

    n_patterns_default = n_samples
    obs_patterns_default: list[list[int]] = []
    return n_patterns_default, obs_patterns_default, None


# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------


def _initialize_parameters(
    ctx: InitContext,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    float,
    Matrix,
    Matrix | None,
]:
    """Initialize loadings, scores, mu, V, Av, Sv, Muv, Va, Vmu and center data.

    Returns:
        Initialized parameters and centered data/probe matrices.
    """
    # Use a deterministic RNG only when we fall back to random init; when
    # init is provided (e.g., MATLAB fixture), pass through without forcing a
    # new seed.
    init_value = cast("str | Mapping[str, Any] | None", ctx.opts.get("init"))
    init_result: InitResult = init_params(
        init_value,
        ctx.shapes,
        score_pattern_index=ctx.pattern_index,
        rng=None if init_value else np.random.default_rng(),
    )
    loadings = init_result.a
    scores = init_result.s
    mu = init_result.mu.reshape(-1, 1)
    noise_var = float(init_result.v)
    loading_covariances = init_result.av
    score_covariances = init_result.sv
    mu_variances = init_result.muv.reshape(-1, 1)

    # Priors on loadings and mu
    if ctx.use_prior:
        va = np.full(ctx.shapes.n_components, 1000.0, dtype=float)
        vmu = 1000.0
    else:
        va = np.full(ctx.shapes.n_components, np.inf, dtype=float)
        vmu = float("inf")

    # MAP / PPCA: disable posterior variances if needed
    if not ctx.use_postvar:
        mu_variances = np.array([], dtype=float)
        loading_covariances = []

    if not bool(ctx.opts.get("bias", 1)):
        mu_variances = np.array([], dtype=float)
        vmu = 0.0

    # Initialize mu from data if empty
    if mu.size == 0:
        if bool(ctx.opts.get("bias", 1)):
            if sp.issparse(ctx.x_data):
                mu_num = np.array(ctx.x_data.sum(axis=1)).ravel()
            else:
                mu_num = np.sum(
                    np.asarray(ctx.x_data, dtype=float),
                    axis=1,
                )
            # Avoid division by zero: rows with no observations get zero mean.
            with np.errstate(divide="ignore", invalid="ignore"):
                mu_vec = np.divide(
                    mu_num,
                    ctx.n_obs_row,
                    out=np.zeros_like(mu_num, dtype=float),
                    where=ctx.n_obs_row > 0,
                )
            mu = mu_vec.reshape(-1, 1)
        else:
            mu = np.zeros((ctx.shapes.n_features, 1), dtype=float)

    # Initial centering using subtract_mu helper.
    x_data_centered, x_probe_centered = subtract_mu(
        mu,
        ctx.x_data,
        ctx.mask,
        probe=(
            ProbeMatrices(x=ctx.x_probe, mask=ctx.mask_probe)
            if ctx.x_probe is not None and ctx.mask_probe is not None
            else None
        ),
        update_bias=bool(ctx.opts.get("bias", 1)),
    )

    return (
        loadings,
        scores,
        mu,
        noise_var,
        loading_covariances,
        score_covariances,
        mu_variances,
        va,
        vmu,
        x_data_centered,
        x_probe_centered,
    )


def _update_hyperpriors(ctx: HyperpriorContext) -> tuple[np.ndarray, float]:
    """Update Va and Vmu after a broad-prior warmup period.

    Returns:
        Updated ``va`` and ``vmu`` hyperpriors.
    """
    if not ctx.use_prior or ctx.iteration <= ctx.niter_broadprior:
        return ctx.va, ctx.vmu

    denom = float(ctx.n_features + 2.0 * ctx.hp_vb)

    vmu = ctx.vmu
    if ctx.bias_enabled:
        vmu_val = float(np.sum(ctx.mu**2))
        if ctx.mu_variances.size > 0:
            vmu_val += float(np.sum(ctx.mu_variances))
        vmu = max((vmu_val + 2.0 * ctx.hp_va) / denom, _EPS_VAR)

    va_new = np.sum(ctx.loadings**2, axis=0)
    if _has_covariances(ctx.loading_covariances):
        for row_cov in ctx.loading_covariances:
            va_new += np.diag(row_cov)

    va_new = (va_new + 2.0 * ctx.hp_va) / denom
    va_new = np.maximum(va_new, _EPS_VAR)

    return va_new, vmu


def _row_sums(err_mx: np.ndarray | sp.spmatrix) -> np.ndarray:
    if sp.issparse(err_mx):
        err_sparse = (
            err_mx
            if isinstance(err_mx, sp.csr_matrix)
            else sp.csr_matrix(cast("Any", err_mx))
        )
        return np.asarray(err_sparse.sum(axis=1)).ravel()
    return np.sum(np.asarray(err_mx, dtype=float), axis=1)


# ---------------------------------------------------------------------------
# Bias / mean update
# ---------------------------------------------------------------------------


def _update_bias(
    *,
    bias_enabled: bool,
    bias_state: BiasState,
    err_mx: np.ndarray | sp.spmatrix,
    centering: CenteringState,
) -> tuple[BiasState, CenteringState]:
    """Bias / mean update, including in-place centering of x / x_probe.

    Returns:
        Updated bias state and centering state.
    """
    if not bias_enabled:
        return bias_state, centering

    mu = bias_state.mu
    mu_variances = bias_state.muv
    noise_var = float(bias_state.noise_var)
    vmu = float(bias_state.vmu)
    n_obs_row = bias_state.n_obs_row

    # Safe d_mu: zero update where no observations.
    err_sum_rows = _row_sums(err_mx)

    with np.errstate(divide="ignore", invalid="ignore"):
        d_mu = np.divide(
            err_sum_rows,
            n_obs_row,
            out=np.zeros_like(n_obs_row, dtype=float),
            where=n_obs_row > 0,
        )

    if mu_variances.size > 0 and vmu > 0.0:
        denom = n_obs_row + noise_var / vmu
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_variances = np.divide(
                noise_var,
                denom,
                out=np.zeros_like(denom, dtype=float),
                where=denom > 0,
            ).reshape(-1, 1)

    # Shrinkage factor for mu update, stable even for zero n_obs_row.
    if vmu > 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            shrink = 1.0 / (1.0 + noise_var / (n_obs_row * vmu))
        # Rows with no observations → shrink = 0
        shrink = np.where(n_obs_row > 0, shrink, 0.0)
    else:
        shrink = np.zeros_like(n_obs_row, dtype=float)
    shrink = shrink.reshape(-1, 1)

    mu_new = shrink * (mu + d_mu.reshape(-1, 1))
    d_mu_update = mu_new - mu
    mu = mu_new

    # Recentre data and probe using the updated mean increment.
    x_data_new, x_probe_new = subtract_mu(
        d_mu_update,
        centering.x_data,
        centering.mask,
        probe=(
            ProbeMatrices(x=centering.x_probe, mask=centering.mask_probe)
            if centering.x_probe is not None and centering.mask_probe is not None
            else None
        ),
        update_bias=True,
    )

    updated_bias_state = BiasState(
        mu=mu,
        muv=mu_variances,
        noise_var=noise_var,
        vmu=vmu,
        n_obs_row=n_obs_row,
    )
    updated_centering = CenteringState(
        x_data=x_data_new,
        x_probe=x_probe_new,
        mask=centering.mask,
        mask_probe=centering.mask_probe,
    )
    return updated_bias_state, updated_centering


# ---------------------------------------------------------------------------
# Score / covariance update
# ---------------------------------------------------------------------------


def _score_update_fast_dense_no_av(state: ScoreState) -> ScoreState:
    """Fast path: fully observed dense data, no loading covariances.

    Returns:
        Updated score state.
    """
    x_arr = np.asarray(state.x_data, dtype=float)
    _n_features, n_samples = x_arr.shape

    result = _score_update_dense_ext(
        x_data=np.asarray(x_arr, dtype=np.float64),
        loadings=np.asarray(state.loadings, dtype=np.float64),
        noise_var=float(state.noise_var),
        return_covariance=_has_covariances(state.score_covariances),
    )
    state.scores[:, :] = np.asarray(result["scores"], dtype=float)

    log_stride = int(getattr(state, "log_progress_stride", 1))
    cov_writeback_mode = str(getattr(state, "cov_writeback_mode", "python"))

    if _has_covariances(state.score_covariances) and "score_covariance" in result:
        cov_common = np.asarray(result["score_covariance"], dtype=float)

        if cov_writeback_mode != "python":
            state.score_covariances = [cov_common] * n_samples
            _log_progress_if_needed(
                state.verbose,
                log_stride,
                n_samples,
                n_samples,
                phase="Updating scores",
            )
            return state

        for j in range(n_samples):
            if isinstance(state.score_covariances, np.ndarray):
                state.score_covariances[j, :, :] = cov_common
            else:
                state.score_covariances[j] = cov_common
            _log_progress_if_needed(
                state.verbose,
                log_stride,
                j + 1,
                n_samples,
                phase="Updating scores",
            )
    return state


def _symmetrize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def _safe_cholesky(mat: np.ndarray, eye: np.ndarray) -> ChoFactor:
    mat_sym = _symmetrize(mat)
    try:
        return cho_factor(mat_sym, lower=True, check_finite=False)
    except LinAlgError:
        return cho_factor(mat_sym + _EPS_VAR * eye, lower=True, check_finite=False)


def _csc_column_accessor(
    matrix: sp.csc_matrix,
    *,
    use_buffered: bool,
) -> Callable[[int], np.ndarray]:
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    n_rows = matrix.shape[0]

    if use_buffered:
        buf = np.zeros(n_rows, dtype=float)

        def col_buffered(j: int) -> np.ndarray:
            buf.fill(0.0)
            start = int(indptr[j])
            end = int(indptr[j + 1])
            if end > start:
                buf[indices[start:end]] = data[start:end]
            return buf

        return col_buffered

    def col_allocating(j: int) -> np.ndarray:
        out = np.zeros(n_rows, dtype=float)
        start = int(indptr[j])
        end = int(indptr[j + 1])
        if end > start:
            out[indices[start:end]] = data[start:end]
        return out

    return col_allocating


def _csc_mask_column_accessor(
    matrix: sp.csc_matrix,
    *,
    use_buffered: bool,
) -> Callable[[int], np.ndarray]:
    indices = matrix.indices
    indptr = matrix.indptr
    n_rows = matrix.shape[0]

    if use_buffered:
        buf = np.zeros(n_rows, dtype=float)

        def col_buffered(j: int) -> np.ndarray:
            buf.fill(0.0)
            start = int(indptr[j])
            end = int(indptr[j + 1])
            if end > start:
                buf[indices[start:end]] = 1.0
            return buf

        return col_buffered

    def col_allocating(j: int) -> np.ndarray:
        out = np.zeros(n_rows, dtype=float)
        start = int(indptr[j])
        end = int(indptr[j + 1])
        if end > start:
            out[indices[start:end]] = 1.0
        return out

    return col_allocating


def _csr_row_accessor(
    matrix: sp.csr_matrix,
    *,
    use_buffered: bool,
) -> Callable[[int], np.ndarray]:
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    n_cols = matrix.shape[1]

    if use_buffered:
        buf = np.zeros(n_cols, dtype=float)

        def row_buffered(i: int) -> np.ndarray:
            buf.fill(0.0)
            start = int(indptr[i])
            end = int(indptr[i + 1])
            if end > start:
                buf[indices[start:end]] = data[start:end]
            return buf

        return row_buffered

    def row_allocating(i: int) -> np.ndarray:
        out = np.zeros(n_cols, dtype=float)
        start = int(indptr[i])
        end = int(indptr[i + 1])
        if end > start:
            out[indices[start:end]] = data[start:end]
        return out

    return row_allocating


def _csr_mask_row_accessor(
    matrix: sp.csr_matrix,
    *,
    use_buffered: bool,
) -> Callable[[int], np.ndarray]:
    indices = matrix.indices
    indptr = matrix.indptr
    n_cols = matrix.shape[1]

    if use_buffered:
        buf = np.zeros(n_cols, dtype=float)

        def row_buffered(i: int) -> np.ndarray:
            buf.fill(0.0)
            start = int(indptr[i])
            end = int(indptr[i + 1])
            if end > start:
                buf[indices[start:end]] = 1.0
            return buf

        return row_buffered

    def row_allocating(i: int) -> np.ndarray:
        out = np.zeros(n_cols, dtype=float)
        start = int(indptr[i])
        end = int(indptr[i + 1])
        if end > start:
            out[indices[start:end]] = 1.0
        return out

    return row_allocating


def _dense_column_accessor(arr: np.ndarray) -> Callable[[int], np.ndarray]:
    def _col(j: int) -> np.ndarray:
        return np.asarray(arr[:, j], dtype=float)

    return _col


def _dense_mask_column_accessor(arr: np.ndarray) -> Callable[[int], np.ndarray]:
    def _col(j: int) -> np.ndarray:
        return np.asarray(arr[:, j], dtype=float)

    return _col


def _dense_row_accessor(arr: np.ndarray) -> Callable[[int], np.ndarray]:
    def _row(i: int) -> np.ndarray:
        return np.asarray(arr[i, :], dtype=float)

    return _row


def _dense_mask_row_accessor(arr: np.ndarray) -> Callable[[int], np.ndarray]:
    def _row(i: int) -> np.ndarray:
        return np.asarray(arr[i, :], dtype=float)

    return _row


def _score_accessors(
    state: ScoreState,
) -> tuple[Callable[[int], np.ndarray], Callable[[int], np.ndarray], int]:
    accessor_mode = str(getattr(state, "accessor_mode", "legacy"))
    use_buffered = (
        accessor_mode == "buffered"
        and int(getattr(state, "pattern_batch_size", 0)) <= 1
    )

    if sp.issparse(state.x_data):
        x_csc = state.x_csc if state.x_csc is not None else sp.csc_matrix(state.x_data)
        x_col = _csc_column_accessor(x_csc, use_buffered=use_buffered)

    else:
        x_arr = np.asarray(state.x_data, dtype=float)
        x_col = _dense_column_accessor(x_arr)

    if sp.issparse(state.mask):
        mask_csc = sp.csc_matrix(state.mask)
        mask_col = _csc_mask_column_accessor(mask_csc, use_buffered=use_buffered)

    else:
        mask_arr = np.asarray(state.mask, dtype=float)
        mask_col = _dense_mask_column_accessor(mask_arr)

    return x_col, mask_col, int(state.x_data.shape[1])


def _score_cholesky(
    state: ScoreState, mask_col: np.ndarray
) -> tuple[ChoFactor, np.ndarray, np.ndarray]:
    loadings_masked = mask_col[:, None] * state.loadings
    psi = loadings_masked.T @ loadings_masked + state.noise_var * state.eye_components

    if _has_covariances(state.loading_covariances):
        observed_rows = np.where(mask_col > 0)[0]
        for row_index in observed_rows:
            psi += _covariance_at(state.loading_covariances, int(row_index))

    cho = _safe_cholesky(psi, state.eye_components)
    sv_pattern = state.noise_var * cho_solve(
        cho,
        state.eye_components,
        check_finite=False,
    )
    return cho, loadings_masked, sv_pattern


def _update_scores_for_columns(  # noqa: PLR0913
    state: ScoreState,
    cols: list[int],
    x_col: Callable[[int], np.ndarray],
    mask_col: Callable[[int], np.ndarray],
    cov_index: int,
    *,
    batch_size: int = 0,
) -> None:
    mask_vec = mask_col(cols[0])
    cho, loadings_masked, sv_pattern = _score_cholesky(
        state,
        mask_vec,
    )
    if isinstance(state.score_covariances, np.ndarray):
        state.score_covariances[cov_index, :, :] = sv_pattern
    else:
        state.score_covariances[cov_index] = sv_pattern

    if batch_size <= 1:
        for j_idx in cols:
            rhs = loadings_masked.T @ x_col(j_idx)
            state.scores[:, j_idx] = cho_solve(cho, rhs, check_finite=False)
        return

    for start in range(0, len(cols), batch_size):
        block = cols[start : start + batch_size]
        x_block = np.column_stack([x_col(j_idx) for j_idx in block])
        rhs_block = loadings_masked.T @ x_block
        solved = cho_solve(cho, rhs_block, check_finite=False)
        state.scores[:, block] = solved


def _score_update_general_no_patterns(state: ScoreState) -> ScoreState:
    """General score update when there is no pattern sharing.

    Returns:
        Updated score state.
    """
    if not sp.issparse(state.x_data) and not sp.issparse(state.mask):
        return _score_update_general_dense_ext(state)

    x_col, mask_col, n_samples = _score_accessors(state)

    log_stride = int(getattr(state, "log_progress_stride", 1))

    for j in range(n_samples):
        mask_vec = mask_col(j)
        cho, loadings_masked, sv_pattern = _score_cholesky(
            state,
            mask_vec,
        )
        if isinstance(state.score_covariances, np.ndarray):
            state.score_covariances[j, :, :] = sv_pattern
        else:
            state.score_covariances[j] = sv_pattern

        rhs = loadings_masked.T @ x_col(j)
        state.scores[:, j] = cho_solve(cho, rhs, check_finite=False)

        _log_progress_if_needed(
            state.verbose,
            log_stride,
            j + 1,
            n_samples,
            phase="Updating scores",
        )

    cov_mode = str(getattr(state, "cov_writeback_mode", "python"))
    has_cov = _has_covariances(state.score_covariances)
    if cov_mode != "python" and has_cov:
        if isinstance(state.score_covariances, np.ndarray):
            state.score_covariances = np.asarray(state.score_covariances, dtype=float)
        else:
            state.score_covariances = _covariances_stack(state.score_covariances)
    elif cov_mode == "python" and isinstance(state.score_covariances, np.ndarray):
        state.score_covariances = _covariances_list(state.score_covariances)

    return state


def _score_update_sparse_no_patterns(state: ScoreState) -> ScoreState:
    """Sparse-native score update for pattern-free branch.

    Uses CSC index/value arrays directly to avoid constructing dense
    per-column vectors in the inner loop.

    Returns:
        Updated score state.
    """
    x_csc = state.x_csc if state.x_csc is not None else sp.csc_matrix(state.x_data)
    _score_update_sparse_ext_apply(state, x_csc)
    return state


def _score_update_with_patterns(state: ScoreState) -> ScoreState:
    """Score update when multiple columns share covariance patterns.

    Returns:
        Updated score state.
    """
    x_col, mask_col, _ = _score_accessors(state)

    log_stride = int(getattr(state, "log_progress_stride", 1))

    for pattern_index, cols in enumerate(state.obs_patterns):
        if not cols:
            continue

        _update_scores_for_columns(
            state,
            cols=cols,
            x_col=x_col,
            mask_col=mask_col,
            cov_index=pattern_index,
            batch_size=int(state.pattern_batch_size),
        )

        _log_progress_if_needed(
            state.verbose,
            log_stride,
            pattern_index + 1,
            len(state.obs_patterns),
            phase="Updating scores (patterns)",
        )

    return state


def _update_scores(state: ScoreState) -> ScoreState:
    """Update scores and score covariances.

    Returns:
        Updated score state.
    """
    # Pattern-free branch
    if state.pattern_index is None:
        if state.use_python_scores:
            return _score_update_general_no_patterns(state)

        if sp.issparse(state.x_data) and sp.issparse(state.mask):
            return _score_update_sparse_no_patterns(state)

        dense_mask = None
        fully_observed = False
        if not sp.issparse(state.mask):
            dense_mask = np.asarray(state.mask, dtype=float)
            fully_observed = bool(np.all(dense_mask > 0))

        if (
            dense_mask is not None
            and fully_observed
            and not _has_covariances(state.loading_covariances)
            and not sp.issparse(state.x_data)
        ):
            return _score_update_fast_dense_no_av(state)

        return _score_update_general_no_patterns(state)

    # Pattern-sharing branch
    return _score_update_with_patterns(state)


# ---------------------------------------------------------------------------
# Loadings update
# ---------------------------------------------------------------------------


def _loadings_accessors(
    state: LoadingsUpdateState,
) -> tuple[Callable[[int], np.ndarray], Callable[[int], np.ndarray], int]:
    accessor_mode = str(getattr(state, "accessor_mode", "legacy"))
    use_buffered = accessor_mode == "buffered"

    if sp.issparse(state.x_data):
        x_csr = state.x_csr if state.x_csr is not None else sp.csr_matrix(state.x_data)
        x_row = _csr_row_accessor(x_csr, use_buffered=use_buffered)

    else:
        x_arr = np.asarray(state.x_data, dtype=float)
        x_row = _dense_row_accessor(x_arr)

    if sp.issparse(state.mask):
        mask_csr = sp.csr_matrix(state.mask)
        mask_row = _csr_mask_row_accessor(mask_csr, use_buffered=use_buffered)

    else:
        mask_arr = np.asarray(state.mask, dtype=float)
        mask_row = _dense_mask_row_accessor(mask_arr)

    return x_row, mask_row, int(state.x_data.shape[0])


def _prior_precision_matrix(va: np.ndarray, noise_var: float) -> np.ndarray:
    prior_prec_diag = np.zeros_like(va, dtype=float)
    finite_mask = np.isfinite(va) & (va > 0)
    prior_prec_diag[finite_mask] = noise_var / va[finite_mask]
    return np.diag(prior_prec_diag)


def _score_covariance_for_column(state: LoadingsUpdateState, j_idx: int) -> np.ndarray:
    if state.pattern_index is None:
        return _covariance_at(state.score_covariances, j_idx)
    return _covariance_at(state.score_covariances, int(state.pattern_index[j_idx]))


def _phi_for_row(
    mask_vec: np.ndarray,
    state: LoadingsUpdateState,
    prior_prec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores_masked = mask_vec[None, :] * state.scores
    phi = scores_masked @ scores_masked.T + prior_prec

    if _has_covariances(state.score_covariances):
        for j_idx in np.where(mask_vec > 0)[0]:
            phi += _score_covariance_for_column(state, int(j_idx))

    return phi, scores_masked


def _loadings_update_fast_dense_no_sv(
    state: LoadingsUpdateState,
) -> tuple[
    np.ndarray,
    CovarianceStore,
]:
    """Fast path: fully observed dense data, no score covariances.

    Returns:
        Updated loadings and loading covariances.

    Raises:
        ValueError: If ``x_data`` is sparse.
    """
    if sp.issparse(state.x_data):
        msg = "fast dense loadings path requires dense x_data"
        raise ValueError(msg)

    x_arr = np.asarray(state.x_data, dtype=float)

    n_features = x_arr.shape[0]

    if state.verbose == 2:
        logger.info("Updating loadings")

    prior_prec_diag = np.zeros_like(state.va, dtype=float)
    finite_mask = np.isfinite(state.va) & (state.va > 0)
    prior_prec_diag[finite_mask] = state.noise_var / state.va[finite_mask]
    prior_prec = np.diag(prior_prec_diag)

    return _loadings_update_fast_dense_ext(
        state=state,
        x_arr=x_arr,
        prior_prec=prior_prec,
        n_features=n_features,
    )


def _loadings_update_fast_dense_ext(
    state: LoadingsUpdateState,
    x_arr: np.ndarray,
    prior_prec: np.ndarray,
    n_features: int,
) -> tuple[np.ndarray, CovarianceStore]:
    result = _loadings_update_dense_ext(
        x_data=np.asarray(x_arr, dtype=np.float64),
        scores=np.asarray(state.scores, dtype=np.float64),
        prior_prec=np.asarray(prior_prec, dtype=np.float64),
        noise_var=float(state.noise_var),
        return_covariance=_has_covariances(state.loading_covariances),
    )

    loadings = np.asarray(result["loadings"], dtype=float)
    if _has_covariances(state.loading_covariances) and "loading_covariance" in result:
        loading_cov_common = np.asarray(result["loading_covariance"], dtype=float)
    else:
        loading_cov_common = None

    log_stride = int(getattr(state, "log_progress_stride", 1))
    cov_writeback_mode = str(getattr(state, "cov_writeback_mode", "python"))

    if loading_cov_common is not None and cov_writeback_mode != "python":
        state.loading_covariances = [loading_cov_common] * n_features
        if not _progress_should_log(state.verbose, log_stride, n_features, n_features):
            return np.asarray(loadings, dtype=float), state.loading_covariances

    for i in range(n_features):
        if loading_cov_common is not None and cov_writeback_mode == "python":
            if isinstance(state.loading_covariances, np.ndarray):
                state.loading_covariances[i, :, :] = loading_cov_common
            else:
                state.loading_covariances[i] = loading_cov_common
        _log_progress_if_needed(
            state.verbose,
            log_stride,
            i + 1,
            n_features,
            phase="Updating loadings",
        )

    has_cov = _has_covariances(state.loading_covariances)
    if cov_writeback_mode != "python" and has_cov:
        if isinstance(state.loading_covariances, np.ndarray):
            state.loading_covariances = np.asarray(
                state.loading_covariances, dtype=float
            )
        else:
            state.loading_covariances = _covariances_stack(state.loading_covariances)
    elif cov_writeback_mode == "python" and isinstance(
        state.loading_covariances, np.ndarray
    ):
        state.loading_covariances = _covariances_list(state.loading_covariances)

    return np.asarray(loadings, dtype=float), state.loading_covariances


def _loadings_update_general(
    state: LoadingsUpdateState,
) -> tuple[
    np.ndarray,
    CovarianceStore,
]:
    """General loading update, including masks and score covariances.

    Returns:
        Updated loadings and loading covariances.
    """
    if (
        state.pattern_index is None
        and sp.issparse(state.x_data)
        and sp.issparse(state.mask)
    ):
        return _loadings_update_sparse_no_patterns(state)

    if (
        state.pattern_index is None
        and not sp.issparse(state.x_data)
        and not sp.issparse(state.mask)
    ):
        return _loadings_update_general_dense_ext(state)

    x_row, mask_row, n_features = _loadings_accessors(state)
    prior_prec = _prior_precision_matrix(state.va, state.noise_var)
    eye_components = np.eye(state.scores.shape[0])
    has_cov = _has_covariances(state.loading_covariances)

    if state.verbose == 2:
        logger.info("Updating loadings")

    loadings = np.empty((n_features, eye_components.shape[0]), dtype=float)

    # Cache factorizations for repeated mask patterns to avoid recomputing Phi.
    cho_cache: dict[bytes, tuple[ChoFactor, np.ndarray, np.ndarray | None]] = {}

    for i in range(n_features):
        mask_vec = mask_row(i)

        cached = cho_cache.get(mask_vec.tobytes())
        if cached is None:
            phi, scores_masked = _phi_for_row(mask_vec, state, prior_prec)
            cho = _safe_cholesky(phi, eye_components)

            cov_template: np.ndarray | None = None
            if has_cov:
                cov_template = state.noise_var * cho_solve(
                    cho,
                    eye_components,
                    check_finite=False,
                )

            cho_cache[mask_vec.tobytes()] = (cho, scores_masked, cov_template)
        else:
            cho, scores_masked, cov_template = cached

        loadings[i, :] = cho_solve(
            cho,
            scores_masked @ x_row(i),
            check_finite=False,
        )

        if has_cov:
            if isinstance(state.loading_covariances, np.ndarray):
                state.loading_covariances[i, :, :] = (
                    np.array(cov_template, copy=True)
                    if cached is not None and cov_template is not None
                    else state.noise_var
                    * cho_solve(
                        cho,
                        eye_components,
                        check_finite=False,
                    )
                )
            else:
                state.loading_covariances[i] = (
                    np.array(cov_template, copy=True)
                    if cached is not None and cov_template is not None
                    else state.noise_var
                    * cho_solve(
                        cho,
                        eye_components,
                        check_finite=False,
                    )
                )

        _log_progress_if_needed(
            state.verbose,
            int(getattr(state, "log_progress_stride", 1)),
            i + 1,
            n_features,
            phase="Updating loadings",
        )

    return loadings, state.loading_covariances


def _loadings_update_sparse_no_patterns(
    state: LoadingsUpdateState,
) -> tuple[
    np.ndarray,
    CovarianceStore,
]:
    """Sparse-native loadings update for pattern-free branch.

    Uses CSR index/value arrays directly to avoid constructing dense
    per-row vectors in the inner loop.

    Returns:
        Updated loadings and loading covariances.
    """
    x_csr = state.x_csr if state.x_csr is not None else sp.csr_matrix(state.x_data)
    loadings = _loadings_update_sparse_ext_apply(state, x_csr)

    cov_mode = str(getattr(state, "cov_writeback_mode", "python"))
    has_cov = _has_covariances(state.loading_covariances)
    if cov_mode != "python" and has_cov:
        bulk_cov = (
            np.asarray(state.loading_covariances, dtype=float)
            if isinstance(state.loading_covariances, np.ndarray)
            else _covariances_stack(state.loading_covariances)
        )
        return loadings, bulk_cov

    if cov_mode == "python" and isinstance(state.loading_covariances, np.ndarray):
        state.loading_covariances = _covariances_list(state.loading_covariances)

    return loadings, state.loading_covariances


def _score_update_sparse_ext_apply(state: ScoreState, x_csc: sp.csc_matrix) -> None:
    has_loading_covariances = _has_covariances(state.loading_covariances)
    return_score_covariances = _has_covariances(state.score_covariances)
    cov_mode = str(getattr(state, "cov_writeback_mode", "python"))

    av_arg: np.ndarray | None
    if has_loading_covariances:
        av_arg = _covariances_stack(state.loading_covariances)
    else:
        av_arg = None

    result = _score_update_sparse_ext(
        x_data=x_csc.data.astype(np.float64, copy=False),
        x_indices=x_csc.indices.astype(np.int32, copy=False),
        x_indptr=x_csc.indptr.astype(np.int32, copy=False),
        loadings=np.asarray(state.loadings, dtype=np.float64),
        loading_covariances=av_arg,
        noise_var=float(state.noise_var),
        return_covariances=return_score_covariances,
        num_cpu=int(state.sparse_num_cpu),
    )

    scores_arr = np.asarray(result["scores"], dtype=float)
    state.scores[:, :] = scores_arr

    if return_score_covariances and "score_covariances" in result:
        score_covariances_arr = np.asarray(result["score_covariances"], dtype=float)
        if cov_mode != "python":
            state.score_covariances = score_covariances_arr
        else:
            state.score_covariances = [
                score_covariances_arr[j, :, :]
                for j in range(score_covariances_arr.shape[0])
            ]

    if state.verbose == 2:
        logger.info("Updating scores")


def _score_update_general_dense_ext(state: ScoreState) -> ScoreState:
    x_arr = np.asarray(state.x_data, dtype=np.float64)
    mask_arr = np.asarray(state.mask, dtype=np.float64)
    has_loading_covariances = _has_covariances(state.loading_covariances)
    return_score_covariances = _has_covariances(state.score_covariances)
    cov_mode = str(getattr(state, "cov_writeback_mode", "python"))

    av_arg: np.ndarray | None
    if has_loading_covariances:
        av_arg = _covariances_stack(state.loading_covariances)
    else:
        av_arg = None

    result = _score_update_dense_masked_ext(
        x_data=x_arr,
        mask=mask_arr,
        loadings=np.asarray(state.loadings, dtype=np.float64),
        loading_covariances=av_arg,
        noise_var=float(state.noise_var),
        return_covariances=return_score_covariances,
        num_cpu=int(state.dense_num_cpu),
    )

    state.scores[:, :] = np.asarray(result["scores"], dtype=float)

    n_samples = x_arr.shape[1]
    if return_score_covariances and "score_covariances" in result:
        score_covariances_arr = np.asarray(result["score_covariances"], dtype=float)
        if cov_mode != "python":
            state.score_covariances = score_covariances_arr
        else:
            state.score_covariances = [
                score_covariances_arr[j, :, :]
                for j in range(score_covariances_arr.shape[0])
            ]

    log_stride = int(getattr(state, "log_progress_stride", 1))
    if cov_mode != "python":
        _log_progress_if_needed(
            state.verbose,
            log_stride,
            n_samples,
            n_samples,
            phase="Updating scores",
        )
    elif _progress_should_log(state.verbose, log_stride, 1, n_samples):
        for j in range(n_samples):
            _log_progress_if_needed(
                state.verbose,
                log_stride,
                j + 1,
                n_samples,
                phase="Updating scores",
            )

    return state


def _loadings_update_general_dense_ext(
    state: LoadingsUpdateState,
) -> tuple[np.ndarray, CovarianceStore]:
    x_arr = np.asarray(state.x_data, dtype=np.float64)
    mask_arr = np.asarray(state.mask, dtype=np.float64)
    prior_prec = _prior_precision_matrix(state.va, state.noise_var)
    has_score_covariances = _has_covariances(state.score_covariances)
    return_loading_covariances = _has_covariances(state.loading_covariances)
    cov_mode = str(getattr(state, "cov_writeback_mode", "python"))

    sv_arg: np.ndarray | None
    if has_score_covariances:
        sv_arg = _covariances_stack(state.score_covariances)
    else:
        sv_arg = None

    result = _loadings_update_dense_masked_ext(
        x_data=x_arr,
        mask=mask_arr,
        scores=np.asarray(state.scores, dtype=np.float64),
        score_covariances=sv_arg,
        prior_prec=np.asarray(prior_prec, dtype=np.float64),
        noise_var=float(state.noise_var),
        return_covariances=return_loading_covariances,
        num_cpu=int(state.dense_num_cpu),
    )

    loadings = np.asarray(result["loadings"], dtype=float)

    n_features = x_arr.shape[0]
    if return_loading_covariances and "loading_covariances" in result:
        loading_covariances_arr = np.asarray(result["loading_covariances"], dtype=float)
        if cov_mode != "python":
            state.loading_covariances = loading_covariances_arr
        else:
            state.loading_covariances = [
                loading_covariances_arr[i, :, :]
                for i in range(loading_covariances_arr.shape[0])
            ]

    log_stride = int(getattr(state, "log_progress_stride", 1))
    if cov_mode != "python":
        _log_progress_if_needed(
            state.verbose,
            log_stride,
            n_features,
            n_features,
            phase="Updating loadings",
        )
    elif _progress_should_log(state.verbose, log_stride, 1, n_features):
        for i in range(n_features):
            _log_progress_if_needed(
                state.verbose,
                log_stride,
                i + 1,
                n_features,
                phase="Updating loadings",
            )

    return loadings, state.loading_covariances


def _loadings_update_sparse_ext_apply(
    state: LoadingsUpdateState,
    x_csr: sp.csr_matrix,
) -> np.ndarray:
    has_score_covariances = _has_covariances(state.score_covariances)
    return_loading_covariances = _has_covariances(state.loading_covariances)
    cov_mode = str(getattr(state, "cov_writeback_mode", "python"))

    sv_arg: np.ndarray | None
    if has_score_covariances:
        sv_arg = _covariances_stack(state.score_covariances)
    else:
        sv_arg = None

    prior_prec = _prior_precision_matrix(state.va, state.noise_var)
    result = _loadings_update_sparse_ext(
        x_data=x_csr.data.astype(np.float64, copy=False),
        x_indices=x_csr.indices.astype(np.int32, copy=False),
        x_indptr=x_csr.indptr.astype(np.int32, copy=False),
        scores=np.asarray(state.scores, dtype=np.float64),
        score_covariances=sv_arg,
        prior_prec=np.asarray(prior_prec, dtype=np.float64),
        noise_var=float(state.noise_var),
        return_covariances=return_loading_covariances,
        num_cpu=int(state.sparse_num_cpu),
    )

    loadings_arr = np.asarray(result["loadings"], dtype=float)
    loadings = loadings_arr

    if return_loading_covariances and "loading_covariances" in result:
        loading_covariances_arr = np.asarray(
            result["loading_covariances"],
            dtype=float,
        )
        if cov_mode != "python":
            state.loading_covariances = loading_covariances_arr
        else:
            state.loading_covariances = [
                loading_covariances_arr[i, :, :]
                for i in range(loading_covariances_arr.shape[0])
            ]

    if state.verbose == 2:
        logger.info("Updating loadings")

    return loadings


def _update_loadings(
    state: LoadingsUpdateState,
) -> tuple[
    np.ndarray,
    CovarianceStore,
]:
    """Update loadings and loading covariances.

    Returns:
        Updated loadings and loading covariances.
    """
    dense_mask = None
    fully_observed = False
    sv_contrib = _has_covariances(state.score_covariances)

    if not sp.issparse(state.mask):
        dense_mask = np.asarray(state.mask, dtype=float)
        fully_observed = bool(np.all(dense_mask > 0))

    if (
        state.pattern_index is None
        and sp.issparse(state.x_data)
        and sp.issparse(state.mask)
    ):
        loadings, covariances = _loadings_update_sparse_no_patterns(state)
        cov_mode = str(getattr(state, "cov_writeback_mode", "python"))
        if (
            cov_mode != "python"
            and _has_covariances(covariances)
            and not isinstance(covariances, np.ndarray)
        ):
            covariances = _covariances_stack(covariances)
        return loadings, covariances

    if (
        dense_mask is not None
        and fully_observed
        and not sv_contrib
        and not sp.issparse(state.x_data)
    ):
        return _loadings_update_fast_dense_no_sv(state)

    return _loadings_update_general(state)


# ---------------------------------------------------------------------------
# RMS / probe RMS
# ---------------------------------------------------------------------------


def _recompute_rms(ctx: RmsContext) -> tuple[float, float, Matrix]:
    """Recompute RMS and probe RMS after an update of loadings/scores.

    Returns:
        RMS on data, RMS on probe, and error matrix.
    """
    cfg_data = RmsConfig(n_observed=int(ctx.n_data), num_cpu=int(ctx.num_cpu))
    rms, err_mx_raw = compute_rms(
        ctx.x_data,
        ctx.loadings,
        ctx.scores,
        ctx.mask,
        cfg_data,
    )
    err_mx: Matrix
    if sp.issparse(err_mx_raw):
        err_mx = (
            err_mx_raw
            if isinstance(err_mx_raw, sp.csr_matrix)
            else sp.csr_matrix(cast("Any", err_mx_raw))
        )
    else:
        err_mx = np.asarray(err_mx_raw)

    if ctx.n_probe > 0 and ctx.x_probe is not None and ctx.mask_probe is not None:
        cfg_probe = RmsConfig(n_observed=int(ctx.n_probe), num_cpu=int(ctx.num_cpu))
        prms, _ = compute_rms(
            ctx.x_probe,
            ctx.loadings,
            ctx.scores,
            ctx.mask_probe,
            cfg_probe,
        )
    else:
        prms = float("nan")

    return float(rms), float(prms), err_mx


# ---------------------------------------------------------------------------
# Noise variance update
# ---------------------------------------------------------------------------


def _update_noise_variance(
    noise_state: NoiseState,
    rms: float,
    ix: np.ndarray,
    jx: np.ndarray,
    hp_v: float = 0.001,
) -> tuple[NoiseState, float]:
    """Update noise variance using current posterior covariances.

    Returns:
        ``(noise_state, s_xv)`` after updating noise variance.
    """
    loadings = noise_state.loadings
    scores = noise_state.scores
    loading_covariances = noise_state.loading_covariances
    score_covariances = noise_state.score_covariances

    n_samples = int(scores.shape[1])
    fully_observed = int(ix.size) == (int(loadings.shape[0]) * n_samples)

    if fully_observed and _has_covariances(score_covariances):
        sv_by_col = _score_covariances_by_column(
            score_covariances,
            noise_state.pattern_index,
            n_samples,
        )
        s_xv = _fully_observed_s_xv(
            noise_state=noise_state,
            sv_by_col=sv_by_col,
            n_samples=n_samples,
        )
        s_xv += (rms**2) * noise_state.n_data
        updated_state, _ = _finalize_noise_state(noise_state, s_xv, hp_v)
        return updated_state, s_xv

    sv_by_col = _score_covariances_by_column(
        score_covariances,
        noise_state.pattern_index,
        n_samples,
    )

    has_loading_covariances = _has_covariances(loading_covariances)

    av_arg: np.ndarray | None
    if has_loading_covariances:
        av_arg = _covariances_stack(loading_covariances)
    else:
        av_arg = None

    sv_arr = np.stack(sv_by_col, axis=0).astype(np.float64, copy=False)
    s_xv = float(
        _noise_sxv_sum_ext(
            ix=np.asarray(ix, dtype=np.int32),
            jx=np.asarray(jx, dtype=np.int32),
            loadings=np.asarray(loadings, dtype=np.float64),
            scores=np.asarray(scores, dtype=np.float64),
            sv_by_col=sv_arr,
            loading_covariances=av_arg,
            num_cpu=int(noise_state.sparse_num_cpu),
        )
    )

    if noise_state.mu_variances.size > 0:
        muv_term = float(np.sum(noise_state.mu_variances[ix, 0]).item())
        s_xv += muv_term

    s_xv += (rms**2) * noise_state.n_data
    updated_state, _ = _finalize_noise_state(noise_state, s_xv, hp_v)
    return updated_state, s_xv


def _score_covariances_by_column(
    score_covariances: CovarianceStore,
    pattern_index: np.ndarray | None,
    n_samples: int,
) -> list[np.ndarray]:
    if pattern_index is None:
        return _covariances_list(score_covariances)
    return [
        _covariance_at(score_covariances, int(pattern_index[j]))
        for j in range(n_samples)
    ]


def _fully_observed_s_xv(
    *,
    noise_state: NoiseState,
    sv_by_col: list[np.ndarray],
    n_samples: int,
) -> float:
    loadings = noise_state.loadings
    scores = noise_state.scores
    loading_covariances = noise_state.loading_covariances
    mu_variances = noise_state.mu_variances

    gram = loadings.T @ loadings
    sv_term = float(sum(np.trace(sv_j @ gram) for sv_j in sv_by_col))
    s_xv = sv_term

    if _has_covariances(loading_covariances):
        av_sum = np.sum(_covariances_stack(loading_covariances), axis=0)
        av_cov_term = float(np.sum(scores * (av_sum @ scores)))
        av_trace_term = float(sum(np.sum(sv_j * av_sum) for sv_j in sv_by_col))
        s_xv += av_cov_term
        s_xv += av_trace_term

    muv_term = 0.0
    if mu_variances.size > 0:
        muv_term = float(np.sum(mu_variances[:, 0]) * n_samples)
        s_xv += muv_term

    return s_xv


def _finalize_noise_state(
    noise_state: NoiseState,
    s_xv: float,
    hp_v: float,
) -> tuple[NoiseState, float]:
    v_new = (s_xv + 2.0 * hp_v) / (noise_state.n_data + 2.0 * hp_v)
    v_new = max(float(v_new), _EPS_VAR)

    updated_state = NoiseState(
        loadings=noise_state.loadings,
        scores=noise_state.scores,
        loading_covariances=noise_state.loading_covariances,
        score_covariances=noise_state.score_covariances,
        mu_variances=noise_state.mu_variances,
        pattern_index=noise_state.pattern_index,
        n_data=noise_state.n_data,
        noise_var=v_new,
        sparse_num_cpu=noise_state.sparse_num_cpu,
    )
    return updated_state, float(s_xv)


# ---------------------------------------------------------------------------
# Final PCA rotation
# ---------------------------------------------------------------------------


def _final_rotation(
    ctx: RotationContext,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
]:
    """Final PCA rotation if no iterative rotation was applied.

    Returns:
        Rotated loadings, loading covariances, scores, score covariances, and mu.
    """
    # Only pass loadings covariances if we actually maintain them.
    loading_covariances = (
        _covariances_list(ctx.loading_covariances)
        if _has_covariances(ctx.loading_covariances)
        else None
    )

    params = RotateParams(
        loading_covariances=loading_covariances,
        score_covariances=_covariances_list(ctx.score_covariances),
        isv=ctx.pattern_index,
        obscombj=ctx.obs_patterns,
        update_bias=ctx.bias_enabled,
    )

    d_mu_final, loadings_rot, av_rot, scores_rot, sv_rot = rotate_to_pca(
        ctx.loadings,
        ctx.scores,
        params,
    )

    # Ensure we always return a list for av, even if rotation used None.
    av_out: list[np.ndarray] = av_rot if av_rot is not None else []

    mu_out = ctx.mu
    if ctx.bias_enabled:
        mu_out += d_mu_final

    return loadings_rot, av_out, scores_rot, sv_rot, mu_out
