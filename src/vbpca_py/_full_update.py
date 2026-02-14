"""Helper functions for full VB-PCA update.

This module factors out pieces of the :func:`pca_full` implementation
to keep the main loop readable while preserving the original algorithmic
behaviour. The helpers are intentionally small and focused; shared
state is carried via simple data classes rather than long parameter
lists so that static analysis tools remain happy.
"""

from __future__ import annotations

import logging
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

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping

logger = logging.getLogger(__name__)

_EPS_VAR = 1e-15  # minimum variance for numerical safety
ChoFactor = tuple[np.ndarray, bool]


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
    loading_covariances: list[np.ndarray]  # Av
    score_covariances: list[np.ndarray]  # Sv
    pattern_index: np.ndarray | None
    obs_patterns: list[list[int]]
    noise_var: float
    eye_components: np.ndarray
    verbose: int


@dataclass
class NoiseState:
    """State required for updating the noise variance."""

    loadings: np.ndarray
    scores: np.ndarray
    loading_covariances: list[np.ndarray]
    score_covariances: list[np.ndarray]
    mu_variances: np.ndarray
    pattern_index: np.ndarray | None
    n_data: float
    noise_var: float


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
    loading_covariances: list[np.ndarray]
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
    loading_covariances: list[np.ndarray]
    score_covariances: list[np.ndarray]
    pattern_index: np.ndarray | None
    va: np.ndarray
    noise_var: float
    verbose: int


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


@dataclass
class RotationContext:
    """Context for final PCA-style rotation."""

    loadings: np.ndarray
    loading_covariances: list[np.ndarray]
    scores: np.ndarray
    score_covariances: list[np.ndarray]
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
) -> tuple[Matrix, Matrix | None, int, int, np.ndarray, np.ndarray]:
    """Copy input, apply probe handling, and remove empty rows/cols.

    Returns:
        Data, probe data, original shapes, and kept row/col indices.
    """
    x_probe_opt = opts.get("xprobe", None)

    if isinstance(x, sp.csr_matrix):
        x_data: Matrix = x.copy()
    else:
        x_data = np.array(x, dtype=float).copy()

    x_probe: Matrix | None = None
    if x_probe_opt is not None:
        x_probe_array = np.asarray(x_probe_opt, dtype=float)
        if x_probe_array.size != 0:
            if sp.issparse(x_probe_opt):
                x_probe = sp.csr_matrix(cast("Any", x_probe_opt))
            else:
                x_probe = x_probe_array.copy()

    n_features_original, n_samples_original = x_data.shape

    x_data, x_probe, row_idx, col_idx, init_opt = remove_empty_entries(
        x_data,
        x_probe,
        opts["init"],
        compat_mode=str(opts.get("compat_mode", "strict_legacy")),
    )
    # Carry updated init option forward.
    opts["init"] = init_opt

    return x_data, x_probe, n_features_original, n_samples_original, row_idx, col_idx


# -- mask helpers -------------------------------------------------------------


def _build_masks_sparse(
    x_data: Matrix,
    x_probe: Matrix | None,
) -> tuple[Matrix, Matrix | None, Matrix, Matrix | None]:
    """Build masks and normalized data for sparse inputs.

    Returns:
        Tuple of data, probe, mask, and probe mask.
    """
    x_csr = sp.csr_matrix(x_data)
    mask_csr = x_csr.copy()
    mask_csr.data[:] = 1.0
    mask: Matrix = mask_csr

    if x_probe is not None:
        if sp.issparse(x_probe):
            x_probe_csr = sp.csr_matrix(x_probe)
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
        x_arr[x_arr == 0.0] = eps
    x_arr[np.isnan(x_arr)] = 0.0
    x_data = x_arr

    if x_probe is not None:
        x_probe_arr = np.asarray(x_probe, dtype=float)
        x_probe = x_probe_arr
        mask_probe = ~np.isnan(x_probe_arr)
        if use_eps_for_zeros:
            x_probe_arr[x_probe_arr == 0.0] = eps
        x_probe_arr[np.isnan(x_probe_arr)] = 0.0
    else:
        mask_probe = None

    return x_data, x_probe, mask, mask_probe


def _build_masks_and_counts(
    x_data: Matrix,
    x_probe: Matrix | None,
    opts: MutableMapping[str, object],
) -> tuple[Matrix, Matrix | None, Matrix, Matrix | None, np.ndarray, float, int]:
    """Build missingness masks, handle NaNs/zeros, and count observations.

    Returns:
        Data, probe, masks, per-row counts, total data count, probe count.
    """
    if sp.issparse(x_data):
        x_data, x_probe, mask, mask_probe = _build_masks_sparse(x_data, x_probe)
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


def _missing_patterns_info(
    mask: Matrix,
    opts: MutableMapping[str, object],
    n_samples: int,
) -> tuple[int, list[list[int]], np.ndarray | None]:
    """Compute missingness-pattern information and Isv mapping.

    Returns:
        Number of patterns, pattern lists, and optional pattern index map.
    """
    if opts.get("uniquesv"):
        mask_arr = (
            mask.toarray() if isinstance(mask, sp.csr_matrix) else np.asarray(mask)
        )
        n_patterns, obs_patterns, isv = _missing_patterns(mask_arr)
        isv_arr = np.array(isv, dtype=int)
        return int(n_patterns), obs_patterns, isv_arr

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
    if ctx.loading_covariances:
        for row_cov in ctx.loading_covariances:
            va_new += np.diag(row_cov)

    va_new = (va_new + 2.0 * ctx.hp_va) / denom
    va_new = np.maximum(va_new, _EPS_VAR)

    return va_new, vmu


# ---------------------------------------------------------------------------
# Bias / mean update
# ---------------------------------------------------------------------------


def _update_bias(
    *,
    bias_enabled: bool,
    bias_state: BiasState,
    err_mx: np.ndarray,
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
    with np.errstate(divide="ignore", invalid="ignore"):
        d_mu = np.divide(
            np.sum(err_mx, axis=1),
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
    if sp.issparse(state.x_data):
        x_csr = sp.csr_matrix(state.x_data)

        def _x_col(j: int) -> np.ndarray:
            return np.asarray(x_csr[:, j].toarray()).ravel()

    else:
        x_arr = np.asarray(state.x_data, dtype=float)

        def _x_col(j: int) -> np.ndarray:
            return np.asarray(x_arr[:, j], dtype=float)

    _n_features, n_samples = state.x_data.shape
    n_components = state.loadings.shape[1]

    # Common psi = A^T A + noise_var I
    ata = state.loadings.T @ state.loadings
    psi = ata + state.noise_var * state.eye_components
    psi = 0.5 * (psi + psi.T)
    try:
        cho = cho_factor(psi, lower=True, check_finite=False)
    except LinAlgError:
        jitter = _EPS_VAR * np.eye(n_components)
        cho = cho_factor(psi + jitter, lower=True, check_finite=False)

    cov_common = state.noise_var * cho_solve(
        cho,
        state.eye_components,
        check_finite=False,
    )

    for j in range(n_samples):
        x_col = _x_col(j)
        rhs = state.loadings.T @ x_col
        state.scores[:, j] = cho_solve(cho, rhs, check_finite=False)
        state.score_covariances[j] = cov_common

        log_progress(
            state.verbose,
            current=j + 1,
            total=n_samples,
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


def _score_accessors(
    state: ScoreState,
) -> tuple[Callable[[int], np.ndarray], Callable[[int], np.ndarray], int]:
    if sp.issparse(state.x_data):
        x_csr = sp.csr_matrix(state.x_data)

        def _x_col(j: int) -> np.ndarray:
            return np.asarray(x_csr[:, j].toarray()).ravel()

    else:
        x_arr = np.asarray(state.x_data, dtype=float)

        def _x_col(j: int) -> np.ndarray:
            return np.asarray(x_arr[:, j], dtype=float)

    if sp.issparse(state.mask):
        mask_csr = sp.csr_matrix(state.mask)

        def _mask_col(j: int) -> np.ndarray:
            return np.asarray(mask_csr[:, j].toarray()).ravel()

    else:
        mask_arr = np.asarray(state.mask, dtype=float)

        def _mask_col(j: int) -> np.ndarray:
            return np.asarray(mask_arr[:, j], dtype=float)

    return _x_col, _mask_col, int(state.x_data.shape[1])


def _score_cholesky(
    state: ScoreState, mask_col: np.ndarray
) -> tuple[ChoFactor, np.ndarray, np.ndarray]:
    loadings_masked = mask_col[:, None] * state.loadings
    psi = loadings_masked.T @ loadings_masked + state.noise_var * state.eye_components

    if state.loading_covariances:
        observed_rows = np.where(mask_col > 0)[0]
        for row_index in observed_rows:
            psi += state.loading_covariances[row_index]

    cho = _safe_cholesky(psi, state.eye_components)
    sv_pattern = state.noise_var * cho_solve(
        cho,
        state.eye_components,
        check_finite=False,
    )
    return cho, loadings_masked, sv_pattern


def _update_scores_for_columns(
    state: ScoreState,
    cols: list[int],
    x_col: Callable[[int], np.ndarray],
    mask_col: Callable[[int], np.ndarray],
    cov_index: int,
) -> None:
    mask_vec = mask_col(cols[0])
    cho, loadings_masked, sv_pattern = _score_cholesky(
        state,
        mask_vec,
    )
    state.score_covariances[cov_index] = sv_pattern

    for j_idx in cols:
        rhs = loadings_masked.T @ x_col(j_idx)
        state.scores[:, j_idx] = cho_solve(cho, rhs, check_finite=False)


def _score_update_general_no_patterns(state: ScoreState) -> ScoreState:
    """General score update when there is no pattern sharing.

    Returns:
        Updated score state.
    """
    x_col, mask_col, n_samples = _score_accessors(state)

    for j in range(n_samples):
        _update_scores_for_columns(
            state,
            cols=[j],
            x_col=x_col,
            mask_col=mask_col,
            cov_index=j,
        )

        log_progress(
            state.verbose,
            current=j + 1,
            total=n_samples,
            phase="Updating scores",
        )

    return state


def _score_update_with_patterns(state: ScoreState) -> ScoreState:
    """Score update when multiple columns share covariance patterns.

    Returns:
        Updated score state.
    """
    x_col, mask_col, _ = _score_accessors(state)

    for pattern_index, cols in enumerate(state.obs_patterns):
        if not cols:
            continue

        _update_scores_for_columns(
            state,
            cols=cols,
            x_col=x_col,
            mask_col=mask_col,
            cov_index=pattern_index,
        )

        log_progress(
            state.verbose,
            current=pattern_index + 1,
            total=len(state.obs_patterns),
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
        dense_mask = None
        fully_observed = False
        if not sp.issparse(state.mask):
            dense_mask = np.asarray(state.mask, dtype=float)
            fully_observed = bool(np.all(dense_mask > 0))

        if dense_mask is not None and fully_observed and not state.loading_covariances:
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
    if sp.issparse(state.x_data):
        x_csr = sp.csr_matrix(state.x_data)

        def _x_row(i: int) -> np.ndarray:
            return np.asarray(x_csr[i, :].toarray()).ravel()

    else:
        x_arr = np.asarray(state.x_data, dtype=float)

        def _x_row(i: int) -> np.ndarray:
            return np.asarray(x_arr[i, :], dtype=float)

    if sp.issparse(state.mask):
        mask_csr = sp.csr_matrix(state.mask)

        def _mask_row(i: int) -> np.ndarray:
            return np.asarray(mask_csr[i, :].toarray()).ravel()

    else:
        mask_arr = np.asarray(state.mask, dtype=float)

        def _mask_row(i: int) -> np.ndarray:
            return np.asarray(mask_arr[i, :], dtype=float)

    return _x_row, _mask_row, int(state.x_data.shape[0])


def _prior_precision_matrix(va: np.ndarray, noise_var: float) -> np.ndarray:
    prior_prec_diag = np.zeros_like(va, dtype=float)
    finite_mask = np.isfinite(va) & (va > 0)
    prior_prec_diag[finite_mask] = noise_var / va[finite_mask]
    return np.diag(prior_prec_diag)


def _score_covariance_for_column(state: LoadingsUpdateState, j_idx: int) -> np.ndarray:
    if state.pattern_index is None:
        return state.score_covariances[j_idx]
    return state.score_covariances[state.pattern_index[j_idx]]


def _phi_for_row(
    mask_vec: np.ndarray,
    state: LoadingsUpdateState,
    prior_prec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores_masked = mask_vec[None, :] * state.scores
    phi = scores_masked @ scores_masked.T + prior_prec

    if state.score_covariances:
        for j_idx in np.where(mask_vec > 0)[0]:
            phi += _score_covariance_for_column(state, int(j_idx))

    return phi, scores_masked


def _loadings_update_fast_dense_no_sv(
    state: LoadingsUpdateState,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
]:
    """Fast path: fully observed dense data, no score covariances.

    Returns:
        Updated loadings and loading covariances.
    """
    if sp.issparse(state.x_data):
        x_csr = sp.csr_matrix(state.x_data)

        def _x_row(i: int) -> np.ndarray:
            return np.asarray(x_csr[i, :].toarray()).ravel()

    else:
        x_arr = np.asarray(state.x_data, dtype=float)

        def _x_row(i: int) -> np.ndarray:
            return np.asarray(x_arr[i, :], dtype=float)

    n_features = state.x_data.shape[0]
    n_components = state.scores.shape[0]

    if state.verbose == 2:
        logger.info("Updating loadings")

    prior_prec_diag = np.zeros_like(state.va, dtype=float)
    finite_mask = np.isfinite(state.va) & (state.va > 0)
    prior_prec_diag[finite_mask] = state.noise_var / state.va[finite_mask]
    prior_prec = np.diag(prior_prec_diag)

    scores_masked = state.scores
    phi = scores_masked @ scores_masked.T + prior_prec
    phi = 0.5 * (phi + phi.T)
    try:
        cho = cho_factor(phi, lower=True, check_finite=False)
    except LinAlgError:
        jitter = _EPS_VAR * np.eye(n_components)
        cho = cho_factor(phi + jitter, lower=True, check_finite=False)

    loadings = np.empty((n_features, n_components), dtype=float)
    for i in range(n_features):
        x_row = _x_row(i)

        rhs = scores_masked @ x_row
        loadings[i, :] = cho_solve(cho, rhs, check_finite=False)

        if state.loading_covariances:
            state.loading_covariances[i] = state.noise_var * cho_solve(
                cho,
                np.eye(n_components),
                check_finite=False,
            )

        log_progress(
            state.verbose,
            current=i + 1,
            total=n_features,
            phase="Updating loadings",
        )

    return loadings, state.loading_covariances


def _loadings_update_general(
    state: LoadingsUpdateState,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
]:
    """General loading update, including masks and score covariances.

    Returns:
        Updated loadings and loading covariances.
    """
    x_row, mask_row, n_features = _loadings_accessors(state)
    n_components = state.scores.shape[0]
    prior_prec = _prior_precision_matrix(state.va, state.noise_var)
    eye_components = np.eye(n_components)

    if state.verbose == 2:
        logger.info("Updating loadings")

    loadings = np.empty((n_features, n_components), dtype=float)

    for i in range(n_features):
        mask_vec = mask_row(i)
        phi, scores_masked = _phi_for_row(mask_vec, state, prior_prec)
        cho = _safe_cholesky(phi, eye_components)

        rhs = scores_masked @ x_row(i)
        loadings[i, :] = cho_solve(cho, rhs, check_finite=False)

        if state.loading_covariances:
            state.loading_covariances[i] = state.noise_var * cho_solve(
                cho,
                eye_components,
                check_finite=False,
            )

        log_progress(
            state.verbose,
            current=i + 1,
            total=n_features,
            phase="Updating loadings",
        )

    return loadings, state.loading_covariances


def _update_loadings(
    state: LoadingsUpdateState,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
]:
    """Update loadings and loading covariances.

    Returns:
        Updated loadings and loading covariances.
    """
    dense_mask = None
    fully_observed = False
    sv_contrib = bool(state.score_covariances)

    if not sp.issparse(state.mask):
        dense_mask = np.asarray(state.mask, dtype=float)
        fully_observed = bool(np.all(dense_mask > 0))

    if dense_mask is not None and fully_observed and not sv_contrib:
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
    cfg_data = RmsConfig(n_observed=int(ctx.n_data), num_cpu=1)
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
        cfg_probe = RmsConfig(n_observed=int(ctx.n_probe), num_cpu=1)
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
        Updated noise state and accumulated ``s_xv``.
    """
    s_xv = 0.0
    loadings = noise_state.loadings
    scores = noise_state.scores
    loading_covariances = noise_state.loading_covariances
    score_covariances = noise_state.score_covariances
    pattern_index = noise_state.pattern_index

    def _accumulate_entry(i: int, j: int, sv_j: np.ndarray) -> None:
        nonlocal s_xv
        a_i = loadings[i, :][None, :]  # (1, k)
        s_xv += float((a_i @ sv_j @ a_i.T)[0, 0].item())
        if not loading_covariances:
            return

        s_j = scores[:, j][:, None]
        cov_term = s_j.T @ loading_covariances[i] @ s_j
        cov_term += np.sum(sv_j * loading_covariances[i])
        s_xv += float(np.asarray(cov_term).item())

    if pattern_index is None:
        for i, j in zip(ix, jx, strict=True):
            _accumulate_entry(int(i), int(j), score_covariances[int(j)])
    else:
        for i, j in zip(ix, jx, strict=True):
            sv_j = score_covariances[int(pattern_index[int(j)])]
            _accumulate_entry(int(i), int(j), sv_j)

    if noise_state.mu_variances.size > 0:
        s_xv += float(np.sum(noise_state.mu_variances[ix, 0]).item())

    s_xv += (rms**2) * noise_state.n_data
    v_new = (s_xv + 2.0 * hp_v) / (noise_state.n_data + 2.0 * hp_v)
    v_new = max(float(v_new), _EPS_VAR)

    updated_state = NoiseState(
        loadings=loadings,
        scores=scores,
        loading_covariances=loading_covariances,
        score_covariances=score_covariances,
        mu_variances=noise_state.mu_variances,
        pattern_index=pattern_index,
        n_data=noise_state.n_data,
        noise_var=v_new,
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
    loading_covariances = ctx.loading_covariances or None

    params = RotateParams(
        loading_covariances=loading_covariances,
        score_covariances=ctx.score_covariances,
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
