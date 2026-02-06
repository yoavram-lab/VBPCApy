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
from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import issparse

from ._mean import Matrix, ProbeMatrices, subtract_mu
from ._missing import _missing_patterns
from ._monitoring import InitResult, InitShapes, init_params, log_progress
from ._remove_empty import remove_empty_entries
from ._rms import RmsConfig, compute_rms
from ._rotate import RotateParams, rotate_to_pca

if TYPE_CHECKING:
    from collections.abc import MutableMapping

logger = logging.getLogger(__name__)

_EPS_VAR = 1e-15  # minimum variance for numerical safety


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

    if issparse(x):
        x_data: Matrix = x.copy()
    else:
        x_data = np.array(x, dtype=float).copy()

    if x_probe_opt is not None and np.size(x_probe_opt) != 0:
        if issparse(x_probe_opt):
            x_probe: Matrix | None = x_probe_opt.copy()  # type: ignore[assignment]
        else:
            x_probe = np.array(x_probe_opt, dtype=float).copy()
    else:
        x_probe = None

    n_features_original, n_samples_original = x_data.shape

    x_data, x_probe, row_idx, col_idx, init_opt = remove_empty_entries(
        x_data,
        x_probe,
        opts["init"],
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
    mask: Matrix = (x_data != 0).astype(float)

    if x_probe is not None and issparse(x_probe):
        mask_probe: Matrix | None = (x_probe != 0).astype(float)
    elif x_probe is not None:
        x_probe_arr = np.array(x_probe, dtype=float)
        x_probe = x_probe_arr
        mask_probe = (x_probe_arr != 0).astype(float)
    else:
        mask_probe = None

    return x_data, x_probe, mask, mask_probe


def _build_masks_dense(
    x_data: Matrix,
    x_probe: Matrix | None,
) -> tuple[Matrix, Matrix | None, Matrix, Matrix | None]:
    """Build masks and normalized data for dense inputs.

    Returns:
        Tuple of data, probe, mask, and probe mask.
    """
    x_arr = np.array(x_data, dtype=float, copy=False)
    mask = ~np.isnan(x_arr)

    eps = np.finfo(float).eps
    x_arr[x_arr == 0.0] = eps
    x_arr[np.isnan(x_arr)] = 0.0
    x_data = x_arr

    if x_probe is not None:
        x_probe_arr = np.array(x_probe, dtype=float, copy=False)
        x_probe = x_probe_arr
        mask_probe = ~np.isnan(x_probe_arr)
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
    if issparse(x_data):
        x_data, x_probe, mask, mask_probe = _build_masks_sparse(x_data, x_probe)
    else:
        x_data, x_probe, mask, mask_probe = _build_masks_dense(x_data, x_probe)

    if issparse(mask):
        n_obs_row = np.array(mask.sum(axis=1)).ravel()
    else:
        n_obs_row = np.sum(mask, axis=1)
    n_data = float(np.sum(n_obs_row))

    if mask_probe is None:
        n_probe = 0
    elif issparse(mask_probe):
        n_probe = int(mask_probe.count_nonzero())
    else:
        n_probe = int(np.count_nonzero(mask_probe))

    if n_probe == 0:
        x_probe = None
        mask_probe = None
        # Turn off early stopping if there is no probe set.
        opts["earlystop"] = 0

    return x_data, x_probe, mask, mask_probe, n_obs_row, n_data, n_probe


def _observed_indices(x_data: Matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return indices of observed entries in ``x_data``."""
    if issparse(x_data):
        i_idx, j_idx = x_data.nonzero()
    else:
        i_idx, j_idx = np.nonzero(x_data)
    return np.array(i_idx), np.array(j_idx)


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
        n_patterns, obs_patterns, isv = _missing_patterns(mask)
        isv_arr = np.array(isv, dtype=int)
        return int(n_patterns), obs_patterns, isv_arr

    n_patterns = n_samples
    obs_patterns: list[list[int]] = []
    return n_patterns, obs_patterns, None


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
    shapes = ctx.shapes

    # Use a deterministic RNG only when we fall back to random init; when
    # init is provided (e.g., MATLAB fixture), pass through without forcing a
    # new seed.
    rng = None if ctx.opts.get("init") else np.random.default_rng()
    init_result: InitResult = init_params(
        ctx.opts["init"],
        shapes,
        score_pattern_index=ctx.pattern_index,
        rng=rng,
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
        va = np.full(shapes.n_components, 1000.0, dtype=float)
        vmu = 1000.0
    else:
        va = np.full(shapes.n_components, np.inf, dtype=float)
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
            if issparse(ctx.x_data):
                mu_num = np.array(ctx.x_data.sum(axis=1)).ravel()
            else:
                mu_num = np.sum(
                    np.array(ctx.x_data, dtype=float, copy=False),
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
            mu = np.zeros((shapes.n_features, 1), dtype=float)

    # Initial centering using subtract_mu helper.
    probe_container: ProbeMatrices | None = None
    if ctx.x_probe is not None and ctx.mask_probe is not None:
        probe_container = ProbeMatrices(x=ctx.x_probe, mask=ctx.mask_probe)

    x_data_centered, x_probe_centered = subtract_mu(
        mu,
        ctx.x_data,
        ctx.mask,
        probe=probe_container,
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
            muv_vec = np.divide(
                noise_var,
                denom,
                out=np.zeros_like(denom, dtype=float),
                where=denom > 0,
            )
        mu_variances = muv_vec.reshape(-1, 1)

    # Shrinkage factor for mu update, stable even for zero n_obs_row.
    if vmu > 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            shrink = 1.0 / (1.0 + noise_var / (n_obs_row * vmu))
        # Rows with no observations → shrink = 0
        shrink = np.where(n_obs_row > 0, shrink, 0.0)
    else:
        shrink = np.zeros_like(n_obs_row, dtype=float)
    shrink = shrink.reshape(-1, 1)

    d_mu_vec = d_mu.reshape(-1, 1)
    mu_old = mu
    mu = shrink * (mu + d_mu_vec)
    d_mu_update = mu - mu_old

    # Recentre data and probe using the updated mean increment.
    probe_container: ProbeMatrices | None = None
    if centering.x_probe is not None and centering.mask_probe is not None:
        probe_container = ProbeMatrices(
            x=centering.x_probe,
            mask=centering.mask_probe,
        )

    x_data_new, x_probe_new = subtract_mu(
        d_mu_update,
        centering.x_data,
        centering.mask,
        probe=probe_container,
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
    x_is_sparse = issparse(state.x_data)
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
        if x_is_sparse:
            x_col = state.x_data[:, j].toarray().ravel()
        else:
            x_col = np.array(state.x_data[:, j], dtype=float, copy=False)
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


def _score_update_general_no_patterns(state: ScoreState) -> ScoreState:
    """General score update when there is no pattern sharing.

    Returns:
        Updated score state.
    """
    x_is_sparse = issparse(state.x_data)
    _n_features, n_samples = state.x_data.shape
    n_components = state.loadings.shape[1]

    for j in range(n_samples):
        if issparse(state.mask):
            mask_col = np.array(state.mask[:, j].toarray()).ravel()
        else:
            mask_col = np.array(state.mask[:, j], dtype=float, copy=False)

        loadings_masked = mask_col[:, None] * state.loadings
        psi = (
            loadings_masked.T @ loadings_masked + state.noise_var * state.eye_components
        )

        if state.loading_covariances:
            observed_rows = np.where(mask_col > 0)[0]
            for row_index in observed_rows:
                psi += state.loading_covariances[row_index]

        psi = 0.5 * (psi + psi.T)
        try:
            cho = cho_factor(psi, lower=True, check_finite=False)
        except LinAlgError:
            jitter = _EPS_VAR * np.eye(n_components)
            cho = cho_factor(psi + jitter, lower=True, check_finite=False)

        if x_is_sparse:
            x_col = state.x_data[:, j].toarray().ravel()
        else:
            x_col = np.array(state.x_data[:, j], dtype=float, copy=False)

        rhs = loadings_masked.T @ x_col
        state.scores[:, j] = cho_solve(cho, rhs, check_finite=False)
        state.score_covariances[j] = state.noise_var * cho_solve(
            cho,
            state.eye_components,
            check_finite=False,
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
    x_is_sparse = issparse(state.x_data)
    _n_features, _n_samples = state.x_data.shape
    n_components = state.loadings.shape[1]
    n_patterns = len(state.obs_patterns)

    for pattern_index, cols in enumerate(state.obs_patterns):
        if not cols:
            continue

        j_rep = cols[0]
        if issparse(state.mask):
            mask_col = np.array(state.mask[:, j_rep].toarray()).ravel()
        else:
            mask_col = np.array(state.mask[:, j_rep], dtype=float, copy=False)

        loadings_masked = mask_col[:, None] * state.loadings
        psi = (
            loadings_masked.T @ loadings_masked + state.noise_var * state.eye_components
        )

        if state.loading_covariances:
            observed_rows = np.where(mask_col > 0)[0]
            for row_index in observed_rows:
                psi += state.loading_covariances[row_index]

        psi = 0.5 * (psi + psi.T)
        try:
            cho = cho_factor(psi, lower=True, check_finite=False)
        except LinAlgError:
            jitter = _EPS_VAR * np.eye(n_components)
            cho = cho_factor(psi + jitter, lower=True, check_finite=False)

        sv_pattern = state.noise_var * cho_solve(
            cho,
            state.eye_components,
            check_finite=False,
        )
        state.score_covariances[pattern_index] = sv_pattern

        loadings_masked_t = loadings_masked.T
        for j_idx in cols:
            if x_is_sparse:
                x_col = state.x_data[:, j_idx].toarray().ravel()
            else:
                x_col = np.array(
                    state.x_data[:, j_idx],
                    dtype=float,
                    copy=False,
                )
            rhs = loadings_masked_t @ x_col
            state.scores[:, j_idx] = cho_solve(cho, rhs, check_finite=False)

        log_progress(
            state.verbose,
            current=pattern_index + 1,
            total=n_patterns,
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
        if not issparse(state.mask):
            dense_mask = np.array(state.mask, dtype=float, copy=False)
            fully_observed = np.all(dense_mask > 0)

        if dense_mask is not None and fully_observed and not state.loading_covariances:
            return _score_update_fast_dense_no_av(state)

        return _score_update_general_no_patterns(state)

    # Pattern-sharing branch
    return _score_update_with_patterns(state)


# ---------------------------------------------------------------------------
# Loadings update
# ---------------------------------------------------------------------------


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
    x_is_sparse = issparse(state.x_data)
    n_features, _ = state.x_data.shape
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
        if x_is_sparse:
            x_row = state.x_data[i, :].toarray().ravel()
        else:
            x_row = np.array(state.x_data[i, :], dtype=float, copy=False)

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
    x_is_sparse = issparse(state.x_data)
    n_features, _ = state.x_data.shape
    n_components = state.scores.shape[0]

    if state.verbose == 2:
        logger.info("Updating loadings")

    prior_prec_diag = np.zeros_like(state.va, dtype=float)
    finite_mask = np.isfinite(state.va) & (state.va > 0)
    prior_prec_diag[finite_mask] = state.noise_var / state.va[finite_mask]
    prior_prec = np.diag(prior_prec_diag)

    loadings = np.empty((n_features, n_components), dtype=float)

    for i in range(n_features):
        if issparse(state.mask):
            mask_row = np.array(state.mask[i, :].toarray()).ravel()
        else:
            mask_row = np.array(state.mask[i, :], dtype=float, copy=False)

        scores_masked = mask_row[None, :] * state.scores
        phi = scores_masked @ scores_masked.T + prior_prec

        observed_cols = np.where(mask_row > 0)[0]
        for j_idx in observed_cols:
            if not state.score_covariances:
                continue
            sv_j = (
                state.score_covariances[j_idx]
                if state.pattern_index is None
                else state.score_covariances[state.pattern_index[j_idx]]
            )
            phi += sv_j

        phi = 0.5 * (phi + phi.T)
        try:
            cho = cho_factor(phi, lower=True, check_finite=False)
        except LinAlgError:
            jitter = _EPS_VAR * np.eye(n_components)
            cho = cho_factor(phi + jitter, lower=True, check_finite=False)

        if x_is_sparse:
            x_row = state.x_data[i, :].toarray().ravel()
        else:
            x_row = np.array(state.x_data[i, :], dtype=float, copy=False)

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

    if not issparse(state.mask):
        dense_mask = np.array(state.mask, dtype=float, copy=False)
        fully_observed = np.all(dense_mask > 0)

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
    rms, err_mx = compute_rms(ctx.x_data, ctx.loadings, ctx.scores, ctx.mask, cfg_data)

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

    if pattern_index is None:
        for i, j in zip(ix, jx, strict=True):
            a_i = loadings[i, :][None, :]  # (1, k)
            sv_j = score_covariances[j]
            s_xv += float((a_i @ sv_j @ a_i.T)[0, 0])
            if loading_covariances:
                s_j = scores[:, j][:, None]
                s_xv += float(
                    s_j.T @ loading_covariances[i] @ s_j
                    + np.sum(sv_j * loading_covariances[i]),
                )
    else:
        for i, j in zip(ix, jx, strict=True):
            a_i = loadings[i, :][None, :]
            sv_j = score_covariances[pattern_index[j]]
            s_xv += float((a_i @ sv_j @ a_i.T)[0, 0])
            if loading_covariances:
                s_j = scores[:, j][:, None]
                s_xv += float(
                    s_j.T @ loading_covariances[i] @ s_j
                    + np.sum(sv_j * loading_covariances[i]),
                )

    if noise_state.mu_variances.size > 0:
        s_xv += float(np.sum(noise_state.mu_variances[ix, 0]))

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
