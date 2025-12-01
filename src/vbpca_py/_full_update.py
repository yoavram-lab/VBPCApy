# vbpca_py/_full_update.py
"""Helper functions for full VB-PCA update.

This module factors out pieces of the :func:`pca_full` implementation
to keep the main loop readable while preserving the original algorithmic
behaviour.  The helpers are intentionally small and focused; shared
state is carried via simple data classes rather than long parameter
lists so that static analysis tools remain happy.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from dataclasses import dataclass

import numpy as np
from scipy.sparse import issparse

from ._mean import Matrix, subtract_mu
from ._missing import _missing_patterns
from ._monitoring import (
    InitResult,
    InitShapes,
    init_params,  # ← add this
    log_progress,
)
from ._remove_empty import remove_empty_entries
from ._rms import compute_rms
from ._rotate import rotate_to_pca

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small state containers
# ---------------------------------------------------------------------------


@dataclass
class BiasState:
    """State required for updating the bias/mean parameter.

    Attributes:
    ----------
    mu:
        Current mean vector of shape (n_features, 1).
    muv:
        Posterior variances of the mean, shape (n_features, 1) or empty.
    noise_var:
        Current noise variance :math:`σ²`.
    vmu:
        Prior variance of the mean.
    n_obs_row:
        Number of observations per feature (row).
    """

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
    """State required for updating scores S and their covariances."""

    x_data: Matrix
    mask: Matrix
    a: np.ndarray
    s: np.ndarray
    av: list[np.ndarray]
    sv: list[np.ndarray]
    pattern_index: np.ndarray | None
    obs_patterns: list[list[int]]
    noise_var: float
    eye_components: np.ndarray
    verbose: int


@dataclass
class NoiseState:
    """State required for updating the noise variance."""

    a: np.ndarray
    s: np.ndarray
    av: list[np.ndarray]
    sv: list[np.ndarray]
    muv: np.ndarray
    pattern_index: np.ndarray | None
    n_data: float
    noise_var: float


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _prepare_data(
    x: Matrix,
    opts: MutableMapping[str, object],
) -> tuple[Matrix, Matrix | None, int, int, np.ndarray, np.ndarray]:
    """Copy input, apply probe handling, and remove empty rows/cols."""
    x_probe_opt = opts.get("xprobe", None)

    if issparse(x):
        x_data: Matrix = x.copy()
    else:
        x_data = np.asarray(x, dtype=float).copy()

    if x_probe_opt is not None and np.size(x_probe_opt) != 0:
        if issparse(x_probe_opt):
            x_probe: Matrix | None = x_probe_opt.copy()  # type: ignore[assignment]
        else:
            x_probe = np.asarray(x_probe_opt, dtype=float).copy()
    else:
        x_probe = None

    n_features_original, n_samples_original = x_data.shape

    x_data, x_probe, row_idx, col_idx, init_opt = remove_empty_entries(
        x_data,
        x_probe,
        opts["init"],
        opts["verbose"],
    )
    # Carry updated init option forward.
    opts["init"] = init_opt

    return x_data, x_probe, n_features_original, n_samples_original, row_idx, col_idx


def _build_masks_and_counts(
    x_data: Matrix,
    x_probe: Matrix | None,
    opts: MutableMapping[str, object],
) -> tuple[Matrix, Matrix | None, Matrix, Matrix | None, np.ndarray, float, int]:
    """Build missingness masks, handle NaNs/zeros, and count observations.

    Returns:
    -------
    x_data, x_probe :
        Possibly modified copies of the input data / probe matrices.
    mask, mask_probe :
        Boolean or numeric masks indicating observed entries.
    n_obs_row :
        Number of observations per feature (row).
    n_data :
        Total number of observed entries.
    n_probe :
        Number of observed probe entries.
    """
    if issparse(x_data):
        mask: Matrix = (x_data != 0).astype(float)
        if x_probe is not None and issparse(x_probe):
            mask_probe: Matrix | None = (x_probe != 0).astype(float)
        elif x_probe is not None:
            x_probe_arr = np.asarray(x_probe, dtype=float)
            x_probe = x_probe_arr
            mask_probe = (x_probe_arr != 0).astype(float)
        else:
            mask_probe = None
    else:
        x_arr = np.asarray(x_data, dtype=float)
        x_data = x_arr
        mask = ~np.isnan(x_arr)

        eps = np.finfo(float).eps
        x_arr[x_arr == 0] = eps
        x_arr[np.isnan(x_arr)] = 0.0

        if x_probe is not None:
            x_probe_arr = np.asarray(x_probe, dtype=float)
            x_probe = x_probe_arr
            mask_probe = ~np.isnan(x_probe_arr)
            x_probe_arr[x_probe_arr == 0] = eps
            x_probe_arr[np.isnan(x_probe_arr)] = 0.0
        else:
            mask_probe = None

    if issparse(mask):
        n_obs_row = np.asarray(mask.sum(axis=1)).ravel()
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
    return np.asarray(i_idx), np.asarray(j_idx)


def _missing_patterns_info(
    mask: Matrix,
    opts: MutableMapping[str, object],
    n_samples: int,
) -> tuple[int, list[list[int]], np.ndarray | None]:
    """Compute missingness-pattern information and Isv mapping."""
    if opts["uniquesv"]:
        n_patterns, obs_patterns, isv = _missing_patterns(mask)
        isv_arr = np.asarray(isv, dtype=int)
        return int(n_patterns), obs_patterns, isv_arr

    n_patterns = n_samples
    obs_patterns: list[list[int]] = []
    return n_patterns, obs_patterns, None


# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------


def _initialize_parameters(
    *,
    x_data: Matrix,
    mask: Matrix,
    n_features: int,
    n_samples: int,
    n_components: int,
    n_patterns: int,
    pattern_index: np.ndarray | None,
    n_obs_row: np.ndarray,
    use_prior: bool,
    use_postvar: bool,
    opts: MutableMapping[str, object],
    x_probe: Matrix | None,
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
    """Initialize A, S, Mu, V, Av, Sv, Muv, Va, Vmu and center data."""
    shapes = InitShapes(
        n_features=n_features,
        n_samples=n_samples,
        n_components=n_components,
        n_obs_patterns=n_patterns,
    )

    init_result: InitResult = init_params(
        opts["init"],
        shapes,
        score_pattern_index=pattern_index,
        rng=np.random.default_rng(),
    )
    a = init_result.a
    s = init_result.s
    mu = init_result.mu.reshape(-1, 1)
    noise_var = float(init_result.v)
    av = init_result.av
    sv = init_result.sv
    muv = init_result.muv.reshape(-1, 1)

    # Priors on A and Mu
    if use_prior:
        va = 1000.0 * np.ones(n_components, dtype=float)
        vmu = 1000.0
    else:
        va = np.full(n_components, np.inf, dtype=float)
        vmu = float("inf")

    # MAP / PPCA: disable posterior variances if needed
    if not use_postvar:
        muv = np.array([], dtype=float)
        av = []

    if not bool(opts["bias"]):
        muv = np.array([], dtype=float)
        vmu = 0.0

    # Initialize Mu from data if empty
    if mu.size == 0:
        if bool(opts["bias"]):
            if issparse(x_data):
                mu_num = np.asarray(x_data.sum(axis=1)).ravel()
            else:
                mu_num = np.sum(x_data, axis=1)
            mu = (mu_num / n_obs_row).reshape(-1, 1)
        else:
            mu = np.zeros((n_features, 1), dtype=float)

    # Initial centering
    x_data, x_probe = subtract_mu(
        mu,
        x_data,
        mask,
        x_probe,
        None,
        update_bias=bool(opts["bias"]),
    )

    return a, s, mu, noise_var, av, sv, muv, va, vmu, x_data, x_probe


def _update_hyperpriors(
    *,
    iteration: int,
    use_prior: bool,
    niter_broadprior: int,
    bias: bool,
    mu: np.ndarray,
    muv: np.ndarray,
    a: np.ndarray,
    av: list[np.ndarray],
    n_features: int,
    hp_va: float,
    hp_vb: float,
    va: np.ndarray,
    vmu: float,
) -> tuple[np.ndarray, float]:
    """Update Va and Vmu after a broad-prior warmup period."""
    if not use_prior or iteration <= niter_broadprior:
        return va, vmu

    if bias:
        vmu_val = float(np.sum(mu**2))
        if muv.size > 0:
            vmu_val += float(np.sum(muv))
        vmu = (vmu_val + 2.0 * hp_va) / (n_features + 2.0 * hp_vb)

    va_new = np.sum(a**2, axis=0)
    if av:
        for row_cov in av:
            va_new = va_new + np.diag(row_cov)
    va_new = (va_new + 2.0 * hp_va) / (n_features + 2.0 * hp_vb)

    return va_new, vmu


# ---------------------------------------------------------------------------
# Bias / mean update
# ---------------------------------------------------------------------------


def _update_bias(
    bias: bool,
    bias_state: BiasState,
    err_mx: np.ndarray,
    centering: CenteringState,
) -> tuple[BiasState, CenteringState]:
    """Bias / mean update, including in-place centering of x / x_probe."""
    if not bias:
        return bias_state, centering

    mu = bias_state.mu
    muv = bias_state.muv
    noise_var = bias_state.noise_var
    vmu = bias_state.vmu
    n_obs_row = bias_state.n_obs_row

    d_mu = np.sum(err_mx, axis=1) / n_obs_row  # (n_features,)

    if muv.size > 0:
        muv = (noise_var / (n_obs_row + noise_var / vmu)).reshape(-1, 1)

    shrink = 1.0 / (1.0 + noise_var / (n_obs_row * vmu))
    shrink = shrink.reshape(-1, 1)
    d_mu_vec = d_mu.reshape(-1, 1)

    mu_old = mu
    mu = shrink * (mu + d_mu_vec)
    d_mu_update = mu - mu_old

    x_data, x_probe = subtract_mu(
        d_mu_update,
        centering.x_data,
        centering.mask,
        centering.x_probe,
        centering.mask_probe,
        update_bias=True,
    )

    updated_bias_state = BiasState(
        mu=mu,
        muv=muv,
        noise_var=noise_var,
        vmu=vmu,
        n_obs_row=n_obs_row,
    )
    updated_centering = CenteringState(
        x_data=x_data,
        x_probe=x_probe,
        mask=centering.mask,
        mask_probe=centering.mask_probe,
    )
    return updated_bias_state, updated_centering


# ---------------------------------------------------------------------------
# Score / covariance update
# ---------------------------------------------------------------------------


def _update_scores(state: ScoreState) -> ScoreState:
    """Update S (scores) and Sv (score covariances)."""
    x_is_sparse = issparse(state.x_data)
    _, n_samples = state.x_data.shape

    if state.pattern_index is None:
        # No pattern sharing: Sv[j] is specific to column j
        for j in range(n_samples):
            if issparse(state.mask):
                m_col = np.asarray(state.mask[:, j].toarray()).ravel()
            else:
                m_col = state.mask[:, j].astype(float)

            a_masked = m_col[:, None] * state.a  # (n_features, n_components)
            psi = a_masked.T @ a_masked + state.noise_var * state.eye_components
            if state.av:
                for i_row in np.where(m_col > 0)[0]:
                    psi = psi + state.av[i_row]

            inv_psi = np.linalg.inv(psi)

            x_col = (
                state.x_data[:, j].toarray().ravel()
                if x_is_sparse
                else state.x_data[:, j]
            )

            state.s[:, j] = inv_psi @ (a_masked.T @ x_col)
            state.sv[j] = state.noise_var * inv_psi

            log_progress(
                state.verbose,
                current=j + 1,
                total=n_samples,
                phase="Updating S",
            )
    else:
        # Pattern sharing: Sv[k] shared for all j in obs_patterns[k]
        n_patterns = len(state.obs_patterns)
        for k, cols in enumerate(state.obs_patterns):
            if not cols:
                continue
            j_rep = cols[0]

            if issparse(state.mask):
                m_col = np.asarray(state.mask[:, j_rep].toarray()).ravel()
            else:
                m_col = state.mask[:, j_rep].astype(float)

            a_masked = m_col[:, None] * state.a
            psi = a_masked.T @ a_masked + state.noise_var * state.eye_components
            if state.av:
                for i_row in np.where(m_col > 0)[0]:
                    psi = psi + state.av[i_row]

            inv_psi = np.linalg.inv(psi)
            state.sv[k] = state.noise_var * inv_psi
            tmp = inv_psi @ a_masked.T  # (n_components, n_features)

            for j_idx in cols:
                x_col = (
                    state.x_data[:, j_idx].toarray().ravel()
                    if x_is_sparse
                    else state.x_data[:, j_idx]
                )
                state.s[:, j_idx] = tmp @ x_col

            log_progress(
                state.verbose,
                current=k + 1,
                total=n_patterns,
                phase="Updating S (patterns)",
            )

    return state


# ---------------------------------------------------------------------------
# Loadings update
# ---------------------------------------------------------------------------


def _update_loadings(
    *,
    x_data: Matrix,
    mask: Matrix,
    s: np.ndarray,
    av: list[np.ndarray],
    sv: list[np.ndarray],
    pattern_index: np.ndarray | None,
    va: np.ndarray,
    noise_var: float,
    verbose: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Update A (loadings) and Av (loading covariances)."""
    x_is_sparse = issparse(x_data)
    n_features, _ = x_data.shape
    n_components = s.shape[0]

    if verbose == 2:
        logger.info("Updating A")

    prior_prec = np.diag(noise_var / va)
    a = np.empty((n_features, n_components), dtype=float)

    for i in range(n_features):
        if issparse(mask):
            m_row = np.asarray(mask[i, :].toarray()).ravel()
        else:
            m_row = mask[i, :].astype(float)

        s_masked = m_row[None, :] * s  # (n_components, n_samples)
        phi = s_masked @ s_masked.T + prior_prec
        obs_cols = np.where(m_row > 0)[0]

        for j_idx in obs_cols:
            phi = phi + (
                sv[j_idx] if pattern_index is None else sv[pattern_index[j_idx]]
            )

        inv_phi = np.linalg.inv(phi)

        x_row = x_data[i, :].toarray().ravel() if x_is_sparse else x_data[i, :]

        a[i, :] = x_row @ s_masked.T @ inv_phi

        if av:
            av[i] = noise_var * inv_phi

        log_progress(verbose, current=i + 1, total=n_features, phase="Updating A")

    return a, av


# ---------------------------------------------------------------------------
# RMS / probe RMS
# ---------------------------------------------------------------------------


def _recompute_rms(
    *,
    x_data: Matrix,
    x_probe: Matrix | None,
    mask: Matrix,
    mask_probe: Matrix | None,
    n_data: float,
    n_probe: int,
    a: np.ndarray,
    s: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Recompute RMS and probe RMS after an update of A/S."""
    rms, err_mx = compute_rms(x_data, a, s, mask, n_data)
    if n_probe > 0 and x_probe is not None and mask_probe is not None:
        prms, _ = compute_rms(x_probe, a, s, mask_probe, n_probe)
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
    """Update noise variance using current posterior covariances."""
    s_xv = 0.0
    if noise_state.pattern_index is None:
        for i, j in zip(ix, jx, strict=True):
            a_i = noise_state.a[i, :][None, :]  # (1, n_components)
            s_xv += float(a_i @ noise_state.sv[j] @ a_i.T)
            if noise_state.av:
                s_j = noise_state.s[:, j][:, None]
                s_xv += float(
                    s_j.T @ noise_state.av[i] @ s_j
                    + np.sum(noise_state.sv[j] * noise_state.av[i])
                )
    else:
        for i, j in zip(ix, jx, strict=True):
            a_i = noise_state.a[i, :][None, :]
            sv_j = noise_state.sv[noise_state.pattern_index[j]]
            s_xv += float(a_i @ sv_j @ a_i.T)
            if noise_state.av:
                s_j = noise_state.s[:, j][:, None]
                s_xv += float(
                    s_j.T @ noise_state.av[i] @ s_j + np.sum(sv_j * noise_state.av[i])
                )

    if noise_state.muv.size > 0:
        s_xv += float(np.sum(noise_state.muv[ix, 0]))

    s_xv = s_xv + (rms**2) * noise_state.n_data
    v_new = (s_xv + 2.0 * hp_v) / (noise_state.n_data + 2.0 * hp_v)

    updated_state = NoiseState(
        a=noise_state.a,
        s=noise_state.s,
        av=noise_state.av,
        sv=noise_state.sv,
        muv=noise_state.muv,
        pattern_index=noise_state.pattern_index,
        n_data=noise_state.n_data,
        noise_var=float(v_new),
    )
    return updated_state, float(s_xv)


# ---------------------------------------------------------------------------
# Final PCA rotation
# ---------------------------------------------------------------------------


def _final_rotation(
    *,
    a: np.ndarray,
    av: list[np.ndarray],
    s: np.ndarray,
    sv: list[np.ndarray],
    mu: np.ndarray,
    pattern_index: np.ndarray | None,
    obs_patterns: list[list[int]],
    bias: bool,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    """Final PCA rotation if no iterative rotation was applied."""
    av_for_rotation = av if av else None
    d_mu_final, a, av, s, sv = rotate_to_pca(
        a,
        av_for_rotation,
        s,
        sv,
        pattern_index,
        obs_patterns,
        update_bias=bias,
    )
    if bias:
        mu = mu + d_mu_final
    return a, av, s, sv, mu
