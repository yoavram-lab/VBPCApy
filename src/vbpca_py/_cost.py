"""
Full variational cost (free energy) computation for VB PCA.

This module implements `compute_full_cost`, which evaluates the
variational free energy (or negative ELBO) for the VB PCA model
given current posterior parameters.

MATLAB semantic alignment notes (cf_full.m):
- The data term is computed on X that has been mean-centered on the observed set:
    Dense:  Xc = X_clean - repmat(Mu,1,n2) .* M
    Sparse: stored entries in row i are shifted by Mu[i] (mask ignored; observed set is structure)
- Missing dense entries are encoded as NaN (when mask is None) and treated as unobserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.linalg import slogdet

from ._mean import (
    subtract_mu as _subtract_mu_mean,  # sparse path uses stored-entry subtraction
)
from ._rms import RmsConfig, compute_rms

# ============================================================
# Common validation error messages
# ============================================================

ERR_A_S_2D = "A and S must be 2D arrays."
ERR_A_ROWS = "A rows must match X rows."
ERR_S_COLS = "S columns must match X columns."
ERR_LATENT_DIM = "A and S latent dimensions are incompatible."

ERR_MU_SHAPE = "mu must be a 1D array of length n_features."
ERR_MASK_SHAPE = "mask must have the same shape as X."

ERR_MU_PRIOR_VAR = (
    "mu_prior_variance must be non-zero when using a mu prior with mu_variances."
)

ERR_LOADING_PRIORS_REQUIRED = (
    "loading_priors must be provided when priors are enabled for A."
)

ERR_PATTERN_INDEX_SHAPE = "score_pattern_index must have shape (n_samples,)."
ERR_PATTERN_INDEX_BOUNDS = (
    "score_pattern_index contains out-of-range indices for score_covariances."
)
ERR_LOADING_COV_LEN = "loading_covariances length must equal number of features."
ERR_LOADING_COV_SHAPE = (
    "Each loading covariance must have shape (n_components, n_components)."
)


# ============================================================
# Parameter containers
# ============================================================


@dataclass(slots=True)
class CostParams:
    """Grouped parameters for `compute_full_cost`."""

    mu: np.ndarray
    noise_variance: float
    loading_covariances: list[np.ndarray] | None = None
    score_covariances: list[np.ndarray] | None = None
    score_pattern_index: np.ndarray | None = None
    mu_variances: np.ndarray | None = None
    loading_priors: np.ndarray | float | None = None
    mu_prior_variance: float | None = None
    mask: np.ndarray | sp.spmatrix | None = None
    s_xv: float | None = None
    n_data: int | None = None
    num_cpu: int = 1


@dataclass(slots=True)
class ObservationInfo:
    """Observed-entry structure for the data term."""

    mask: np.ndarray | sp.spmatrix
    row_idx: np.ndarray
    col_idx: np.ndarray
    n_data: int | None


@dataclass(slots=True)
class CovarianceInfo:
    """Covariance and variance structure for the data term."""

    n_components: int
    loading_covariances: list[np.ndarray] | None
    score_covariances: list[np.ndarray] | None
    score_pattern_index: np.ndarray | None
    mu_variances: np.ndarray | None
    s_xv: float | None
    num_cpu: int


# ============================================================
# Internal helpers
# ============================================================


def _normalize_mu(mu: np.ndarray, n_features: int) -> np.ndarray:
    """Ensure `mu` is a 1D array of length `n_features`."""
    if mu.ndim == 2 and mu.shape[1] == 1:
        mu = mu[:, 0]
    if mu.ndim != 1 or mu.shape[0] != n_features:
        raise ValueError(ERR_MU_SHAPE)
    return mu


def _coerce_dense_mask_to_bool(mask: np.ndarray) -> np.ndarray:
    """Normalize dense masks to boolean 'observed' semantics."""
    m = np.asarray(mask)
    if m.dtype == bool:
        return m
    # MATLAB uses numeric 0/1 masks frequently; treat nonzero as observed.
    return m != 0


def _build_mask_and_clean_x(
    x: np.ndarray | sp.spmatrix,
    mask: np.ndarray | sp.spmatrix | None,
) -> tuple[np.ndarray | sp.spmatrix, np.ndarray | sp.spmatrix, np.ndarray, np.ndarray]:
    """Construct a mask and clean x for missing values.

    Returns:
        x_clean:
            Dense: NaNs replaced by zeros when mask is None; otherwise unchanged.
            Sparse: unchanged.
        mask_out:
            Dense: boolean observed mask.
            Sparse: sparse boolean mask with the same sparsity structure as x.
        row_idx, col_idx:
            Observed coordinates.
    """
    if mask is None:
        if sp.issparse(x):
            mask_out = x.copy()
            mask_out.data = np.ones_like(mask_out.data, dtype=bool)
            x_clean = x
        else:
            mask_out = ~np.isnan(x)
            x_clean = np.where(mask_out, x, 0.0)
    else:
        mask_out = mask
        x_clean = x

    if mask_out.shape != x_clean.shape:
        raise ValueError(ERR_MASK_SHAPE)

    if sp.issparse(mask_out):
        row_idx, col_idx = mask_out.nonzero()
    else:
        mask_bool = _coerce_dense_mask_to_bool(np.asarray(mask_out))
        mask_out = mask_bool  # type: ignore[assignment]
        row_idx, col_idx = np.where(mask_bool)

    # Reject NaNs on observed entries when caller provides a mask.
    if mask is not None:
        if sp.issparse(mask_out):
            if not sp.issparse(x_clean):
                rows, cols = mask_out.nonzero()
                if rows.size and np.isnan(np.asarray(x_clean))[rows, cols].any():
                    raise ValueError(
                        "X contains NaN on observed entries specified by mask."
                    )
        elif np.isnan(np.asarray(x_clean))[row_idx, col_idx].any():
            raise ValueError("X contains NaN on observed entries specified by mask.")

    return x_clean, mask_out, row_idx, col_idx


def _validate_score_pattern_index(
    score_pattern_index: np.ndarray | None,
    score_covariances: list[np.ndarray] | None,
    n_samples: int,
) -> None:
    if score_pattern_index is None:
        return
    idx = np.asarray(score_pattern_index)
    if idx.shape != (n_samples,):
        raise ValueError(ERR_PATTERN_INDEX_SHAPE)
    if score_covariances is None or len(score_covariances) == 0:
        return
    if idx.size > 0:
        if np.min(idx) < 0 or np.max(idx) >= len(score_covariances):
            raise ValueError(ERR_PATTERN_INDEX_BOUNDS)


def _validate_score_covariances(
    score_covariances: list[np.ndarray] | None,
    n_components: int,
    n_samples: int,
    score_pattern_index: np.ndarray | None,
) -> None:
    if score_covariances is None:
        return

    for sv in score_covariances:
        if sv is None:
            continue
        if sv.shape != (n_components, n_components):
            raise ValueError(
                "Each score covariance must have shape (n_components, n_components)."
            )

    if score_pattern_index is None:
        if len(score_covariances) != n_samples:
            raise ValueError(
                "score_covariances length must equal number of samples when no pattern index is provided."
            )
        return

    idx = np.asarray(score_pattern_index)
    if idx.size == 0:
        return
    if len(score_covariances) <= int(np.max(idx)):
        raise ValueError("score_covariances length must cover all pattern indices.")


def _validate_loading_covariances(
    loading_covariances: list[np.ndarray] | None,
    n_features: int,
    n_components: int,
) -> None:
    if loading_covariances is None:
        return

    if len(loading_covariances) != n_features:
        raise ValueError(ERR_LOADING_COV_LEN)

    for av in loading_covariances:
        if av.shape != (n_components, n_components):
            raise ValueError(ERR_LOADING_COV_SHAPE)


def _center_x_by_mu(
    x_clean: np.ndarray | sp.spmatrix,
    mu: np.ndarray,
    mask_out: np.ndarray | sp.spmatrix,
) -> np.ndarray | sp.spmatrix:
    """
    MATLAB-aligned mean centering used for the data term (cost_x):

    Dense:  Xc = X_clean - mu[:,None] * M   where M is boolean/0-1 observed mask.
    Sparse: subtract mu[row] from each *stored* entry (mask ignored; structure is observed set).
    """
    # If Mu is empty in MATLAB, they skip; in Python we treat "all zeros" as a no-op fast path.
    if mu.size == 0:
        return x_clean

    if sp.issparse(x_clean):
        # Use the same semantics as subtract_mu MEX: shift stored entries by Mu[row].
        # mask is ignored for sparse, but the API requires something; pass mask_out.
        x_out, _ = _subtract_mu_mean(
            mu, x_clean, mask_out, probe=None, update_bias=True
        )
        return x_out

    # Dense
    m_bool = mask_out if isinstance(mask_out, np.ndarray) else np.asarray(mask_out)
    m_bool = _coerce_dense_mask_to_bool(np.asarray(m_bool))
    x_arr = np.asarray(x_clean, dtype=float)
    return x_arr - (mu[:, None] * m_bool)


def _sxv_identity_contrib(
    a: np.ndarray,
    s: np.ndarray,
    obs: ObservationInfo,
    loading_covs: list[np.ndarray] | None,
) -> float:
    """Contribution to s_xv when Sv = I for all columns."""
    _ = s  # scores only used when Av is provided
    contrib = 0.0

    n_features = a.shape[0]
    row_counts = np.bincount(obs.row_idx, minlength=n_features)
    row_norm_sq = np.sum(a * a, axis=1)
    contrib += float(np.sum(row_counts * row_norm_sq))

    if loading_covs is not None and len(loading_covs) > 0:
        for i, j in zip(obs.row_idx, obs.col_idx, strict=True):
            if i >= len(loading_covs):
                continue
            av_i = loading_covs[i]
            s_j = s[:, j]
            contrib += float(s_j @ av_i @ s_j)
            contrib += float(np.trace(av_i))

    return contrib


def _sxv_patterned_fastpath(
    a: np.ndarray,
    obs: ObservationInfo,
    score_covs: list[np.ndarray],
    pattern_index: np.ndarray,
) -> float:
    """Fast path for Sv with pattern indices and no Av."""
    n_features = a.shape[0]
    n_patterns = len(score_covs)

    pat = pattern_index[obs.col_idx]
    counts = np.zeros((n_features, n_patterns), dtype=np.int64)
    np.add.at(counts, (obs.row_idx, pat), 1)

    contrib = 0.0
    for p, sv_p in enumerate(score_covs):
        if sv_p is None:
            continue
        a_sv = a @ sv_p
        per_row = np.sum(a_sv * a, axis=1)
        contrib += float(np.sum(counts[:, p] * per_row))

    return contrib


def _sxv_general_per_observation(
    a: np.ndarray,
    s: np.ndarray,
    obs: ObservationInfo,
    cov: CovarianceInfo,
) -> float:
    """General per-observation Sv/Av contribution."""
    contrib = 0.0
    score_covs = cov.score_covariances
    loading_covs = cov.loading_covariances
    pattern_index = cov.score_pattern_index

    if score_covs is None:
        return 0.0

    for i, j in zip(obs.row_idx, obs.col_idx, strict=True):
        sv_j = (
            score_covs[pattern_index[j]] if pattern_index is not None else score_covs[j]
        )
        a_i = a[i, :]
        contrib += float(a_i @ sv_j @ a_i.T)

        if loading_covs is not None and i < len(loading_covs):
            av_i = loading_covs[i]
            s_j = s[:, j]
            contrib += float(s_j @ av_i @ s_j)
            contrib += float(np.sum(sv_j * av_i))

    return contrib


def _sxv_score_covariance_contrib(
    a: np.ndarray,
    s: np.ndarray,
    obs: ObservationInfo,
    cov: CovarianceInfo,
) -> float:
    """Variance contribution from Sv, Av, or both."""
    score_covs = cov.score_covariances
    loading_covs = cov.loading_covariances
    pattern_index = cov.score_pattern_index

    if score_covs is None:
        return _sxv_identity_contrib(a, s, obs, loading_covs)

    if loading_covs is None and pattern_index is not None:
        return _sxv_patterned_fastpath(a, obs, score_covs, pattern_index)

    return _sxv_general_per_observation(a, s, obs, cov)


def _sxv_mu_variance_contrib(
    obs: ObservationInfo,
    cov: CovarianceInfo,
) -> float:
    mu_vars = cov.mu_variances
    if mu_vars is None or mu_vars.size == 0:
        return 0.0
    return float(np.sum(mu_vars[obs.row_idx]))


def _compute_sxv(
    x_centered: np.ndarray | sp.spmatrix,
    a: np.ndarray,
    s: np.ndarray,
    obs: ObservationInfo,
    cov: CovarianceInfo,
) -> tuple[float, int]:
    """
    Compute expected squared reconstruction error s_xv:

      s_xv = (rms^2) * ndata  +  covariance contributions (Sv/Av/Muv)

    Important:
    `x_centered` must already be mean-centered on observed entries to match MATLAB cf_full semantics.
    """
    n_data_effective = len(obs.row_idx) if obs.n_data is None else obs.n_data

    if n_data_effective <= 0:
        raise ValueError("n_data must be positive for cost computation.")

    if cov.s_xv is not None:
        return float(cov.s_xv), n_data_effective

    rms_config = RmsConfig(
        n_observed=n_data_effective,
        num_cpu=max(int(cov.num_cpu), 1),
    )
    rms, _ = compute_rms(x_centered, a, s, obs.mask, rms_config)
    s_xv_local = float((rms**2) * n_data_effective)

    s_xv_local += _sxv_score_covariance_contrib(a, s, obs, cov)
    s_xv_local += _sxv_mu_variance_contrib(obs, cov)

    return s_xv_local, n_data_effective


def _compute_mu_cost(
    mu: np.ndarray,
    mu_variances: np.ndarray | None,
    n_features: int,
    mu_prior_variance: float | None,
    loading_priors: np.ndarray | float | None,
) -> float:
    cost_mu = 0.0
    use_prior = loading_priors is not None and not np.any(np.isinf(loading_priors))

    if not use_prior:
        if mu_variances is not None and mu_variances.size > 0:
            cost_mu = -0.5 * float(np.sum(np.log(2.0 * np.pi * mu_variances))) - (
                n_features / 2.0
            )
        return cost_mu

    if mu_variances is not None and mu_variances.size > 0:
        if mu_prior_variance is None or mu_prior_variance == 0:
            raise ValueError(ERR_MU_PRIOR_VAR)
        cost_mu = (
            0.5 / mu_prior_variance * float(np.sum(mu**2 + mu_variances))
            - 0.5 * float(np.sum(np.log(mu_variances)))
            + (n_features / 2.0) * np.log(mu_prior_variance)
            - (n_features / 2.0)
        )
    elif mu_prior_variance not in (None, 0):
        cost_mu = 0.5 / mu_prior_variance * float(np.sum(mu**2)) + (
            n_features / 2.0
        ) * np.log(2.0 * np.pi * mu_prior_variance)

    return cost_mu


def _compute_loading_cost(
    a: np.ndarray,
    loading_covariances: list[np.ndarray] | None,
    loading_priors: np.ndarray | float | None,
    n_features: int,
    n_components: int,
) -> float:
    cost_a = 0.0
    use_prior = loading_priors is not None and not np.any(np.isinf(loading_priors))

    if loading_covariances is not None:
        if len(loading_covariances) != n_features:
            raise ValueError(
                "loading_covariances length must equal number of features."
            )
        for av in loading_covariances:
            if av.shape != (n_components, n_components):
                raise ValueError(
                    "Each loading covariance must have shape (n_components, n_components)."
                )

    if not use_prior:
        if loading_covariances is not None and len(loading_covariances) > 0:
            cost_a = -(n_features * n_components / 2.0) * (1.0 + np.log(2.0 * np.pi))
            for i in range(n_features):
                av_i = loading_covariances[i]
                sign, logdet = slogdet(av_i)
                if sign > 0:
                    cost_a -= 0.5 * logdet
                else:
                    cost_a -= 0.5 * (-np.inf)
        return cost_a

    if loading_priors is None:
        raise ValueError(ERR_LOADING_PRIORS_REQUIRED)

    va_arr = np.asarray(loading_priors)

    if loading_covariances is not None and len(loading_covariances) > 0:
        cost_a = (
            0.5 * float(np.sum(np.sum(a**2, axis=0) / va_arr))
            + (n_features / 2.0) * float(np.sum(np.log(va_arr)))
            - (n_features * n_components / 2.0)
        )
        for i in range(n_features):
            av_i = loading_covariances[i]
            trace_term = 0.5 * float(np.sum(np.diag(av_i) / va_arr))
            sign, logdet = slogdet(av_i)
            if sign > 0:
                cost_a += trace_term - 0.5 * logdet
            else:
                cost_a += trace_term - 0.5 * (-np.inf)
    else:
        cost_a = 0.5 * float(np.sum(a**2 / va_arr)) + (n_features / 2.0) * float(
            np.sum(np.log(2.0 * np.pi * va_arr))
        )

    return cost_a


def _compute_score_cost(
    s: np.ndarray,
    score_covariances: list[np.ndarray] | None,
    score_pattern_index: np.ndarray | None,
    n_components: int,
    n_samples: int,
) -> float:
    cost_s = 0.5 * float(np.sum(s**2))

    if score_covariances is not None and len(score_covariances) > 0:
        if score_pattern_index is not None and len(score_pattern_index) > 0:
            for j in range(n_samples):
                sv_idx = score_pattern_index[j]
                sv_j = score_covariances[sv_idx]
                if sv_j is not None:
                    trace_svj = 0.5 * float(np.trace(sv_j))
                    sign, logdet_svj = slogdet(sv_j)
                    if sign > 0:
                        cost_s += trace_svj - 0.5 * logdet_svj
                    else:
                        cost_s += trace_svj - 0.5 * (-np.inf)
        else:
            for j in range(n_samples):
                sv_j = score_covariances[j]
                if sv_j is not None:
                    trace_svj = 0.5 * float(np.trace(sv_j))
                    sign, logdet_svj = slogdet(sv_j)
                    if sign > 0:
                        cost_s += trace_svj - 0.5 * logdet_svj
                    else:
                        cost_s += trace_svj - 0.5 * (-np.inf)

    cost_s -= (n_components * n_samples) / 2.0
    return cost_s


# ============================================================
# Public API
# ============================================================


def compute_full_cost(
    x: np.ndarray | sp.spmatrix,
    a: np.ndarray,
    s: np.ndarray,
    params: CostParams,
) -> tuple[float, float, float, float, float]:
    """
    Compute total cost and components (cost, cost_x, cost_a, cost_mu, cost_s).

    IMPORTANT MATLAB alignment:
    - The data term is computed on mean-centered X (centered on observed set).
    """
    n_features, n_samples = x.shape

    if params.noise_variance <= 0:
        raise ValueError("noise_variance must be positive.")

    if a.ndim != 2 or s.ndim != 2:
        raise ValueError(ERR_A_S_2D)
    if a.shape[0] != n_features:
        raise ValueError(ERR_A_ROWS)
    if s.shape[1] != n_samples:
        raise ValueError(ERR_S_COLS)
    if a.shape[1] != s.shape[0]:
        raise ValueError(ERR_LATENT_DIM)

    n_components = a.shape[1]
    mu = _normalize_mu(params.mu, n_features)

    _validate_score_pattern_index(
        params.score_pattern_index, params.score_covariances, n_samples
    )
    _validate_score_covariances(
        params.score_covariances, n_components, n_samples, params.score_pattern_index
    )
    _validate_loading_covariances(params.loading_covariances, n_features, n_components)

    x_clean, mask_out, row_idx, col_idx = _build_mask_and_clean_x(x, params.mask)

    # Mean-center X on observed entries (MATLAB cf_full semantics)
    x_centered = _center_x_by_mu(x_clean, mu, mask_out)

    obs = ObservationInfo(
        mask=mask_out,
        row_idx=row_idx,
        col_idx=col_idx,
        n_data=params.n_data,
    )

    cov = CovarianceInfo(
        n_components=n_components,
        loading_covariances=params.loading_covariances,
        score_covariances=params.score_covariances,
        score_pattern_index=params.score_pattern_index,
        mu_variances=params.mu_variances,
        s_xv=params.s_xv,
        num_cpu=params.num_cpu,
    )

    s_xv_val, n_data_effective = _compute_sxv(
        x_centered,
        a,
        s,
        obs,
        cov,
    )

    cost_x = 0.5 * s_xv_val / params.noise_variance + 0.5 * n_data_effective * (
        np.log(2.0 * np.pi * params.noise_variance)
    )

    cost_mu = _compute_mu_cost(
        mu=mu,
        mu_variances=params.mu_variances,
        n_features=n_features,
        mu_prior_variance=params.mu_prior_variance,
        loading_priors=params.loading_priors,
    )

    cost_a = _compute_loading_cost(
        a=a,
        loading_covariances=params.loading_covariances,
        loading_priors=params.loading_priors,
        n_features=n_features,
        n_components=n_components,
    )

    cost_s = _compute_score_cost(
        s=s,
        score_covariances=params.score_covariances,
        score_pattern_index=params.score_pattern_index,
        n_components=n_components,
        n_samples=n_samples,
    )

    cost = cost_mu + cost_a + cost_x + cost_s
    return cost, cost_x, cost_a, cost_mu, cost_s
