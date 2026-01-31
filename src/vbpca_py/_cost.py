"""
Full variational cost (free energy) computation for VB PCA.

This module implements `compute_full_cost`, which evaluates the
variational free energy (or negative ELBO) for the VB PCA model
given current posterior parameters:

- x : observed data matrix
- a, s : factor loadings and scores
- params : grouped mean, noise, prior, and covariance parameters

The function returns the total cost and its decomposed components:
data term, mean term, loading term, and latent score term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.linalg import slogdet

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


# ============================================================
# Parameter containers
# ============================================================


@dataclass(slots=True)
class CostParams:
    """Grouped parameters for `compute_full_cost`.

    Attributes:
        mu:
            Mean vector of the observation model. Shape (n_features,) or
            (n_features, 1).
        noise_variance:
            Observation noise variance (V in the original code).
        loading_covariances:
            Optional list of Av[i] covariances for rows of A,
            each of shape (n_components, n_components).
        score_covariances:
            Optional list of Sv[j] or Sv[pattern] covariances
            for columns of S, each of shape (n_components, n_components).
        score_pattern_index:
            Optional mapping from sample index j into
            score_covariances (pattern index).
        mu_variances:
            Optional posterior variances for mu (per feature).
        loading_priors:
            Prior variances for rows/columns of A (Va). Scalar or
            array broadcastable to A.
        mu_prior_variance:
            Prior variance for mu (Vmu).
        mask:
            Optional mask matrix for X; ones/True mark observed entries.
        s_xv:
            Optional precomputed expected squared reconstruction error term.
        n_data:
            Optional number of observed entries; if None, inferred from
            the mask.
        num_cpu:
            Number of CPU threads to use for the RMS computation on the
            sparse path. Values < 1 are coerced to 1.
    """

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


def _build_mask_and_clean_x(
    x: np.ndarray | sp.spmatrix,
    mask: np.ndarray | sp.spmatrix | None,
) -> tuple[np.ndarray | sp.spmatrix, np.ndarray | sp.spmatrix, np.ndarray, np.ndarray]:
    """Construct a mask and clean x for missing values.

    Returns:
        x_clean:
            x with NaNs replaced by zeros in the dense case; unchanged
            for sparse.
        mask_out:
            Mask with ones/True for observed entries and zeros/False
            for missing.
        row_idx:
            Row indices of observed entries.
        col_idx:
            Column indices of observed entries.
    """
    if mask is None:
        if sp.issparse(x):
            # Sparse x: treat all stored entries as observed.
            mask_out = x.copy()
            mask_out.data = np.ones_like(mask_out.data, dtype=bool)
            x_clean = x
        else:
            # Dense x: NaN indicates missing.
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
        row_idx, col_idx = np.where(mask_out)

    return x_clean, mask_out, row_idx, col_idx


def _sxv_identity_contrib(
    a: np.ndarray,
    s: np.ndarray,
    obs: ObservationInfo,
    loading_covs: list[np.ndarray] | None,
) -> float:
    """Contribution to s_xv when Sv = I for all columns."""
    _ = s  # scores only used when Av is provided

    contrib = 0.0

    # a_i^T I a_i = ||a_i||^2, aggregated by row occurrence counts.
    n_features = a.shape[0]
    row_counts = np.bincount(obs.row_idx, minlength=n_features)
    row_norm_sq = np.sum(a * a, axis=1)
    contrib += float(np.sum(row_counts * row_norm_sq))

    # If Av is present, per-observation terms depending on both i and j.
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
    """Fast path for Sv with pattern indices and no Av.

    Aggregates contributions a_i^T Sv_p a_i over (row, pattern) counts.
    """
    n_features = a.shape[0]
    n_patterns = len(score_covs)

    # For each observation (i, j), determine its pattern p and count (i, p).
    pat = pattern_index[obs.col_idx]
    counts = np.zeros((n_features, n_patterns), dtype=np.int64)
    np.add.at(counts, (obs.row_idx, pat), 1)

    contrib = 0.0
    for p, sv_p in enumerate(score_covs):
        if sv_p is None:
            continue
        a_sv = a @ sv_p  # (n_features, k)
        per_row = np.sum(a_sv * a, axis=1)  # a_i^T Sv_p a_i
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

    # Case 1: No Sv provided => Sv = I.
    if score_covs is None:
        return _sxv_identity_contrib(a, s, obs, loading_covs)

    # Case 2: Sv present, no Av, and pattern index available => fast path.
    if loading_covs is None and pattern_index is not None:
        return _sxv_patterned_fastpath(a, obs, score_covs, pattern_index)

    # Case 3: General per-observation path.
    return _sxv_general_per_observation(a, s, obs, cov)


def _sxv_mu_variance_contrib(
    obs: ObservationInfo,
    cov: CovarianceInfo,
) -> float:
    """Contribution from posterior variances of mu."""
    mu_vars = cov.mu_variances
    if mu_vars is None or mu_vars.size == 0:
        return 0.0
    return float(np.sum(mu_vars[obs.row_idx]))


def _compute_sxv(
    x: np.ndarray | sp.spmatrix,
    a: np.ndarray,
    s: np.ndarray,
    obs: ObservationInfo,
    cov: CovarianceInfo,
) -> tuple[float, int]:
    """
    Compute or refine the expected squared reconstruction error s_xv.

    This includes:
    - the deterministic residual contribution (via RMS), and
    - contributions from the posterior covariances of A, S, and mu.
    """
    # Effective number of observed entries
    n_data_effective = len(obs.row_idx) if obs.n_data is None else obs.n_data

    # If caller supplied s_xv explicitly, just use it.
    if cov.s_xv is not None:
        return float(cov.s_xv), n_data_effective

    # RMS reconstruction error (handles sparse/dense internally),
    # using the specified number of CPU threads.
    rms_config = RmsConfig(
        n_observed=n_data_effective,
        num_cpu=max(int(cov.num_cpu), 1),
    )
    rms, _ = compute_rms(x, a, s, obs.mask, rms_config)
    s_xv_local = float((rms**2) * n_data_effective)

    # Add covariance contributions
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
    """Compute the cost contribution from the mean parameter mu."""
    cost_mu = 0.0

    use_prior = loading_priors is not None and not np.any(np.isinf(loading_priors))

    if not use_prior:
        if mu_variances is not None and mu_variances.size > 0:
            # cost if no prior
            cost_mu = -0.5 * float(np.sum(np.log(2.0 * np.pi * mu_variances))) - (
                n_features / 2.0
            )
        return cost_mu

    # Priors enabled
    if mu_variances is not None and mu_variances.size > 0:
        # Full posterior variances given
        if mu_prior_variance is None or mu_prior_variance == 0:
            raise ValueError(ERR_MU_PRIOR_VAR)
        cost_mu = (
            0.5 / mu_prior_variance * float(np.sum(mu**2 + mu_variances))
            - 0.5 * float(np.sum(np.log(mu_variances)))
            + (n_features / 2.0) * np.log(mu_prior_variance)
            - (n_features / 2.0)
        )
    elif mu_prior_variance not in (None, 0):
        # Simple Gaussian prior: mu ~ N(0, mu_prior_variance I)
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
    """Compute the cost contribution from the loading matrix A."""
    cost_a = 0.0

    use_prior = loading_priors is not None and not np.any(np.isinf(loading_priors))

    if not use_prior:
        if loading_covariances is not None and len(loading_covariances) > 0:
            # cost if no prior
            cost_a = -(n_features * n_components / 2.0) * (1.0 + np.log(2.0 * np.pi))
            for i in range(n_features):
                av_i = loading_covariances[i]
                sign, logdet = slogdet(av_i)
                if sign > 0:
                    cost_a -= 0.5 * logdet
                else:
                    cost_a -= 0.5 * (-np.inf)
        return cost_a

    # Priors enabled
    if loading_priors is None:
        raise ValueError(ERR_LOADING_PRIORS_REQUIRED)

    va_arr = np.asarray(loading_priors)

    if loading_covariances is not None and len(loading_covariances) > 0:
        # Full Av given per row
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
        # No Av given: simple Gaussian prior on A with variance Va.
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
    """Compute the cost contribution from the latent scores S."""
    cost_s = 0.5 * float(np.sum(s**2))

    if score_covariances is not None and len(score_covariances) > 0:
        if score_pattern_index is not None and len(score_pattern_index) > 0:
            # Pattern-indexed Sv
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
            # Direct Sv[j] per column
            for j in range(n_samples):
                sv_j = score_covariances[j]
                if sv_j is not None:
                    trace_svj = 0.5 * float(np.trace(sv_j))
                    sign, logdet_svj = slogdet(sv_j)
                    if sign > 0:
                        cost_s += trace_svj - 0.5 * logdet_svj
                    else:
                        cost_s += trace_svj - 0.5 * (-np.inf)

    # Subtract normalizing constant
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
    """Compute the full variational cost (free energy) for VB PCA.

    This is a structured, typed port of Illin & Raiko's 2010 VB PCA
    cost function, decomposed into:

    - cost_x: data term (expected squared reconstruction error)
    - cost_mu: mean term
    - cost_a: loading matrix term
    - cost_s: latent score term

    Args:
        x:
            Observed data matrix of shape (n_features, n_samples). May contain
            NaNs to indicate missing values when `params.mask` is None and
            x is dense. For sparse x, missing entries are typically implicit
            (unstored zeros).
        a:
            Factor loadings of shape (n_features, n_components).
        s:
            Factor scores of shape (n_components, n_samples).
        params:
            Grouped parameters controlling priors, posterior covariances,
            masking, noise variance, and RMS threading.

    Returns:
        cost:
            Total cost (sum of all components).
        cost_x:
            Data term.
        cost_a:
            Loading matrix term.
        cost_mu:
            Mean term.
        cost_s:
            Latent score term.
    """
    n_features, n_samples = x.shape

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

    # Mask / missing handling and observed indices
    x_clean, mask_out, row_idx, col_idx = _build_mask_and_clean_x(x, params.mask)

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

    # Compute s_xv and n_data (data term core)
    s_xv_val, n_data_effective = _compute_sxv(
        x_clean,
        a,
        s,
        obs,
        cov,
    )

    # Data term
    cost_x = 0.5 * s_xv_val / params.noise_variance + 0.5 * n_data_effective * (
        np.log(2.0 * np.pi * params.noise_variance)
    )

    # Mean term
    cost_mu = _compute_mu_cost(
        mu=mu,
        mu_variances=params.mu_variances,
        n_features=n_features,
        mu_prior_variance=params.mu_prior_variance,
        loading_priors=params.loading_priors,
    )

    # Loading term
    cost_a = _compute_loading_cost(
        a=a,
        loading_covariances=params.loading_covariances,
        loading_priors=params.loading_priors,
        n_features=n_features,
        n_components=n_components,
    )

    # Latent score term
    cost_s = _compute_score_cost(
        s=s,
        score_covariances=params.score_covariances,
        score_pattern_index=params.score_pattern_index,
        n_components=n_components,
        n_samples=n_samples,
    )

    cost = cost_mu + cost_a + cost_x + cost_s
    return cost, cost_x, cost_a, cost_mu, cost_s
