"""
Full variational cost (free energy) computation for VB PCA.

This module implements `compute_full_cost`, which evaluates the
variational free energy (negative ELBO) for the VB PCA model given the
current posterior parameters.

MATLAB semantic alignment notes (cf_full.m):
- Data term uses observed-set mean-centering:
    Dense: ``Xc = X_clean - repmat(Mu,1,n2) .* M``.
    Sparse: stored entries in row ``i`` are shifted by ``Mu[i]``; the mask is
    ignored because sparsity encodes the observed set.
- Missing dense entries are encoded as NaN (when mask is None) and treated as
    unobserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.linalg import slogdet

from ._mean import (
    subtract_mu as _subtract_mu_mean,  # sparse path uses stored-entry subtraction
)
from ._rms import RmsConfig, compute_rms
from ._sparsity import validate_mask_compatibility

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
ERR_SCORE_COV_SHAPE = (
    "Each score covariance must have shape (n_components, n_components)."
)
ERR_SCORE_COV_LEN_NO_PATTERN = (
    "score_covariances length must equal number of samples when no pattern "
    "index is provided."
)
ERR_SCORE_COV_LEN_PATTERN = "score_covariances length must cover all pattern indices."
ERR_LOADING_COV_LEN = "loading_covariances length must equal number of features."
ERR_LOADING_COV_SHAPE = (
    "Each loading covariance must have shape (n_components, n_components)."
)
ERR_NAN_ON_OBSERVED = "X contains NaN on observed entries specified by mask."
ERR_NOISE_VARIANCE = "noise_variance must be positive."
ERR_N_DATA_POSITIVE = "n_data must be positive for cost computation."


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
    """Ensure ``mu`` is a 1D array of length ``n_features``.

    Args:
        mu: Candidate mean vector.
        n_features: Expected length of the mean vector.

    Returns:
        Flattened mean vector of length ``n_features``.

    Raises:
        ValueError: If ``mu`` cannot be coerced to the expected shape.
    """
    if mu.ndim == 2 and mu.shape[1] == 1:
        mu = mu[:, 0]
    if mu.ndim != 1 or mu.shape[0] != n_features:
        msg = ERR_MU_SHAPE
        raise ValueError(msg)
    return mu


def _coerce_dense_mask_to_bool(mask: np.ndarray) -> np.ndarray:
    """Normalize dense masks to boolean "observed" semantics.

    Args:
        mask: Dense mask with arbitrary dtype.

    Returns:
        Boolean mask where non-zero entries are treated as observed.
    """
    m = np.asarray(mask)
    if m.dtype == bool:
        return m
    # MATLAB uses numeric 0/1 masks frequently; treat nonzero as observed.
    return m != 0  # type: ignore[no-any-return]


def _ensure_no_nan_on_observed(
    mask_out: np.ndarray | sp.spmatrix,
    x_clean: np.ndarray | sp.spmatrix,
    mask: np.ndarray | sp.spmatrix | None,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
) -> None:
    """Validate that observed entries do not contain NaNs.

    Raises:
        ValueError: If a NaN is found on observed coordinates.
    """
    if mask is None:
        return

    if sp.issparse(mask_out):
        mask_csr = sp.csr_matrix(cast("Any", mask_out))
        if sp.issparse(x_clean):
            x_csr = sp.csr_matrix(cast("Any", x_clean))
            if np.isnan(np.asarray(x_csr.data)).any():
                msg = ERR_NAN_ON_OBSERVED
                raise ValueError(msg)
            return

        rows, cols = mask_csr.nonzero()
        has_nan = rows.size and np.isnan(np.asarray(x_clean))[rows, cols].any()
        if has_nan:
            msg = ERR_NAN_ON_OBSERVED
            raise ValueError(msg)
        return

    if np.isnan(np.asarray(x_clean))[row_idx, col_idx].any():
        msg = ERR_NAN_ON_OBSERVED
        raise ValueError(msg)


def _build_mask_and_clean_x(
    x: np.ndarray | sp.spmatrix,
    mask: np.ndarray | sp.spmatrix | None,
) -> tuple[np.ndarray | sp.spmatrix, np.ndarray | sp.spmatrix, np.ndarray, np.ndarray]:
    """Construct a mask and cleaned data for missing values.

    Args:
        x: Dense or sparse data matrix.
        mask: Optional observed mask (dense or sparse).

    Returns:
        Tuple ``(x_clean, mask_out, row_idx, col_idx)`` where ``x_clean`` has
        NaNs replaced by zeros when ``mask`` is ``None`` (dense case),
        ``mask_out`` is a boolean mask aligned with ``x``, and
        ``row_idx``/``col_idx`` give observed coordinates.

    Raises:
        ValueError: If ``mask`` and ``x`` shapes differ or observed entries
        contain NaNs.
    """
    x_clean: np.ndarray | sp.csr_matrix
    mask_out: np.ndarray | sp.csr_matrix

    if mask is None:
        if sp.issparse(x):
            x_csr = sp.csr_matrix(cast("Any", x))
            mask_csr = sp.csr_matrix(x_csr, copy=True)
            mask_csr.data = np.ones_like(mask_csr.data, dtype=bool)
            x_clean = x_csr
            mask_out = mask_csr
        else:
            x_arr = np.asarray(x, dtype=float)
            mask_dense = ~np.isnan(x_arr)
            mask_out = mask_dense
            x_clean = np.where(mask_dense, x_arr, 0.0)
    elif sp.issparse(x):
        x_csr = sp.csr_matrix(cast("Any", x))
        mask_csr = (
            sp.csr_matrix(cast("Any", mask))
            if sp.issparse(mask)
            else sp.csr_matrix(np.asarray(mask))
        )
        x_clean = x_csr
        mask_out = mask_csr
    else:
        x_arr = np.asarray(x, dtype=float)
        mask_out = np.asarray(mask)
        x_clean = x_arr

    if mask_out.shape != x_clean.shape:
        msg = ERR_MASK_SHAPE
        raise ValueError(msg)

    if sp.issparse(mask_out):
        mask_csr = sp.csr_matrix(cast("Any", mask_out))
        row_idx_sparse, col_idx_sparse = mask_csr.nonzero()
        row_idx_out = np.asarray(row_idx_sparse, dtype=np.intp)
        col_idx_out = np.asarray(col_idx_sparse, dtype=np.intp)
    else:
        mask_dense = np.asarray(mask_out)
        mask_bool = _coerce_dense_mask_to_bool(mask_dense)
        mask_out = mask_bool
        row_idx_dense, col_idx_dense = np.where(mask_bool)
        row_idx_out = np.asarray(row_idx_dense, dtype=np.intp)
        col_idx_out = np.asarray(col_idx_dense, dtype=np.intp)

    _ensure_no_nan_on_observed(mask_out, x_clean, mask, row_idx_out, col_idx_out)

    return x_clean, mask_out, row_idx_out, col_idx_out


def _validate_score_pattern_index(
    score_pattern_index: np.ndarray | None,
    score_covariances: list[np.ndarray] | None,
    n_samples: int,
) -> None:
    """Validate ``score_pattern_index`` against provided covariances.

    Raises:
        ValueError: If shape or bounds are invalid.
    """
    if score_pattern_index is None:
        return
    idx = np.asarray(score_pattern_index)
    if idx.shape != (n_samples,):
        msg = ERR_PATTERN_INDEX_SHAPE
        raise ValueError(msg)
    if score_covariances is None or len(score_covariances) == 0:
        return
    if idx.size > 0 and (np.min(idx) < 0 or np.max(idx) >= len(score_covariances)):
        msg = ERR_PATTERN_INDEX_BOUNDS
        raise ValueError(msg)


def _validate_score_covariances(
    score_covariances: list[np.ndarray] | None,
    n_components: int,
    n_samples: int,
    score_pattern_index: np.ndarray | None,
) -> None:
    """Validate score covariance shapes and lengths.

    Raises:
        ValueError: If shapes or lengths are inconsistent.
    """
    if score_covariances is None:
        return

    for sv in score_covariances:
        if sv is None:
            continue
        if sv.shape != (n_components, n_components):
            msg = ERR_SCORE_COV_SHAPE
            raise ValueError(msg)

    if score_pattern_index is None:
        if len(score_covariances) != n_samples:
            msg = ERR_SCORE_COV_LEN_NO_PATTERN
            raise ValueError(msg)
        return

    idx = np.asarray(score_pattern_index)
    if idx.size == 0:
        return
    if len(score_covariances) <= int(np.max(idx)):
        msg = ERR_SCORE_COV_LEN_PATTERN
        raise ValueError(msg)


def _validate_loading_covariances(
    loading_covariances: list[np.ndarray] | None,
    n_features: int,
    n_components: int,
) -> None:
    """Validate loading covariance shapes and lengths.

    Raises:
        ValueError: If shapes or lengths are inconsistent.
    """
    if loading_covariances is None:
        return

    if len(loading_covariances) != n_features:
        msg = ERR_LOADING_COV_LEN
        raise ValueError(msg)

    for av in loading_covariances:
        if av.shape != (n_components, n_components):
            msg = ERR_LOADING_COV_SHAPE
            raise ValueError(msg)


def _center_x_by_mu(
    x_clean: np.ndarray | sp.spmatrix,
    mu: np.ndarray,
    mask_out: np.ndarray | sp.spmatrix,
) -> np.ndarray | sp.spmatrix:
    """Mean-center ``x_clean`` using MATLAB-aligned cost semantics.

    Dense: ``Xc = X_clean - mu[:, None] * M`` where ``M`` is a boolean/0-1
    observed mask.
    Sparse: subtract ``mu[row]`` from each stored entry (mask ignored; the
    sparsity pattern is the observed set).

    Args:
        x_clean: Data matrix with NaNs already handled.
        mu: Mean vector (1D or column vector).
        mask_out: Observed mask aligned with ``x_clean``.

    Returns:
        Mean-centered data with the same sparsity/density structure as
        ``x_clean``.
    """
    # If Mu is empty in MATLAB, they skip; treat all-zeros as a no-op fast path.
    if mu.size == 0:
        return x_clean

    if sp.issparse(x_clean):
        # Use the same semantics as subtract_mu MEX: shift stored entries by Mu[row].
        # mask is ignored for sparse, but the API requires something; pass mask_out.
        x_sparse = sp.csr_matrix(cast("Any", x_clean))
        mask_sparse = (
            sp.csr_matrix(cast("Any", mask_out))
            if sp.issparse(mask_out)
            else sp.csr_matrix(np.asarray(mask_out))
        )
        x_out, _ = _subtract_mu_mean(
            mu, x_sparse, mask_sparse, probe=None, update_bias=True
        )
        return x_out

    # Dense
    m_bool = mask_out if isinstance(mask_out, np.ndarray) else np.asarray(mask_out)
    m_bool = _coerce_dense_mask_to_bool(np.asarray(m_bool))
    x_arr = np.asarray(x_clean, dtype=float)
    return x_arr - (mu[:, None] * m_bool)  # type: ignore[no-any-return]


def _sxv_identity_contrib(
    a: np.ndarray,
    s: np.ndarray,
    obs: ObservationInfo,
    loading_covs: list[np.ndarray] | None,
) -> float:
    """Contribution to ``s_xv`` when ``Sv = I`` for all columns.

    Args:
        a: Loadings matrix ``(n_features, k)``.
        s: Scores matrix ``(k, n_samples)``.
        obs: Observed-entry metadata.
        loading_covs: Optional per-feature loading covariances.

    Returns:
        Scalar contribution to ``s_xv``.
    """
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
    """Fast path for Sv with pattern indices and no Av.

    Args:
        a: Loadings matrix ``(n_features, k)``.
        obs: Observed-entry metadata.
        score_covs: Patterned score covariances.
        pattern_index: Pattern index per sample.

    Returns:
        Scalar contribution to ``s_xv``.
    """
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
    """General per-observation Sv/Av contribution.

    Args:
        a: Loadings matrix ``(n_features, k)``.
        s: Scores matrix ``(k, n_samples)``.
        obs: Observed-entry metadata.
        cov: Covariance-related configuration.

    Returns:
        Scalar contribution to ``s_xv`` from Sv/Av/Mu covariance terms.
    """
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
    """Variance contribution from Sv, Av, or both.

    Args:
        a: Loadings matrix.
        s: Scores matrix.
        obs: Observed-entry metadata.
        cov: Covariance-related configuration.

    Returns:
        Scalar ``s_xv`` contribution from Sv/Av terms.
    """
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
    """Contribution from posterior mean variance to ``s_xv``.

    Args:
        obs: Observed-entry metadata.
        cov: Covariance-related configuration.

    Returns:
        Scalar contribution from ``mu_variances``.
    """
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
    """Compute expected squared reconstruction error ``s_xv``.

    s_xv = (rms^2) * ndata  +  covariance contributions (Sv/Av/Muv)

    Important:
    ``x_centered`` must already be mean-centered on observed entries to
    match MATLAB cf_full semantics.

    Args:
        x_centered: Mean-centered data matrix.
        a: Loadings matrix.
        s: Scores matrix.
        obs: Observed-entry metadata.
        cov: Covariance-related configuration.

    Returns:
        Tuple ``(s_xv, n_data_effective)`` where ``s_xv`` is the expected
        squared reconstruction error and ``n_data_effective`` counts
        observed entries.

    Raises:
        ValueError: If ``n_data`` is non-positive.
    """
    n_data_effective = len(obs.row_idx) if obs.n_data is None else obs.n_data

    if n_data_effective <= 0:
        msg = ERR_N_DATA_POSITIVE
        raise ValueError(msg)

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
    """Compute the mean-term contribution to the variational cost.

    Args:
        mu: Mean vector (flattened).
        mu_variances: Posterior variances of the mean.
        n_features: Number of observed features.
        mu_prior_variance: Prior variance for ``mu`` when priors enabled.
        loading_priors: Loading priors, used to decide whether mu prior applies.

    Returns:
        Scalar cost contribution from the bias term.

    Raises:
        ValueError: If a mean prior variance is required but missing/zero.
    """
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
            msg = ERR_MU_PRIOR_VAR
            raise ValueError(msg)
        mu_prior_var = float(mu_prior_variance)
        cost_mu = (
            0.5 / mu_prior_var * float(np.sum(mu**2 + mu_variances))
            - 0.5 * float(np.sum(np.log(mu_variances)))
            + (n_features / 2.0) * np.log(mu_prior_var)
            - (n_features / 2.0)
        )
    elif mu_prior_variance is not None and mu_prior_variance != 0:
        mu_prior_var = float(mu_prior_variance)
        cost_mu = 0.5 / mu_prior_var * float(np.sum(mu**2)) + (
            n_features / 2.0
        ) * np.log(2.0 * np.pi * mu_prior_var)
    else:
        cost_mu = 0.0

    return cost_mu


def _loading_cost_no_prior(
    loading_covariances: list[np.ndarray] | None,
    n_features: int,
    n_components: int,
) -> float:
    """Cost for the no-prior case.

    Returns:
        Scalar cost contribution when loadings have no prior.
    """
    if loading_covariances is None or len(loading_covariances) == 0:
        return 0.0

    cost_a = -(n_features * n_components / 2.0) * (1.0 + np.log(2.0 * np.pi))
    for i in range(n_features):
        av_i = loading_covariances[i]
        sign, logdet = slogdet(av_i)
        if sign > 0:
            cost_a -= 0.5 * logdet
        else:
            cost_a -= 0.5 * (-np.inf)
    return cost_a  # type: ignore[no-any-return]


def _loading_cost_with_prior(
    a: np.ndarray,
    loading_covariances: list[np.ndarray] | None,
    va_arr: np.ndarray,
    n_features: int,
    n_components: int,
) -> float:
    """Cost when priors are provided.

    Returns:
        Scalar cost contribution when loadings use priors.
    """
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
        return cost_a

    return 0.5 * float(np.sum(a**2 / va_arr)) + (n_features / 2.0) * float(
        np.sum(np.log(2.0 * np.pi * va_arr))
    )


def _compute_loading_cost(
    a: np.ndarray,
    loading_covariances: list[np.ndarray] | None,
    loading_priors: np.ndarray | float | None,
    n_features: int,
    n_components: int,
) -> float:
    """Compute the loading-term contribution to the variational cost.

    Args:
        a: Loadings matrix.
        loading_covariances: Optional posterior covariances per feature.
        loading_priors: Prior variances for loadings.
        n_features: Number of features.
        n_components: Latent dimensionality.

    Returns:
        Scalar cost contribution from the loadings.

    Raises:
        ValueError: If covariance lengths/shapes are invalid or priors missing.
    """
    use_prior = loading_priors is not None and not np.any(np.isinf(loading_priors))

    if loading_covariances is not None:
        if len(loading_covariances) != n_features:
            msg = ERR_LOADING_COV_LEN
            raise ValueError(msg)
        for av in loading_covariances:
            if av.shape != (n_components, n_components):
                msg = ERR_LOADING_COV_SHAPE
                raise ValueError(msg)

    if not use_prior:
        return _loading_cost_no_prior(loading_covariances, n_features, n_components)

    va_arr = np.asarray(loading_priors)
    return _loading_cost_with_prior(
        a, loading_covariances, va_arr, n_features, n_components
    )


def _compute_score_cost(
    s: np.ndarray,
    score_covariances: list[np.ndarray] | None,
    score_pattern_index: np.ndarray | None,
    n_components: int,
    n_samples: int,
) -> float:
    """Compute the score-term contribution to the variational cost.

    Args:
        s: Scores matrix.
        score_covariances: Optional posterior covariances per sample/pattern.
        score_pattern_index: Optional pattern indices for ``score_covariances``.
        n_components: Latent dimensionality.
        n_samples: Number of samples.

    Returns:
        Scalar cost contribution from the scores.
    """
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
    """Compute total cost and components (cost, cost_x, cost_a, cost_mu, cost_s).

    IMPORTANT MATLAB alignment:
    - The data term is computed on mean-centered X (centered on observed set).

    Args:
        x: Data matrix (dense or sparse).
        a: Loadings matrix.
        s: Scores matrix.
        params: Grouped cost parameters.

    Returns:
        Tuple of total cost and component costs ``(cost, cost_x, cost_a,
        cost_mu, cost_s)``.

    Raises:
        ValueError: If shapes are inconsistent or parameters are invalid.
    """
    n_features, n_samples = x.shape

    if params.noise_variance <= 0:
        msg = ERR_NOISE_VARIANCE
        raise ValueError(msg)

    if a.ndim != 2 or s.ndim != 2:
        msg = ERR_A_S_2D
        raise ValueError(msg)
    if a.shape[0] != n_features:
        msg = ERR_A_ROWS
        raise ValueError(msg)
    if s.shape[1] != n_samples:
        msg = ERR_S_COLS
        raise ValueError(msg)
    if a.shape[1] != s.shape[0]:
        msg = ERR_LATENT_DIM
        raise ValueError(msg)

    n_components = a.shape[1]
    mu = _normalize_mu(params.mu, n_features)

    _validate_score_pattern_index(
        params.score_pattern_index, params.score_covariances, n_samples
    )
    _validate_score_covariances(
        params.score_covariances, n_components, n_samples, params.score_pattern_index
    )
    _validate_loading_covariances(params.loading_covariances, n_features, n_components)

    if params.mask is not None:
        validate_mask_compatibility(
            x,
            params.mask,
            allow_sparse_mask_for_dense=False,
            allow_dense_mask_for_sparse=False,
            context="compute_full_cost",
        )

    mask_data = _build_mask_and_clean_x(x, params.mask)

    # Mean-center X on observed entries (MATLAB cf_full semantics)
    x_centered = _center_x_by_mu(mask_data[0], mu, mask_data[1])

    obs = ObservationInfo(
        mask=mask_data[1],
        row_idx=mask_data[2],
        col_idx=mask_data[3],
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

    return cost_mu + cost_a + cost_x + cost_s, cost_x, cost_a, cost_mu, cost_s
