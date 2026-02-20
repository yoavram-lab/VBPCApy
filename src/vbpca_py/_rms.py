# compute_rms.py (updated to match MATLAB semantics)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import scipy.sparse as sp

from ._sparse_error import sparse_reconstruction_error
from ._sparsity import validate_mask_compatibility


@dataclass(frozen=True)
class RmsConfig:
    # If provided, we validate it; for sparse we primarily derive from data.nnz
    n_observed: int | None = None
    num_cpu: int = 1
    # If True, enforce MATLAB legacy rule that sparse observation set is the
    # structure of X.
    validate_sparse_mask: bool = True


def compute_rms(
    data: np.ndarray | sp.spmatrix,
    loadings: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray | sp.spmatrix | None,
    config: RmsConfig,
) -> tuple[float, np.ndarray | sp.spmatrix]:
    """MATLAB-compatible compute_rms.

    Dense:
        err = (X - A @ S) ⊙ M
        rms = sqrt(sum(err^2) / ndata)   where ndata = sum(M)

    Sparse:
        err = errpca_pt(X, A, S, numCPU)   (mask ignored)
        rms = sqrt(sum(err^2) / ndata)     where ndata = nnz(X)

    Notes:
        - For sparse, the observation set is the sparsity pattern of ``data``.
          (Observed zeros must be stored explicitly as eps, as in legacy code.)

    Returns:
        rms: Root-mean-squared reconstruction error.
        err: Residual matrix (sparse for sparse input, dense otherwise).
    """
    # MATLAB: if isempty(X), errMx=[]; rms=NaN
    data_is_sparse = sp.issparse(data)
    if data_is_sparse:
        if isinstance(data, sp.csr_matrix):
            data_csr = data
        else:
            data_csr = sp.csr_matrix(cast("Any", data))
        if data_csr.size == 0:
            return float("nan"), np.array([])
        _validate_shapes(data_csr, loadings, scores)
        if mask is not None:
            validate_mask_compatibility(
                data_csr,
                mask,
                allow_dense_mask_for_sparse=False,
                allow_sparse_mask_for_dense=True,
                context="compute_rms",
            )
        return _compute_rms_sparse(
            data_csr=data_csr,
            loadings=loadings,
            scores=scores,
            mask=mask,
            config=config,
        )

    data_arr = np.asarray(data, dtype=float)
    if data_arr.size == 0:
        return float("nan"), np.array([])
    _validate_shapes(data_arr, loadings, scores)
    if mask is not None:
        validate_mask_compatibility(
            data_arr,
            mask,
            allow_sparse_mask_for_dense=False,
            allow_dense_mask_for_sparse=False,
            context="compute_rms",
        )
    return _compute_rms_dense(
        data_arr=data_arr,
        loadings=loadings,
        scores=scores,
        mask=mask,
        config=config,
    )


def _compute_rms_sparse(
    *,
    data_csr: sp.csr_matrix,
    loadings: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray | sp.spmatrix | None,
    config: RmsConfig,
) -> tuple[float, sp.csr_matrix]:
    """Compute RMS for sparse data.

    Returns:
        Tuple of ``(rms, err)`` where ``err`` is CSR residual matrix.

    Raises:
        ValueError: If provided mask conflicts with sparse structure or
            n_observed validation fails.
    """
    num_cpu = max(int(config.num_cpu), 1)
    if config.validate_sparse_mask and mask is not None:
        _validate_sparse_mask_matches_structure(data_csr, mask)

    err = sparse_reconstruction_error(data_csr, loadings, scores, num_cpu=num_cpu)

    n_obs = int(data_csr.nnz)
    if config.n_observed is not None and int(config.n_observed) != n_obs:
        msg = (
            "n_observed mismatch for sparse data: "
            f"config={config.n_observed}, data.nnz={n_obs}"
        )
        raise ValueError(msg)

    rms = float(np.sqrt(np.sum(err.data**2) / n_obs))
    return rms, err


def _compute_rms_dense(
    *,
    data_arr: np.ndarray,
    loadings: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray | sp.spmatrix | None,
    config: RmsConfig,
) -> tuple[float, np.ndarray]:
    """Compute RMS for dense data with mask application.

    Returns:
        Tuple of ``(rms, err)`` where ``err`` is the masked residual.

    Raises:
        ValueError: If mask is missing or incompatible with data shape, or if
            observed-count validation fails.
    """
    if mask is None:
        msg = "mask must be provided for dense data."
        raise ValueError(msg)

    mask_arr = np.asarray(mask, dtype=float)

    if mask_arr.shape != data_arr.shape:
        msg = "mask shape must match data shape."
        raise ValueError(msg)

    n_obs = int(np.count_nonzero(mask_arr))
    if n_obs <= 0:
        msg = "mask must mark at least one observed entry (n_obs > 0)."
        raise ValueError(msg)

    if config.n_observed is not None and int(config.n_observed) != n_obs:
        msg = (
            "n_observed mismatch for dense data: "
            f"config={config.n_observed}, sum(mask)={n_obs}"
        )
        raise ValueError(msg)

    residual = data_arr - (loadings @ scores)
    err = residual * mask_arr

    rms = float(np.sqrt(np.sum(err**2) / n_obs))
    return rms, err


def _validate_shapes(
    data: np.ndarray | sp.spmatrix,
    loadings: np.ndarray,
    scores: np.ndarray,
) -> None:
    """Validate matrix shapes for RMS computation.

    Raises:
        ValueError: If dimensionalities are incompatible.
    """
    if loadings.ndim != 2 or scores.ndim != 2:
        msg = "loadings and scores must be 2-D arrays."
        raise ValueError(msg)

    n_features, n_samples = data.shape
    if loadings.shape[0] != n_features:
        msg = f"loadings has {loadings.shape[0]} rows but data has {n_features} rows."
        raise ValueError(msg)
    if scores.shape[1] != n_samples:
        msg = f"scores has {scores.shape[1]} columns but data has {n_samples} columns."
        raise ValueError(msg)
    if loadings.shape[1] != scores.shape[0]:
        msg = (
            "Incompatible latent dims: "
            f"loadings.shape[1]={loadings.shape[1]}, "
            f"scores.shape[0]={scores.shape[0]}."
        )
        raise ValueError(msg)


def _validate_sparse_mask_matches_structure(
    x: sp.csr_matrix,
    mask: np.ndarray | sp.spmatrix,
) -> None:
    """Enforce MATLAB-legacy sparse semantics.

    Observed set is encoded by ``X`` itself. Therefore a provided mask must
    match ``spones(X)`` exactly (or be ``None``).

    Raises:
        ValueError: If the provided mask does not match the sparsity pattern.
    """
    if sp.issparse(mask):
        mask_csr = (
            mask
            if isinstance(mask, sp.csr_matrix)
            else sp.csr_matrix(cast("Any", mask))
        )
        # Compare structure (indptr/indices); mask values are irrelevant beyond
        # nonzero-ness.
        if not (
            np.array_equal(mask_csr.indptr, x.indptr)
            and np.array_equal(mask_csr.indices, x.indices)
        ):
            msg = (
                "For sparse data, mask must match the sparsity pattern of "
                "data (spones(X))."
            )
            raise ValueError(msg)
    else:
        # Dense mask: must be 1 on all nonzeros of X and 0 elsewhere to match
        # structure exactly. Strict check (exact spones): mask must be 0/1 and
        # sum must equal nnz and cover all nnz coords.
        m = np.asarray(mask)
        if m.shape != x.shape:
            msg = "mask shape must match data shape."
            raise ValueError(msg)
        rows, cols = x.nonzero()
        if not np.all(m[rows, cols] == 1):
            msg = (
                "For sparse data, mask must match the sparsity pattern of "
                "data (spones(X)): it must be 1 on all observed (stored) "
                "entries of X."
            )
            raise ValueError(msg)
        if int(np.sum(m != 0)) != int(x.nnz):
            msg = (
                "For sparse data, mask must match the sparsity pattern of "
                "data (spones(X)): it must be zero everywhere except stored "
                "entries of X."
            )
            raise ValueError(msg)
