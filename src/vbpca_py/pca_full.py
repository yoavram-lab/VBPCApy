# src/vbpca_py/pca_full.py

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping
from scipy.sparse import spmatrix

from ._expand import _add_m_cols, _add_m_rows
from ._monitoring import (
    _initial_monitoring,
)
from ._options import _options

logger = logging.getLogger(__name__)

Array = np.ndarray
Sparse = spmatrix
Matrix = Array | Sparse


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pca_full(x: Matrix, n_components: int, **kwargs: object) -> dict[str, Any]:
    """Variational Bayesian PCA with full posterior covariances (VBPCA).

    This is an idiomatic Python translation of Ilin & Raiko's PCA_FULL
    (JMLR 2010). It supports dense and sparse inputs with missing values,
    MAP/VB/PPCA variants, and optional uniqueness of score covariances.

    Parameters
    ----------
    x :
        Data matrix of shape (n_features, n_samples). May be a NumPy
        array (with NaNs marking missing values) or a SciPy sparse
        matrix with only observed values stored (zeros are treated as
        missing unless explicitly set to eps by the caller).
    n_components :
        Number of latent components.

    Other Parameters (via **kwargs)
    --------------------------------
    init : {"random", dict, str}, optional
        Initialization. "random" uses random orthogonal loadings;
        a dict is passed to the initializer; a string is treated as a
        MATLAB .mat filename (if supported by your init code).
    maxiters : int, optional
        Maximum number of iterations (default 1000).
    algorithm : {"vb", "map", "ppca"}, optional
        Inference scheme: variational Bayes (default), MAP, or PPCA.
    bias : bool, optional
        Whether to estimate a bias/mean term (default True).
    uniquesv : bool, optional
        Whether to compute only unique covariance matrices for scores.
    xprobe : array-like, optional
        Probe/validation data of the same shape as x.
    rotate2pca : bool, optional
        If True, apply rotation towards PCA basis each iteration.
        If False, rotate once at the end.
    verbose : int, optional
        Verbosity level for logging (0, 1, or 2).
    display : int, optional
        If non-zero and matplotlib is available, show RMS plots.
    autosave : float, optional
        Accepted for compatibility but ignored (no .mat autosaving).
    filename : str, optional
        Accepted for compatibility but ignored.
    niter_broadprior : int, optional
        Number of iterations before hyperprior updates.

    Returns:
    -------
    result : dict
        Dictionary with keys:
        - "A", "S", "Mu", "V", "Av", "Sv", "Isv", "Muv"
        - "Va", "Vmu", "lc"
        - "cv": {"A", "S", "Isv", "Mu"}
        - "hp": {"Va", "Vmu"}
    """
    opts = _build_options(kwargs)
    use_prior, use_postvar = _select_algorithm(opts)

    X, Xprobe, n1x, n2x, Ir, Ic = _prepare_data(x, opts)
    M, Mprobe, Nobs_i, ndata, nprobe = _build_masks_and_counts(X, Xprobe, opts)
    IX, JX = _observed_indices(X)

    n1, n2 = X.shape
    nobscomb, obscombj, Isv_use = _missing_patterns_info(M, opts, n2)

    (
        A,
        S,
        Mu,
        V,
        Av,
        Sv,
        Muv,
        Va,
        Vmu,
        X,
        Xprobe,
    ) = _initialize_parameters(
        X=X,
        Xprobe=Xprobe,
        M=M,
        Nobs_i=Nobs_i,
        n1=n1,
        n2=n2,
        n_components=n_components,
        nobscomb=nobscomb,
        Isv_use=Isv_use,
        use_prior=use_prior,
        use_postvar=use_postvar,
        opts=opts,
    )

    # Initial RMS / logging state
    rms, errMx, prms, lc, dsph = _initial_monitoring(
        X, Xprobe, M, ndata, nprobe, A, S, opts
    )
    Aold = A.copy()

    # Hyperparams for Va, Vmu, V
    hpVa = 0.001
    hpVb = 0.001
    hpV = 0.001

    time_start = time.time()
    verbose = int(opts["verbose"])
    rotate_each_iter = int(opts["rotate2pca"])
    eye_components = np.eye(n_components, dtype=float)

    # ------------------------------------------------------------------
    # Main EM / VB loop
    # ------------------------------------------------------------------
    for iteration in range(1, int(opts["maxiters"]) + 1):
        Va, Vmu = _update_hyperpriors(
            iteration=iteration,
            use_prior=use_prior,
            niter_broadprior=int(opts["niter_broadprior"]),
            bias=bool(opts["bias"]),
            Mu=Mu,
            Muv=Muv,
            A=A,
            Av=Av,
            n1=n1,
            hpVa=hpVa,
            hpVb=hpVb,
            Va=Va,
            Vmu=Vmu,
        )

        Mu, Muv, X, Xprobe = _update_bias(
            bias=bool(opts["bias"]),
            Mu=Mu,
            Muv=Muv,
            V=V,
            Vmu=Vmu,
            Nobs_i=Nobs_i,
            errMx=errMx,
            X=X,
            Xprobe=Xprobe,
            M=M,
            Mprobe=Mprobe,
        )

        A, S, Sv = _update_scores(
            X=X,
            M=M,
            A=A,
            S=S,
            Av=Av,
            Sv=Sv,
            Isv_use=Isv_use,
            obscombj=obscombj,
            V=V,
            eye_components=eye_components,
            verbose=verbose,
        )

        if rotate_each_iter:
            A, Av, S, Sv, Mu, X, Xprobe = _maybe_rotate(
                A=A,
                Av=Av,
                S=S,
                Sv=Sv,
                Isv_use=Isv_use,
                obscombj=obscombj,
                bias=bool(opts["bias"]),
                X=X,
                Xprobe=Xprobe,
                M=M,
                Mprobe=Mprobe,
            )

        A, Av = _update_loadings(
            X=X,
            M=M,
            S=S,
            Av=Av,
            Sv=Sv,
            Isv_use=Isv_use,
            Va=Va,
            V=V,
            verbose=verbose,
        )

        rms, prms, errMx = _recompute_rms(
            X=X,
            Xprobe=Xprobe,
            M=M,
            Mprobe=Mprobe,
            ndata=ndata,
            nprobe=nprobe,
            A=A,
            S=S,
        )

        V, sXv = _update_noise_variance(
            V=V,
            A=A,
            S=S,
            Av=Av,
            Sv=Sv,
            Muv=Muv,
            rms=rms,
            IX=IX,
            JX=JX,
            Isv_use=Isv_use,
            ndata=ndata,
        )

        lc, angleA, convmsg, Aold = _log_and_check_convergence(
            opts=opts,
            X=X,
            A=A,
            S=S,
            Mu=Mu,
            V=V,
            Va=Va,
            Av=Av,
            Vmu=Vmu,
            Muv=Muv,
            Sv=Sv,
            Isv_use=Isv_use,
            M=M,
            sXv=sXv,
            ndata=ndata,
            time_start=time_start,
            lc=lc,
            rms=rms,
            prms=prms,
            Aold=Aold,
            dsph=dsph,
        )

        if convmsg:
            if use_prior and iteration <= int(opts["niter_broadprior"]):
                # Prior never updated: ignore convergence and continue
                pass
            else:
                if verbose:
                    logger.info("%s", convmsg)
                break

    # ------------------------------------------------------------------
    # Final PCA rotation (if not rotated during iterations)
    # ------------------------------------------------------------------
    if not rotate_each_iter:
        A, Av, S, Sv, Mu = _final_rotation(
            A=A,
            Av=Av,
            S=S,
            Sv=Sv,
            Mu=Mu,
            Isv_use=Isv_use,
            obscombj=obscombj,
            bias=bool(opts["bias"]),
        )

    # ------------------------------------------------------------------
    # Restore removed rows/columns (add missing rows/cols back)
    # ------------------------------------------------------------------
    if n1 < n1x:
        A, Av = _add_m_rows(A, Av, Ir, n1x, Va)
        Mu, Muv = _add_m_rows(Mu, Muv, Ir, n1x, Vmu)
    if n2 < n2x:
        S, Sv, Isv_use = _add_m_cols(S, Sv, Ic, n2x, Isv_use)

    # ------------------------------------------------------------------
    # Pack result
    # ------------------------------------------------------------------
    result: dict[str, Any] = {
        "A": A,
        "S": S,
        "Mu": Mu,
        "V": float(V),
        "Av": Av,
        "Sv": Sv,
        "Isv": Isv_use,
        "Muv": Muv,
        "Va": Va,
        "Vmu": float(Vmu),
        "lc": lc,
        # Posterior covariances / variances, grouped:
        "cv": {"A": Av, "S": Sv, "Isv": Isv_use, "Mu": Muv},
        "hp": {"Va": Va, "Vmu": float(Vmu)},
    }

    return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_options(kwargs: Mapping[str, object]) -> dict[str, object]:
    """Merge user kwargs with defaults using _options (case-insensitive)."""
    opts_default: dict[str, object] = {
        "init": "random",
        "maxiters": 1000,
        "bias": 1,
        "uniquesv": 0,
        "autosave": 600,  # accepted but ignored
        "filename": "pca_f_autosave",  # accepted but ignored
        "minangle": 1e-8,
        "algorithm": "vb",
        "niter_broadprior": 100,
        "earlystop": 0,
        "rmsstop": np.array([100, 1e-4, 1e-3]),
        "cfstop": np.array([]),
        "verbose": 1,
        "xprobe": None,
        "rotate2pca": 1,
        "display": 0,
    }

    opts, wrnmsg = _options(opts_default, **kwargs)
    if wrnmsg:
        logger.warning("pca_full options warning: %s", wrnmsg)
    return opts


def _select_algorithm(opts: Mapping[str, object]) -> tuple[bool, bool]:
    """Decode algorithm mode into (use_prior, use_postvar)."""
    algorithm = str(opts["algorithm"]).lower()
    if algorithm == "ppca":
        return False, False
    if algorithm == "map":
        return True, False
    if algorithm == "vb":
        return True, True
    raise ValueError(f"Wrong value of the argument 'algorithm': {opts['algorithm']}")
