# src/vbpca_py/pca_full.py

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import spmatrix

if TYPE_CHECKING:
    from collections.abc import Mapping

from ._converge import ConvergenceState, _log_and_check_convergence
from ._expand import _add_m_cols, _add_m_rows
from ._full_update import (
    BiasState,
    CenteringState,
    NoiseState,
    ScoreState,
    _build_masks_and_counts,
    _final_rotation,
    _initialize_parameters,
    _missing_patterns_info,
    _observed_indices,
    _prepare_data,
    _recompute_rms,
    _update_bias,
    _update_hyperpriors,
    _update_loadings,
    _update_noise_variance,
    _update_scores,
)
from ._mean import subtract_mu
from ._monitoring import InitialMonitoringInputs, _initial_monitoring
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

    # ------------------------------------------------------------------
    # Data preparation and missingness masks
    # ------------------------------------------------------------------
    x_data, x_probe, n1x, n2x, row_idx, col_idx = _prepare_data(x, opts)
    x_data, x_probe, mask, mask_probe, n_obs_row, n_data, n_probe = (
        _build_masks_and_counts(x_data, x_probe, opts)
    )

    ix_obs, jx_obs = _observed_indices(x_data)

    n_features, n_samples = x_data.shape
    n_patterns, obs_patterns, pattern_index = _missing_patterns_info(
        mask,
        opts,
        n_samples=n_samples,
    )

    # ------------------------------------------------------------------
    # Parameter initialisation + initial centering
    # ------------------------------------------------------------------
    (
        a,
        s,
        mu,
        noise_var,
        av,
        sv,
        muv,
        va,
        vmu,
        x_data,
        x_probe,
    ) = _initialize_parameters(
        x_data=x_data,
        x_probe=x_probe,
        mask=mask,
        mask_probe=mask_probe,
        n_features=n_features,
        n_samples=n_samples,
        n_components=n_components,
        n_patterns=n_patterns,
        pattern_index=pattern_index,
        n_obs_row=n_obs_row,
        use_prior=use_prior,
        use_postvar=use_postvar,
        opts=opts,
    )

    # ------------------------------------------------------------------
    # Initial monitoring (RMS / probe RMS / logging)
    # ------------------------------------------------------------------
    init_inputs = InitialMonitoringInputs(
        x_data=x_data,
        x_probe=x_probe,
        mask=mask,
        n_data=n_data,
        n_probe=n_probe,
        a=a,
        s=s,
        opts=opts,
    )
    rms, err_mx, prms, lc, dsph = _initial_monitoring(init_inputs)
    a_old = a.copy()

    # ------------------------------------------------------------------
    # Hyperparameters for Va, Vmu, V
    # ------------------------------------------------------------------
    hp_va = 0.001
    hp_vb = 0.001
    hp_v = 0.001

    time_start = time.time()
    verbose = int(opts["verbose"])
    rotate_each_iter = bool(opts["rotate2pca"])
    eye_components = np.eye(n_components, dtype=float)

    # Bias / centering state
    bias_state = BiasState(
        mu=mu,
        muv=muv,
        noise_var=noise_var,
        vmu=vmu,
        n_obs_row=n_obs_row,
    )
    centering_state = CenteringState(
        x_data=x_data,
        x_probe=x_probe,
        mask=mask,
        mask_probe=mask_probe,
    )

    # ------------------------------------------------------------------
    # Main VB / EM loop
    # ------------------------------------------------------------------
    for iteration in range(1, int(opts["maxiters"]) + 1):
        # 1) Update Va, Vmu after broad-prior warmup
        va, vmu = _update_hyperpriors(
            iteration=iteration,
            use_prior=use_prior,
            niter_broadprior=int(opts["niter_broadprior"]),
            bias=bool(opts["bias"]),
            mu=bias_state.mu,
            muv=bias_state.muv,
            a=a,
            av=av,
            n_features=n_features,
            hp_va=hp_va,
            hp_vb=hp_vb,
            va=va,
            vmu=vmu,
        )

        # 2) Bias / mean update (Mu, Muv) and recenter X / Xprobe
        bias_state.noise_var = noise_var
        bias_state.vmu = vmu

        bias_state, centering_state = _update_bias(
            bias=bool(opts["bias"]),
            bias_state=bias_state,
            err_mx=err_mx,
            centering=centering_state,
        )

        mu = bias_state.mu
        muv = bias_state.muv
        # noise_var and vmu remain as scalars above

        x_data = centering_state.x_data
        x_probe = centering_state.x_probe

        # 3) Update S and Sv (scores and their covariances)
        score_state = ScoreState(
            x_data=x_data,
            mask=mask,
            a=a,
            s=s,
            av=av,
            sv=sv,
            pattern_index=pattern_index,
            obs_patterns=obs_patterns,
            noise_var=noise_var,
            eye_components=eye_components,
            verbose=verbose,
        )
        score_state = _update_scores(score_state)
        s = score_state.s
        sv = score_state.sv

        # Optional rotate-to-PCA step each iteration, including bias shift
        if rotate_each_iter:
            mu_before = mu.copy()
            a, av, s, sv, mu = _final_rotation(
                a=a,
                av=av,
                s=s,
                sv=sv,
                mu=mu,
                pattern_index=pattern_index,
                obs_patterns=obs_patterns,
                bias=bool(opts["bias"]),
            )
            if bool(opts["bias"]):
                d_mu_iter = mu - mu_before
                x_data, x_probe = subtract_mu(
                    d_mu_iter,
                    x_data,
                    mask,
                    x_probe,
                    mask_probe,
                    update_bias=True,
                )
                centering_state.x_data = x_data
                centering_state.x_probe = x_probe
                bias_state.mu = mu

        # 4) Update A and Av (loadings and their covariances)
        a, av = _update_loadings(
            x_data=x_data,
            mask=mask,
            s=s,
            av=av,
            sv=sv,
            pattern_index=pattern_index,
            va=va,
            noise_var=noise_var,
            verbose=verbose,
        )

        # 5) Recompute RMS / probe RMS and error matrix
        rms, prms, err_mx = _recompute_rms(
            x_data=x_data,
            x_probe=x_probe,
            mask=mask,
            mask_probe=mask_probe,
            n_data=n_data,
            n_probe=n_probe,
            a=a,
            s=s,
        )

        # 6) Update noise variance V
        noise_state = NoiseState(
            a=a,
            s=s,
            av=av,
            sv=sv,
            muv=muv,
            pattern_index=pattern_index,
            n_data=n_data,
            noise_var=noise_var,
        )
        noise_state, s_xv = _update_noise_variance(
            noise_state,
            rms,
            ix_obs,
            jx_obs,
            hp_v=hp_v,
        )
        noise_var = noise_state.noise_var
        bias_state.noise_var = noise_var  # keep in sync

        # 7) Logging, cost, convergence check
        conv_state = ConvergenceState(
            opts=opts,
            x_data=x_data,
            a=a,
            s=s,
            mu=mu,
            noise_var=noise_var,
            va=va,
            av=av,
            vmu=vmu,
            muv=muv,
            sv=sv,
            pattern_index=pattern_index,
            mask=mask,
            s_xv=s_xv,
            n_data=n_data,
            time_start=time_start,
            lc=lc,
            a_old=a_old,
            dsph=dsph,
        )

        lc, angle_a, convmsg, a_old = _log_and_check_convergence(
            conv_state,
            rms,
            prms,
        )

        if convmsg:
            # In VB mode, ignore convergence if prior never updated
            if use_prior and iteration <= int(opts["niter_broadprior"]):
                pass
            else:
                if verbose:
                    logger.info("%s", convmsg)
                break

    # ------------------------------------------------------------------
    # Final PCA rotation (if not rotated during iterations)
    # ------------------------------------------------------------------
    if not rotate_each_iter:
        a, av, s, sv, mu = _final_rotation(
            a=a,
            av=av,
            s=s,
            sv=sv,
            mu=mu,
            pattern_index=pattern_index,
            obs_patterns=obs_patterns,
            bias=bool(opts["bias"]),
        )

    # ------------------------------------------------------------------
    # Restore removed rows/columns (add missing rows/cols back)
    # ------------------------------------------------------------------
    if n_features < n1x:
        a, av = _add_m_rows(a, av, row_idx, n1x, va)
        mu, muv = _add_m_rows(mu, muv, row_idx, n1x, vmu)

    if n_samples < n2x:
        s, sv, pattern_index = _add_m_cols(s, sv, col_idx, n2x, pattern_index)

    # ------------------------------------------------------------------
    # Pack result
    # ------------------------------------------------------------------
    result: dict[str, Any] = {
        "A": a,
        "S": s,
        "Mu": mu,
        "V": float(noise_var),
        "Av": av,
        "Sv": sv,
        "Isv": pattern_index,
        "Muv": muv,
        "Va": va,
        "Vmu": float(vmu),
        "lc": lc,
        # Posterior covariances / variances, grouped:
        "cv": {"A": av, "S": sv, "Isv": pattern_index, "Mu": muv},
        "hp": {"Va": va, "Vmu": float(vmu)},
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
    msg = f"Wrong value of the argument 'algorithm': {opts['algorithm']}"
    raise ValueError(msg)
