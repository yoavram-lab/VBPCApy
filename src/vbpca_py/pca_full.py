"""Top-level PCA_FULL routine (Variational Bayesian PCA).

This module provides :func:`vbpca_py.pca_full.pca_full`, a refactored, helper-
driven orchestration of the original MATLAB *PCA_FULL* algorithm (Ilin & Raiko,
JMLR 2010).

Implementation note: helper modules (e.g. :mod:`vbpca_py._full_update`) are
treated as the source of truth for data structures and function signatures. If
there is an API conflict, this module adapts to the helper API.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import issparse, spmatrix

if TYPE_CHECKING:
    from collections.abc import Mapping

from ._converge import ConvergenceState, _log_and_check_convergence
from ._expand import _add_m_cols, _add_m_rows
from ._full_update import (
    BiasState,
    CenteringState,
    HyperpriorContext,
    InitContext,
    LoadingsUpdateState,
    NoiseState,
    RmsContext,
    RotationContext,
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
from ._mean import ProbeMatrices, subtract_mu
from ._monitoring import InitialMonitoringInputs, InitShapes, _initial_monitoring
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

    Orchestrates the algorithm, delegating update steps to helpers in
    :mod:`vbpca_py._full_update`. Helper APIs are authoritative.
    """
    opts = _build_options(kwargs)
    use_prior, use_postvar = _select_algorithm(opts)

    prep = _prepare_problem(x, opts)
    init = _initialize_and_monitor(
        prep,
        n_components=int(n_components),
        use_prior=use_prior,
        use_postvar=use_postvar,
        opts=opts,
    )

    a, s, mu, noise_var, av, sv, muv, va, vmu = _run_main_loop(
        init,
        prep,
        use_prior=use_prior,
        opts=opts,
    )

    a, av, s, sv, mu = _finalize_rotation(
        a,
        av,
        s,
        sv,
        mu,
        prep,
        bias_enabled=bool(opts["bias"]),
        rotate_each_iter=bool(opts["rotate2pca"]),
    )

    a, av, s, sv, mu, muv, pattern_index = _restore_original_shape(
        a,
        av,
        s,
        sv,
        mu,
        muv,
        prep,
        va=va,
        vmu=float(vmu),
    )

    return _pack_result(
        a=a,
        s=s,
        mu=mu,
        noise_var=float(noise_var),
        av=av,
        sv=sv,
        pattern_index=pattern_index,
        muv=muv,
        va=va,
        vmu=float(vmu),
        lc=init["lc"],
    )


# ---------------------------------------------------------------------------
# Internal orchestration helpers
# ---------------------------------------------------------------------------


def _prepare_problem(x: Matrix, opts: Mapping[str, object]) -> dict[str, Any]:
    """Prepare data, masks, and missingness metadata."""
    x_data, x_probe, n1x, n2x, row_idx, col_idx = _prepare_data(x, opts)
    x_data, x_probe, mask, mask_probe, n_obs_row, n_data, n_probe = (
        _build_masks_and_counts(x_data, x_probe, opts)
    )
    ix_obs, jx_obs = _observed_indices(x_data)

    n_features, n_samples = x_data.shape
    n_patterns, obs_patterns, pattern_index = _missing_patterns_info(
        mask, opts, n_samples=n_samples
    )

    return {
        "x_data": x_data,
        "x_probe": x_probe,
        "mask": mask,
        "mask_probe": mask_probe,
        "n_obs_row": n_obs_row,
        "n_data": n_data,
        "n_probe": n_probe,
        "ix_obs": ix_obs,
        "jx_obs": jx_obs,
        "n_features": int(n_features),
        "n_samples": int(n_samples),
        "n1x": int(n1x),
        "n2x": int(n2x),
        "row_idx": row_idx,
        "col_idx": col_idx,
        "n_patterns": int(n_patterns),
        "obs_patterns": obs_patterns,
        "pattern_index": pattern_index,
    }


def _initialize_and_monitor(
    prep: Mapping[str, Any],
    *,
    n_components: int,
    use_prior: bool,
    use_postvar: bool,
    opts: Mapping[str, object],
) -> dict[str, Any]:
    """Initialize parameters and run initial monitoring/logging."""
    shapes = InitShapes(
        n_features=int(prep["n_features"]),
        n_samples=int(prep["n_samples"]),
        n_components=int(n_components),
        n_obs_patterns=int(prep["n_patterns"]),
    )

    init_ctx = InitContext(
        x_data=prep["x_data"],
        x_probe=prep["x_probe"],
        mask=prep["mask"],
        mask_probe=prep["mask_probe"],
        shapes=shapes,
        pattern_index=prep["pattern_index"],
        n_obs_row=prep["n_obs_row"],
        use_prior=use_prior,
        use_postvar=use_postvar,
        opts=opts,
    )

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
    ) = _initialize_parameters(init_ctx)

    init_inputs = InitialMonitoringInputs(
        x_data=x_data,
        x_probe=x_probe,
        mask=prep["mask"],
        n_data=prep["n_data"],
        n_probe=prep["n_probe"],
        a=a,
        s=s,
        opts=opts,
    )
    rms, err_mx, prms, lc, dsph = _initial_monitoring(init_inputs)

    return {
        "a": a,
        "s": s,
        "mu": mu,
        "noise_var": float(noise_var),
        "av": av,
        "sv": sv,
        "muv": muv,
        "va": va,
        "vmu": float(vmu),
        "x_data": x_data,
        "x_probe": x_probe,
        "rms": float(rms),
        "prms": float(prms),
        "err_mx": err_mx,
        "lc": lc,
        "dsph": dsph,
    }


def _run_main_loop(
    init: dict[str, Any],
    prep: Mapping[str, Any],
    *,
    use_prior: bool,
    opts: Mapping[str, object],
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
]:
    """Run VB/EM iterations and update `init['lc']` in-place."""
    a = init["a"]
    s = init["s"]
    mu = init["mu"]
    noise_var = float(init["noise_var"])
    av = init["av"]
    sv = init["sv"]
    muv = init["muv"]
    va = init["va"]
    vmu = float(init["vmu"])
    x_data = init["x_data"]
    x_probe = init["x_probe"]
    err_mx = init["err_mx"]
    lc = init["lc"]
    dsph = init["dsph"]

    hp_va = 0.001
    hp_vb = 0.001
    hp_v = 0.001

    time_start = time.time()
    verbose = int(opts["verbose"])
    rotate_each_iter = bool(opts["rotate2pca"])
    eye_components = np.eye(
        int(prep["n_components"]) if "n_components" in prep else a.shape[1], dtype=float
    )
    # Prefer actual k from A if prep didn't store it
    eye_components = np.eye(a.shape[1], dtype=float)

    bias_state = BiasState(
        mu=mu,
        muv=muv,
        noise_var=float(noise_var),
        vmu=float(vmu),
        n_obs_row=prep["n_obs_row"],
    )
    centering_state = CenteringState(
        x_data=x_data,
        x_probe=x_probe,
        mask=prep["mask"],
        mask_probe=prep["mask_probe"],
    )

    loadings_old = a.copy()

    for iteration in range(1, int(opts["maxiters"]) + 1):
        hp_ctx = HyperpriorContext(
            iteration=iteration,
            use_prior=use_prior,
            niter_broadprior=int(opts["niter_broadprior"]),
            bias_enabled=bool(opts["bias"]),
            mu=bias_state.mu,
            mu_variances=bias_state.muv,
            loadings=a,
            loading_covariances=av,
            n_features=int(prep["n_features"]),
            hp_va=hp_va,
            hp_vb=hp_vb,
            va=va,
            vmu=float(vmu),
        )
        va, vmu = _update_hyperpriors(hp_ctx)

        bias_state.noise_var = float(noise_var)
        bias_state.vmu = float(vmu)

        # _update_bias expects a dense 2D error matrix; compute_rms may return sparse.
        err_mx_arr = err_mx.toarray() if issparse(err_mx) else np.asarray(err_mx)

        bias_state, centering_state = _update_bias(
            bias_enabled=bool(opts["bias"]),
            bias_state=bias_state,
            err_mx=err_mx_arr,
            centering=centering_state,
        )
        mu = bias_state.mu
        muv = bias_state.muv
        x_data = centering_state.x_data
        x_probe = centering_state.x_probe

        score_state = ScoreState(
            x_data=x_data,
            mask=prep["mask"],
            loadings=a,
            scores=s,
            loading_covariances=av,
            score_covariances=sv,
            pattern_index=prep["pattern_index"],
            obs_patterns=prep["obs_patterns"],
            noise_var=float(noise_var),
            eye_components=eye_components,
            verbose=verbose,
        )
        score_state = _update_scores(score_state)
        s = score_state.scores
        sv = score_state.score_covariances

        if rotate_each_iter:
            mu_before = mu.copy()
            rot_ctx = RotationContext(
                loadings=a,
                loading_covariances=av,
                scores=s,
                score_covariances=sv,
                mu=mu,
                pattern_index=prep["pattern_index"],
                obs_patterns=prep["obs_patterns"],
                bias_enabled=bool(opts["bias"]),
            )
            a, av, s, sv, mu = _final_rotation(rot_ctx)

            if bool(opts["bias"]):
                d_mu_iter = mu - mu_before
                probe_container = (
                    ProbeMatrices(x=x_probe, mask=prep["mask_probe"])
                    if x_probe is not None and prep["mask_probe"] is not None
                    else None
                )
                x_data, x_probe = subtract_mu(
                    d_mu_iter,
                    x_data,
                    prep["mask"],
                    probe=probe_container,
                    update_bias=True,
                )
                centering_state.x_data = x_data
                centering_state.x_probe = x_probe
                bias_state.mu = mu

        load_state = LoadingsUpdateState(
            x_data=x_data,
            mask=prep["mask"],
            scores=s,
            loading_covariances=av,
            score_covariances=sv,
            pattern_index=prep["pattern_index"],
            va=va,
            noise_var=float(noise_var),
            verbose=verbose,
        )
        a, av = _update_loadings(load_state)

        rms_ctx = RmsContext(
            x_data=x_data,
            x_probe=x_probe,
            mask=prep["mask"],
            mask_probe=prep["mask_probe"],
            n_data=float(prep["n_data"]),
            n_probe=int(prep["n_probe"]),
            loadings=a,
            scores=s,
        )
        rms, prms, err_mx = _recompute_rms(rms_ctx)

        noise_state = NoiseState(
            loadings=a,
            scores=s,
            loading_covariances=av,
            score_covariances=sv,
            mu_variances=muv,
            pattern_index=prep["pattern_index"],
            n_data=float(prep["n_data"]),
            noise_var=float(noise_var),
        )
        noise_state, s_xv = _update_noise_variance(
            noise_state,
            float(rms),
            prep["ix_obs"],
            prep["jx_obs"],
            hp_v=float(hp_v),
        )
        noise_var = float(noise_state.noise_var)
        bias_state.noise_var = float(noise_var)

        conv_state = ConvergenceState(
            opts=dict(opts),  # ConvergenceState expects MutableMapping
            x_data=x_data,
            loadings=a,
            scores=s,
            mu=mu,
            noise_var=float(noise_var),
            va=va,
            loading_covariances=av,
            vmu=float(vmu),
            mu_variances=muv,
            score_covariances=sv,
            pattern_index=prep["pattern_index"],
            mask=prep["mask"],
            s_xv=float(s_xv),
            n_data=float(prep["n_data"]),
            time_start=time_start,
            lc=lc,
            loadings_old=loadings_old,
            dsph=dsph,
        )

        lc, _angle_a, convmsg, loadings_old = _log_and_check_convergence(
            conv_state,
            float(rms),
            float(prms),
        )

        if convmsg:
            if use_prior and iteration <= int(opts["niter_broadprior"]):
                pass
            else:
                if verbose:
                    logger.info("%s", convmsg)
                break

    # write back latest state into init for packing
    init["lc"] = lc

    return a, s, mu, float(noise_var), av, sv, muv, va, float(vmu)


def _finalize_rotation(
    a: np.ndarray,
    av: list[np.ndarray],
    s: np.ndarray,
    sv: list[np.ndarray],
    mu: np.ndarray,
    prep: Mapping[str, Any],
    *,
    bias_enabled: bool,
    rotate_each_iter: bool,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    """Apply final rotation if rotation wasn't applied each iteration."""
    if rotate_each_iter:
        return a, av, s, sv, mu

    rot_ctx = RotationContext(
        loadings=a,
        loading_covariances=av,
        scores=s,
        score_covariances=sv,
        mu=mu,
        pattern_index=prep["pattern_index"],
        obs_patterns=prep["obs_patterns"],
        bias_enabled=bias_enabled,
    )
    return _final_rotation(rot_ctx)


def _restore_original_shape(
    a: np.ndarray,
    av: list[np.ndarray],
    s: np.ndarray,
    sv: list[np.ndarray],
    mu: np.ndarray,
    muv: np.ndarray,
    prep: Mapping[str, Any],
    *,
    va: np.ndarray,
    vmu: float,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    """Restore removed rows/cols back to original X shape."""
    pattern_index = prep["pattern_index"]

    if int(prep["n_features"]) < int(prep["n1x"]):
        a, av = _add_m_rows(a, av, prep["row_idx"], int(prep["n1x"]), va)
        mu, muv = _add_m_rows(mu, muv, prep["row_idx"], int(prep["n1x"]), vmu)

    if int(prep["n_samples"]) < int(prep["n2x"]):
        s, sv, pattern_index = _add_m_cols(
            s, sv, prep["col_idx"], int(prep["n2x"]), pattern_index
        )

    return a, av, s, sv, mu, muv, pattern_index


def _pack_result(
    *,
    a: np.ndarray,
    s: np.ndarray,
    mu: np.ndarray,
    noise_var: float,
    av: list[np.ndarray],
    sv: list[np.ndarray],
    pattern_index: np.ndarray | None,
    muv: np.ndarray,
    va: np.ndarray,
    vmu: float,
    lc: dict[str, list[float]],
) -> dict[str, Any]:
    """Package outputs in the legacy-compatible result dict."""
    return {
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
        "cv": {"A": av, "S": sv, "Isv": pattern_index, "Mu": muv},
        "hp": {"Va": va, "Vmu": float(vmu)},
    }


# ---------------------------------------------------------------------------
# Options helpers
# ---------------------------------------------------------------------------


def _build_options(kwargs: Mapping[str, object]) -> dict[str, object]:
    """Merge user kwargs with defaults using _options (case-insensitive)."""
    opts_default: dict[str, object] = {
        "init": "random",
        "maxiters": 1000,
        "bias": 1,
        "uniquesv": 0,
        "autosave": 600,
        "filename": "pca_f_autosave",
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
