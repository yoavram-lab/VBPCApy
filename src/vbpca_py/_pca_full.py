"""Top-level orchestration for Variational Bayesian PCA (PCA_FULL).

This module provides :func:`vbpca_py._pca_full.pca_full`, an idiomatic Python
translation of Ilin & Raiko's PCA_FULL (JMLR 2010).

The function itself is intentionally thin: the mathematical update steps live
in helper modules (notably :mod:`vbpca_py._full_update`). If there is an API
conflict, the helper modules are authoritative.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

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


def pca_full(x: Matrix, n_components: int, **kwargs: object) -> dict[str, object]:
    """Run VBPCA with full posterior covariances.

    Parameters
    ----------
    x
        Data matrix of shape (n_features, n_samples). Dense arrays may contain
        NaNs to denote missing values. Sparse matrices store only observed
        entries.
    n_components
        Number of latent components.
    **kwargs
        Options; see :func:`_build_options`.

    Returns:
    -------
    dict
        Result dictionary matching the historical PCA_FULL API.
    """
    opts = _build_options(kwargs)
    use_prior, use_postvar = _select_algorithm(opts)

    prepared = _prepare_problem(x, opts)
    training = _initialize_model(
        prepared=prepared,
        n_components=int(n_components),
        use_prior=use_prior,
        use_postvar=use_postvar,
        opts=opts,
    )
    training = _run_training_loop(
        prepared=prepared, training=training, use_prior=use_prior, opts=opts
    )
    training = _maybe_finalize_rotation(prepared=prepared, training=training, opts=opts)
    final = _restore_original_shape(prepared=prepared, training=training)
    return _pack_result(final)


# ---------------------------------------------------------------------------
# Small data containers (to reduce arg lists + improve readability)
# ---------------------------------------------------------------------------


@dataclass
class PreparedProblem:
    """Prepared data and masks for a PCA_FULL run.

    This collects the various derived arrays that are shared across update
    steps (masks, pattern index, observed indices, etc.) so we can avoid
    extremely long argument lists in orchestration helpers.
    """

    x_data: Matrix
    x_probe: Matrix | None
    mask: Matrix
    mask_probe: Matrix | None
    n_obs_row: np.ndarray
    n_data: float
    n_probe: int
    ix_obs: np.ndarray
    jx_obs: np.ndarray
    n_features: int
    n_samples: int
    n1x: int
    n2x: int
    row_idx: np.ndarray | None
    col_idx: np.ndarray | None
    n_patterns: int
    obs_patterns: list[np.ndarray]
    pattern_index: np.ndarray | None


@dataclass
class ModelState:
    """Current model parameters during training."""

    a: np.ndarray
    s: np.ndarray
    mu: np.ndarray
    noise_var: float
    av: list[np.ndarray]
    sv: list[np.ndarray]
    muv: np.ndarray
    va: np.ndarray
    vmu: float


@dataclass
class TrainingState:
    """Mutable training state that evolves over iterations."""

    model: ModelState
    lc: dict[str, list[float]]
    dsph: dict[str, object]
    err_mx: object
    a_old: np.ndarray
    time_start: float


@dataclass(frozen=True)
class FinalState:
    """Final objects after shape restoration, ready for packing."""

    a: np.ndarray
    s: np.ndarray
    mu: np.ndarray
    noise_var: float
    av: list[np.ndarray]
    sv: list[np.ndarray]
    pattern_index: np.ndarray | None
    muv: np.ndarray
    va: np.ndarray
    vmu: float
    lc: dict[str, list[float]]


@dataclass(frozen=True)
class IterationConfig:
    """Constant configuration used within the training loop."""

    hp_va: float
    hp_vb: float
    hp_v: float
    eye_components: np.ndarray
    use_prior: bool
    rotate_each_iter: bool
    verbose: int
    opts: Mapping[str, object]


@dataclass
class IterationContext:
    """Inputs for a single training iteration."""

    iteration: int
    prepared: PreparedProblem
    training: TrainingState
    bias_state: BiasState
    centering_state: CenteringState
    cfg: IterationConfig


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------


def _prepare_problem(x: Matrix, opts: Mapping[str, object]) -> PreparedProblem:
    x_data, x_probe, n1x, n2x, row_idx, col_idx = _prepare_data(x, opts)
    x_data, x_probe, mask, mask_probe, n_obs_row, n_data, n_probe = (
        _build_masks_and_counts(x_data, x_probe, opts)
    )
    ix_obs, jx_obs = _observed_indices(x_data)

    pattern_info = _missing_patterns_info(mask, opts, n_samples=x_data.shape[1])

    return PreparedProblem(
        x_data=x_data,
        x_probe=x_probe,
        mask=mask,
        mask_probe=mask_probe,
        n_obs_row=n_obs_row,
        n_data=float(n_data),
        n_probe=int(n_probe),
        ix_obs=ix_obs,
        jx_obs=jx_obs,
        n_features=int(x_data.shape[0]),
        n_samples=int(x_data.shape[1]),
        n1x=int(n1x),
        n2x=int(n2x),
        row_idx=row_idx,
        col_idx=col_idx,
        n_patterns=int(pattern_info[0]),
        obs_patterns=pattern_info[1],
        pattern_index=pattern_info[2],
    )


def _adjust_opts_for_explicit_init(opts: Mapping[str, object]) -> None:
    """Align hyperparameters when an explicit init is provided.

    Avoids drifting away from supplied MATLAB-style initializations by
    extending ``niter_broadprior`` and optionally capping ``maxiters`` when
    a learning curve is present in ``init``.
    """
    init_val = opts.get("init")
    if init_val is None or (isinstance(init_val, str) and init_val.lower() == "random"):
        return

    maxiters = int(opts.get("maxiters", 0) or 0)
    nbp = int(opts.get("niter_broadprior", 0) or 0)
    if nbp <= maxiters:
        opts["niter_broadprior"] = maxiters + 1

    lc = getattr(init_val, "lc", None)
    lc_rms = getattr(lc, "rms", None) if lc is not None else None
    if lc_rms is None:
        return

    try:
        opts["maxiters"] = 0
    except (TypeError, ValueError):
        logger.debug("init.lc.rms length not usable; leaving maxiters unchanged.")


def _initialize_model(
    *,
    prepared: PreparedProblem,
    n_components: int,
    use_prior: bool,
    use_postvar: bool,
    opts: Mapping[str, object],
) -> TrainingState:
    _adjust_opts_for_explicit_init(opts)

    shapes = InitShapes(
        n_features=prepared.n_features,
        n_samples=prepared.n_samples,
        n_components=n_components,
        n_obs_patterns=prepared.n_patterns,
    )

    init_ctx = InitContext(
        x_data=prepared.x_data,
        x_probe=prepared.x_probe,
        mask=prepared.mask,
        mask_probe=prepared.mask_probe,
        shapes=shapes,
        pattern_index=prepared.pattern_index,
        n_obs_row=prepared.n_obs_row,
        use_prior=use_prior,
        use_postvar=use_postvar,
        opts=opts,
    )

    init_params = _initialize_parameters(init_ctx)

    # Ensure the prepared data seen by the training loop is already centered by
    # the initial Mu. This mirrors the MATLAB flow where SubtractMu is called
    # once before the first iteration. PreparedProblem is mutable, so keep the
    # centered views here to avoid double-centering on the first bias update.
    prepared.x_data = init_params[9]
    prepared.x_probe = init_params[10]

    # Initial monitoring returns: (rms, err_mx, prms, lc, dsph).
    _, err_mx, _, lc, dsph = _initial_monitoring(
        InitialMonitoringInputs(
            x_data=init_params[9],
            x_probe=init_params[10],
            mask=prepared.mask,
            n_data=float(prepared.n_data),
            n_probe=int(prepared.n_probe),
            a=init_params[0],
            s=init_params[1],
            opts=opts,
        )
    )

    model = ModelState(
        a=init_params[0],
        s=init_params[1],
        mu=init_params[2],
        noise_var=float(init_params[3]),
        av=init_params[4],
        sv=init_params[5],
        muv=init_params[6],
        va=init_params[7],
        vmu=float(init_params[8]),
    )
    return TrainingState(
        model=model,
        lc=lc,
        dsph=dsph,
        err_mx=err_mx,
        a_old=init_params[0].copy(),
        time_start=time.time(),
    )


def _run_training_loop(
    *,
    prepared: PreparedProblem,
    training: TrainingState,
    use_prior: bool,
    opts: Mapping[str, object],
) -> TrainingState:
    """Run the VB / EM iterations, mutating and returning the training state.

    Returns:
        The updated ``TrainingState`` after running iterations or hitting a stop.
    """
    # Hyperparameters for Va, Vmu, V (kept here for compatibility).
    cfg = IterationConfig(
        hp_va=0.001,
        hp_vb=0.001,
        hp_v=0.001,
        eye_components=np.eye(training.model.s.shape[0], dtype=float),
        use_prior=use_prior,
        rotate_each_iter=bool(opts["rotate2pca"]),
        verbose=int(opts["verbose"]),
        opts=opts,
    )

    bias_state = BiasState(
        mu=training.model.mu,
        muv=training.model.muv,
        noise_var=float(training.model.noise_var),
        vmu=float(training.model.vmu),
        n_obs_row=prepared.n_obs_row,
    )
    centering_state = CenteringState(
        x_data=prepared.x_data,
        x_probe=prepared.x_probe,
        mask=prepared.mask,
        mask_probe=prepared.mask_probe,
    )

    for iteration in range(1, int(opts["maxiters"]) + 1):
        ctx = IterationContext(
            iteration=iteration,
            prepared=prepared,
            training=training,
            bias_state=bias_state,
            centering_state=centering_state,
            cfg=cfg,
        )
        _iteration_step(ctx)

        # Stopping condition is recorded by _iteration_step in lc/convmsg.
        if training.lc.get("_stop", [""])[-1]:
            break

    # Cleanup internal stop marker to keep lc stable for external callers.
    training.lc.pop("_stop", None)
    return training


def _iteration_step(ctx: IterationContext) -> None:
    """One full iteration of updates; mutates ``ctx.training`` in place."""
    training = ctx.training
    m = training.model
    cfg = ctx.cfg

    # 1) Hyperpriors
    m.va, m.vmu = _update_hyperpriors(
        HyperpriorContext(
            iteration=ctx.iteration,
            use_prior=cfg.use_prior,
            niter_broadprior=int(cfg.opts["niter_broadprior"]),
            bias_enabled=bool(cfg.opts["bias"]),
            mu=ctx.bias_state.mu,
            mu_variances=ctx.bias_state.muv,
            loadings=m.a,
            loading_covariances=m.av,
            n_features=ctx.prepared.n_features,
            hp_va=cfg.hp_va,
            hp_vb=cfg.hp_vb,
            va=m.va,
            vmu=float(m.vmu),
        )
    )

    # 2) Bias / mean update + recenter.
    ctx.bias_state.noise_var = float(m.noise_var)
    ctx.bias_state.vmu = float(m.vmu)

    err_mx_arr = _as_dense_err_matrix(training.err_mx)
    ctx.bias_state, ctx.centering_state = _update_bias(
        bias_enabled=bool(cfg.opts["bias"]),
        bias_state=ctx.bias_state,
        err_mx=err_mx_arr,
        centering=ctx.centering_state,
    )
    m.mu = ctx.bias_state.mu
    m.muv = ctx.bias_state.muv

    x_data = ctx.centering_state.x_data
    x_probe = ctx.centering_state.x_probe

    # 3) Scores
    score_state = ScoreState(
        x_data=x_data,
        mask=ctx.prepared.mask,
        loadings=m.a,
        scores=m.s,
        loading_covariances=m.av,
        score_covariances=m.sv,
        pattern_index=ctx.prepared.pattern_index,
        obs_patterns=ctx.prepared.obs_patterns,
        noise_var=float(m.noise_var),
        eye_components=cfg.eye_components,
        verbose=cfg.verbose,
    )
    score_state = _update_scores(score_state)
    m.s = score_state.scores
    m.sv = score_state.score_covariances

    # 3b) Optional rotate-to-PCA
    if cfg.rotate_each_iter:
        x_data, x_probe = _rotate_towards_pca(
            prepared=ctx.prepared,
            training=training,
            bias_state=ctx.bias_state,
            x_data=x_data,
            x_probe=x_probe,
            opts=cfg.opts,
        )
        ctx.centering_state.x_data = x_data
        ctx.centering_state.x_probe = x_probe

    # 4) Loadings
    m.a, m.av = _update_loadings(
        LoadingsUpdateState(
            x_data=x_data,
            mask=ctx.prepared.mask,
            scores=m.s,
            loading_covariances=m.av,
            score_covariances=m.sv,
            pattern_index=ctx.prepared.pattern_index,
            va=m.va,
            noise_var=float(m.noise_var),
            verbose=cfg.verbose,
        )
    )

    # 5) RMS (and error matrix)
    rms, prms, err_mx = _recompute_rms(
        RmsContext(
            x_data=x_data,
            x_probe=x_probe,
            mask=ctx.prepared.mask,
            mask_probe=ctx.prepared.mask_probe,
            n_data=float(ctx.prepared.n_data),
            n_probe=int(ctx.prepared.n_probe),
            loadings=m.a,
            scores=m.s,
        )
    )
    training.err_mx = err_mx

    # 6) Noise variance
    noise_state = NoiseState(
        loadings=m.a,
        scores=m.s,
        loading_covariances=m.av,
        score_covariances=m.sv,
        mu_variances=m.muv,
        pattern_index=ctx.prepared.pattern_index,
        n_data=float(ctx.prepared.n_data),
        noise_var=float(m.noise_var),
    )
    noise_state, s_xv = _update_noise_variance(
        noise_state,
        float(rms),
        ctx.prepared.ix_obs,
        ctx.prepared.jx_obs,
        hp_v=float(cfg.hp_v),
    )
    m.noise_var = float(noise_state.noise_var)
    ctx.bias_state.noise_var = float(m.noise_var)

    # 7) Logging + convergence
    training.lc, _, convmsg, training.a_old = _log_and_check_convergence(
        ConvergenceState(
            opts=dict(cfg.opts),
            x_data=x_data,
            loadings=m.a,
            scores=m.s,
            mu=m.mu,
            noise_var=float(m.noise_var),
            va=m.va,
            loading_covariances=m.av,
            vmu=float(m.vmu),
            mu_variances=m.muv,
            score_covariances=m.sv,
            pattern_index=ctx.prepared.pattern_index,
            mask=ctx.prepared.mask,
            s_xv=float(s_xv),
            n_data=float(ctx.prepared.n_data),
            time_start=float(training.time_start),
            lc=training.lc,
            loadings_old=training.a_old,
            dsph=training.dsph,
        ),
        float(rms),
        float(prms),
    )

    # Store stop message internally to keep loop logic simple without touching
    # the public learning-curve schema.
    stop_now = ""
    if convmsg:
        if cfg.use_prior and ctx.iteration <= int(cfg.opts["niter_broadprior"]):
            stop_now = ""
        else:
            if cfg.verbose:
                logger.info("%s", convmsg)
            stop_now = convmsg

    training.lc.setdefault("_stop", []).append(stop_now)


def _rotate_towards_pca(  # noqa: PLR0913
    *,
    prepared: PreparedProblem,
    training: TrainingState,
    bias_state: BiasState,
    x_data: Matrix,
    x_probe: Matrix | None,
    opts: Mapping[str, object],
) -> tuple[Matrix, Matrix | None]:
    """Apply rotate-to-PCA step.

    Returns the (possibly re-centered) x_data and x_probe so the caller can keep
    a consistent view of the centered matrices.

    Returns:
        Tuple of rotated ``(x_data, x_probe)`` with bias adjustment applied.
    """
    m = training.model
    mu_before = m.mu.copy()

    rot_ctx = RotationContext(
        loadings=m.a,
        loading_covariances=m.av,
        scores=m.s,
        score_covariances=m.sv,
        mu=m.mu,
        pattern_index=prepared.pattern_index,
        obs_patterns=prepared.obs_patterns,
        bias_enabled=bool(opts["bias"]),
    )
    m.a, m.av, m.s, m.sv, m.mu = _final_rotation(rot_ctx)

    if not bool(opts["bias"]):
        return x_data, x_probe

    d_mu_iter = m.mu - mu_before
    probe_container = (
        ProbeMatrices(x=x_probe, mask=prepared.mask_probe)
        if x_probe is not None and prepared.mask_probe is not None
        else None
    )
    x_new, probe_new = subtract_mu(
        d_mu_iter,
        x_data,
        prepared.mask,
        probe=probe_container,
        update_bias=True,
    )
    bias_state.mu = m.mu
    return x_new, probe_new


def _maybe_finalize_rotation(
    *,
    prepared: PreparedProblem,
    training: TrainingState,
    opts: Mapping[str, object],
) -> TrainingState:
    if bool(opts["rotate2pca"]):
        return training

    m = training.model
    rot_ctx = RotationContext(
        loadings=m.a,
        loading_covariances=m.av,
        scores=m.s,
        score_covariances=m.sv,
        mu=m.mu,
        pattern_index=prepared.pattern_index,
        obs_patterns=prepared.obs_patterns,
        bias_enabled=bool(opts["bias"]),
    )
    m.a, m.av, m.s, m.sv, m.mu = _final_rotation(rot_ctx)
    return training


def _restore_original_shape(
    *, prepared: PreparedProblem, training: TrainingState
) -> FinalState:
    """Undo row/column removal performed during preparation.

    Returns:
        FinalState containing parameters expanded back to original dimensions.
    """
    m = training.model
    a, av, s, sv, mu, muv, pattern_index = (
        m.a,
        m.av,
        m.s,
        m.sv,
        m.mu,
        m.muv,
        prepared.pattern_index,
    )

    if prepared.n_features < prepared.n1x:
        a, av = _add_m_rows(a, av, prepared.row_idx, prepared.n1x, m.va)
        mu, muv = _add_m_rows(mu, muv, prepared.row_idx, prepared.n1x, float(m.vmu))

    if prepared.n_samples < prepared.n2x:
        s, sv, pattern_index = _add_m_cols(
            s, sv, prepared.col_idx, prepared.n2x, pattern_index
        )

    return FinalState(
        a=a,
        s=s,
        mu=mu,
        noise_var=float(m.noise_var),
        av=av,
        sv=sv,
        pattern_index=pattern_index,
        muv=muv,
        va=m.va,
        vmu=float(m.vmu),
        lc=training.lc,
    )


def _pack_result(final: FinalState) -> dict[str, object]:
    """Pack final values into the historical PCA_FULL result dictionary.

    Returns:
        Dictionary mirroring the legacy MATLAB output structure.
    """
    return {
        "A": final.a,
        "S": final.s,
        "Mu": final.mu,
        "V": float(final.noise_var),
        "Av": final.av,
        "Sv": final.sv,
        "Isv": final.pattern_index,
        "Muv": final.muv,
        "Va": final.va,
        "Vmu": float(final.vmu),
        "lc": final.lc,
        "cv": {
            "A": final.av,
            "S": final.sv,
            "Isv": final.pattern_index,
            "Mu": final.muv,
        },
        "hp": {"Va": final.va, "Vmu": float(final.vmu)},
    }


def _as_dense_err_matrix(err_mx: object) -> np.ndarray:
    """Ensure the bias updater sees a dense numeric array for err_mx.

    In sparse-input modes, the RMS helper may return sparse matrices or other
    wrappers. :func:`_update_bias` expects a dense array with shape
    (n_features, n_samples) so it can sum over axis=1.

    Returns:
        Dense ndarray version of ``err_mx`` (empty array when ``err_mx`` is None).
    """
    if err_mx is None:
        return np.zeros((0, 0), dtype=float)
    if issparse(err_mx):
        return np.asarray(cast("spmatrix", err_mx).toarray(), dtype=float)
    return np.asarray(err_mx, dtype=float)


# ---------------------------------------------------------------------------
# Options + algorithm selection (publicly relied upon)
# ---------------------------------------------------------------------------


def _build_options(kwargs: Mapping[str, object]) -> dict[str, object]:
    """Merge user kwargs with defaults using _options (case-insensitive).

    Returns:
        Options dictionary after applying defaults and user overrides.
    """
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
    """Decode algorithm mode into (use_prior, use_postvar).

    Returns:
        Tuple ``(use_prior, use_postvar)`` decoded from ``opts['algorithm']``.

    Raises:
        ValueError: If ``algorithm`` is not one of {"ppca", "map", "vb"}.
    """
    algorithm = str(opts["algorithm"]).lower()
    if algorithm == "ppca":
        return False, False
    if algorithm == "map":
        return True, False
    if algorithm == "vb":
        return True, True
    msg = f"Wrong value of the argument 'algorithm': {opts['algorithm']}"
    raise ValueError(msg)
