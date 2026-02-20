"""
Top-level orchestration for Variational Bayesian PCA (PCA_FULL).

This module provides :func:`vbpca_py._pca_full.pca_full`, an idiomatic Python
translation of Ilin & Raiko's PCA_FULL (JMLR 2010).

The function itself is intentionally thin: the mathematical update steps live
in helper modules (notably :mod:`vbpca_py._full_update`). If there is an API
conflict, the helper modules are authoritative.
"""

from __future__ import annotations

import logging
import time
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsFloat, SupportsIndex, SupportsInt, cast

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

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
    _observed_indices_with_mode,
    _prepare_data,
    _recompute_rms,
    _update_bias,
    _update_hyperpriors,
    _update_loadings,
    _update_noise_variance,
    _update_scores,
)
from ._mean import ProbeMatrices, subtract_mu
from ._memory import exceeds_budget, format_bytes, resolve_max_dense_bytes
from ._monitoring import InitialMonitoringInputs, InitShapes, _initial_monitoring
from ._options import _options
from ._runtime_policy import (
    RuntimeThreadConfig,
    RuntimeWorkloadProfile,
    apply_runtime_policy_defaults,
    resolve_runtime_thread_config_with_report,
)

logger = logging.getLogger(__name__)

Array = np.ndarray
Sparse = sp.csr_matrix
Matrix = Array | Sparse
_ALLOWED_COMPAT_MODES = {"strict_legacy", "modern"}
_PHASE_TIMING_KEYS = (
    "phase_scores_sec",
    "phase_loadings_sec",
    "phase_rms_sec",
    "phase_noise_sec",
    "phase_converge_sec",
    "phase_total_sec",
)


def _coerce_int(
    val: SupportsInt | SupportsIndex | str | bytes | bytearray | None,
    default: int = 0,
) -> int:
    """Safely coerce common int-like inputs with a fallback.

    Returns:
        Integer conversion of ``val`` when possible; ``default`` otherwise.
    """
    if val is None:
        return default
    try:
        return int(
            cast("SupportsInt | SupportsIndex | str | bytes | bytearray", val)
        )
    except (TypeError, ValueError):
        return default


def _int_opt(val: object, default: int = 0) -> int:
    if isinstance(
        val, (int, str, bytes, bytearray, np.integer, SupportsInt, SupportsIndex)
    ):
        return _coerce_int(val, default=default)
    return default


def _float_opt(val: object, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (bytes, bytearray)):
        try:
            return float(val.decode())
        except (TypeError, ValueError):
            return default
    try:
        return float(cast("str | SupportsFloat | SupportsIndex", val))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pca_full(
    x: Matrix,
    n_components: int,
    *,
    mask: Matrix | None = None,
    **kwargs: object,
) -> dict[str, object]:
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
    opts: MutableMapping[str, object] = _build_options(kwargs)
    use_prior, use_postvar = _select_algorithm(opts)

    prepared = _prepare_problem(x, opts, mask_override=mask)
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
    return _pack_result(
        final,
        include_diagnostics=bool(opts.get("return_diagnostics", True)),
        explained_var_solver=str(opts.get("explained_var_solver", "auto")),
        explained_var_gram_ratio=_float_opt(
            opts.get("explained_var_gram_ratio", 4.0), default=4.0
        ),
        runtime_report=(
            final.runtime_report if bool(opts.get("runtime_report", 0)) else None
        ),
    )


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
    obs_patterns: list[list[int]]
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
    err_mx: np.ndarray | sp.spmatrix | None
    a_old: np.ndarray
    time_start: float
    runtime_report: dict[str, object] | None


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
    runtime_report: dict[str, object] | None


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
    num_cpu: int
    runtime_threads: RuntimeThreadConfig
    opts: MutableMapping[str, object]


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


def _validate_dense_mask_budget(
    x: Matrix, mask_override: Matrix | None, opts: MutableMapping[str, object]
) -> None:
    """Guard against densifying masks beyond the configured budget.

    Raises:
        ValueError: If the dense mask would exceed ``max_dense_bytes``.
    """
    if not sp.issparse(x) or mask_override is None or sp.issparse(mask_override):
        return

    max_dense_bytes = resolve_max_dense_bytes(opts.get("max_dense_bytes"))
    over, est_bytes = exceeds_budget(
        np.shape(mask_override), np.bool_, max_dense_bytes
    )
    preflight_obj = opts.setdefault("_runtime_preflight", [])
    preflight = cast("list[dict[str, object]]", preflight_obj)
    preflight.append(
        {
            "check": "dense_mask_budget",
            "is_sparse_input": True,
            "mask_sparse": False,
            "estimate_bytes": int(est_bytes),
            "max_dense_bytes": max_dense_bytes,
            "over_budget": bool(over),
        }
    )
    if not over:
        return

    budget = 0 if max_dense_bytes is None else max_dense_bytes
    msg = (
        "Dense mask would exceed max_dense_bytes: "
        f"{format_bytes(est_bytes)} > {format_bytes(int(budget))}"
    )
    raise ValueError(msg)


def _prepare_problem(
    x: Matrix,
    opts: MutableMapping[str, object],
    *,
    mask_override: Matrix | None = None,
) -> PreparedProblem:
    _validate_dense_mask_budget(x, mask_override, opts)
    x_data, x_probe, mask_prepared, n1x, n2x, row_idx, col_idx = _prepare_data(
        x, opts, mask_override=mask_override
    )
    x_data, x_probe, mask, mask_probe, n_obs_row, n_data, n_probe = (
        _build_masks_and_counts(
            x_data,
            x_probe,
            opts,
            mask_override=mask_prepared,
        )
    )
    ix_obs, jx_obs = _observed_indices_with_mode(
        x_data,
        mask,
        str(opts.get("compat_mode", "strict_legacy")),
    )

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


def _auto_masked_batch_size(
    prepared: PreparedProblem, opts: MutableMapping[str, object]
) -> int:
    """Choose a masked_batch_size when user did not override it.

    Heuristic: enable batching only when pattern count is moderately large to
    reduce Python overhead while keeping per-batch work sizable.

    Returns:
        Auto-chosen batch size; ``0`` leaves batching disabled.
    """
    user_val = _int_opt(opts.get("masked_batch_size", 0))
    if user_val > 0:
        return user_val

    n_patterns = prepared.n_patterns
    if n_patterns <= 64:
        return 0

    # Use pattern count as the primary signal; cap to keep batches balanced.
    if n_patterns >= 512:
        return 512
    if n_patterns >= 256:
        return 256
    if prepared.n_data > 200_000 or n_patterns >= 128:
        return min(192, n_patterns)
    return 0


def _adjust_opts_for_explicit_init(opts: MutableMapping[str, object]) -> None:
    """Align hyperparameters when an explicit init is provided.

    Avoids drifting away from supplied MATLAB-style initializations by
    extending ``niter_broadprior`` and optionally capping ``maxiters`` when
    a learning curve is present in ``init``.
    """
    init_val = opts.get("init")
    if init_val is None or (isinstance(init_val, str) and init_val.lower() == "random"):
        return

    maxiters = _int_opt(opts.get("maxiters", 0))
    nbp = _int_opt(opts.get("niter_broadprior", 0))
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
    opts: MutableMapping[str, object],
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
        runtime_report=None,
    )


def _run_training_loop(
    *,
    prepared: PreparedProblem,
    training: TrainingState,
    use_prior: bool,
    opts: MutableMapping[str, object],
) -> TrainingState:
    """Run the VB / EM iterations, mutating and returning the training state.

    Returns:
        The updated ``TrainingState`` after running iterations or hitting a stop.
    """
    auto_batch = _auto_masked_batch_size(prepared, opts)
    if auto_batch > 0:
        opts["masked_batch_size"] = auto_batch

    workload = RuntimeWorkloadProfile(
        n_features=int(prepared.n_features),
        n_samples=int(prepared.n_samples),
        n_components=int(training.model.s.shape[0]),
        n_observed=max(0, int(prepared.n_data)),
        is_sparse=bool(sp.issparse(prepared.x_data)),
    )

    _ensure_phase_timing_keys(training.lc)

    runtime_threads, runtime_report = resolve_runtime_thread_config_with_report(
        dict(opts),
        workload=workload,
    )
    preflight = opts.pop("_runtime_preflight", None)
    if preflight:
        runtime_report["preflight"] = preflight
    training.runtime_report = runtime_report

    # Hyperparameters for Va, Vmu, V (kept here for compatibility).
    cfg = IterationConfig(
        hp_va=0.001,
        hp_vb=0.001,
        hp_v=0.001,
        eye_components=np.eye(training.model.s.shape[0], dtype=float),
        use_prior=use_prior,
        rotate_each_iter=bool(opts.get("rotate2pca", 0)),
        verbose=_int_opt(opts.get("verbose", 0)),
        num_cpu=max(0, _int_opt(opts.get("num_cpu", 1), default=1)),
        runtime_threads=runtime_threads,
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

    maxiters_int = _int_opt(opts.get("maxiters", 0))

    for iteration in range(1, maxiters_int + 1):
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
        if training.lc.get("_stop", [0.0])[-1]:
            break

    # Cleanup internal stop marker to keep lc stable for external callers.
    training.lc.pop("_stop", None)
    return training


def _ensure_phase_timing_keys(lc: dict[str, list[float]]) -> None:
    n_points = max(1, len(lc.get("rms", [])))
    for key in _PHASE_TIMING_KEYS:
        values = lc.get(key)
        if not values:
            lc[key] = [0.0] * n_points
            continue
        if len(values) < n_points:
            values.extend([0.0] * (n_points - len(values)))
        elif len(values) > n_points:
            del values[n_points:]


def _iteration_step(ctx: IterationContext) -> None:  # noqa: PLR0914,PLR0915
    """One full iteration of updates; mutates ``ctx.training`` in place."""
    training = ctx.training
    m = training.model
    cfg = ctx.cfg
    iter_start = time.perf_counter()

    # 1) Hyperpriors
    m.va, m.vmu = _update_hyperpriors(
        HyperpriorContext(
            iteration=ctx.iteration,
            use_prior=cfg.use_prior,
            niter_broadprior=_int_opt(cfg.opts.get("niter_broadprior", 0)),
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

    ctx.bias_state, ctx.centering_state = _update_bias(
        bias_enabled=bool(cfg.opts["bias"]),
        bias_state=ctx.bias_state,
        err_mx=(
            training.err_mx
            if training.err_mx is not None
            else np.zeros((0, 0), dtype=float)
        ),
        centering=ctx.centering_state,
    )
    m.mu = ctx.bias_state.mu
    m.muv = ctx.bias_state.muv

    x_data = ctx.centering_state.x_data
    x_probe = ctx.centering_state.x_probe

    score_x_csr, score_x_csc = _coerce_sparse_views(x_data)

    # 3) Scores
    score_start = time.perf_counter()
    masked_batch_size = max(0, _int_opt(cfg.opts.get("masked_batch_size", 0)))

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
        pattern_batch_size=masked_batch_size,
        x_csr=score_x_csr,
        x_csc=score_x_csc,
        sparse_num_cpu=cfg.runtime_threads.score_update_sparse,
    )
    score_state = _update_scores(score_state)
    m.s = score_state.scores
    m.sv = score_state.score_covariances
    phase_scores_sec = time.perf_counter() - score_start

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
    loadings_x_csr, loadings_x_csc = _coerce_sparse_views(x_data)

    loadings_start = time.perf_counter()
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
            x_csr=loadings_x_csr,
            x_csc=loadings_x_csc,
            sparse_num_cpu=cfg.runtime_threads.loadings_update_sparse,
        )
    )
    phase_loadings_sec = time.perf_counter() - loadings_start

    # 5) RMS (and error matrix)
    rms_start = time.perf_counter()
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
            num_cpu=cfg.runtime_threads.rms,
        )
    )
    training.err_mx = err_mx
    phase_rms_sec = time.perf_counter() - rms_start

    # 6) Noise variance
    noise_start = time.perf_counter()
    noise_state = NoiseState(
        loadings=m.a,
        scores=m.s,
        loading_covariances=m.av,
        score_covariances=m.sv,
        mu_variances=m.muv,
        pattern_index=ctx.prepared.pattern_index,
        n_data=float(ctx.prepared.n_data),
        noise_var=float(m.noise_var),
        sparse_num_cpu=cfg.runtime_threads.noise_sxv_sum,
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
    phase_noise_sec = time.perf_counter() - noise_start

    # 7) Logging + convergence
    converge_start = time.perf_counter()
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
    phase_converge_sec = time.perf_counter() - converge_start
    phase_total_sec = time.perf_counter() - iter_start

    _append_phase_timings(
        training.lc,
        {
            "phase_scores_sec": phase_scores_sec,
            "phase_loadings_sec": phase_loadings_sec,
            "phase_rms_sec": phase_rms_sec,
            "phase_noise_sec": phase_noise_sec,
            "phase_converge_sec": phase_converge_sec,
            "phase_total_sec": phase_total_sec,
        },
    )

    # Store stop message internally to keep loop logic simple without touching
    # the public learning-curve schema.
    stop_now = 0.0
    if convmsg:
        if cfg.use_prior and ctx.iteration <= _int_opt(
            cfg.opts.get("niter_broadprior", 0)
        ):
            stop_now = 0.0
        else:
            if cfg.verbose:
                logger.info("%s", convmsg)
            stop_now = 1.0

    training.lc.setdefault("_stop", []).append(float(stop_now))


def _append_phase_timings(
    lc: dict[str, list[float]],
    timings: dict[str, float],
) -> None:
    for key, value in timings.items():
        lc.setdefault(key, [0.0])
        lc[key].append(max(0.0, float(value)))


def _coerce_sparse_views(
    x_data: Matrix,
) -> tuple[sp.csr_matrix | None, sp.csc_matrix | None]:
    if not sp.issparse(x_data):
        return None, None

    x_csr = x_data if isinstance(x_data, sp.csr_matrix) else sp.csr_matrix(x_data)
    return x_csr, x_csr.tocsc()


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
    a, av, s, sv, mu, muv = m.a, m.av, m.s, m.sv, m.mu, m.muv
    pattern_index: np.ndarray | list[int] | None = prepared.pattern_index

    row_idx = (
        prepared.row_idx
        if prepared.row_idx is not None
        else np.arange(prepared.n_features, dtype=int)
    )
    col_idx = (
        prepared.col_idx
        if prepared.col_idx is not None
        else np.arange(prepared.n_samples, dtype=int)
    )

    if prepared.n_features < prepared.n1x:
        a, av_out = _add_m_rows(a, av, row_idx, prepared.n1x, m.va)
        av = cast("list[np.ndarray]", av_out)
        mu, muv_out = _add_m_rows(mu, muv, row_idx, prepared.n1x, float(m.vmu))
        muv = cast("np.ndarray", muv_out)

    if prepared.n_samples < prepared.n2x:
        s, sv_out, pattern_index = _add_m_cols(
            s, sv, col_idx, prepared.n2x, pattern_index
        )
        sv = cast("list[np.ndarray]", sv_out)
        if isinstance(pattern_index, list):
            pattern_index_out: np.ndarray | None = np.asarray(pattern_index, dtype=int)
        elif pattern_index is None:
            pattern_index_out = None
        else:
            pattern_index_out = np.asarray(pattern_index, dtype=int)
        pattern_index = pattern_index_out

    # Ensure FinalState receives ndarray or None.
    if isinstance(pattern_index, list):
        pattern_index_final: np.ndarray | None = np.asarray(pattern_index, dtype=int)
    elif pattern_index is None:
        pattern_index_final = None
    else:
        pattern_index_final = np.asarray(pattern_index, dtype=int)
    pattern_index = pattern_index_final

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
        runtime_report=training.runtime_report,
    )


def _reconstruct_data(a: np.ndarray, s: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Reconstruct data matrix using loadings, scores, and bias.

    Args:
        a: Loadings matrix of shape (n_features, n_components).
        s: Scores matrix of shape (n_components, n_samples).
        mu: Bias vector of shape (n_features,) or (n_features, 1).

    Returns:
        Reconstructed data matrix of shape (n_features, n_samples).
    """
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.ndim == 1:
        mu_arr = mu_arr[:, None]

    recon = np.asarray(a, dtype=float) @ np.asarray(s, dtype=float)
    if mu_arr.size:
        recon += mu_arr
    return recon


def _marginal_variance(final: FinalState) -> np.ndarray:
    """
    Compute marginal variance for each observed entry.

    Vectorized version of the legacy MATLAB reference:
    ``Vr(i,j) = a_i Sv_j a_i' + s_j' Av_i s_j + sum(sum(Sv_j .* Av_i)) + Muv(i)``.

    Args:
        final: Final state containing loadings, scores, covariances, and masks.

    Returns:
        Marginal variance matrix of shape (n_features, n_samples).
    """
    loadings = np.asarray(final.a, dtype=float)
    scores = np.asarray(final.s, dtype=float)
    n_features, n_samples = loadings.shape[0], scores.shape[1]
    n_components = loadings.shape[1]

    av_stack = (
        np.stack([np.asarray(x, dtype=float) for x in final.av], axis=0)
        if final.av
        else np.zeros((n_features, n_components, n_components), dtype=float)
    )

    if not final.sv:
        sv_stack = np.zeros((n_samples, n_components, n_components), dtype=float)
    elif final.pattern_index is None:
        sv_stack = np.stack([np.asarray(x, dtype=float) for x in final.sv], axis=0)
    else:
        pattern_stack = np.stack([np.asarray(x, dtype=float) for x in final.sv], axis=0)
        pattern_index_arr = np.asarray(final.pattern_index, dtype=int)
        sv_stack = pattern_stack[pattern_index_arr]

    term_loadings = np.einsum("ik,jkl,il->ij", loadings, sv_stack, loadings)
    term_scores = np.einsum("ikl,kj,lj->ij", av_stack, scores, scores)
    term_cross = np.einsum("ikl,jkl->ij", av_stack, sv_stack)

    mu_var = np.asarray(final.muv, dtype=float)
    if mu_var.size == 0:
        mu_var = np.zeros((n_features, 1), dtype=float)
    elif mu_var.ndim == 1:
        mu_var = mu_var[:, None]

    return term_loadings + term_scores + term_cross + mu_var


def _explained_variance(
    xrec: np.ndarray,
    n_components: int,
    *,
    solver: str = "auto",
    gram_ratio: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-component explained variance from the reconstruction.

    Mirrors the legacy MATLAB workflow (eigenvalues of cov(Xrec')).
    Returns raw eigenvalues and their normalized ratios.

    Returns:
        Tuple of (eigenvalues, explained variance ratios) truncated to
        ``n_components`` entries each. Empty arrays are returned when no
        variance can be computed.
    """
    if xrec.size == 0 or n_components <= 0:
        empty = np.zeros((0,), dtype=float)
        return empty, empty

    xrec_arr = np.asarray(xrec, dtype=float)
    n_features, n_samples = xrec_arr.shape

    solver_norm = str(solver).strip().lower()
    if solver_norm not in {"auto", "svd", "gram"}:
        solver_norm = "auto"

    if n_samples > 1 and n_features > n_samples:
        x_centered = xrec_arr - np.mean(xrec_arr, axis=1, keepdims=True)
        eigvals_sorted = _explained_variance_tall(
            x_centered=x_centered,
            solver=solver_norm,
            gram_ratio=gram_ratio,
        )
    else:
        cov = np.cov(xrec_arr)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals_sorted = np.flip(np.sort(np.real(eigvals)))

    eigvals_sorted = np.maximum(eigvals_sorted, 0.0)

    eigvals_top = eigvals_sorted[:n_components]
    total = float(np.sum(eigvals_sorted))
    ratios = np.zeros_like(eigvals_top) if total <= 0.0 else eigvals_top / total

    return eigvals_top, ratios


def _explained_variance_tall(
    *,
    x_centered: np.ndarray,
    solver: str,
    gram_ratio: float,
) -> np.ndarray:
    n_features, n_samples = x_centered.shape
    safe_ratio = gram_ratio if gram_ratio > 0.0 else 3.0
    feature_sample_ratio = n_features / float(max(n_samples, 1))
    use_gram = solver == "gram" or (
        solver == "auto" and feature_sample_ratio >= safe_ratio
    )

    if use_gram:
        gram = (x_centered.T @ x_centered) / float(n_samples - 1)
        eigvals = np.linalg.eigvalsh(gram)
        return np.flip(np.sort(np.real(eigvals)))

    singular_values = np.linalg.svd(x_centered, compute_uv=False)
    return (singular_values**2) / float(n_samples - 1)


def _last_metric(lc: dict[str, list[float]], key: str) -> float:
    values = lc.get(key, [])
    if not values:
        return float("nan")
    try:
        return float(values[-1])
    except (TypeError, ValueError):
        return float("nan")


def _pack_result(
    final: FinalState,
    *,
    include_diagnostics: bool = True,
    explained_var_solver: str = "auto",
    explained_var_gram_ratio: float = 4.0,
    runtime_report: dict[str, object] | None = None,
) -> dict[str, object]:
    """Pack final values into the historical PCA_FULL result dictionary.

    Returns:
        Dictionary mirroring the legacy MATLAB output structure.
    """
    if include_diagnostics:
        xrec = _reconstruct_data(final.a, final.s, final.mu)
        vr = _marginal_variance(final)
        ev, evr = _explained_variance(
            xrec,
            final.a.shape[1],
            solver=explained_var_solver,
            gram_ratio=explained_var_gram_ratio,
        )
    else:
        xrec = None
        vr = None
        ev = None
        evr = None

    lc = final.lc

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
        "Xrec": xrec,
        "Vr": vr,
        "ExplainedVar": ev,
        "ExplainedVarRatio": evr,
        "RMS": _last_metric(lc, "rms"),
        "PRMS": _last_metric(lc, "prms"),
        "Cost": _last_metric(lc, "cost"),
        "cv": {
            "A": final.av,
            "S": final.sv,
            "Isv": final.pattern_index,
            "Mu": final.muv,
        },
        "hp": {"Va": final.va, "Vmu": float(final.vmu)},
        "RuntimeReport": runtime_report,
    }


# ---------------------------------------------------------------------------
# Options + algorithm selection (publicly relied upon)
# ---------------------------------------------------------------------------


def _build_options(kwargs: Mapping[str, object]) -> dict[str, object]:
    """Merge user kwargs with defaults using _options (case-insensitive).

    Returns:
        Options dictionary after applying defaults and user overrides.

    Raises:
        ValueError: If ``compat_mode`` is not one of
            ``{"strict_legacy", "modern"}``.
    """
    opts_default: dict[str, object] = {
        "init": "random",
        "maxiters": 1000,
        "bias": True,
        "max_dense_bytes": 2_000_000_000,
        "uniquesv": False,
        "auto_pattern_masked": False,
        "angle_every": 1,
        "minangle": 1e-8,
        "algorithm": "vb",
        "niter_broadprior": 100,
        "earlystop": False,
        "rmsstop": np.array([100, 1e-4, 1e-3]),
        "cfstop": np.array([]),
        "verbose": 1,
        "num_cpu": 1,
        "num_cpu_score_update": None,
        "num_cpu_loadings_update": None,
        "num_cpu_noise_update": None,
        "num_cpu_rms": None,
        "runtime_tuning": "off",
        "runtime_profile": None,
        "masked_batch_size": 0,
        "xprobe": None,
        "rotate2pca": True,
        "display": False,
        "compat_mode": "strict_legacy",
        "return_diagnostics": True,
        "runtime_report": False,
        "explained_var_solver": "auto",
        "explained_var_gram_ratio": 4.0,
    }

    opts, wrnmsg = _options(opts_default, **kwargs)
    opts["_num_cpu_user_set"] = any(str(key).lower() == "num_cpu" for key in kwargs)

    compat_mode_raw = opts.get("compat_mode", "strict_legacy")
    compat_mode = str(compat_mode_raw).strip().lower()
    if compat_mode not in _ALLOWED_COMPAT_MODES:
        msg = (
            "compat_mode must be one of {'strict_legacy', 'modern'} "
            f"(got {compat_mode_raw!r})."
        )
        raise ValueError(msg)
    opts["compat_mode"] = compat_mode
    opts["angle_every"] = max(1, _int_opt(opts.get("angle_every", 1), default=1))
    opts["return_diagnostics"] = int(bool(opts.get("return_diagnostics", 1)))
    opts = apply_runtime_policy_defaults(opts)

    solver_raw = str(opts.get("explained_var_solver", "auto"))
    solver = solver_raw.strip().lower()
    if solver not in {"auto", "svd", "gram"}:
        msg = (
            "explained_var_solver must be one of {'auto', 'svd', 'gram'} "
            f"(got {solver_raw!r})."
        )
        raise ValueError(msg)
    opts["explained_var_solver"] = solver

    ratio = _float_opt(opts.get("explained_var_gram_ratio", 4.0), default=4.0)
    if ratio <= 0.0:
        ratio = 4.0
    opts["explained_var_gram_ratio"] = ratio

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
