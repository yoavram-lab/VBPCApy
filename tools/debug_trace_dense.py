"""Debug trace of one iteration starting from MATLAB dense fixture.

Run: python tools/debug_trace_dense.py
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from vbpca_py._full_update import (
    BiasState,
    CenteringState,
    HyperpriorContext,
    LoadingsUpdateState,
    NoiseState,
    RmsContext,
    RotationContext,
    ScoreState,
    _build_masks_and_counts,
    _missing_patterns_info,
    _recompute_rms,
    _update_bias,
    _update_hyperpriors,
    _update_loadings,
    _update_noise_variance,
    _update_scores,
)
from vbpca_py._options import _options
from vbpca_py._rms import RmsConfig, compute_rms
from vbpca_py._rotate import RotateParams, rotate_to_pca
from vbpca_py.pca_full import (
    ModelState,
    PreparedProblem,
    TrainingState,
    _prepare_data,
)


def _load_fixture(path: str) -> tuple[np.ndarray, object]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    return np.asarray(mat["x"], float), mat["result"]


def _build_prepared(x: np.ndarray, opts: dict[str, object]) -> PreparedProblem:
    x_data, x_probe, n1x, n2x, row_idx, col_idx = _prepare_data(x, opts)
    x_data, x_probe, mask, mask_probe, n_obs_row, n_data, n_probe = (
        _build_masks_and_counts(x_data, x_probe, opts)
    )
    # pattern info
    n_patterns, obs_patterns, pattern_index = _missing_patterns_info(
        mask, opts, n_samples=x_data.shape[1]
    )
    ix_obs, jx_obs = np.nonzero(x_data)
    return PreparedProblem(
        x_data=x_data,
        x_probe=x_probe,
        mask=mask,
        mask_probe=mask_probe,
        n_obs_row=n_obs_row,
        n_data=float(n_data),
        n_probe=int(n_probe),
        ix_obs=np.array(ix_obs),
        jx_obs=np.array(jx_obs),
        n_features=int(x_data.shape[0]),
        n_samples=int(x_data.shape[1]),
        n1x=int(n1x),
        n2x=int(n2x),
        row_idx=row_idx,
        col_idx=col_idx,
        n_patterns=int(n_patterns),
        obs_patterns=obs_patterns,
        pattern_index=pattern_index,
    )


def _model_from_mat(res: object) -> ModelState:
    a = np.asarray(res.A, float)
    s = np.asarray(res.S, float)
    mu = np.asarray(res.Mu, float)
    if mu.ndim == 1:
        mu = mu[:, None]
    mu = mu.astype(float)
    av = [np.asarray(m, float) for m in res.Av]
    sv = [np.asarray(m, float) for m in res.Sv]
    muv = np.asarray(res.Muv, float)
    if muv.ndim == 1:
        muv = muv[:, None]
    va = np.asarray(res.Va, float)
    vmu = float(res.Vmu)
    return ModelState(
        a=a,
        s=s,
        mu=mu,
        noise_var=float(res.V),
        av=av,
        sv=sv,
        muv=muv,
        va=va,
        vmu=vmu,
    )


def trace_dense() -> None:
    x, res = _load_fixture("tests/data/legacy_pca_full_dense.mat")
    # mirror fixture options
    defopts = {
        "init": "random",
        "maxiters": int(getattr(res, "maxiters", 200)),
        "bias": int(getattr(res, "bias", 1)),
        "uniquesv": int(getattr(res, "uniquesv", 0)),
        "autosave": 0,
        "filename": "pca_f_autosave",
        "minangle": 1e-8,
        "algorithm": "vb",
        "niter_broadprior": 100,
        "earlystop": 0,
        "rmsstop": np.array([100, 1e-4, 1e-3]),
        "cfstop": np.array([]),
        "verbose": 0,
        "xprobe": None,
        "rotate2pca": 1,
        "display": 0,
    }
    opts, _ = _options(defopts)

    prepared = _build_prepared(x, opts)
    model = _model_from_mat(res)

    # initial err matrix from reconstruction (matches MATLAB compute_rms)
    rms_cfg = RmsConfig(n_observed=int(prepared.n_data))
    rms0, err_mx0 = compute_rms(
        prepared.x_data, model.a, model.s, prepared.mask, rms_cfg
    )

    lc = {
        "rms": [float(res.lc.rms[-1])],
        "prms": [float(res.lc.prms[-1])],
        "time": [float(res.lc.time[-1])],
        "cost": [],
    }
    training = TrainingState(
        model=model,
        lc=lc,
        dsph={},
        err_mx=err_mx0,
        a_old=model.a.copy(),
        time_start=time.time(),
    )

    bias_state = BiasState(
        mu=model.mu,
        muv=model.muv,
        noise_var=float(model.noise_var),
        vmu=float(model.vmu),
        n_obs_row=prepared.n_obs_row,
    )
    centering_state = CenteringState(
        x_data=prepared.x_data,
        x_probe=prepared.x_probe,
        mask=prepared.mask,
        mask_probe=prepared.mask_probe,
    )

    cfg = SimpleNamespace(
        hp_va=0.001,
        hp_vb=0.001,
        hp_v=0.001,
        eye_components=np.eye(model.s.shape[0], dtype=float),
        use_prior=True,
        rotate_each_iter=bool(opts["rotate2pca"]),
        verbose=0,
        opts=opts,
    )

    def diff(tag: str, new: np.ndarray, ref: np.ndarray) -> str:
        return f"{tag}: norm={np.linalg.norm(new - ref):.6e}, rel={np.linalg.norm(new - ref) / (np.linalg.norm(ref) + 1e-12):.6e}"

    print("=== Starting from MATLAB params (dense) ===")

    # 1) Hyperpriors
    hp_ctx = HyperpriorContext(
        iteration=1,
        use_prior=cfg.use_prior,
        niter_broadprior=int(opts["niter_broadprior"]),
        bias_enabled=bool(opts["bias"]),
        mu=bias_state.mu,
        mu_variances=bias_state.muv,
        loadings=model.a,
        loading_covariances=model.av,
        n_features=prepared.n_features,
        hp_va=cfg.hp_va,
        hp_vb=cfg.hp_vb,
        va=model.va,
        vmu=float(model.vmu),
    )
    va_new, vmu_new = _update_hyperpriors(hp_ctx)
    print(diff("Va", va_new, model.va))
    print(f"Vmu delta: {vmu_new - model.vmu:.6e}")

    # 2) Bias
    bias_state.noise_var = float(model.noise_var)
    bias_state.vmu = float(model.vmu)
    bias_state, centering_state = _update_bias(
        bias_enabled=bool(opts["bias"]),
        bias_state=bias_state,
        err_mx=np.asarray(
            err_mx0 if err_mx0 is not None else np.zeros_like(prepared.x_data)
        ),
        centering=centering_state,
    )
    print(diff("Mu", bias_state.mu, model.mu))

    x_data = centering_state.x_data
    x_probe = centering_state.x_probe

    # 3) Scores
    score_state = ScoreState(
        x_data=x_data,
        mask=prepared.mask,
        loadings=model.a,
        scores=model.s.copy(),
        loading_covariances=model.av,
        score_covariances=model.sv.copy(),
        pattern_index=prepared.pattern_index,
        obs_patterns=prepared.obs_patterns,
        noise_var=float(model.noise_var),
        eye_components=cfg.eye_components,
        verbose=0,
    )
    score_state = _update_scores(score_state)
    print(diff("S", score_state.scores, model.s))

    # 3b) Rotate-to-PCA (per-iter)
    rot_ctx = RotationContext(
        loadings=model.a,
        loading_covariances=model.av,
        scores=score_state.scores,
        score_covariances=score_state.score_covariances,
        mu=bias_state.mu,
        pattern_index=prepared.pattern_index,
        obs_patterns=prepared.obs_patterns,
        bias_enabled=bool(opts["bias"]),
    )
    rot_params = RotateParams(
        loading_covariances=rot_ctx.loading_covariances,
        score_covariances=rot_ctx.score_covariances,
        isv=rot_ctx.pattern_index,
        obscombj=rot_ctx.obs_patterns,
        update_bias=rot_ctx.bias_enabled,
    )
    d_mu_rot, a_rot, av_rot, s_rot, sv_rot = rotate_to_pca(
        rot_ctx.loadings,
        rot_ctx.scores,
        rot_params,
    )
    mu_after_rot = rot_ctx.mu + d_mu_rot if rot_ctx.bias_enabled else rot_ctx.mu
    print(diff("A after rot", a_rot, model.a))
    print(diff("S after rot", s_rot, score_state.scores))
    print(diff("Mu after rot", mu_after_rot, bias_state.mu))

    # 4) Loadings
    load_state = LoadingsUpdateState(
        x_data=x_data,
        mask=prepared.mask,
        scores=s_rot,
        loading_covariances=av_rot,
        score_covariances=sv_rot,
        pattern_index=prepared.pattern_index,
        va=model.va,
        noise_var=float(model.noise_var),
        verbose=0,
    )
    a_new, av_new = _update_loadings(load_state)
    print(diff("A update", a_new, a_rot))

    # 5) RMS
    rms_ctx = RmsContext(
        x_data=x_data,
        x_probe=x_probe,
        mask=prepared.mask,
        mask_probe=prepared.mask_probe,
        n_data=float(prepared.n_data),
        n_probe=int(prepared.n_probe),
        loadings=a_new,
        scores=s_rot,
    )
    rms, prms, err_mx = _recompute_rms(rms_ctx)
    print(f"rms after updates: {rms:.6f}, fixture rms: {float(res.lc.rms[-1]):.6f}")

    # 6) Noise
    noise_state = NoiseState(
        loadings=a_new,
        scores=s_rot,
        loading_covariances=av_new,
        score_covariances=sv_rot,
        mu_variances=bias_state.muv,
        pattern_index=prepared.pattern_index,
        n_data=float(prepared.n_data),
        noise_var=float(model.noise_var),
    )
    noise_state, s_xv = _update_noise_variance(
        noise_state,
        float(rms),
        prepared.ix_obs,
        prepared.jx_obs,
        hp_v=cfg.hp_v,
    )
    print(f"V new: {noise_state.noise_var:.6f}, fixture V: {float(res.V):.6f}")
    print(f"s_xv: {s_xv:.6f}")


if __name__ == "__main__":
    trace_dense()
