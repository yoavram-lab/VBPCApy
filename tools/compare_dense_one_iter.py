import numpy as np
from scipy.io import loadmat

from vbpca_py._full_update import (
    BiasState,
    CenteringState,
    LoadingsUpdateState,
    NoiseState,
    RmsContext,
    ScoreState,
    _build_masks_and_counts,
    _missing_patterns_info,
    _recompute_rms,
    _update_bias,
    _update_loadings,
    _update_noise_variance,
    _update_scores,
)
from vbpca_py._mean import ProbeMatrices, subtract_mu
from vbpca_py._options import _options
from vbpca_py._rms import RmsConfig, compute_rms
from vbpca_py._rotate import RotateParams, rotate_to_pca
from vbpca_py.pca_full import _prepare_data


def run_python_one_iter():
    mat = loadmat(
        "tests/data/legacy_pca_full_dense.mat", squeeze_me=True, struct_as_record=False
    )
    res = mat["result"]
    x = np.asarray(mat["x"], float)
    opts, _ = _options(
        {
            "init": "random",
            "maxiters": int(res.maxiters),
            "bias": int(res.bias),
            "uniquesv": int(res.uniquesv),
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
    )

    xd, xp, *_ = _prepare_data(x, opts)
    xd, xp, mask, mask_probe, n_obs_row, n_data, n_probe = _build_masks_and_counts(
        xd, xp, opts
    )
    _, obs_patterns, pattern_index = _missing_patterns_info(
        mask, opts, n_samples=xd.shape[1]
    )

    A = np.asarray(res.A, float)
    S = np.asarray(res.S, float)
    Mu = np.asarray(res.Mu, float)
    if Mu.ndim == 1:
        Mu = Mu[:, None]
    Av = [np.asarray(a, float) for a in res.Av]
    Sv = [np.asarray(s, float) for s in res.Sv]
    Muv = np.asarray(res.Muv, float)
    if Muv.ndim == 1:
        Muv = Muv[:, None]

    # Center data using the starting Mu, matching MATLAB SubtractMu.
    probe = ProbeMatrices(x=xp, mask=mask_probe) if xp is not None else None
    xd_centered, xp_centered = subtract_mu(Mu, xd, mask, probe=probe, update_bias=True)

    rms_cfg = RmsConfig(n_observed=int(n_data))
    rms0, err_mx0 = compute_rms(xd_centered, A, S, mask, rms_cfg)

    bias = BiasState(
        mu=Mu, muv=Muv, noise_var=float(res.V), vmu=float(res.Vmu), n_obs_row=n_obs_row
    )
    cent = CenteringState(
        x_data=xd_centered, x_probe=xp_centered, mask=mask, mask_probe=mask_probe
    )
    bias, cent = _update_bias(
        bias_enabled=True,
        bias_state=bias,
        err_mx=np.asarray(err_mx0),
        centering=cent,
    )

    score = ScoreState(
        x_data=cent.x_data,
        mask=mask,
        loadings=A,
        scores=S.copy(),
        loading_covariances=Av,
        score_covariances=Sv.copy(),
        pattern_index=pattern_index,
        obs_patterns=obs_patterns,
        noise_var=float(res.V),
        eye_components=np.eye(S.shape[0]),
        verbose=0,
    )
    score = _update_scores(score)

    rot_params = RotateParams(
        loading_covariances=score.loading_covariances,
        score_covariances=score.score_covariances,
        isv=pattern_index,
        obscombj=obs_patterns,
        update_bias=True,
    )
    d_mu_rot, A_rot, Av_rot, S_rot, Sv_rot = rotate_to_pca(
        score.loadings, score.scores, rot_params
    )
    Mu_rot = bias.mu + d_mu_rot

    load_state = LoadingsUpdateState(
        x_data=cent.x_data,
        mask=mask,
        scores=S_rot,
        loading_covariances=Av_rot,
        score_covariances=Sv_rot,
        pattern_index=pattern_index,
        va=res.Va,
        noise_var=float(res.V),
        verbose=0,
    )
    A_new, Av_new = _update_loadings(load_state)

    rms_ctx = RmsContext(
        x_data=cent.x_data,
        x_probe=cent.x_probe,
        mask=mask,
        mask_probe=mask_probe,
        n_data=float(n_data),
        n_probe=int(n_probe),
        loadings=A_new,
        scores=S_rot,
    )
    rms, prms, err = _recompute_rms(rms_ctx)

    noise_state = NoiseState(
        loadings=A_new,
        scores=S_rot,
        loading_covariances=Av_new,
        score_covariances=Sv_rot,
        mu_variances=bias.muv,
        pattern_index=pattern_index,
        n_data=float(n_data),
        noise_var=float(res.V),
    )
    noise_state, sxv = _update_noise_variance(
        noise_state,
        float(rms),
        np.nonzero(xd)[0],
        np.nonzero(xd)[1],
        hp_v=0.001,
    )

    return {
        "A": A_new,
        "S": S_rot,
        "Mu": Mu_rot,
        "V": float(noise_state.noise_var),
        "rms": float(rms),
    }


def run_octave_one_iter():
    mat = loadmat("/tmp/oct_trace_dense.mat", squeeze_me=True, struct_as_record=False)
    A = np.asarray(mat["A"], float)
    S = np.asarray(mat["S"], float)
    Mu = np.asarray(mat["Mu"], float)
    if Mu.ndim == 1:
        Mu = Mu[:, None]
    V = float(mat["V"])
    return {"A": A, "S": S, "Mu": Mu, "V": V}


def main():
    py = run_python_one_iter()
    oc = run_octave_one_iter()

    print(
        "Python: A norm",
        np.linalg.norm(py["A"]),
        "S norm",
        np.linalg.norm(py["S"]),
        "Mu norm",
        np.linalg.norm(py["Mu"]),
        "V",
        py["V"],
        "rms",
        py["rms"],
    )
    print(
        "Octave: A norm",
        np.linalg.norm(oc["A"]),
        "S norm",
        np.linalg.norm(oc["S"]),
        "Mu norm",
        np.linalg.norm(oc["Mu"]),
        "V",
        oc["V"],
    )
    print(
        "Deltas: A",
        np.linalg.norm(py["A"] - oc["A"]),
        "S",
        np.linalg.norm(py["S"] - oc["S"]),
        "Mu",
        np.linalg.norm(py["Mu"] - oc["Mu"]),
        "V",
        py["V"] - oc["V"],
    )


if __name__ == "__main__":
    main()
