"""Fixed-iteration and bounded-selection qualification of the noise rearrangement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py import _pca_full as pf
from vbpca_py.model_selection import (
    CVConfig,
    SelectionConfig,
    cross_validate_components,
    select_n_components,
)

if TYPE_CHECKING:
    from vbpca_py._full_update import NoiseState

_CONVERGENCE_OFF = {
    "angle": False,
    "earlystop": False,
    "rms_plateau": False,
    "cost": False,
    "composite": False,
    "slowing_down": False,
}


def _options() -> dict[str, object]:
    return {
        "maxiters": 6,
        "niter_broadprior": 0,
        "runtime_tuning": "off",
        "num_cpu": 1,
        "random_state": 289,
        "convergence_criteria": dict(_CONVERGENCE_OFF),
        "record_cost": True,
        "runtime_report": True,
        "verbose": 0,
    }


def _data(sparse: bool, patterns: bool) -> tuple:
    rng = np.random.default_rng(419)
    x = rng.normal(size=(40, 30))
    mask = np.ones(x.shape, dtype=bool)
    if patterns:
        mask[2, ::2] = False
        mask[8, 1::2] = False
    else:
        mask.ravel()[rng.choice(x.size, 35, replace=False)] = False
    mask[0, 0] = True
    x[0, 0] = 0.0
    x[3, 4] = 0.0
    if sparse:
        rows, columns = np.nonzero(mask)
        data = sp.csr_matrix((x[rows, columns], (rows, columns)), shape=x.shape)
        observed = data.copy()
        observed.data = np.ones(data.nnz, dtype=bool)
        return data, observed
    return x, mask


@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("patterns", [False, True])
@pytest.mark.parametrize("compat", ["modern", "strict_legacy"])
def test_fixed_iteration_factor_covariance_noise_and_cost_trajectories(
    algorithm: str,
    sparse: bool,
    patterns: bool,
    compat: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, mask = _data(sparse, patterns)
    original = pf._update_noise_variance
    outputs, trajectories = [], []
    for mode in ("observed", "complement"):
        trace = []

        def capture(state: NoiseState, rms, ix, jx, hp_v=0.001, *, _trace=trace):
            updated, covariance_sum = original(state, rms, ix, jx, hp_v=hp_v)
            _trace.append({
                "loadings": state.loadings.copy(),
                "scores": state.scores.copy(),
                "av": np.asarray(state.loading_covariances).copy(),
                "sv": np.asarray(state.score_covariances).copy(),
                "muv": state.mu_variances.copy(),
                "ix": ix.copy(),
                "jx": jx.copy(),
                "noise": updated.noise_var,
                "covariance_sum": covariance_sum,
            })
            return updated, covariance_sum

        monkeypatch.setattr(pf, "_update_noise_variance", capture)
        outputs.append(
            pf.pca_full(
                data,
                3,
                mask=mask,
                algorithm=algorithm,
                uniquesv=int(patterns),
                compat_mode=compat,
                noise_sufficient_statistics=mode,
                **_options(),
            )
        )
        trajectories.append(trace)
    assert len(trajectories[0]) == len(trajectories[1])
    for observed, candidate in zip(*trajectories, strict=True):
        for name in observed:
            np.testing.assert_allclose(
                candidate[name], observed[name], rtol=1e-9, atol=1e-10, err_msg=name
            )
    for key in ("A", "S", "Mu", "V", "Va", "Vmu", "Xrec", "Vr", "Vpred"):
        np.testing.assert_allclose(
            outputs[1][key], outputs[0][key], rtol=1e-9, atol=1e-10
        )
    for key in ("rms", "prms", "cost", "angle"):
        np.testing.assert_allclose(
            outputs[1]["lc"][key], outputs[0]["lc"][key], rtol=1e-9, atol=1e-10
        )
    report = outputs[1]["RuntimeReport"]["noise_sufficient_statistics"]
    assert report["requested"] == "complement"
    if compat == "modern":
        assert report["complement_iterations"] > 0
    assert (
        sum(
            report[key]
            for key in (
                "complement_iterations",
                "complete_iterations",
                "observed_iterations",
            )
        )
        == 6
    )


@pytest.mark.parametrize("seed", [17, 29])
def test_bounded_cv_selection_and_convergence_agree(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(12, 2)) @ rng.normal(size=(2, 30))
    x += rng.normal(scale=0.2, size=x.shape)
    outputs = [
        cross_validate_components(
            x,
            components=[0, 1, 2, 3],
            config=CVConfig(n_splits=10, seed=seed),
            noise_sufficient_statistics=mode,
            maxiters=40,
            niter_broadprior=0,
            runtime_tuning="off",
            num_cpu=1,
            random_state=seed,
        )
        for mode in ("observed", "complement")
    ]
    assert outputs[0][0] == outputs[1][0]
    for observed, candidate in zip(outputs[0][1], outputs[1][1], strict=True):
        assert observed.keys() == candidate.keys()
        for key, value in observed.items():
            if isinstance(value, float):
                np.testing.assert_allclose(candidate[key], value, rtol=1e-8, atol=1e-10)
            else:
                assert candidate[key] == value


def test_auto_dispatch_is_eligible_by_density_rank_and_size() -> None:
    rng = np.random.default_rng(423)
    x = rng.normal(size=(64, 64))
    x[3, 7] = np.nan
    result = pf.pca_full(x, 8, noise_sufficient_statistics="auto", **_options())
    report = result["RuntimeReport"]["noise_sufficient_statistics"]
    assert report["planned"] == "complement"
    assert report["missing_coordinates"] == 1
    assert report["coordinate_bytes"] == 8
    assert report["complement_iterations"] == 6
    low_rank = pf.pca_full(x, 2, noise_sufficient_statistics="auto", **_options())
    assert (
        low_rank["RuntimeReport"]["noise_sufficient_statistics"]["planned"]
        == "observed_or_complete"
    )


def test_default_dispatch_and_explicit_observed_override_preserve_sparse_policy():
    rng = np.random.default_rng(425)
    x = rng.normal(size=(64, 64))
    x[3, 7] = np.nan
    default = pf.pca_full(x, 8, **_options())
    report = default["RuntimeReport"]["noise_sufficient_statistics"]
    assert report["requested"] == "auto"
    assert report["complement_iterations"] == 6
    observed = pf.pca_full(x, 8, noise_sufficient_statistics="observed", **_options())
    assert (
        observed["RuntimeReport"]["noise_sufficient_statistics"]["observed_iterations"]
        == 6
    )
    for key in ("Xrec", "Vr", "Vpred", "V"):
        np.testing.assert_allclose(default[key], observed[key], rtol=1e-10, atol=1e-11)
    mask = np.isfinite(x)
    ix, jx = np.nonzero(mask)
    data = sp.csr_matrix((x[ix, jx], (ix, jx)), shape=x.shape)
    sparse = pf.pca_full(data, 8, compat_mode="modern", **_options())
    report = sparse["RuntimeReport"]["noise_sufficient_statistics"]
    assert report["requested"] == "auto"
    assert report["observed_iterations"] == 6
    assert report["coordinate_bytes"] == 0


@pytest.mark.parametrize("seed", [71, 89])
def test_automatic_noise_path_preserves_bounded_prms_sweep(monkeypatch, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(96, 3)) @ rng.normal(size=(3, 128))
    x += 0.3 * rng.normal(size=x.shape)
    x[rng.random(x.shape) < 0.01] = np.nan
    x[0, 0] = 0.0
    real = pf._update_noise_variance
    calls = []

    def record(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result[0].statistics_mode)
        return result

    monkeypatch.setattr(pf, "_update_noise_variance", record)
    results = []
    for mode in ("observed", "auto"):
        calls.clear()
        results.append(
            select_n_components(
                x,
                components=[4, 8, 12],
                xprobe_fraction=0.01,
                config=SelectionConfig(
                    compute_explained_variance=False, return_best_model=True
                ),
                noise_sufficient_statistics=mode,
                **{**_options(), "maxiters": 20},
            )
        )
        assert ("complement" in calls) == (mode == "auto")
    assert results[0][0] == results[1][0]
    for a, b in zip(results[0][2], results[1][2], strict=True):
        for key in ("rms", "prms", "cost", "n_iter"):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        results[0][3].noise_variance_, results[1][3].noise_variance_, rtol=1e-10
    )
