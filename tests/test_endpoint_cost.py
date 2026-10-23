"""Endpoint cost preserves fitting and returned-state metrics without a trace."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import vbpca_py._converge as convergence
from vbpca_py._missing import make_xprobe_mask
from vbpca_py._pca_full import pca_full
from vbpca_py._selection_core import (
    CVConfig,
    SelectionConfig,
    cross_validate_components,
    select_n_components,
)

OFF = dict.fromkeys(
    ("angle", "earlystop", "rms_plateau", "cost", "composite", "slowing_down"), False
)
OPTS = {
    "maxiters": 5,
    "niter_broadprior": 0,
    "convergence_criteria": OFF,
    "runtime_tuning": "off",
    "num_cpu": 1,
    "random_state": 295,
    "verbose": 0,
    "return_diagnostics": False,
    "runtime_report": True,
}


def _problem():
    rng = np.random.default_rng(295)
    x = rng.normal(size=(17, 23))
    x[rng.random(x.shape) < 0.15] = np.nan
    x[0] = rng.normal(size=23)
    x[:, 0] = rng.normal(size=17)
    return x


@pytest.mark.parametrize("algorithm", ["vb", "map", "ppca"])
@pytest.mark.parametrize("order", ["legacy", "post_factor"])
@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("uniquesv", [False, True])
def test_endpoint_preserves_factors_and_cost(algorithm, order, rotate, uniquesv):
    opts = {
        **OPTS,
        "algorithm": algorithm,
        "bias_update_order": order,
        "rotate2pca": rotate,
        "uniquesv": uniquesv,
    }
    trace = pca_full(_problem(), 3, record_cost=True, **opts)
    endpoint = pca_full(_problem(), 3, cost_diagnostics="endpoint", **opts)
    for key in ("A", "S", "Mu", "V", "Av", "Sv", "Va", "Muv", "RMS", "PRMS", "Cost"):
        assert_allclose(endpoint[key], trace[key], rtol=0, atol=0)
    assert_allclose(endpoint["lc"]["rms"], trace["lc"]["rms"], rtol=0, atol=0)
    assert np.count_nonzero(np.isfinite(endpoint["lc"]["cost"])) == 1
    assert endpoint["RuntimeReport"]["cost_diagnostics"]["endpoint_evaluations"] == 1


def test_selection_endpoint_reduces_cost_calls_and_keeps_rank_zero(monkeypatch):
    calls = []
    real = convergence.compute_full_cost

    def record(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(convergence, "compute_full_cost", record)
    results = []
    for mode in ("trace", "endpoint"):
        calls.clear()
        result = select_n_components(
            _problem(),
            components=[0, 2, 4],
            config=SelectionConfig(
                cost_diagnostics=mode,
                return_best_model=True,
                compute_explained_variance=False,
            ),
            record_cost=False,
            **OPTS,
        )
        assert len(calls) == (10 if mode == "trace" else 2)
        assert result[2][0]["cost_diagnostics"] == "not_applicable"
        assert all(row["cost_diagnostics"] == mode for row in result[2][1:])
        results.append(result)
    assert results[0][0] == results[1][0]
    for a, b in zip(results[0][2], results[1][2], strict=True):
        for key in ("rms", "prms", "cost", "n_iter", "converged"):
            assert_allclose(a[key], b[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "option",
    [
        {"record_cost": True},
        {"cfstop": [2, 1e-5, 1e-5]},
        {"cfstop_rel": 1e-5},
        {"cfstop_curv": 1e-5},
        {"composite_stop": {"elbo_rel": 1e-5}},
    ],
)
def test_endpoint_rejects_cost_consumers(option):
    with pytest.raises(ValueError, match="cost-dependent stopping criteria"):
        pca_full(
            _problem(),
            2,
            cost_diagnostics="endpoint",
            **{
                **OPTS,
                "convergence_criteria": {"cost": True, "composite": True},
                **option,
            },
        )


def test_disabled_cost_event_does_not_require_a_trace():
    result = pca_full(
        _problem(), 2, cost_diagnostics="endpoint", cfstop_rel=1e-3, **OPTS
    )
    assert np.isfinite(result["Cost"])


@pytest.mark.parametrize("mode", ["invalid", "ENDPOINT_UNKNOWN"])
def test_invalid_diagnostic_mode(mode):
    with pytest.raises(ValueError, match="cost_diagnostics"):
        pca_full(_problem(), 2, cost_diagnostics=mode, **OPTS)
    with pytest.raises(ValueError, match="cost_diagnostics"):
        select_n_components(
            _problem(),
            components=[2],
            config=SelectionConfig(cost_diagnostics=mode),
            **OPTS,
        )


def test_cost_selection_preserves_trace_requirement():
    with pytest.raises(ValueError, match="cost selection requires"):
        select_n_components(
            _problem(),
            components=[2],
            config=SelectionConfig(metric="cost", cost_diagnostics="endpoint"),
            **OPTS,
        )


def test_zero_iterations_has_no_invented_endpoint_cost():
    result = pca_full(
        _problem(), 2, cost_diagnostics="endpoint", **{**OPTS, "maxiters": 0}
    )
    assert np.isnan(result["Cost"])
    assert result["RuntimeReport"]["cost_diagnostics"]["endpoint_evaluations"] == 0


def test_probe_restore_cost_matches_the_returned_best_iteration():
    x = np.random.default_rng(0).normal(size=(17, 23))
    x, probe = make_xprobe_mask(x, fraction=0.3, rng=np.random.default_rng(295))
    opts = {
        **OPTS,
        "maxiters": 30,
        "earlystop": True,
        "xprobe": probe,
        "convergence_criteria": {**OFF, "earlystop": True},
    }
    reference = pca_full(x, 4, record_cost=True, **opts)
    endpoint = pca_full(x, 4, cost_diagnostics="endpoint", **opts)
    lc = endpoint["lc"]
    assert lc["convergence_reason"] == "earlystop"
    assert 0 < lc["returned_iteration"] < lc["terminal_iteration"]
    assert np.isnan(lc["cost"][-1])
    for key in ("A", "S", "V", "Cost", "PRMS"):
        assert_allclose(endpoint[key], reference[key], rtol=0, atol=0)


def test_cross_validation_preserves_endpoint_metrics():
    rows = [
        cross_validate_components(
            _problem(),
            components=[0, 1, 2],
            config=CVConfig(n_splits=2),
            cost_diagnostics=mode,
            **OPTS,
        )
        for mode in ("trace", "endpoint")
    ]
    assert rows[0][0] == rows[1][0]
    for left, right in zip(rows[0][1], rows[1][1], strict=True):
        for key in ("mean_rms", "mean_prms", "mean_cost"):
            assert_allclose(left[key], right[key], rtol=0, atol=0)
