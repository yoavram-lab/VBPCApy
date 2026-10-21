"""Recommendations must not revive the qualified premature cost-stop policy."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from vbpca_py import VBPCA, recommend_config
from vbpca_py._pca_full import _build_options


def _recommend(n, p, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return recommend_config(n=n, p=p, **kwargs)


@pytest.mark.parametrize(
    ("n", "p"),
    [
        (100, 20),
        (100, 50),
        (200, 150),
        (30, 100),
        (30, 2000),
        (1000, 50),
        (3000, 30),
        (400, 400),
    ],
)
@pytest.mark.parametrize("priority", ["balanced", "accuracy", "speed"])
def test_every_recommendation_leaves_relative_cost_stopping_inactive(n, p, priority):
    opts = _build_options(_recommend(n, p, priority=priority))
    assert opts["cfstop_rel"] is None
    criteria = opts["convergence_criteria"]
    if p > 200 or p / n > 2 or n / p > 15:
        assert criteria["cost"] is False
        assert criteria["composite"] is False
        assert criteria["angle"] is True
        assert criteria["rms_plateau"] is True


def test_cost_stop_regression_against_stable_fixed_iteration_reconstruction():
    # Independent rank-one signal + Gaussian noise, with data and fit seeds
    # fixed separately from the recommendation-policy choice.
    rng = np.random.default_rng(247)
    p = n = 400
    u, v = rng.normal(size=p), rng.normal(size=n)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    x = rng.normal(size=(p, n)) + 2.4 * np.sqrt(n) * np.outer(u, v)
    opts = _recommend(n, p)
    opts.update(
        algorithm="ppca",
        xprobe_fraction=0,
        random_state=247,
        runtime_tuning="off",
        num_cpu=1,
        return_diagnostics=False,
        record_cost=True,
        maxiters=300,
    )
    corrected = VBPCA(1, **opts).fit(x)
    legacy = VBPCA(
        1,
        **{
            **opts,
            "cfstop_rel": 0.0009148652415765465,
            "convergence_criteria": {
                **opts["convergence_criteria"],
                "cost": True,
                "composite": True,
            },
        },
    ).fit(x)
    reference_opts = {
        **opts,
        "record_cost": False,
        "convergence_criteria": dict.fromkeys(opts["convergence_criteria"], False),
    }
    reference = VBPCA(1, **reference_opts).fit(x)
    truth = reference.components_ @ reference.scores_

    def error(model):
        return np.linalg.norm(
            model.components_ @ model.scores_ - truth
        ) / np.linalg.norm(truth)

    assert legacy.convergence_reason_ == "cfstop_rel"
    assert corrected.convergence_reason_ == "angle"
    assert corrected.n_iter_ > legacy.n_iter_
    assert error(legacy) > 0.01
    assert error(corrected) < 0.001
    # The reference's endpoint itself must be stable under a larger cap.
    longer = VBPCA(1, **{**reference_opts, "maxiters": 400}).fit(x)
    assert_allclose(longer.components_ @ longer.scores_, truth, rtol=1e-10, atol=1e-10)
    assert_allclose(corrected.noise_variance_, reference.noise_variance_, rtol=1e-5)
    assert np.isfinite(corrected.cost_)
