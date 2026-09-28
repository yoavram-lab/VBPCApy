"""Tests for modern-defaults simulation and trial execution."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from analysis.trade_study import _modern_defaults_trial as trial_module
from analysis.trade_study._modern_defaults_data import (
    apply_missingness,
    generate_simulation,
    support_preserving_holdout,
    trial_seeds,
)
from analysis.trade_study._modern_defaults_metrics import (
    projection_distance,
    rms_tail_diagnostics,
)
from analysis.trade_study._modern_defaults_trial import resolve_candidate, run_trial


@pytest.mark.parametrize(
    "mechanism",
    ["complete", "mcar", "mar", "mnar_censored", "block"],
)
def test_missingness_preserves_row_and_column_support(mechanism: str) -> None:
    rng = np.random.default_rng(8)
    x = rng.standard_normal((80, 100))
    mask = apply_missingness(x, mechanism, 0.15, rng)

    assert mask.shape == x.shape
    assert mask.dtype == np.bool_
    assert np.all(mask.any(axis=1))
    assert np.all(mask.any(axis=0))
    expected = 0.0 if mechanism == "complete" else 0.15
    assert 1.0 - float(mask.mean()) == pytest.approx(expected, abs=0.025)
    if mechanism == "mar":
        assert np.all(mask[0])


def test_holdout_is_observed_disjoint_and_support_preserving() -> None:
    rng = np.random.default_rng(10)
    observation = rng.random((40, 30)) > 0.2
    train, holdout = support_preserving_holdout(observation, 0.1, rng)

    assert not np.any(train & holdout)
    assert np.array_equal(train | holdout, observation)
    assert np.all(train.any(axis=1))
    assert np.all(train.any(axis=0))
    assert int(holdout.sum()) == pytest.approx(0.1 * int(observation.sum()), abs=1)


def test_holdout_respects_protected_cells_and_eligible_fraction() -> None:
    observation = np.ones((20, 10), dtype=bool)
    protected = np.zeros_like(observation)
    protected[0] = True
    train, holdout = support_preserving_holdout(
        observation,
        0.1,
        np.random.default_rng(12),
        protected_mask=protected,
    )

    assert np.all(train[0])
    assert not np.any(holdout[0])
    assert int(holdout.sum()) == 19
    assert np.array_equal(train | holdout, observation)


def test_holdout_zero_fraction_and_invalid_protected_shape() -> None:
    observation = np.ones((5, 4), dtype=bool)
    train, holdout = support_preserving_holdout(
        observation, 0.0, np.random.default_rng(1)
    )

    assert np.array_equal(train, observation)
    assert not np.any(holdout)
    with pytest.raises(ValueError, match="same shape"):
        support_preserving_holdout(
            observation,
            0.1,
            np.random.default_rng(1),
            protected_mask=np.ones((4, 5), dtype=bool),
        )


def test_generated_mar_training_mask_keeps_anchor_observed() -> None:
    regime = {
        "seed": 101,
        "p": 40,
        "n": 30,
        "true_rank": 3,
        "noise_std": 0.5,
        "noise_model": "gaussian",
        "missingness": "mar",
    }
    simulation = generate_simulation(
        regime,
        rep=0,
        missing_fraction=0.15,
        holdout_fraction=0.1,
    )

    assert np.all(simulation.observation_mask[0])
    assert np.all(simulation.train_mask[0])
    assert not np.any(simulation.holdout_mask[0])


def test_simulations_are_paired_across_candidates_and_reproducible() -> None:
    regime = {
        "seed": 99,
        "p": 40,
        "n": 30,
        "true_rank": 3,
        "noise_std": 0.5,
        "noise_model": "student_t3",
        "missingness": "mnar_censored",
    }
    first = generate_simulation(
        regime,
        rep=2,
        missing_fraction=0.15,
        holdout_fraction=0.1,
    )
    second = generate_simulation(
        copy.deepcopy(regime),
        rep=2,
        missing_fraction=0.15,
        holdout_fraction=0.1,
    )

    assert np.array_equal(first.x_true, second.x_true)
    assert np.array_equal(first.train_mask, second.train_mask)
    assert np.array_equal(first.holdout_mask, second.holdout_mask)
    assert first.init_seed == second.init_seed == trial_seeds(99, 2)[1]
    assert np.isfinite(first.x_true).all()


def test_resolve_candidate_applies_relative_and_discrete_options(monkeypatch) -> None:
    monkeypatch.setattr(
        trial_module,
        "recommend_config",
        lambda *, n, p: {
            "maxiters": 100,
            "hp_va": 0.01,
            "hp_vb": 0.02,
            "hp_v": 0.04,
            "va_init": 1000.0,
        },
    )
    candidate = {
        "base": "recommended",
        "hp_va_scale": 2.0,
        "maxiters_scale": 0.75,
        "criterion_policy": "angle_cost",
        "rmsstop_window": 50,
        "rmsstop_atol": 1e-5,
        "rmsstop_rtol": 1e-4,
        "bias_update_order": "legacy",
    }
    options = resolve_candidate(
        candidate,
        {"n": 20, "p": 30},
        init_seed=123,
        num_cpu=4,
    )

    assert options["hp_va"] == pytest.approx(0.02)
    assert options["maxiters"] == 75
    assert options["rmsstop"] == [50, 1e-5, 1e-4]
    assert options["convergence_criteria"]["angle"] is True
    assert options["convergence_criteria"]["rms_plateau"] is False
    assert options["bias_update_order"] == "legacy"
    assert options["random_state"] == 123
    assert options["num_cpu"] == 4
    assert options["runtime_tuning"] == "off"


def test_projection_distance_supports_zero_and_unequal_ranks() -> None:
    empty = np.empty((5, 0))
    basis = np.eye(5)[:, :2]

    assert projection_distance(empty, empty) == pytest.approx(0.0)
    assert projection_distance(basis, basis) == pytest.approx(0.0, abs=1e-8)
    assert projection_distance(empty, basis) == pytest.approx(1.0)


def test_rms_tail_diagnostic_detects_alternation_after_detrending() -> None:
    smooth = np.linspace(2.0, 1.0, 30).tolist()
    alternating = (np.linspace(2.0, 1.0, 30) + 0.1 * (-1.0) ** np.arange(30)).tolist()

    smooth_increases, smooth_amplitude = rms_tail_diagnostics(smooth)
    alternating_increases, alternating_amplitude = rms_tail_diagnostics(alternating)

    assert smooth_increases == pytest.approx(0.0)
    assert alternating_increases > 0.0
    assert alternating_amplitude > smooth_amplitude


def test_run_trial_returns_scalar_rank_zero_aware_metrics(monkeypatch) -> None:
    criteria = dict.fromkeys(
        ("angle", "earlystop", "rms_plateau", "cost", "composite", "slowing_down"),
        False,
    )
    monkeypatch.setattr(
        trial_module,
        "recommend_config",
        lambda *, n, p: {
            "maxiters": 3,
            "niter_broadprior": 0,
            "xprobe_fraction": 0.2,
            "convergence_criteria": criteria,
        },
    )
    manifest = {
        "selection": {
            "missing_fraction": 0.15,
            "external_holdout_fraction": 0.1,
            "component_margin": 1,
            "selection_patience": 2,
        }
    }
    regime = {
        "name": "tiny",
        "seed": 42,
        "p": 15,
        "n": 20,
        "true_rank": 1,
        "noise_std": 0.5,
        "noise_model": "gaussian",
        "missingness": "mcar",
    }
    candidate = {
        "id": "tiny_candidate",
        "base": "recommended",
        "compat_mode": "modern",
        "bias_update_order": "post_factor",
    }

    scores = run_trial(manifest, candidate, regime, rep=0, num_cpu=1)

    expected = {
        "selected_rank",
        "rank_mae",
        "holdout_rmse",
        "coverage_95",
        "interval_score",
        "selection_total_iters",
        "candidate_budget_hit_rate",
        "rms_two_cycle_amplitude",
        "wall_seconds",
    }
    assert expected.issubset(scores)
    assert 0 <= scores["selected_rank"] <= 2
    assert np.isfinite(scores["holdout_rmse"])
    assert 0.0 <= scores["candidate_budget_hit_rate"] <= 1.0
    assert scores["wall_seconds"] >= 0.0
