"""Resolve, fit, and score one modern-defaults study trial."""

from __future__ import annotations

import copy
import time
import warnings
from typing import Any

import numpy as np

from vbpca_py import VBPCA, SelectionConfig, recommend_config, select_n_components

from ._modern_defaults_data import SimulationData, generate_simulation
from ._modern_defaults_metrics import (
    mean_only_fit,
    predictive_scores,
    projection_distance,
    rms_tail_diagnostics,
)
from ._modern_defaults_spec import CRITERION_POLICIES


def resolve_candidate(
    candidate: dict[str, Any],
    regime: dict[str, Any],
    *,
    init_seed: int,
    num_cpu: int,
) -> dict[str, Any]:
    """Resolve one relative candidate into estimator-ready options."""
    base = str(candidate["base"])
    if base == "recommended":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            options = copy.deepcopy(
                recommend_config(n=int(regime["n"]), p=int(regime["p"]))
            )
    elif base == "estimator_default":
        options = {"maxiters": 1000, "niter_broadprior": 100}
    else:
        msg = f"unknown candidate base {base!r}"
        raise ValueError(msg)

    for key in ("hp_va", "hp_vb", "hp_v"):
        scale_key = f"{key}_scale"
        if scale_key in candidate:
            options[key] = float(
                np.clip(
                    float(options.get(key, 0.001)) * float(candidate[scale_key]),
                    1e-5,
                    1.0,
                )
            )
    if "va_init_scale" in candidate:
        options["va_init"] = float(
            np.clip(
                float(options.get("va_init", 1000.0))
                * float(candidate["va_init_scale"]),
                10.0,
                10_000.0,
            )
        )
    if "maxiters_scale" in candidate:
        options["maxiters"] = max(
            1,
            int(
                round(
                    float(options.get("maxiters", 1000)) * candidate["maxiters_scale"]
                )
            ),
        )

    direct_keys = (
        "xprobe_fraction",
        "niter_broadprior",
        "patience",
        "minangle",
        "cfstop_rel",
        "compat_mode",
        "bias_update_order",
    )
    for key in direct_keys:
        if key in candidate:
            options[key] = copy.deepcopy(candidate[key])

    if "rmsstop_window" in candidate:
        options["rmsstop"] = [
            int(candidate["rmsstop_window"]),
            float(candidate["rmsstop_atol"]),
            float(candidate["rmsstop_rtol"]),
        ]
    if "criterion_policy" in candidate:
        options["convergence_criteria"] = copy.deepcopy(
            CRITERION_POLICIES[str(candidate["criterion_policy"])]
        )

    options.update(
        {
            "random_state": init_seed,
            "num_cpu": num_cpu,
            "runtime_tuning": "off",
            "runtime_report": True,
            "verbose": 0,
        }
    )
    return options


def _fit_selected_model(
    data: SimulationData,
    selected_rank: int,
    options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if selected_rank == 0:
        reconstruction, predictive_variance = mean_only_fit(data)
        loadings = np.empty((data.x_true.shape[0], 0), dtype=float)
        diagnostics = {
            "n_iter": 0,
            "convergence_reason": "closed_form_mean_only",
            "selected_budget_hit": False,
            "rms_increase_fraction": float("nan"),
            "rms_two_cycle_amplitude": float("nan"),
        }
        return reconstruction, predictive_variance, loadings, diagnostics

    refit_options = copy.deepcopy(options)
    maxiters = int(refit_options.pop("maxiters", 1000))
    refit_options["return_diagnostics"] = True
    model = VBPCA(selected_rank, maxiters=maxiters, **refit_options)
    model.fit(data.x_true, mask=data.train_mask)
    if model.reconstruction_ is None or model.predictive_variance_ is None:
        msg = "diagnostic refit did not return reconstruction and predictive variance"
        raise RuntimeError(msg)
    increase_fraction, two_cycle = rms_tail_diagnostics(
        list((model.learning_curve_ or {}).get("rms", []))
    )
    diagnostics = {
        "n_iter": int(model.n_iter_ or 0),
        "convergence_reason": model.convergence_reason_ or "maxiters",
        "selected_budget_hit": int(model.n_iter_ or 0) >= maxiters,
        "rms_increase_fraction": increase_fraction,
        "rms_two_cycle_amplitude": two_cycle,
    }
    if model.components_ is None:
        msg = "diagnostic refit did not return loadings"
        raise RuntimeError(msg)
    return (
        model.reconstruction_,
        model.predictive_variance_,
        model.components_,
        diagnostics,
    )


def run_trial(
    manifest: dict[str, Any],
    candidate: dict[str, Any],
    regime: dict[str, Any],
    *,
    rep: int,
    num_cpu: int,
) -> dict[str, Any]:
    """Run and score one paired replicate."""
    selection = manifest["selection"]
    data = generate_simulation(
        regime,
        rep=rep,
        missing_fraction=float(selection["missing_fraction"]),
        holdout_fraction=float(selection["external_holdout_fraction"]),
    )
    options = resolve_candidate(
        candidate,
        regime,
        init_seed=data.init_seed,
        num_cpu=num_cpu,
    )
    maxiters = int(options.pop("maxiters", 1000))
    margin = int(selection["component_margin"])
    max_rank = min(int(regime["true_rank"]) + margin, min(data.x_true.shape) - 1)
    components = range(max_rank + 1)

    started = time.perf_counter()
    selected_rank, best_metrics, trace, _ = select_n_components(
        data.x_true,
        mask=data.train_mask,
        components=components,
        config=SelectionConfig(
            metric="prms",
            patience=int(selection["selection_patience"]),
            max_trials=len(components),
            compute_explained_variance=False,
            return_best_model=False,
        ),
        maxiters=maxiters,
        return_diagnostics=False,
        **options,
    )
    options["maxiters"] = maxiters
    reconstruction, predictive_variance, loadings, diagnostics = _fit_selected_model(
        data,
        int(selected_rank),
        options,
    )
    wall_seconds = time.perf_counter() - started

    trace_iterations = [int(item.get("n_iter", 0)) for item in trace]
    positive_iterations = [
        int(item.get("n_iter", 0)) for item in trace if int(item["k"]) > 0
    ]
    scores = predictive_scores(data, reconstruction, predictive_variance)
    scores.update(
        {
            "rep": rep,
            "selected_rank": int(selected_rank),
            "true_rank": int(regime["true_rank"]),
            "rank_mae": abs(int(selected_rank) - int(regime["true_rank"])),
            "exact_rank": int(selected_rank) == int(regime["true_rank"]),
            "null_selected": int(regime["true_rank"]) == 0 and int(selected_rank) > 0,
            "subspace_distance": projection_distance(
                data.true_loadings,
                loadings,
            ),
            "selection_total_iters": int(sum(trace_iterations)),
            "candidate_budget_hit_rate": float(
                np.mean(np.asarray(positive_iterations) >= maxiters)
                if positive_iterations
                else 0.0
            ),
            "selection_prms": float(best_metrics["prms"]),
            "diagnostic_refit_iters": diagnostics["n_iter"],
            "convergence_reason": diagnostics["convergence_reason"],
            "selected_budget_hit": diagnostics["selected_budget_hit"],
            "rms_increase_fraction": diagnostics["rms_increase_fraction"],
            "rms_two_cycle_amplitude": diagnostics["rms_two_cycle_amplitude"],
            "wall_seconds": wall_seconds,
        }
    )
    return scores
