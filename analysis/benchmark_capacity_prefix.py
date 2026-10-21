"""Bounded first-minimum execution qualification; no default adoption gate.

Registered before execution: seeds 248/249/250; null, strong rank-one and weak
rank-two Gaussian fixtures, 40 x 60, 10% missing plus independent 15% holdout.
Three entrywise folds, capacities 0..6, fixed 80 VB iterations, one native/BLAS
thread, no stopping or runtime tuning. Compare full first-minimum, sequential
first-minimum and full one-SE. Report every risk curve and refit diagnostics.
These small Gaussian predictive checks do not calibrate downstream rank tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from vbpca_py import VBPCA, CVConfig, cross_validate_components, model_selection

OFF = dict.fromkeys(["angle", "rms", "cost", "early_stopping", "composite"], False)
FIT = {
    "maxiters": 80,
    "runtime_tuning": "off",
    "num_cpu": 1,
    "random_state": 248,
    "convergence_criteria": OFF,
    "niter_broadprior": 0,
    "verbose": 0,
}


@contextmanager
def _count_fits():
    original = model_selection._fit_candidate
    counts = {"positive_rank": 0, "rank_zero": 0}

    def fit(k, *args, **kwargs):
        counts["positive_rank" if k else "rank_zero"] += 1
        return original(k, *args, **kwargs)

    model_selection._fit_candidate = fit
    try:
        yield counts
    finally:
        model_selection._fit_candidate = original


def _predictive_check(x, train, hold, k):
    if k:
        model = VBPCA(k, **FIT).fit(x, mask=train)
        mean, variance = model.reconstruction_, model.predictive_variance_
    else:
        count = train.sum(axis=1, keepdims=True)
        mu = np.where(train, x, 0).sum(axis=1, keepdims=True) / count
        mean = np.broadcast_to(mu, x.shape)
        noise = np.mean((x[train] - mean[train]) ** 2)
        variance = np.broadcast_to(noise * (1 + 1 / count), x.shape)
    error = x[hold] - mean[hold]
    std = np.sqrt(np.maximum(variance[hold], 0))
    return {
        "holdout_rmse": float(np.sqrt(np.mean(error**2))),
        "predictive_95_coverage": float(np.mean(np.abs(error) <= 1.96 * std)),
        "downstream_significance_rank": None,
        "rank_calibration_scope": "Not evaluated: requires a separately registered significance procedure.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = []
    for seed in [248, 249, 250]:
        for name, rank, strength in [
            ("null", 0, 0.0),
            ("strong", 1, 2.0),
            ("weak", 2, 0.3),
        ]:
            rng = np.random.default_rng(seed)
            x = strength * rng.normal(size=(40, rank)) @ rng.normal(
                size=(rank, 60)
            ) + rng.normal(size=(40, 60))
            observed = rng.random(x.shape) > 0.1
            hold = observed & (rng.random(x.shape) < 0.15)
            train = observed & ~hold
            results = {}
            modes = [
                ("full_first_minimum", "first_minimum", False),
                ("prefix_first_minimum", "first_minimum", True),
                ("full_one_se", "one_se", False),
            ]
            if seed % 2:
                modes.reverse()
            for label, rule, early in modes:
                wall, cpu = perf_counter(), process_time()
                with _count_fits() as counts:
                    k, curve = cross_validate_components(
                        x,
                        mask=train,
                        components=range(7),
                        config=CVConfig(
                            n_splits=3, seed=seed, selection_rule=rule, early_stop=early
                        ),
                        **FIT,
                    )
                results[label] = {
                    "capacity": k,
                    "curve": curve,
                    "fit_counts": dict(counts),
                    "wall_seconds": perf_counter() - wall,
                    "cpu_seconds": process_time() - cpu,
                    "evaluated_candidates": [row["k"] for row in curve],
                    "selected_refit": _predictive_check(x, train, hold, k),
                }
            full, prefix = (
                results["full_first_minimum"],
                results["prefix_first_minimum"],
            )
            if full["capacity"] != prefix["capacity"]:
                raise AssertionError(f"Prefix target differs: {seed}/{name}")
            for a, b in zip(full["curve"], prefix["curve"], strict=False):
                np.testing.assert_allclose(
                    a["mean_prms"], b["mean_prms"], rtol=1e-8, atol=1e-10
                )
            records.append({
                "seed": seed,
                "fixture": name,
                "true_generative_rank": rank,
                "data_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                "train_mask_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
                "results": results,
            })
    payload = {
        "design": __doc__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "versions": {k: version(k) for k in ["vbpca-py", "numpy", "scipy"]},
        "default_adoption": False,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
