"""Benchmark observed-cell and complement dense VB-PCA update kernels.

Run from the repository root, for example::

    python analysis/benchmark_dense_sufficient_statistics.py \
        --features 1000 --samples 1500 --components 10 --threads 24 \
        --densities 0.25 0.50 0.65 0.82 0.90 1.00 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vbpca_py import dense_update_kernels as kernels

if TYPE_CHECKING:
    from collections.abc import Callable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.65, 0.82, 0.9, 1.0],
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=210)
    parser.add_argument("--skip-covariances", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _paired_timings(
    observed: Callable[[], object],
    complement: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
) -> tuple[float, float]:
    for _ in range(warmups):
        observed()
        complement()

    timings: dict[str, list[float]] = {"observed": [], "complement": []}
    calls = {"observed": observed, "complement": complement}
    for repeat in range(repeats):
        order = ("observed", "complement")
        if repeat % 2:
            order = tuple(reversed(order))
        for mode in order:
            started = time.perf_counter()
            calls[mode]()
            timings[mode].append(time.perf_counter() - started)

    return statistics.median(timings["observed"]), statistics.median(
        timings["complement"]
    )


def _max_abs_difference(
    observed: dict[str, np.ndarray],
    complement: dict[str, np.ndarray],
    keys: tuple[str, ...],
) -> float:
    differences = [
        float(
            np.max(
                np.abs(
                    np.asarray(observed[key], dtype=float)
                    - np.asarray(complement[key], dtype=float)
                )
            )
        )
        for key in keys
        if key in observed and key in complement
    ]
    return max(differences, default=0.0)


def _benchmark_density(
    args: argparse.Namespace,
    density: float,
    rng: np.random.Generator,
    x: np.ndarray,
    loadings: np.ndarray,
    scores: np.ndarray,
    loading_covariances: np.ndarray,
    score_covariances: np.ndarray,
    prior_precision: np.ndarray,
) -> list[dict[str, float | int | str | list[int]]]:
    mask = rng.random(x.shape) < density
    mask[0, :] = True
    mask[:, 0] = True
    return_covariances = not args.skip_covariances

    def score(*, use_complement: bool) -> dict[str, np.ndarray]:
        return kernels.score_update_dense_masked_nopattern(
            x_data=x,
            mask=mask,
            loadings=loadings,
            loading_covariances=loading_covariances,
            noise_var=0.5,
            return_covariances=return_covariances,
            use_complement=use_complement,
            num_cpu=args.threads,
        )

    def loading(*, use_complement: bool) -> dict[str, np.ndarray]:
        return kernels.loadings_update_dense_masked_nopattern(
            x_data=x,
            mask=mask,
            scores=scores,
            score_covariances=score_covariances,
            prior_prec=prior_precision,
            noise_var=0.5,
            return_covariances=return_covariances,
            use_complement=use_complement,
            num_cpu=args.threads,
        )

    rows: list[dict[str, float | int | str | list[int]]] = []
    for phase, call, keys in (
        ("scores", score, ("scores", "score_covariances")),
        ("loadings", loading, ("loadings", "loading_covariances")),
    ):
        observed_call = partial(call, use_complement=False)
        complement_call = partial(call, use_complement=True)
        observed_result = observed_call()
        complement_result = complement_call()
        max_abs = _max_abs_difference(observed_result, complement_result, keys)
        np.testing.assert_allclose(
            np.asarray(observed_result[keys[0]]),
            np.asarray(complement_result[keys[0]]),
            rtol=2e-11,
            atol=2e-12,
        )
        observed_seconds, complement_seconds = _paired_timings(
            observed_call,
            complement_call,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        rows.append({
            "shape": [args.features, args.samples],
            "components": args.components,
            "threads": args.threads,
            "phase": phase,
            "target_density": density,
            "observed_fraction": float(np.mean(mask)),
            "observed_seconds": observed_seconds,
            "complement_seconds": complement_seconds,
            "complement_over_observed": complement_seconds / observed_seconds,
            "max_abs_difference": max_abs,
        })
    return rows


def main() -> None:
    args = _parser().parse_args()
    if min(args.features, args.samples, args.components, args.repeats) <= 0:
        raise ValueError("shape, components, and repeats must be positive")
    if any(not 0.0 < density <= 1.0 for density in args.densities):
        raise ValueError("densities must be in (0, 1]")

    rng = np.random.default_rng(args.seed)
    x = np.asarray(rng.normal(size=(args.features, args.samples)), order="C")
    loadings = np.asarray(
        rng.normal(scale=0.4, size=(args.features, args.components)),
        order="C",
    )
    scores = np.asarray(
        rng.normal(scale=0.5, size=(args.components, args.samples)),
        order="C",
    )
    eye = np.eye(args.components)
    loading_covariances = np.repeat((0.02 * eye)[None, :, :], args.features, axis=0)
    score_covariances = np.repeat((0.03 * eye)[None, :, :], args.samples, axis=0)
    prior_precision = 0.1 * eye

    rows: list[dict[str, float | int | str | list[int]]] = []
    for density in args.densities:
        rows.extend(
            _benchmark_density(
                args,
                density,
                rng,
                x,
                loadings,
                scores,
                loading_covariances,
                score_covariances,
                prior_precision,
            )
        )

    payload = json.dumps(rows, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
