"""Compare dense masked and explicit-zero CSR storage for genotype dosage data.

The defaults are a local smoke benchmark. Use the study dimensions with::

    python analysis/benchmark_genomics_storage.py \
        --features 5846 --samples 2504 --iterations 3 --repeats 3

For process peak memory, run each representation separately under
``/usr/bin/time -v``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from vbpca_py import VBPCA

_CONVERGENCE_OFF = {
    "angle": False,
    "earlystop": False,
    "rms_plateau": False,
    "cost": False,
    "composite": False,
    "slowing_down": False,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=600)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--missing-rate", type=float, default=0.18)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=213)
    parser.add_argument(
        "--representation",
        choices=("both", "dense", "sparse"),
        default="both",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _make_problem(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    rng = np.random.default_rng(args.seed)
    maf = rng.uniform(0.05, 0.5, size=(args.features, 1))
    dosage = rng.binomial(2, maf, size=(args.features, args.samples)).astype(float)
    observed = rng.random(dosage.shape) >= args.missing_rate

    # Avoid empty features or samples while retaining the requested density.
    observed[:, 0] = True
    observed[0, :] = True

    dense = np.array(dosage, order="F")
    dense[~observed] = np.nan

    rows, cols = np.nonzero(observed)
    sparse = sp.csr_matrix(
        (dosage[rows, cols], (rows, cols)),
        shape=dosage.shape,
    )
    if sparse.nnz != int(observed.sum()):
        msg = "CSR construction dropped explicitly stored genotype zeros"
        raise RuntimeError(msg)
    return dense, np.array(observed, order="F"), sparse


def _dense_bytes(x: np.ndarray, mask: np.ndarray) -> int:
    return int(x.nbytes + mask.nbytes)


def _sparse_bytes(x: sp.csr_matrix) -> int:
    return int(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes)


def _fit(
    x: np.ndarray | sp.csr_matrix,
    *,
    mask: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, float | int | str | None]:
    model = VBPCA(
        n_components=args.components,
        maxiters=args.iterations,
        niter_broadprior=0,
        convergence_criteria=_CONVERGENCE_OFF,
        random_state=args.seed,
        runtime_tuning="off",
        num_cpu=args.threads,
        rotate2pca=False,
        return_diagnostics=False,
    )
    started = time.perf_counter()
    model.fit(x, mask=mask)
    return {
        "seconds": time.perf_counter() - started,
        "iterations": model.n_iter_,
        "rms": model.rms_,
        "convergence_reason": model.convergence_reason_,
    }


def _median_seconds(results: list[dict[str, Any]]) -> float:
    return float(np.median([float(result["seconds"]) for result in results]))


def main() -> None:
    args = _parser().parse_args()
    dense, mask, sparse = _make_problem(args)
    report: dict[str, Any] = {
        "shape": [args.features, args.samples],
        "components": args.components,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "threads": args.threads,
        "observed_fraction": float(mask.mean()),
        "observed_zero_fraction": float(np.mean((dense == 0) & mask)),
        "input_bytes": {
            "dense_plus_bool_mask": _dense_bytes(dense, mask),
            "explicit_zero_csr": _sparse_bytes(sparse),
        },
    }

    if args.representation in {"both", "dense"}:
        report["dense"] = [
            _fit(dense, mask=mask, args=args) for _ in range(args.repeats)
        ]
    if args.representation in {"both", "sparse"}:
        report["sparse"] = [
            _fit(sparse, mask=None, args=args) for _ in range(args.repeats)
        ]

    if "dense" in report and "sparse" in report:
        dense_seconds = _median_seconds(report["dense"])
        sparse_seconds = _median_seconds(report["sparse"])
        report["comparison"] = {
            "dense_median_seconds": dense_seconds,
            "sparse_median_seconds": sparse_seconds,
            "sparse_over_dense_runtime": sparse_seconds / dense_seconds,
            "sparse_over_dense_input_bytes": (
                report["input_bytes"]["explicit_zero_csr"]
                / report["input_bytes"]["dense_plus_bool_mask"]
            ),
        }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
