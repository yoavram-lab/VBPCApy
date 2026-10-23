"""Snapshot and compare bounded fixed-policy fits in two installed environments.

See docs/releases/0.4.6.md. The caller supplies the registered fitting-policy
module, uses the same numerical dependencies/builds, and sets numerical threads
to one. This checks fixed matrices, not equivalence of preprocessing methods.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy

import vbpca_py
from vbpca_py import AutoEncoder, CVConfig, cross_validate_components

# A deliberate version boundary: compare historical feature-first fits in both
# environments, including installations that predate the explicit legacy API.
try:
    from vbpca_py.legacy import VBPCA
except ModuleNotFoundError as error:
    if error.name != "vbpca_py.legacy":
        raise
    from vbpca_py import VBPCA

if TYPE_CHECKING:
    from types import ModuleType

CASES = (
    (12, 24, 0.0, 1),
    (12, 24, 0.3, 2),
    (24, 12, 0.0, 2),
    (24, 12, 0.3, 1),
    (16, 16, 0.0, 1),
    (16, 16, 0.3, 2),
)
FIELDS = (
    "components_",
    "scores_",
    "mean_",
    "variance_",
    "predictive_variance_",
    "prior_variances_",
    "component_relevance_",
    "reconstruction_",
    "rms_",
    "noise_variance_",
    "n_iter_",
    "converged_",
)


def _load_policy(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("registered_fit_policy", path)
    if spec is None or spec.loader is None:
        msg = f"Cannot load registered policy module {path}"
        raise ValueError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fit_snapshots(policy: ModuleType) -> tuple[dict[str, np.ndarray], list[dict]]:
    arrays = {}
    records = []
    for index, (p, n, missing, rank) in enumerate(CASES):
        seed = 20261266 + index
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(p, rank)) @ rng.normal(size=(rank, n)) + rng.normal(
            size=(p, n)
        )
        x[rng.random(x.shape) < missing] = np.nan
        options = policy.resolve_fitting_options(
            vbpca_py.recommend_config(n, p),
            policy.FROZEN_FITTING_POLICY,
            overrides={"runtime_tuning": "off"},
        )
        options.update(random_state=seed, verbose=0, xprobe_fraction=0.0)
        model = VBPCA(rank, **options).fit(x)
        for field in FIELDS:
            value = getattr(model, field)
            if value is not None:
                arrays[f"case{index}_{field}"] = np.asarray(value)
        records.append({
            "p": p,
            "n": n,
            "missingness": missing,
            "rank": rank,
            "seed": seed,
            "options": options,
            "reason": model.convergence_reason_,
        })
    return arrays, records


def _legacy_cv_snapshots() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261300)
    raw = rng.integers(0, 3, size=(36, 4)).astype(float)
    raw[rng.random(raw.shape) < 0.1] = np.nan
    encoder = AutoEncoder(column_types=["categorical"] * 4, binary="both")
    z = encoder.fit_transform(raw).T
    arrays = {}
    for metric in ("prms", "brier"):
        best, results = cross_validate_components(
            z,
            components=[0, 1, 2],
            config=CVConfig(
                n_splits=2,
                feature_groups=encoder.feature_groups_,
                metric=metric,
                seed=20261301,
            ),
            maxiters=30,
            niter_broadprior=0,
            runtime_tuning="off",
            verbose=0,
        )
        arrays[f"cv_{metric}_best"] = np.asarray(best)
        for name in ("prms", "brier", "log_score", "accuracy", "rms"):
            for prefix in ("mean", "se"):
                arrays[f"cv_{metric}_{prefix}_{name}"] = np.asarray([
                    row[f"{prefix}_{name}"] for row in results
                ])
    return arrays


def snapshot(policy_path: Path, output: Path) -> None:
    """Export fixed-policy fit/CV arrays and their reproducibility metadata.

    Raises:
        ValueError: If output lacks an .npz suffix or the policy cannot load.
    """
    if output.suffix != ".npz":
        msg = "snapshot output must end in .npz"
        raise ValueError(msg)
    policy = _load_policy(policy_path)
    arrays, cases = _fit_snapshots(policy)
    arrays.update(_legacy_cv_snapshots())
    metadata = {
        "schema": 1,
        "version": vbpca_py.__version__,
        "source": vbpca_py.__file__,
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "policy": policy.FROZEN_FITTING_POLICY,
        "environment": {
            "python": list(sys.version_info[:2]),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "threads": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "cases": cases,
        "legacy_cv": {
            "data_seed": 20261300,
            "fit_seed": 20261301,
            "splits": 2,
            "components": [0, 1, 2],
            "maxiters": 30,
            "warmup": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{metadata['version']} from {metadata['source']}: {len(arrays)} arrays")


def compare_snapshots(baseline: Path, candidate: Path) -> dict[str, object]:
    """Require matching protocols and byte-identical dtype/shape/value arrays.

    Returns:
        A bounded compatibility report; passed is false for any array mismatch.

    Raises:
        ValueError: If metadata, environment or the exported field sets differ.
    """
    before = json.loads(baseline.with_suffix(".json").read_text())
    after = json.loads(candidate.with_suffix(".json").read_text())
    for field in (
        "schema",
        "policy_sha256",
        "policy",
        "environment",
        "cases",
        "legacy_cv",
    ):
        if field not in before or before[field] != after.get(field):
            msg = f"snapshot protocol differs in {field}"
            raise ValueError(msg)
    differences = []
    with (
        np.load(baseline, allow_pickle=False) as old,
        np.load(candidate, allow_pickle=False) as new,
    ):
        if not old.files or set(old.files) != set(new.files):
            msg = "snapshot field sets differ or are empty"
            raise ValueError(msg)
        for field in old.files:
            left, right = old[field], new[field]
            if (
                left.dtype != right.dtype
                or left.shape != right.shape
                or left.tobytes() != right.tobytes()
            ):
                differences.append(field)
        count = len(old.files)
    return {
        "baseline_version": before["version"],
        "candidate_version": after["version"],
        "arrays_checked": count,
        "byte_identical_arrays": count - len(differences),
        "differences": differences,
        "passed": not differences,
        "scope": "Bounded fixed-input fits and legacy full-one-hot CV; not exhaustive identity or preprocessing-method equivalence.",
    }


def main() -> None:
    """Run snapshot or compare; comparisons exit nonzero on differences.

    Raises:
        SystemExit: If any compared array differs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    capture = subcommands.add_parser("snapshot")
    capture.add_argument("--policy-module", required=True, type=Path)
    capture.add_argument("--output", required=True, type=Path)
    compare = subcommands.add_parser("compare")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.policy_module, args.output)
    else:
        report = compare_snapshots(args.baseline, args.candidate)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
