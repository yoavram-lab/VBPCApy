# ruff: noqa: INP001, I001
# pyright: reportUnknownMemberType=false
"""Replay legacy cultural VBPCA analysis using public sklearn-style APIs.

This script intentionally mirrors the high-level flow of tools/VPCACulturalCheck.m
while using the public Python package interfaces:
- vbpca_py.VBPCA
- vbpca_py.select_n_components / SelectionConfig
- vbpca_py.preprocessing.MissingAwareOneHotEncoder

Notes:
-----
- Genetics dataset is omitted (not present in tools/datasets).
- Outputs are written locally for side-by-side reassurance checks.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from vbpca_py import SelectionConfig, VBPCA, select_n_components
from vbpca_py.preprocessing import MissingAwareOneHotEncoder

logger = logging.getLogger(__name__)


def _as_float(value: object) -> float:
    """Safely coerce a metric-like value to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (np.floating, np.integer)):
        return float(value.item())
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return float(value)
        except ValueError:
            return float("nan")
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return float("nan")


@dataclass(frozen=True)
class DatasetPolicy:
    """Dataset-specific preprocessing and scaling policy."""

    name: str
    file_name: str
    apply_legacy_oh: bool
    scaling: str  # center | standard | none


DATASET_POLICIES: dict[str, DatasetPolicy] = {
    "kinship": DatasetPolicy(
        name="kinship",
        file_name="KinshipOrgDataRaw.csv",
        apply_legacy_oh=True,
        scaling="center",
    ),
    "subsistence": DatasetPolicy(
        name="subsistence",
        file_name="SubsistenceDataRaw.csv",
        apply_legacy_oh=True,
        scaling="center",
    ),
    "religion": DatasetPolicy(
        name="religion",
        file_name="RelDataRaw.csv",
        apply_legacy_oh=True,
        scaling="center",
    ),
    "isolation": DatasetPolicy(
        name="isolation",
        file_name="IsoDataRaw.csv",
        apply_legacy_oh=True,
        scaling="center",
    ),
    "binford": DatasetPolicy(
        name="binford",
        file_name="Binford_all.csv",
        apply_legacy_oh=False,
        scaling="none",
    ),
    "seshat": DatasetPolicy(
        name="seshat",
        file_name="Seshat.csv",
        apply_legacy_oh=False,
        scaling="standard",
    ),
    "birds": DatasetPolicy(
        name="birds",
        file_name="data_birds_OH.csv",
        apply_legacy_oh=False,
        scaling="center",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default="kinship,subsistence,religion,isolation,binford,seshat,birds",
        help="Comma-separated dataset keys.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("tools/datasets"),
        help="Directory containing CSV datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/cultural_replay"),
        help="Output directory for per-dataset artifacts.",
    )
    parser.add_argument("--maxiters", type=int, default=80)
    parser.add_argument(
        "--max-k-cap",
        type=int,
        default=0,
        help=(
            "Optional extra cap on k; 0 or less uses min(n_features, n_samples)."
        ),
    )
    parser.add_argument(
        "--compat-mode",
        type=str,
        default="modern",
        choices=["modern", "strict_legacy"],
    )
    parser.add_argument(
        "--runtime-tuning",
        type=str,
        default="safe",
        choices=["off", "safe", "aggressive"],
        help="Runtime autotuning policy (safe enables RMS thread autotune).",
    )
    parser.add_argument(
        "--num-cpu",
        type=int,
        default=None,
        help="Threads for core kernels; overrides env defaults when > 0.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rms",
        choices=["rms", "prms", "cost"],
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=-1,
        help="Selection patience; <=0 disables.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=-1,
        help="Selection max trials; <=0 disables.",
    )
    parser.add_argument(
        "--drop-missing-threshold",
        type=float,
        default=0.5,
        help="Drop feature columns with missing fraction > threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for deterministic initialization path.",
    )
    parser.add_argument(
        "--selection-verbose",
        type=int,
        default=1,
        help=(
            "Verbosity passed to select_n_components "
            "(k-sweep progress logs when > 0)."
        ),
    )
    parser.add_argument(
        "--selection-quiet",
        action="store_true",
        help="Silence selection logs (overrides --selection-verbose).",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help=(
            "Apply tuned defaults for faster sweeps (runtime_tuning='safe', "
            "smaller k window, patience/max_trials tightening)."
        ),
    )
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def _split_csv_list(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _load_csv_as_float(path: Path) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    return frame.to_numpy(dtype=float), frame.columns.astype(str).tolist()


def _drop_missing_heavy_columns(
    x: np.ndarray,
    feature_names: list[str],
    threshold: float,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    missing_frac = np.mean(~np.isfinite(x), axis=0)
    keep_mask = missing_frac <= threshold
    kept = x[:, keep_mask]
    kept_names = [
        name for name, keep in zip(feature_names, keep_mask, strict=True) if keep
    ]
    return kept, kept_names, keep_mask


def _component_grid(
    n_features: int,
    n_samples: int,
    max_k_cap: int,
    *,
    fast_mode: bool,
) -> list[int]:
    natural_cap = max(1, min(n_features, n_samples))
    k_cap = min(natural_cap, max_k_cap) if max_k_cap > 0 else natural_cap
    if not fast_mode:
        return list(range(2, k_cap + 1)) if k_cap >= 2 else [1]

    span = max(n_features, n_samples)
    if span >= 800:
        tuned_cap = min(k_cap, 25)
    elif span >= 400:
        tuned_cap = min(k_cap, 35)
    elif span >= 200:
        tuned_cap = min(k_cap, 45)
    else:
        tuned_cap = k_cap

    return list(range(2, tuned_cap + 1)) if tuned_cap >= 2 else [1]


def _scale_by_policy(
    x: np.ndarray,
    scaling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    stds[~np.isfinite(stds)] = 1.0
    stds[stds == 0.0] = 1.0

    if scaling == "center":
        z = x - means
        return z, means, np.ones_like(stds)
    if scaling == "standard":
        z = (x - means) / stds
        return z, means, stds
    if scaling == "none":
        return x.copy(), np.zeros_like(means), np.ones_like(stds)

    msg = f"Unknown scaling policy: {scaling!r}"
    raise ValueError(msg)




def _run_dataset(  # noqa: PLR0913
    policy: DatasetPolicy,
    data_dir: Path,
    out_dir: Path,
    maxiters: int,
    max_k_cap: int,
    metric: str,
    patience: int | None,
    max_trials: int | None,
    drop_missing_threshold: float,
    seed: int,
    compat_mode: str,
    runtime_tuning: str,
    num_cpu: int | None,
    fast_mode: bool,  # noqa: FBT001
    selection_verbose: int,
    verbose: int,
) -> dict[str, Any]:
    data_path = data_dir / policy.file_name
    if not data_path.exists():
        msg = f"Dataset not found: {data_path}"
        raise FileNotFoundError(msg)

    x_raw, raw_columns = _load_csv_as_float(data_path)

    if policy.apply_legacy_oh:
        encoder = MissingAwareOneHotEncoder(handle_unknown="ignore", mean_center=False)
        obs_mask = np.isfinite(x_raw)
        x_model_input = encoder.fit_transform(x_raw, mask=obs_mask)
        model_columns = encoder.feature_names_out_
    else:
        x_model_input = x_raw
        model_columns = raw_columns

    x_filtered, _filtered_columns, keep_mask = _drop_missing_heavy_columns(
        x_model_input,
        model_columns,
        threshold=drop_missing_threshold,
    )
    del keep_mask

    z, _means, _scales = _scale_by_policy(x_filtered, policy.scaling)

    x_f_by_n = z.T
    observed_mask = np.isfinite(x_f_by_n)

    n_features, n_samples = x_f_by_n.shape
    components = _component_grid(
        n_features,
        n_samples,
        max_k_cap,
        fast_mode=fast_mode,
    )

    np.random.seed(seed)  # noqa: NPY002
    cfg = SelectionConfig(
        metric=metric,  # type: ignore[arg-type]
        stop_on_metric_reversal=True,
        patience=patience,
        max_trials=max_trials,
        compute_explained_variance=True,
        return_best_model=True,
    )

    select_kwargs: dict[str, object] = {
        "maxiters": maxiters,
        "algorithm": "vb",
        "uniquesv": False,
        "rmsstop": [80, np.finfo(float).eps, np.finfo(float).eps],
        "cfstop": [80, 0, 0],
        "minangle": 0,
        "compat_mode": compat_mode,
        "rotate2pca": True,
        "runtime_tuning": runtime_tuning,
        "selection_verbose": int(selection_verbose),
        "verbose": 0,
    }
    if num_cpu is not None:
        select_kwargs["num_cpu"] = int(num_cpu)

    best_k, best_metrics, trace, best_model = select_n_components(
        x_f_by_n,
        mask=observed_mask,
        components=components,
        config=cfg,
        **select_kwargs,
    )

    model = best_model
    if model is None:
        model_kwargs: dict[str, object] = {
            "maxiters": maxiters,
            "algorithm": "vb",
            "uniquesv": False,
            "rmsstop": [80, np.finfo(float).eps, np.finfo(float).eps],
            "cfstop": [80, 0, 0],
            "minangle": 0,
            "compat_mode": compat_mode,
            "rotate2pca": True,
            "runtime_tuning": runtime_tuning,
            "verbose": verbose,
        }
        if num_cpu is not None:
            model_kwargs["num_cpu"] = int(num_cpu)

        model = VBPCA(
            n_components=int(best_k),
            **model_kwargs,
        )
        model.fit(x_f_by_n, mask=observed_mask)

    evr = (
        np.asarray(model.explained_variance_ratio_, dtype=float)
        if model.explained_variance_ratio_ is not None
        else np.full(int(best_k), np.nan, dtype=float)
    )

    dataset_out = out_dir / policy.name
    dataset_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace).to_csv(dataset_out / "selection_trace.csv", index=False)

    summary: dict[str, Any] = {
        "dataset": policy.name,
        "file": policy.file_name,
        "n_samples": int(x_filtered.shape[0]),
        "n_features": int(x_filtered.shape[1]),
        "selected_k": int(best_k),
        "best_metric": metric,
        "best_metrics": {
            "rms": _as_float(best_metrics.get("rms", np.nan)),
            "prms": _as_float(best_metrics.get("prms", np.nan)),
            "cost": _as_float(best_metrics.get("cost", np.nan)),
        },
        "explained_variance_ratio": evr.tolist(),
        "total_variance_explained": float(np.nansum(evr)),
        "noise_variance": _as_float(getattr(model, "noise_variance_", np.nan)),
    }
    (dataset_out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Run local cultural replay across selected datasets."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()

    if args.maxiters < 1:
        msg = "--maxiters must be >= 1"
        raise ValueError(msg)
    if args.max_k_cap < 0:
        msg = "--max-k-cap must be >= 0 (0 uses min(n_features, n_samples))"
        raise ValueError(msg)
    if not (0.0 <= args.drop_missing_threshold <= 1.0):
        msg = "--drop-missing-threshold must be in [0, 1]"
        raise ValueError(msg)

    selected = _split_csv_list(args.datasets)
    unknown = [name for name in selected if name not in DATASET_POLICIES]
    if unknown:
        known = ", ".join(sorted(DATASET_POLICIES))
        msg = f"Unknown dataset keys {unknown}; known keys: {known}"
        raise ValueError(msg)

    fast_mode = bool(args.fast_mode)
    runtime_tuning = str(args.runtime_tuning)
    patience = args.patience if args.patience > 0 else (2 if fast_mode else None)
    max_trials = args.max_trials if args.max_trials > 0 else (5 if fast_mode else None)
    if fast_mode and runtime_tuning == "off":
        runtime_tuning = "safe"
    num_cpu = args.num_cpu if args.num_cpu is not None and args.num_cpu > 0 else None
    selection_verbose = 0 if bool(args.selection_quiet) else int(args.selection_verbose)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for idx, name in enumerate(selected):
        policy = DATASET_POLICIES[name]
        run_seed = int(args.seed + idx)
        summary = _run_dataset(
            policy=policy,
            data_dir=args.data_dir,
            out_dir=out_dir,
            maxiters=int(args.maxiters),
            max_k_cap=int(args.max_k_cap),
            metric=str(args.metric),
            patience=patience,
            max_trials=max_trials,
            drop_missing_threshold=float(args.drop_missing_threshold),
            seed=run_seed,
            compat_mode=str(args.compat_mode),
            runtime_tuning=runtime_tuning,
            num_cpu=num_cpu,
            fast_mode=fast_mode,
            selection_verbose=selection_verbose,
            verbose=int(args.verbose),
        )
        summaries.append(summary)
        logger.info(
            "[%s] k=%s var_exp=%.3f rms=%.4f",
            policy.name,
            summary["selected_k"],
            summary["total_variance_explained"],
            summary["best_metrics"]["rms"],
        )

    summary_df = pd.DataFrame(summaries).sort_values("dataset").reset_index(drop=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    logger.info("Wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
