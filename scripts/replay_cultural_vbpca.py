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
- Genetics dataset uses HGDP_Edge_2017_snp.npz (SNP-only; 0/1/2 genotypes).
- Outputs are written locally for side-by-side reassurance checks.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scipy.sparse as sp

from vbpca_py import SelectionConfig, VBPCA, select_n_components
from vbpca_py.preprocessing import MissingAwareOneHotEncoder, MissingAwareStandardScaler

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
    "genetics": DatasetPolicy(
        name="genetics",
        file_name="HGDP_Edge_2017_snp.npz",
        apply_legacy_oh=False,
        scaling="none",
    ),
}


def _load_npz_pair(path: Path) -> tuple[sp.csr_matrix, sp.csr_matrix | None]:
    data = sp.load_npz(path)
    mask_path = path.with_name(f"{path.stem}_mask.npz")
    mask = sp.load_npz(mask_path) if mask_path.exists() else None
    return sp.csr_matrix(data), sp.csr_matrix(mask) if mask is not None else None


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
        default=75,
        help=(
            "Optional extra cap on k (default 75 to mirror legacy MATLAB runs); "
            "0 or less uses min(n_features, n_samples)."
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
            "Verbosity passed to select_n_components (k-sweep progress logs when > 0)."
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


def _run_dataset(  # noqa: PLR0912, PLR0913, PLR0915, C901
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

    is_npz = data_path.suffix.lower() == ".npz"
    dense_time = 0.0
    sparse_time = 0.0
    use_dense = True
    x_f_by_n: np.ndarray | sp.csr_matrix
    observed_mask: np.ndarray | sp.csr_matrix | None

    def _run_probe(
        x_probe: np.ndarray | sp.csr_matrix,
        mask_probe: np.ndarray | sp.csr_matrix | None,
        *,
        components: list[int],
    ) -> float:
        t0 = time.perf_counter()
        probe_cfg = SelectionConfig(
            metric=metric,  # type: ignore[arg-type]
            stop_on_metric_reversal=True,
            patience=None,
            max_trials=None,
            compute_explained_variance=False,
            return_best_model=False,
        )
        select_kwargs_probe: dict[str, object] = {
            "maxiters": 10,
            "algorithm": "vb",
            "uniquesv": False,
            "rmsstop": [80, np.finfo(float).eps, np.finfo(float).eps],
            "cfstop": [80, 0, 0],
            "minangle": 0,
            "compat_mode": compat_mode,
            "rotate2pca": True,
            "runtime_tuning": runtime_tuning,
            "selection_verbose": 0,
            "verbose": 0,
        }
        if num_cpu is not None:
            select_kwargs_probe["num_cpu"] = int(num_cpu)
        select_n_components(
            x_probe,
            mask=mask_probe,
            components=components,
            config=probe_cfg,
            **select_kwargs_probe,
        )  # type: ignore[arg-type]
        return time.perf_counter() - t0

    if is_npz:
        data_csr, mask_csr = _load_npz_pair(data_path)

        data_csr = sp.csr_matrix(data_csr)
        if mask_csr is not None:
            mask_csr = sp.csr_matrix(mask_csr)
        else:
            mask_csr = sp.csr_matrix(data_csr.copy())
            mask_csr.data = np.ones_like(mask_csr.data, dtype=np.float32)

        # Lightweight sanity: values should lie in [0, 2] for SNP dosages
        finite_data = data_csr.data[np.isfinite(data_csr.data)]
        if finite_data.size:
            if (
                finite_data.min(initial=0.0) < -1e-6
                or finite_data.max(initial=0.0) > 2.0 + 1e-6
            ):
                msg = "Genetics NPZ values outside expected 0/1/2 range"
                raise ValueError(msg)

        scaler = MissingAwareStandardScaler().fit(data_csr)

        # Dense path: use masked means/vars from scaler; zero-fill missing
        x_dense_centered = scaler.transform(data_csr)
        if sp.issparse(x_dense_centered):
            x_dense_centered = x_dense_centered.toarray()
        else:
            x_dense_centered = np.asarray(x_dense_centered)
        dense_mask_raw = mask_csr.toarray().astype(bool, copy=False)
        x_dense_centered[~dense_mask_raw] = 0.0
        dense_time = _run_probe(
            x_dense_centered.T,
            dense_mask_raw.T,
            components=[5],
        )

        # Sparse path: keep centered CSR with matching mask
        x_sparse_centered = scaler.transform(data_csr)
        if not sp.issparse(x_sparse_centered):
            x_sparse_centered = sp.csr_matrix(np.asarray(x_sparse_centered))

        sparse_mask_aligned = sp.csr_matrix(mask_csr)

        sparse_time = _run_probe(
            sp.csr_matrix(x_sparse_centered.T),
            sp.csr_matrix(sparse_mask_aligned.T),
            components=[5],
        )

        use_dense = dense_time <= sparse_time
        logger.info(
            "[genetics] probe dense=%.2fs sparse=%.2fs -> using %s path",
            dense_time,
            sparse_time,
            "dense" if use_dense else "sparse",
        )

        if use_dense:
            x_f_by_n = x_dense_centered.T
            observed_mask = dense_mask_raw.T
            del x_sparse_centered, sparse_mask_aligned
        else:
            x_f_by_n = sp.csr_matrix(x_sparse_centered.T)
            observed_mask = sp.csr_matrix(sparse_mask_aligned.T)
            del x_dense_centered, dense_mask_raw
    else:
        x_raw, raw_columns = _load_csv_as_float(data_path)

        if policy.apply_legacy_oh:
            encoder = MissingAwareOneHotEncoder(
                handle_unknown="ignore",
                mean_center=False,
            )
            obs_mask = np.isfinite(x_raw)
            x_model_input = encoder.fit_transform(x_raw, mask=obs_mask)
            if sp.issparse(x_model_input):
                x_model_input = x_model_input.toarray()
            x_model_input = np.asarray(x_model_input)
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

    n_features_int = int(x_f_by_n.shape[0])
    n_samples_int = int(x_f_by_n.shape[1])
    components = _component_grid(
        n_features_int,
        n_samples_int,
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
    )  # type: ignore[arg-type]

    model = best_model

    # Ensure we have full diagnostics (reconstruction and marginal variance) for exports.
    expected_shape = (n_features_int, n_samples_int)

    def _shape_ok(arr: np.ndarray | None) -> bool:
        return arr is not None and arr.shape == expected_shape

    need_refit = (
        model is None
        or not _shape_ok(getattr(model, "reconstruction_", None))
        or not _shape_ok(getattr(model, "variance_", None))
    )

    if need_refit:
        model = VBPCA(
            n_components=int(best_k),
            maxiters=maxiters,
            algorithm="vb",
            uniquesv=False,
            rmsstop=[80, np.finfo(float).eps, np.finfo(float).eps],
            cfstop=[80, 0, 0],
            minangle=0,
            compat_mode=compat_mode,
            rotate2pca=True,
            runtime_tuning=runtime_tuning,
            verbose=verbose,
            num_cpu=int(num_cpu) if num_cpu is not None else None,
            return_diagnostics=True,
        )
        model.fit(x_f_by_n, mask=observed_mask)  # type: ignore[arg-type]

    evr = (
        np.asarray(model.explained_variance_ratio_, dtype=float)
        if model.explained_variance_ratio_ is not None
        else np.full(int(best_k), np.nan, dtype=float)
    )
    dataset_out = out_dir / policy.name
    dataset_out.mkdir(parents=True, exist_ok=True)
    trace_frame = pd.DataFrame(trace)
    if is_npz:
        np.savez_compressed(
            dataset_out / "selection_trace.npz",
            trace=trace_frame.to_records(index=False),
        )
    else:
        trace_frame.to_csv(dataset_out / "selection_trace.csv", index=False)

    # Persist reconstruction (mean per entry) and marginal variance for downstream testing.
    if model.reconstruction_ is not None and model.variance_ is not None:
        np.savez_compressed(
            dataset_out / "posterior_moments.npz",
            reconstruction=model.reconstruction_,
            variance=model.variance_,
            mean=model.mean_,
            selected_k=int(best_k),
        )

    summary: dict[str, Any] = {
        "dataset": policy.name,
        "file": policy.file_name,
        "n_samples": int(n_samples_int),
        "n_features": int(n_features_int),
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
    if is_npz:
        summary["probe"] = {
            "dense_seconds": float(dense_time),
            "sparse_seconds": float(sparse_time),
            "path": "dense" if use_dense else "sparse",
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
