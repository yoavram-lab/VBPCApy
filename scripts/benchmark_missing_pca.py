#!/usr/bin/env python3
"""Run paired missing-data PCA benchmark sweeps.

This script benchmarks methods under controlled missingness:
- mean imputation + PCA
- IterativeImputer (MICE-style) + PCA
- KNN imputation + PCA
- VBPCApy (algorithm="vb", compat_mode="modern")

Outputs one long-form CSV with one row per (setting, replicate, method).
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from vbpca_py import (
    MissingAwareStandardScaler,
    SelectionConfig,
    VBPCA,
    select_n_components,
)


@dataclass(frozen=True)
class Setting:
    dataset: str
    mechanism: str
    pattern: str
    missing_rate: float
    n_components: int
    synthetic_shape: str


@dataclass(frozen=True)
class ReplicateTask:
    setting: Setting
    replicate_id: int
    seed_data: int
    seed_mask: int
    seed_method: int


@dataclass(frozen=True)
class SelectionPlan:
    anchor_k: int
    local_components: tuple[int, ...] | None


def _parse_csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_shape(shape_raw: str) -> tuple[int, int]:
    left, right = shape_raw.lower().split("x")
    return int(left), int(right)


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == 0:
        msg = "n_jobs must be non-zero"
        raise ValueError(msg)
    if n_jobs > 0:
        return n_jobs
    cpu_count = max(1, (os_cpu_count := (os.cpu_count() or 1)))
    resolved = cpu_count + 1 + n_jobs
    return max(1, resolved)


def _load_dataset(
    dataset: str,
    rng: np.random.Generator,
    synthetic_shape: tuple[int, int],
    latent_rank: int,
    noise_scale: float,
) -> np.ndarray:
    if dataset == "synthetic":
        n_samples, n_features = synthetic_shape
        factors = rng.standard_normal((n_samples, latent_rank))
        loadings = rng.standard_normal((latent_rank, n_features))
        noise = noise_scale * rng.standard_normal((n_samples, n_features))
        return factors @ loadings + noise

    if dataset == "genomics_like":
        n_samples, n_features = synthetic_shape
        rank = max(2, latent_rank)
        factors = rng.standard_normal((n_samples, rank))
        loadings = np.zeros((rank, n_features), dtype=float)
        active_per_factor = max(1, n_features // (rank * 6))
        for idx in range(rank):
            active = rng.choice(n_features, size=active_per_factor, replace=False)
            loadings[idx, active] = rng.normal(loc=0.0, scale=1.0, size=active_per_factor)
        signal = factors @ loadings
        hetero_scale = rng.uniform(0.75, 1.5, size=(1, n_features))
        noise = noise_scale * hetero_scale * rng.standard_normal((n_samples, n_features))
        return signal + noise

    if dataset == "diabetes":
        data = load_diabetes()
        return np.asarray(data.data, dtype=float)

    if dataset == "wine":
        data = load_wine()
        return np.asarray(data.data, dtype=float)

    if dataset == "breast_cancer":
        data = load_breast_cancer()
        return np.asarray(data.data, dtype=float)

    msg = f"Unsupported dataset: {dataset!r}"
    raise ValueError(msg)


def _scale_with_package_standard_scaler(
    x_true: np.ndarray,
    x_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale with package-native missing-aware standardization.

    Fit on observed entries from ``x_obs`` only, then apply the same transform to:
    - full ``x_true`` (for holdout metric ground truth), and
    - masked ``x_obs`` (retaining NaNs on missing entries).
    """
    observed_mask = ~np.isnan(x_obs)
    scaler = MissingAwareStandardScaler()
    scaler.fit(x_obs, mask=observed_mask)

    x_true_scaled = scaler.transform(x_true, mask=np.ones_like(observed_mask, dtype=bool))
    x_obs_scaled = scaler.transform(x_obs, mask=observed_mask)
    return x_true_scaled, x_obs_scaled


def _dataset_vbpca_controls(
    dataset: str,
    *,
    base_maxiters: int,
    genomics_maxiters: int,
) -> int:
    if dataset == "genomics_like":
        return int(genomics_maxiters)
    return int(base_maxiters)


def _make_holdout_mask(
    x: np.ndarray,
    mechanism: str,
    pattern: str,
    missing_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_samples, n_features = x.shape
    n_total = n_samples * n_features
    n_holdout = max(1, int(round(missing_rate * n_total)))

    if pattern == "random":
        candidate = np.ones_like(x, dtype=bool)
    elif pattern == "block":
        block_cols = max(1, int(round(0.25 * n_features)))
        chosen_cols = rng.choice(n_features, size=block_cols, replace=False)
        candidate = np.zeros_like(x, dtype=bool)
        candidate[:, chosen_cols] = True
    else:
        msg = f"Unsupported pattern: {pattern!r}"
        raise ValueError(msg)

    probabilities = np.asarray(candidate, dtype=float)

    if mechanism == "MCAR":
        pass
    elif mechanism == "MAR":
        driver_col = int(rng.integers(0, n_features))
        centered = x[:, driver_col] - float(np.mean(x[:, driver_col]))
        if np.std(centered) > 0:
            centered = centered / float(np.std(centered))
        row_prob = 1.0 / (1.0 + np.exp(-centered))
        probabilities *= row_prob[:, None]
    else:
        msg = f"Unsupported mechanism: {mechanism!r}"
        raise ValueError(msg)

    probabilities = probabilities.ravel()
    if probabilities.sum() <= 0:
        msg = "Holdout probability mass is zero for current setting"
        raise RuntimeError(msg)
    probabilities = probabilities / probabilities.sum()

    chosen_flat = rng.choice(
        n_total,
        size=min(n_holdout, int(np.count_nonzero(probabilities))),
        replace=False,
        p=probabilities,
    )
    holdout = np.zeros(n_total, dtype=bool)
    holdout[chosen_flat] = True
    holdout_2d = holdout.reshape(x.shape)

    # Guard against degenerate masks where an entire feature or sample is hidden.
    fully_missing_cols = np.where(np.all(holdout_2d, axis=0))[0]
    for col in fully_missing_cols:
        row_keep = int(rng.integers(0, n_samples))
        holdout_2d[row_keep, col] = False

    fully_missing_rows = np.where(np.all(holdout_2d, axis=1))[0]
    for row in fully_missing_rows:
        col_keep = int(rng.integers(0, n_features))
        holdout_2d[row, col_keep] = False

    return holdout_2d


def _masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    diff = y_pred[mask] - y_true[mask]
    return float(np.sqrt(np.mean(diff**2)))


def _masked_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(y_pred[mask] - y_true[mask])
    return float(np.mean(diff))


def _validate_same_shape(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> None:
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        msg = (
            f"Shape mismatch in {name}: true={y_true.shape}, "
            f"pred={y_pred.shape}, mask={mask.shape}."
        )
        raise ValueError(msg)


def _run_mean_pca(
    x_obs: np.ndarray,
    n_components: int,
    seed: int,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    x_imp = imputer.fit_transform(x_obs)
    pca = PCA(n_components=n_components, random_state=seed)
    scores = pca.fit_transform(x_imp)
    return pca.inverse_transform(scores)


def _run_mice_pca(
    x_obs: np.ndarray,
    n_components: int,
    seed: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, bool, int, bool]:
    """Run MICE+PCA with convergence-aware warning handling.

    Returns:
        Tuple ``(reconstruction, converged, n_iter, retry_used)``.
    """

    def _fit_with_max_iter(iter_cap: int) -> tuple[np.ndarray, bool, int]:
        imputer = IterativeImputer(
            random_state=seed,
            max_iter=iter_cap,
            sample_posterior=False,
            initial_strategy="mean",
            imputation_order="ascending",
            skip_complete=True,
            keep_empty_features=True,
            tol=tol,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            x_imp_local = imputer.fit_transform(x_obs)
        had_warning = any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )
        n_iter_local = int(getattr(imputer, "n_iter_", iter_cap))
        converged_local = not had_warning
        return x_imp_local, converged_local, n_iter_local

    x_imp, converged, n_iter = _fit_with_max_iter(max_iter)
    retry_used = False
    if not converged:
        retry_cap = max(max_iter + 1, max_iter * 3)
        x_imp_retry, converged_retry, n_iter_retry = _fit_with_max_iter(retry_cap)
        x_imp = x_imp_retry
        converged = converged_retry
        n_iter = n_iter_retry
        retry_used = True

    pca = PCA(n_components=n_components, random_state=seed)
    scores = pca.fit_transform(x_imp)
    return pca.inverse_transform(scores), converged, n_iter, retry_used


def _run_knn_pca(
    x_obs: np.ndarray,
    n_components: int,
    seed: int,
    n_neighbors: int,
) -> np.ndarray:
    imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
    x_imp = imputer.fit_transform(x_obs)
    pca = PCA(n_components=n_components, random_state=seed)
    scores = pca.fit_transform(x_imp)
    return pca.inverse_transform(scores)


def _run_vbpca(
    x_obs: np.ndarray,
    n_components: int,
    maxiters: int,
    seed: int,
    use_model_selection: bool,
    selection_patience: int | None,
    selection_max_trials: int | None,
    selection_components: list[int] | None,
) -> tuple[np.ndarray, float, int, list[dict[str, float]]]:
    # Benchmark convention: (n_samples, n_features). VBPCA expects
    # (n_features, n_samples), so we transpose before fit.
    x_f_by_n = x_obs.T
    observed_mask = ~np.isnan(x_obs)

    n_features, n_samples = x_f_by_n.shape
    selected_k = max(1, min(n_components, n_features, n_samples))
    selection_trace_rows: list[dict[str, float]] = []

    np.random.seed(seed)
    if use_model_selection:
        cfg = SelectionConfig(
            metric="rms",
            stop_on_metric_reversal=True,
            patience=selection_patience,
            max_trials=selection_max_trials,
            compute_explained_variance=False,
            return_best_model=True,
        )
        best_k, _best_metrics, _trace, best_model = select_n_components(
            x_f_by_n,
            mask=observed_mask.T,
            components=selection_components,
            config=cfg,
            maxiters=maxiters,
            verbose=0,
            algorithm="vb",
            compat_mode="modern",
            rotate2pca=1,
        )
        selected_k = int(best_k)
        for entry in _trace:
            selection_trace_rows.append(
                {
                    "k": float(entry["k"]),
                    "rms": float(entry["rms"]),
                    "prms": float(entry["prms"]),
                    "cost": float(entry["cost"]),
                }
            )
        if best_model is not None:
            model = best_model
        else:
            model = VBPCA(
                n_components=selected_k,
                maxiters=maxiters,
                verbose=0,
                algorithm="vb",
                compat_mode="modern",
                rotate2pca=1,
            )
            model.fit(x_f_by_n, mask=observed_mask.T)
    else:
        model = VBPCA(
            n_components=selected_k,
            maxiters=maxiters,
            verbose=0,
            algorithm="vb",
            compat_mode="modern",
            rotate2pca=1,
        )
        model.fit(x_f_by_n, mask=observed_mask.T)

    x_recon = np.asarray(model.inverse_transform(), dtype=float).T
    variance = np.asarray(model.variance_, dtype=float).T
    mean_variance = float(np.mean(variance))

    if not selection_trace_rows:
        rms_val = float(model.rms_) if model.rms_ is not None else float("nan")
        prms_val = float(model.prms_) if model.prms_ is not None else float("nan")
        cost_val = float(model.cost_) if model.cost_ is not None else float("nan")
        selection_trace_rows.append(
            {
                "k": float(selected_k),
                "rms": rms_val,
                "prms": prms_val,
                "cost": cost_val,
            }
        )

    return x_recon, mean_variance, selected_k, selection_trace_rows


def _evaluate_methods(
    x_true: np.ndarray,
    x_obs: np.ndarray,
    holdout_mask: np.ndarray,
    n_components: int,
    seed_method: int,
    vbpca_maxiters: int,
    mice_max_iter: int,
    mice_tol: float,
    include_mice: bool,
    include_knn: bool,
    knn_neighbors: int,
    vbpca_select_components: bool,
    vbpca_selection_patience: int | None,
    vbpca_selection_max_trials: int | None,
    vbpca_selection_components: list[int] | None,
    use_selected_k_for_all_methods: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    rows: list[dict[str, Any]] = []

    # Select VBPCA k first, then optionally reuse that k for all methods.
    start = time.perf_counter()
    recon_vbpca, mean_var, selected_k, selection_trace_rows = _run_vbpca(
        x_obs,
        n_components=n_components,
        maxiters=vbpca_maxiters,
        seed=seed_method,
        use_model_selection=vbpca_select_components,
        selection_patience=vbpca_selection_patience,
        selection_max_trials=vbpca_selection_max_trials,
        selection_components=vbpca_selection_components,
    )
    _validate_same_shape("vbpca_vb_modern", x_true, recon_vbpca, holdout_mask)
    vbpca_time = time.perf_counter() - start

    effective_k = int(selected_k) if use_selected_k_for_all_methods else int(n_components)

    start = time.perf_counter()
    recon_mean = _run_mean_pca(x_obs, n_components=effective_k, seed=seed_method)
    _validate_same_shape("mean_pca", x_true, recon_mean, holdout_mask)
    mean_time = time.perf_counter() - start
    rows.append(
        {
            "method": "mean_pca",
            "rmse": _masked_rmse(x_true, recon_mean, holdout_mask),
            "mae": _masked_mae(x_true, recon_mean, holdout_mask),
            "wall_time_sec": mean_time,
            "vbpca_mean_variance": np.nan,
            "mice_converged": np.nan,
            "mice_n_iter": np.nan,
            "mice_retry_used": np.nan,
            "vbpca_selected_k": int(selected_k),
            "k_used": int(effective_k),
        }
    )

    if include_mice:
        start = time.perf_counter()
        recon_mice, mice_converged, mice_n_iter, mice_retry_used = _run_mice_pca(
            x_obs,
            n_components=effective_k,
            seed=seed_method,
            max_iter=mice_max_iter,
            tol=mice_tol,
        )
        _validate_same_shape("mice_pca", x_true, recon_mice, holdout_mask)
        mice_time = time.perf_counter() - start
        rows.append(
            {
                "method": "mice_pca",
                "rmse": _masked_rmse(x_true, recon_mice, holdout_mask),
                "mae": _masked_mae(x_true, recon_mice, holdout_mask),
                "wall_time_sec": mice_time,
                "vbpca_mean_variance": np.nan,
                "mice_converged": bool(mice_converged),
                "mice_n_iter": int(mice_n_iter),
                "mice_retry_used": bool(mice_retry_used),
                "vbpca_selected_k": int(selected_k),
                "k_used": int(effective_k),
            }
        )

    if include_knn:
        start = time.perf_counter()
        recon_knn = _run_knn_pca(
            x_obs,
            n_components=effective_k,
            seed=seed_method,
            n_neighbors=knn_neighbors,
        )
        _validate_same_shape("knn_pca", x_true, recon_knn, holdout_mask)
        knn_time = time.perf_counter() - start
        rows.append(
            {
                "method": "knn_pca",
                "rmse": _masked_rmse(x_true, recon_knn, holdout_mask),
                "mae": _masked_mae(x_true, recon_knn, holdout_mask),
                "wall_time_sec": knn_time,
                "vbpca_mean_variance": np.nan,
                "mice_converged": np.nan,
                "mice_n_iter": np.nan,
                "mice_retry_used": np.nan,
                "vbpca_selected_k": int(selected_k),
                "k_used": int(effective_k),
            }
        )

    rows.append(
        {
            "method": "vbpca_vb_modern",
            "rmse": _masked_rmse(x_true, recon_vbpca, holdout_mask),
            "mae": _masked_mae(x_true, recon_vbpca, holdout_mask),
            "wall_time_sec": vbpca_time,
            "vbpca_mean_variance": mean_var,
            "mice_converged": np.nan,
            "mice_n_iter": np.nan,
            "mice_retry_used": np.nan,
            "vbpca_selected_k": int(selected_k),
            "k_used": int(selected_k),
        }
    )

    return rows, selection_trace_rows


def _run_one_task(
    task: ReplicateTask,
    latent_rank: int,
    noise_scale: float,
    synthetic_shape: tuple[int, int],
    vbpca_maxiters: int,
    vbpca_maxiters_genomics: int,
    vbpca_selection_patience: int | None,
    vbpca_selection_max_trials: int | None,
    mice_max_iter: int,
    mice_tol: float,
    include_mice: bool,
    include_knn: bool,
    knn_neighbors: int,
    vbpca_select_components: bool,
    use_selected_k_for_all_methods: bool,
    selection_plan: SelectionPlan | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng_data = np.random.default_rng(task.seed_data)
    rng_mask = np.random.default_rng(task.seed_mask)

    x_true = _load_dataset(
        dataset=task.setting.dataset,
        rng=rng_data,
        synthetic_shape=synthetic_shape,
        latent_rank=latent_rank,
        noise_scale=noise_scale,
    )

    holdout_mask = _make_holdout_mask(
        x=x_true,
        mechanism=task.setting.mechanism,
        pattern=task.setting.pattern,
        missing_rate=task.setting.missing_rate,
        rng=rng_mask,
    )

    x_obs = np.asarray(x_true, dtype=float).copy()
    x_obs[holdout_mask] = np.nan

    x_true, x_obs = _scale_with_package_standard_scaler(x_true, x_obs)

    n_samples, n_features = x_true.shape
    k = max(1, min(task.setting.n_components, n_features, n_samples))
    resolved_vbpca_maxiters = _dataset_vbpca_controls(
        task.setting.dataset,
        base_maxiters=vbpca_maxiters,
        genomics_maxiters=vbpca_maxiters_genomics,
    )

    local_components: list[int] | None = None
    anchor_k: int | float = np.nan
    if selection_plan is not None:
        anchor_k = int(selection_plan.anchor_k)
        if selection_plan.local_components is not None:
            local_components = [
                int(comp)
                for comp in selection_plan.local_components
                if 1 <= int(comp) <= int(k)
            ]
            if not local_components:
                local_components = [int(max(1, min(int(anchor_k), int(k))))]

    method_rows, selection_trace_rows = _evaluate_methods(
        x_true=x_true,
        x_obs=x_obs,
        holdout_mask=holdout_mask,
        n_components=k,
        seed_method=task.seed_method,
        vbpca_maxiters=resolved_vbpca_maxiters,
        mice_max_iter=mice_max_iter,
        mice_tol=mice_tol,
        include_mice=include_mice,
        include_knn=include_knn,
        knn_neighbors=knn_neighbors,
        vbpca_select_components=vbpca_select_components,
        vbpca_selection_patience=vbpca_selection_patience,
        vbpca_selection_max_trials=vbpca_selection_max_trials,
        vbpca_selection_components=local_components,
        use_selected_k_for_all_methods=use_selected_k_for_all_methods,
    )

    common_fields: dict[str, Any] = {
        "dataset": task.setting.dataset,
        "mechanism": task.setting.mechanism,
        "pattern": task.setting.pattern,
        "missing_rate": task.setting.missing_rate,
        "n_components_requested": k,
        "synthetic_shape": task.setting.synthetic_shape,
        "replicate_id": task.replicate_id,
        "seed_data": task.seed_data,
        "seed_mask": task.seed_mask,
        "seed_method": task.seed_method,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_holdout": int(np.count_nonzero(holdout_mask)),
        "scaling": "MissingAwareStandardScaler_observed_only",
        "include_mice": bool(include_mice),
        "include_knn": bool(include_knn),
        "knn_neighbors": int(knn_neighbors),
        "vbpca_model_selection": bool(vbpca_select_components),
        "vbpca_maxiters_used": int(resolved_vbpca_maxiters),
        "vbpca_selection_patience": (
            int(vbpca_selection_patience)
            if vbpca_selection_patience is not None
            else np.nan
        ),
        "vbpca_selection_max_trials": (
            int(vbpca_selection_max_trials)
            if vbpca_selection_max_trials is not None
            else np.nan
        ),
        "vbpca_anchor_k": anchor_k,
        "vbpca_local_selection_components": (
            ",".join(str(int(comp)) for comp in local_components)
            if local_components is not None
            else ""
        ),
        "use_selected_k_for_all_methods": bool(use_selected_k_for_all_methods),
    }

    for row in method_rows:
        row.update(
            common_fields
        )

    trace_rows: list[dict[str, Any]] = []
    for trace in selection_trace_rows:
        trace_rows.append(
            {
                **common_fields,
                "trace_k": int(trace["k"]),
                "trace_rms": float(trace["rms"]),
                "trace_prms": float(trace["prms"]),
                "trace_cost": float(trace["cost"]),
            }
        )

    return method_rows, trace_rows


def _build_tasks(
    datasets: list[str],
    mechanisms: list[str],
    patterns: list[str],
    missing_rates: list[float],
    n_components: int,
    n_reps: int,
    random_seed: int,
    synthetic_shape: str,
) -> list[ReplicateTask]:
    tasks: list[ReplicateTask] = []
    master_rng = np.random.default_rng(random_seed)

    for dataset in datasets:
        for mechanism in mechanisms:
            for pattern in patterns:
                for missing_rate in missing_rates:
                    setting = Setting(
                        dataset=dataset,
                        mechanism=mechanism,
                        pattern=pattern,
                        missing_rate=missing_rate,
                        n_components=n_components,
                        synthetic_shape=synthetic_shape,
                    )
                    for rep in range(n_reps):
                        seeds = master_rng.integers(0, np.iinfo(np.int32).max, size=3)
                        tasks.append(
                            ReplicateTask(
                                setting=setting,
                                replicate_id=rep,
                                seed_data=int(seeds[0]),
                                seed_mask=int(seeds[1]),
                                seed_method=int(seeds[2]),
                            )
                        )
    return tasks


def _setting_key(setting: Setting) -> tuple[str, str, str, float, int, str]:
    return (
        setting.dataset,
        setting.mechanism,
        setting.pattern,
        float(setting.missing_rate),
        int(setting.n_components),
        setting.synthetic_shape,
    )


def _build_local_components(anchor_k: int, upper_k: int, window: int) -> tuple[int, ...]:
    if window <= 0:
        return (int(max(1, min(anchor_k, upper_k))),)
    lo = max(1, int(anchor_k) - int(window))
    hi = min(int(upper_k), int(anchor_k) + int(window))
    if lo > hi:
        val = int(max(1, min(anchor_k, upper_k)))
        return (val,)
    return tuple(range(lo, hi + 1))


def _build_selection_plans(
    *,
    tasks: list[ReplicateTask],
    latent_rank: int,
    noise_scale: float,
    synthetic_shape: tuple[int, int],
    vbpca_maxiters: int,
    vbpca_maxiters_genomics: int,
    vbpca_selection_patience: int | None,
    vbpca_selection_max_trials: int | None,
    local_window: int,
) -> dict[tuple[str, str, str, float, int, str], SelectionPlan]:
    plans: dict[tuple[str, str, str, float, int, str], SelectionPlan] = {}
    anchor_tasks: dict[tuple[str, str, str, float, int, str], ReplicateTask] = {}

    for task in tasks:
        key = _setting_key(task.setting)
        if key not in anchor_tasks or task.replicate_id < anchor_tasks[key].replicate_id:
            anchor_tasks[key] = task

    for key, task in anchor_tasks.items():
        rng_data = np.random.default_rng(task.seed_data)
        rng_mask = np.random.default_rng(task.seed_mask)
        x_true = _load_dataset(
            dataset=task.setting.dataset,
            rng=rng_data,
            synthetic_shape=synthetic_shape,
            latent_rank=latent_rank,
            noise_scale=noise_scale,
        )
        holdout_mask = _make_holdout_mask(
            x=x_true,
            mechanism=task.setting.mechanism,
            pattern=task.setting.pattern,
            missing_rate=task.setting.missing_rate,
            rng=rng_mask,
        )
        x_obs = np.asarray(x_true, dtype=float).copy()
        x_obs[holdout_mask] = np.nan
        x_true_scaled, x_obs_scaled = _scale_with_package_standard_scaler(x_true, x_obs)

        n_samples, n_features = x_true_scaled.shape
        k_cap = max(1, min(task.setting.n_components, n_features, n_samples))
        resolved_vbpca_maxiters = _dataset_vbpca_controls(
            task.setting.dataset,
            base_maxiters=vbpca_maxiters,
            genomics_maxiters=vbpca_maxiters_genomics,
        )

        _recon, _var, selected_k, _trace = _run_vbpca(
            x_obs_scaled,
            n_components=int(k_cap),
            maxiters=resolved_vbpca_maxiters,
            seed=task.seed_method,
            use_model_selection=True,
            selection_patience=vbpca_selection_patience,
            selection_max_trials=vbpca_selection_max_trials,
            selection_components=None,
        )

        local_components = _build_local_components(
            anchor_k=int(selected_k),
            upper_k=int(k_cap),
            window=int(local_window),
        )
        plans[key] = SelectionPlan(
            anchor_k=int(selected_k),
            local_components=local_components,
        )

    return plans


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default="synthetic,diabetes,wine,genomics_like",
        help=(
            "Comma-separated datasets from "
            "{synthetic,genomics_like,diabetes,wine,breast_cancer}"
        ),
    )
    parser.add_argument(
        "--mechanisms",
        type=str,
        default="MCAR,MAR",
        help="Comma-separated missingness mechanisms.",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="random,block",
        help="Comma-separated missingness patterns.",
    )
    parser.add_argument(
        "--missing-rates",
        type=str,
        default="0.1,0.3,0.5",
        help="Comma-separated missingness rates in [0,1].",
    )
    parser.add_argument("--n-reps", type=int, default=200)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument(
        "--synthetic-shape",
        type=str,
        default="1000x50",
        help="Shape for synthetic data in <n_samples>x<n_features> format.",
    )
    parser.add_argument("--synthetic-rank", type=int, default=8)
    parser.add_argument("--synthetic-noise", type=float, default=0.15)
    parser.add_argument("--vbpca-maxiters", type=int, default=60)
    parser.add_argument(
        "--vbpca-maxiters-genomics",
        type=int,
        default=45,
        help="Max iterations for genomics_like dataset (controls runtime blow-up).",
    )
    parser.add_argument(
        "--vbpca-select-components",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run VBPCA model selection over 1..n_components.",
    )
    parser.add_argument(
        "--vbpca-selection-patience",
        type=int,
        default=1,
        help="Patience used in VBPCA model selection (set 0/negative to disable).",
    )
    parser.add_argument(
        "--vbpca-selection-max-trials",
        type=int,
        default=8,
        help="Hard cap on number of k-trials in model selection (set 0/negative to disable).",
    )
    parser.add_argument(
        "--vbpca-local-window",
        type=int,
        default=2,
        help=(
            "After anchor selection, evaluate local component candidates in "
            "[anchor-window, anchor+window] per replicate. Set <=0 to use anchor only."
        ),
    )
    parser.add_argument(
        "--use-selected-k-for-all-methods",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use VBPCA-selected k for mean+PCA and MICE+PCA within each replicate.",
    )
    parser.add_argument("--mice-max-iter", type=int, default=12)
    parser.add_argument("--mice-tol", type=float, default=1e-2)
    parser.add_argument(
        "--include-mice",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include MICE+PCA comparator (disable for very large-loci wall-time runs).",
    )
    parser.add_argument(
        "--include-knn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include KNN+PCA comparator.",
    )
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/replicates.csv"),
    )
    parser.add_argument(
        "--selection-trace-output",
        type=Path,
        default=Path("results/vbpca_selection_trace.csv"),
        help="CSV path for per-replicate VBPCA model-selection trace rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    datasets = _parse_csv_values(args.datasets)
    mechanisms = _parse_csv_values(args.mechanisms)
    patterns = _parse_csv_values(args.patterns)
    missing_rates = _parse_float_list(args.missing_rates)

    if not datasets:
        msg = "At least one dataset must be provided"
        raise ValueError(msg)

    if args.knn_neighbors < 1:
        msg = "--knn-neighbors must be >= 1"
        raise ValueError(msg)

    if args.vbpca_maxiters < 1 or args.vbpca_maxiters_genomics < 1:
        msg = "--vbpca-maxiters and --vbpca-maxiters-genomics must be >= 1"
        raise ValueError(msg)

    if int(args.vbpca_local_window) < 0:
        msg = "--vbpca-local-window must be >= 0"
        raise ValueError(msg)

    for rate in missing_rates:
        if not (0 < rate < 1):
            msg = f"Invalid missing rate {rate}; expected 0 < rate < 1"
            raise ValueError(msg)

    synthetic_shape = _parse_shape(args.synthetic_shape)
    selection_patience = (
        int(args.vbpca_selection_patience)
        if int(args.vbpca_selection_patience) > 0
        else None
    )
    selection_max_trials = (
        int(args.vbpca_selection_max_trials)
        if int(args.vbpca_selection_max_trials) > 0
        else None
    )

    tasks = _build_tasks(
        datasets=datasets,
        mechanisms=mechanisms,
        patterns=patterns,
        missing_rates=missing_rates,
        n_components=args.n_components,
        n_reps=args.n_reps,
        random_seed=args.random_seed,
        synthetic_shape=args.synthetic_shape,
    )

    selection_plans: dict[tuple[str, str, str, float, int, str], SelectionPlan] = {}
    if bool(args.vbpca_select_components):
        selection_plans = _build_selection_plans(
            tasks=tasks,
            latent_rank=args.synthetic_rank,
            noise_scale=args.synthetic_noise,
            synthetic_shape=synthetic_shape,
            vbpca_maxiters=int(args.vbpca_maxiters),
            vbpca_maxiters_genomics=int(args.vbpca_maxiters_genomics),
            vbpca_selection_patience=selection_patience,
            vbpca_selection_max_trials=selection_max_trials,
            local_window=int(args.vbpca_local_window),
        )

    resolved_n_jobs = _resolve_n_jobs(args.n_jobs)

    rows_nested = Parallel(n_jobs=resolved_n_jobs, backend="loky", verbose=10)(
        delayed(_run_one_task)(
            task=task,
            latent_rank=args.synthetic_rank,
            noise_scale=args.synthetic_noise,
            synthetic_shape=synthetic_shape,
            vbpca_maxiters=args.vbpca_maxiters,
            vbpca_maxiters_genomics=args.vbpca_maxiters_genomics,
            vbpca_selection_patience=selection_patience,
            vbpca_selection_max_trials=selection_max_trials,
            mice_max_iter=args.mice_max_iter,
            mice_tol=args.mice_tol,
            include_mice=bool(args.include_mice),
            include_knn=bool(args.include_knn),
            knn_neighbors=int(args.knn_neighbors),
            vbpca_select_components=bool(args.vbpca_select_components),
            use_selected_k_for_all_methods=bool(args.use_selected_k_for_all_methods),
            selection_plan=selection_plans.get(_setting_key(task.setting)),
        )
        for task in tasks
    )

    method_rows = [row for method_batch, _trace_batch in rows_nested for row in method_batch]
    trace_rows = [row for _method_batch, trace_batch in rows_nested for row in trace_batch]

    frame = pd.DataFrame(method_rows)
    trace_frame = pd.DataFrame(trace_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    args.selection_trace_output.parent.mkdir(parents=True, exist_ok=True)
    trace_frame.to_csv(args.selection_trace_output, index=False)

    total_settings = len(datasets) * len(mechanisms) * len(patterns) * len(missing_rates)
    print(
        "Completed benchmark run:",
        f"settings={total_settings}",
        f"replicates_per_setting={args.n_reps}",
        f"rows={len(frame)}",
        f"output={args.output}",
        f"selection_trace_rows={len(trace_frame)}",
        f"selection_trace_output={args.selection_trace_output}",
    )


if __name__ == "__main__":
    main()
