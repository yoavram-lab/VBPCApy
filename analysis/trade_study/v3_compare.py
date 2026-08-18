#!/usr/bin/env python
"""V3 Head-to-head comparison: default vs v3 regime-stratified vs sklearn.

Re-runs the paper's stability grid with three conditions and produces
comparison figures (F1-V3 through F5-V3) in the same style as
``v2_compare.py``.  The v3 condition is *regime-gated*: the optimised
config is selected per (n, p) cell using the rules from
``v3_regime_compare.py``:

  - p ≤ 30  → results/v3_smallp/phase3_0_confirmation.json
  - p ≤ 70  → results/v3_trans/phase3_0_confirmation.json
  - p > 70  → results/v2/phase3_1_comparison.json (reused v2 large)

Figures
-------
F1 — Accuracy heatmap (3 rows: default, v3-stratified, sklearn)
F2 — Error decomposition (MAE bars + selected-vs-true)
F3 — Coverage calibration curves (default vs v3, by missingness)
F4 — Holdout RMSE grouped bars (3 conditions × 4 missingness patterns)
F5 — ΔMAE heatmap: v3-stratified-vs-default improvement by (n, p)

Usage
-----
    python -m analysis.trade_study.v3_compare [--smoke] [--fmt png] [--plot-only]
    just trade-v3-vis-compare
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import pathlib
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from vbpca_py import VBPCA, SelectionConfig, select_n_components

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Grid / constants (same as v2_compare.py) ─────────────────────

FULL_GRID: dict[str, list[int]] = {
    "n": [20, 30, 50, 70, 100, 150, 200],
    "p": [10, 20, 30, 50, 70, 100, 200],
    "true_rank": [2, 5, 10],
}
SMOKE_GRID: dict[str, list[int]] = {
    "n": [30],
    "p": [20],
    "true_rank": [2],
}
MISSINGNESS: list[str] = ["complete", "mcar", "mnar_censored", "block"]
MISS_FRACTION = 0.15
NOISE_STD = 0.5
HOLDOUT_FRACTION = 0.10
MAXITERS = 200
MAXITERS_SMOKE = 50
REPS_FULL = 10
REPS_SMOKE = 1
COVERAGE_NOMINALS = [0.50, 0.80, 0.90, 0.95, 0.99]
DPI = 300

_MISS_LABEL = {
    "complete": "Complete",
    "mcar": "MCAR 15%",
    "mnar_censored": "MNAR 15%",
    "block": "Block 15%",
}

METHOD_DEFAULT = "vbpca_default"
METHOD_V3 = "vbpca_v3_optimized"
METHOD_SKLEARN = "sklearn_pca"

METHOD_LABELS = {
    METHOD_DEFAULT: "Default",
    METHOD_V3: "V3-optimized",
    METHOD_SKLEARN: "sklearn PCA",
}
METHOD_COLORS = {
    METHOD_DEFAULT: "#1f77b4",
    METHOD_V3: "#ff7f0e",
    METHOD_SKLEARN: "#2ca02c",
}

# ── Data classes ─────────────────────────────────────────────────


@dataclass
class _Trial:
    n: int
    p: int
    true_rank: int
    missingness: str
    rep: int
    selected_k: int
    method: str  # vbpca_default | vbpca_v3_optimized | sklearn_pca
    over: float
    under: float
    exact: float
    power: float
    abs_error: float


@dataclass
class _CoverageTrial:
    n: int
    p: int
    true_rank: int
    missingness: str
    rep: int
    method: str
    nominal: float
    coverage: float
    holdout_rmse: float
    baseline_rmse: float = 0.0
    mean_interval_width: float = 0.0


# ── Data generation helpers ──────────────────────────────────────


def _generate_low_rank(
    n: int,
    p: int,
    true_rank: int,
    rng: np.random.Generator,
) -> np.ndarray:
    W = rng.standard_normal((n, true_rank))
    S = rng.standard_normal((true_rank, p))
    return W @ S + NOISE_STD * rng.standard_normal((n, p))


def _apply_missingness(
    x: np.ndarray,
    pattern: str,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if pattern == "complete":
        return None
    n, p = x.shape
    mask = np.ones_like(x, dtype=bool)
    if pattern == "mcar":
        mask = rng.random(x.shape) > MISS_FRACTION
    elif pattern == "mnar_censored":
        threshold = np.nanquantile(x, MISS_FRACTION, axis=1, keepdims=True)
        mask = x > threshold
    elif pattern == "block":
        nc = max(1, int(p * MISS_FRACTION))
        nr = max(1, int(n * MISS_FRACTION))
        c0 = rng.integers(0, max(1, p - nc))
        r0 = rng.integers(0, max(1, n - nr))
        mask[r0 : r0 + nr, c0 : c0 + nc] = False
    for i in range(n):
        if not mask[i].any():
            mask[i, rng.integers(p)] = True
    for j in range(p):
        if not mask[:, j].any():
            mask[rng.integers(n), j] = True
    return mask


def _summarize_selection(true_rank: int, selected: int) -> dict[str, float]:
    return {
        "over": float(selected > true_rank),
        "under": float(selected < true_rank),
        "exact": float(selected == true_rank),
        "power": float(true_rank > 0 and selected >= true_rank),
        "abs_error": float(abs(selected - true_rank)),
    }


# ── Config loading (regime-gated v3) ─────────────────────────────

V3_RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
_V2_DIR = V3_RESULTS_DIR / "v2"
_V3_SMALLP_DIR = V3_RESULTS_DIR / "v3_smallp"
_V3_TRANS_DIR = V3_RESULTS_DIR / "v3_trans"


def _load_phase31_config() -> dict[str, Any]:
    path = _V2_DIR / "phase3_1_comparison.json"
    if not path.exists():
        LOGGER.warning("V2 large config not found at %s — using defaults.", path)
        return {}
    data = json.loads(path.read_text())
    return data.get("optimized_config") or {}


def _load_v3_family_config(family_dir: pathlib.Path) -> dict[str, Any]:
    path = family_dir / "phase3_0_confirmation.json"
    if not path.exists():
        LOGGER.warning("V3 family config not found at %s — using defaults.", path)
        return {}
    data = json.loads(path.read_text())
    return data.get("optimized_config") or {}


def _gate_v3_config(
    n: int,
    p: int,
    smallp_cfg: dict[str, Any],
    trans_cfg: dict[str, Any],
    large_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Select the v3 config for an (n, p) cell using regime gating."""
    if p <= 30:
        return smallp_cfg
    if p <= 70:
        return trans_cfg
    return large_cfg


# v4: surrogate-based gating function (set lazily by run_comparison).
_SURROGATE_RECOMMEND: Any = None


def _gate_v3_config_surrogate(
    n: int,
    p: int,
    true_rank: int,
    missingness: str,
    noise_std: float,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """v4 regime-surrogate replacement for :func:`_gate_v3_config`.

    Calls the global surrogate (set by :func:`run_comparison`) to predict
    the best config for the (n, p, true_rank, missingness, noise_std)
    regime.  Returns *fallback* if the surrogate is unavailable.
    """
    if _SURROGATE_RECOMMEND is None:
        return fallback
    try:
        return _SURROGATE_RECOMMEND({
            "n": float(n),
            "p": float(p),
            "true_rank": int(true_rank),
            "missingness": missingness,
            "noise_std": float(noise_std),
        })
    except Exception as exc:
        LOGGER.warning("surrogate gate failed (n=%d,p=%d): %s", n, p, exc)
        return fallback


def _vbpca_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map a trade-study config dict to VBPCA / select_n_components kwargs."""
    kw: dict[str, Any] = {"verbose": 0}
    for key in (
        "hp_va",
        "hp_vb",
        "hp_v",
        "va_init",
        "niter_broadprior",
        "maxiters",
        "minangle",
        "patience",
        "cfstop_rel",
        "rotate2pca",
        "bias",
        "xprobe_fraction",
    ):
        if key in cfg and cfg[key] is not None:
            kw[key] = cfg[key]
    if "rmsstop_window" in cfg:
        kw["rmsstop"] = [
            int(cfg["rmsstop_window"]),
            cfg.get("rmsstop_atol", 1e-4),
            cfg.get("rmsstop_rtol", 1e-3),
        ]
    return kw


# ── Sklearn helpers ──────────────────────────────────────────────


def _sklearn_select_k(
    x: np.ndarray,
    mask: np.ndarray | None,
    max_k: int,
) -> tuple[int, np.ndarray]:
    """Mean-impute then PCA with MLE rank selection (capped at ``max_k``).

    Returns (selected_k, reconstruction).
    """
    x_input = x.copy()
    if mask is not None:
        x_input[~mask] = np.nan
        imp = SimpleImputer(strategy="mean")
        x_imp = imp.fit_transform(x_input)
    else:
        x_imp = x_input

    n_samples, n_features = x_imp.shape
    cap = max(1, min(max_k, min(n_samples, n_features) - 1))
    # Use MLE selection but cap at max_k for fair comparison.
    try:
        pca = PCA(n_components="mle", svd_solver="full")
        pca.fit(x_imp)
        k_mle = int(pca.n_components_)
        k = max(1, min(k_mle, cap))
    except Exception:
        k = cap

    pca_final = PCA(n_components=k)
    z = pca_final.fit_transform(x_imp)
    rec = pca_final.inverse_transform(z)
    return k, rec


# ── Grid runners ─────────────────────────────────────────────────


def _run_vbpca_trial(
    x: np.ndarray,
    mask: np.ndarray | None,
    candidates: list[int],
    opts: dict[str, Any],
) -> int:
    """Run select_n_components and return best_k."""
    maxiters = opts.pop("maxiters", MAXITERS)
    cfg = SelectionConfig(
        metric="cost",
        patience=2,
        max_trials=len(candidates),
        compute_explained_variance=False,
    )
    best_k, _, _, _ = select_n_components(
        x,
        mask=mask,
        components=candidates,
        config=cfg,
        maxiters=maxiters,
        **opts,
    )
    opts["maxiters"] = maxiters
    return best_k


def _run_grid(
    grid: dict[str, list[int]],
    reps: int,
    smallp_cfg: dict[str, Any],
    trans_cfg: dict[str, Any],
    large_cfg: dict[str, Any],
    seed: int = 42,
    maxiters: int = MAXITERS,
) -> list[_Trial]:
    """Run all three conditions across the grid."""
    rng = np.random.default_rng(seed)
    trials: list[_Trial] = []
    settings = list(
        itertools.product(
            grid["n"],
            grid["p"],
            grid["true_rank"],
        )
    )
    total = sum(len(MISSINGNESS) * reps for n, p, r in settings if r < min(n, p))

    done = 0
    for n, p, true_rank in settings:
        if true_rank >= min(n, p):
            continue
        bucket_cfg = _gate_v3_config(n, p, smallp_cfg, trans_cfg, large_cfg)
        for rep in range(reps):
            x_clean = _generate_low_rank(n, p, true_rank, rng)
            max_k = min(true_rank + 6, min(n, p) - 1)
            candidates = list(range(1, max_k + 1))

            for miss in MISSINGNESS:
                x = x_clean.copy()
                mask = _apply_missingness(x, miss, rng)

                # Default
                default_kw = {"maxiters": maxiters, "verbose": 0}
                k_def = _run_vbpca_trial(x, mask, candidates, default_kw)
                trials.append(
                    _Trial(
                        n=n,
                        p=p,
                        true_rank=true_rank,
                        missingness=miss,
                        rep=rep,
                        selected_k=k_def,
                        method=METHOD_DEFAULT,
                        **_summarize_selection(true_rank, k_def),
                    )
                )

                # V3 regime-stratified — surrogate per (n,p,rank,miss) when
                # available, else falls back to bucket gating.
                v3_cfg = _gate_v3_config_surrogate(
                    n=n,
                    p=p,
                    true_rank=true_rank,
                    missingness=miss,
                    noise_std=NOISE_STD,
                    fallback=bucket_cfg,
                )
                v3_kw = _vbpca_kwargs_from_config(v3_cfg)
                if v3_kw:
                    v3_kw_copy = dict(v3_kw)
                    k_v3 = _run_vbpca_trial(x, mask, candidates, v3_kw_copy)
                else:
                    k_v3 = k_def
                trials.append(
                    _Trial(
                        n=n,
                        p=p,
                        true_rank=true_rank,
                        missingness=miss,
                        rep=rep,
                        selected_k=k_v3,
                        method=METHOD_V3,
                        **_summarize_selection(true_rank, k_v3),
                    )
                )

                # sklearn (mean-imputed PCA + MLE rank)
                try:
                    k_sk, _ = _sklearn_select_k(x, mask, max_k)
                except Exception as exc:
                    LOGGER.warning("sklearn failed (n=%d p=%d): %s", n, p, exc)
                    k_sk = max_k
                trials.append(
                    _Trial(
                        n=n,
                        p=p,
                        true_rank=true_rank,
                        missingness=miss,
                        rep=rep,
                        selected_k=k_sk,
                        method=METHOD_SKLEARN,
                        **_summarize_selection(true_rank, k_sk),
                    )
                )

                done += 1
                if done % 50 == 0 or done == total:
                    LOGGER.info("  [%d/%d] cells", done, total)

    return trials


def _run_coverage_grid(
    grid: dict[str, list[int]],
    reps: int,
    smallp_cfg: dict[str, Any],
    trans_cfg: dict[str, Any],
    large_cfg: dict[str, Any],
    seed: int = 42,
    maxiters: int = MAXITERS,
) -> list[_CoverageTrial]:
    """Run coverage analysis for all three methods."""
    rng = np.random.default_rng(seed)
    results: list[_CoverageTrial] = []
    settings = list(
        itertools.product(
            grid["n"],
            grid["p"],
            grid["true_rank"],
        )
    )

    for n, p, true_rank in settings:
        if true_rank >= min(n, p):
            continue
        bucket_cfg = _gate_v3_config(n, p, smallp_cfg, trans_cfg, large_cfg)
        for rep in range(reps):
            x_clean = _generate_low_rank(n, p, true_rank, rng)
            for miss in MISSINGNESS:
                x = x_clean.copy()
                mask = _apply_missingness(x, miss, rng)
                obs_mask = np.ones((n, p), dtype=bool) if mask is None else mask.copy()

                v3_cfg = _gate_v3_config_surrogate(
                    n=n,
                    p=p,
                    true_rank=true_rank,
                    missingness=miss,
                    noise_std=NOISE_STD,
                    fallback=bucket_cfg,
                )
                v3_kw = _vbpca_kwargs_from_config(v3_cfg)

                obs_idx = np.argwhere(obs_mask)
                n_hold = max(1, int(len(obs_idx) * HOLDOUT_FRACTION))
                hold_idx = rng.choice(len(obs_idx), size=n_hold, replace=False)
                hold_coords = obs_idx[hold_idx]
                hold_vals = x_clean[hold_coords[:, 0], hold_coords[:, 1]]

                train_mask = obs_mask.copy()
                train_mask[hold_coords[:, 0], hold_coords[:, 1]] = False
                x_train = x_clean.copy()
                x_train[~train_mask] = np.nan

                for i in range(n):
                    if not train_mask[i].any():
                        j = rng.integers(0, p)
                        train_mask[i, j] = True
                        x_train[i, j] = x_clean[i, j]
                for j in range(p):
                    if not train_mask[:, j].any():
                        i = rng.integers(0, n)
                        train_mask[i, j] = True
                        x_train[i, j] = x_clean[i, j]

                # VBPCA conditions (default + v3)
                for method_name, kw_extra in [
                    (METHOD_DEFAULT, {"verbose": 0}),
                    (METHOD_V3, dict(v3_kw) if v3_kw else {"verbose": 0}),
                ]:
                    mi = kw_extra.pop("maxiters", maxiters)
                    model = VBPCA(
                        n_components=true_rank,
                        maxiters=mi,
                        **kw_extra,
                    )
                    kw_extra["maxiters"] = mi
                    model.fit(x_train, mask=train_mask)
                    xrec = model.reconstruction_
                    # Calibrated intervals use the predictive variance
                    # (latent uncertainty + observation noise).
                    vr = (
                        model.predictive_variance_
                        if model.predictive_variance_ is not None
                        else model.variance_
                    )
                    if xrec is None or vr is None:
                        continue
                    pred = xrec[hold_coords[:, 0], hold_coords[:, 1]]
                    var = vr[hold_coords[:, 0], hold_coords[:, 1]]
                    residuals = np.abs(hold_vals - pred)
                    rmse = float(np.sqrt(np.mean(residuals**2)))
                    std = np.sqrt(np.maximum(var, 1e-12))
                    for nominal in COVERAGE_NOMINALS:
                        z = scipy_stats.norm.ppf(0.5 + nominal / 2.0)
                        half_w = z * std
                        covered = residuals <= half_w
                        results.append(
                            _CoverageTrial(
                                n=n,
                                p=p,
                                true_rank=true_rank,
                                missingness=miss,
                                rep=rep,
                                method=method_name,
                                nominal=nominal,
                                coverage=float(np.mean(covered)),
                                holdout_rmse=rmse,
                                baseline_rmse=float("nan"),
                                mean_interval_width=float(np.mean(2.0 * half_w)),
                            )
                        )

                # sklearn condition (RMSE only; no uncertainty → coverage 0)
                try:
                    _, rec = _sklearn_select_k(x_train, train_mask, max(1, true_rank))
                    pred = rec[hold_coords[:, 0], hold_coords[:, 1]]
                    residuals = np.abs(hold_vals - pred)
                    rmse = float(np.sqrt(np.mean(residuals**2)))
                except Exception:
                    rmse = float("nan")
                results.extend(
                    _CoverageTrial(
                        n=n,
                        p=p,
                        true_rank=true_rank,
                        missingness=miss,
                        rep=rep,
                        method=METHOD_SKLEARN,
                        nominal=nominal,
                        coverage=0.0,
                        holdout_rmse=rmse,
                        baseline_rmse=float("nan"),
                        mean_interval_width=0.0,
                    )
                    for nominal in COVERAGE_NOMINALS
                )

    return results


# ── I/O ──────────────────────────────────────────────────────────


def _save_json(data: list[Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(d) for d in data], indent=2))
    LOGGER.info("Saved %s (%d records)", path, len(data))


def _load_trials(path: pathlib.Path) -> list[_Trial]:
    return [_Trial(**r) for r in json.loads(path.read_text())]


def _load_coverage(path: pathlib.Path) -> list[_CoverageTrial]:
    return [_CoverageTrial(**r) for r in json.loads(path.read_text())]


# ── Plotting helpers ─────────────────────────────────────────────


def _save_fig(fig: plt.Figure, path: pathlib.Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path)


def _rate_matrix_by_miss(
    trials: list[_Trial],
    field: str,
    method: str,
    missingness: str,
) -> tuple[np.ndarray, list[int], list[int]]:
    vals: dict[tuple[int, int], list[float]] = defaultdict(list)
    for t in trials:
        if t.method != method or t.missingness != missingness:
            continue
        vals[t.n, t.p].append(getattr(t, field))
    ns = sorted({k[0] for k in vals})
    ps = sorted({k[1] for k in vals})
    mat = np.full((len(ns), len(ps)), np.nan)
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            v = vals.get((n, p), [])
            if v:
                mat[i, j] = float(np.mean(v))
    return mat, ns, ps


def _annotate_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    *,
    fmt: str = "{:.0%}",
    threshold: float = 0.5,
    fontsize: int = 8,
) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    fmt.format(val),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="white" if val < threshold else "black",
                )


def _annotate_signed(
    ax: plt.Axes,
    mat: np.ndarray,
    abs_max: float,
    fontsize: int = 8,
) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                sign = "+" if val > 0 else ""
                ax.text(
                    j,
                    i,
                    f"{sign}{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="white" if abs(val) > abs_max * 0.6 else "black",
                )


# ── F1: Accuracy heatmaps (3 rows) ───────────────────────────────


def plot_f1_accuracy(
    trials: list[_Trial],
    out_dir: pathlib.Path,
    fmt: str,
) -> None:
    row_defs = [
        (METHOD_DEFAULT, "VBPCApy (default)"),
        (METHOD_V3, "VBPCApy (v3-stratified)"),
        (METHOD_SKLEARN, "sklearn PCA"),
    ]
    n_rows = len(row_defs)
    n_miss = len(MISSINGNESS)
    fig, axes = plt.subplots(
        n_rows,
        n_miss,
        figsize=(4.0 * n_miss + 1.2, 3.2 * n_rows),
        squeeze=False,
    )

    im = None
    for row, (method, label) in enumerate(row_defs):
        for col, miss in enumerate(MISSINGNESS):
            ax = axes[row, col]
            mat, ns, ps = _rate_matrix_by_miss(trials, "exact", method, miss)
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(range(len(ps)))
            ax.set_xticklabels(ps)
            ax.set_yticks(range(len(ns)))
            ax.set_yticklabels(ns)
            if row == n_rows - 1:
                ax.set_xlabel("p (features)")
            if col == 0:
                ax.set_ylabel(f"{label}\nn (samples)")
            ax.set_title(_MISS_LABEL.get(miss, miss) if row == 0 else "")
            _annotate_heatmap(ax, mat)

    fig.subplots_adjust(right=0.88, wspace=0.30, hspace=0.35)
    if im is not None:
        cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cax, label="Exact rate")
    fig.suptitle(
        "F1 — Rank-selection accuracy: default vs v3-stratified vs sklearn",
        fontsize=12,
    )
    _save_fig(fig, out_dir / f"figure_v3_f1_accuracy.{fmt}")


# ── F2: Error decomposition ──────────────────────────────────────


def plot_f2_errors(
    trials: list[_Trial],
    out_dir: pathlib.Path,
    fmt: str,
) -> None:
    fig, (ax_mae, ax_sel) = plt.subplots(1, 2, figsize=(12, 4.5))
    methods = [
        (METHOD_DEFAULT, METHOD_LABELS[METHOD_DEFAULT], METHOD_COLORS[METHOD_DEFAULT]),
        (METHOD_V3, METHOD_LABELS[METHOD_V3], METHOD_COLORS[METHOD_V3]),
        (METHOD_SKLEARN, METHOD_LABELS[METHOD_SKLEARN], METHOD_COLORS[METHOD_SKLEARN]),
    ]
    bar_labels, bar_vals, bar_colors = [], [], []
    for miss in MISSINGNESS:
        for method, mlabel, color in methods:
            vals = [
                t.abs_error
                for t in trials
                if t.missingness == miss and t.method == method
            ]
            if vals:
                bar_labels.append(f"{_MISS_LABEL.get(miss, miss)}\n{mlabel}")
                bar_vals.append(float(np.mean(vals)))
                bar_colors.append(color)

    ax_mae.bar(range(len(bar_labels)), bar_vals, color=bar_colors)
    ax_mae.set_xticks(range(len(bar_labels)))
    ax_mae.set_xticklabels(bar_labels, fontsize=6, rotation=30, ha="right")
    ax_mae.set_ylabel("Mean absolute error")
    ax_mae.set_title("A) MAE by missingness × method")

    all_ranks = sorted({t.true_rank for t in trials})
    for method, mlabel, color in methods:
        means, stds = [], []
        for rank in all_ranks:
            vals = [
                t.selected_k
                for t in trials
                if t.method == method and t.true_rank == rank
            ]
            means.append(float(np.mean(vals)) if vals else float("nan"))
            stds.append(float(np.std(vals)) if vals else 0.0)
        ax_sel.errorbar(
            all_ranks,
            means,
            yerr=stds,
            marker="o",
            label=mlabel,
            color=color,
            capsize=3,
        )
    ax_sel.plot(all_ranks, all_ranks, "k--", alpha=0.4, label="ideal")
    ax_sel.set_xlabel("True rank")
    ax_sel.set_ylabel("Selected k (mean ± std)")
    ax_sel.set_title("B) Selected vs true rank")
    ax_sel.legend(fontsize=8)

    fig.tight_layout()
    _save_fig(fig, out_dir / f"figure_v3_f2_errors.{fmt}")


# ── F3: Coverage calibration ─────────────────────────────────────


def plot_f3_coverage(
    cov: list[_CoverageTrial],
    out_dir: pathlib.Path,
    fmt: str,
) -> None:
    miss_list = sorted({r.missingness for r in cov})
    n_panels = len(miss_list)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)

    method_styles = [
        (METHOD_DEFAULT, METHOD_LABELS[METHOD_DEFAULT], METHOD_COLORS[METHOD_DEFAULT]),
        (METHOD_V3, METHOD_LABELS[METHOD_V3], METHOD_COLORS[METHOD_V3]),
    ]
    for col, miss in enumerate(miss_list):
        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Ideal")
        for method, mlabel, color in method_styles:
            nominals, empiricals = [], []
            for nom in COVERAGE_NOMINALS:
                covs = [
                    r.coverage
                    for r in cov
                    if r.method == method
                    and r.missingness == miss
                    and abs(r.nominal - nom) < 0.001
                ]
                if covs:
                    nominals.append(nom)
                    empiricals.append(float(np.mean(covs)))
            if nominals:
                ax.plot(
                    nominals, empiricals, "o-", label=mlabel, color=color, markersize=4
                )
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(_MISS_LABEL.get(miss, miss))
        ax.legend(fontsize=7)

    fig.suptitle(
        "F3 — Coverage calibration (default vs v3-stratified; sklearn has none)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, out_dir / f"figure_v3_f3_coverage.{fmt}")


# ── F4: Holdout RMSE bars ───────────────────────────────────────


def plot_f4_rmse(
    cov: list[_CoverageTrial],
    out_dir: pathlib.Path,
    fmt: str,
) -> None:
    seen: set[tuple[int, int, int, str, int, str]] = set()
    method_vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in cov:
        key = (r.n, r.p, r.true_rank, r.missingness, r.rep, r.method)
        if key in seen:
            continue
        seen.add(key)
        method_vals[r.method, r.missingness].append(r.holdout_rmse)

    miss_order = [m for m in MISSINGNESS if (METHOD_DEFAULT, m) in method_vals]
    pretty = [_MISS_LABEL.get(m, m) for m in miss_order]
    x = np.arange(len(miss_order))
    w = 0.27

    conditions = [
        (METHOD_DEFAULT, METHOD_LABELS[METHOD_DEFAULT], METHOD_COLORS[METHOD_DEFAULT]),
        (METHOD_V3, METHOD_LABELS[METHOD_V3], METHOD_COLORS[METHOD_V3]),
        (METHOD_SKLEARN, METHOD_LABELS[METHOD_SKLEARN], METHOD_COLORS[METHOD_SKLEARN]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, (method, label, color) in enumerate(conditions):
        vals = [
            float(np.nanmean(method_vals.get((method, m), [float("nan")])))
            for m in miss_order
        ]
        ax.bar(x + (k - 1) * w, vals, w, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(pretty)
    ax.set_ylabel("Holdout RMSE")
    ax.set_title("F4 — Reconstruction RMSE (default vs v3-stratified vs sklearn)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, out_dir / f"figure_v3_f4_rmse.{fmt}")


# ── F5: ΔMAE heatmap (v3-stratified – default) ───────────────────


def plot_f5_delta_mae(
    trials: list[_Trial],
    out_dir: pathlib.Path,
    fmt: str,
) -> None:
    """Heatmap: default MAE − v3-stratified MAE (positive = v3 better)."""
    def_vals: dict[tuple[int, int], list[float]] = defaultdict(list)
    opt_vals: dict[tuple[int, int], list[float]] = defaultdict(list)
    for t in trials:
        if t.method == METHOD_DEFAULT:
            def_vals[t.n, t.p].append(t.abs_error)
        elif t.method == METHOD_V3:
            opt_vals[t.n, t.p].append(t.abs_error)

    keys = sorted(set(def_vals.keys()) & set(opt_vals.keys()))
    if not keys:
        LOGGER.warning("No data for delta-MAE heatmap — skipping.")
        return
    ns = sorted({k[0] for k in keys})
    ps = sorted({k[1] for k in keys})
    mat = np.full((len(ns), len(ps)), np.nan)
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            d = def_vals.get((n, p), [])
            o = opt_vals.get((n, p), [])
            if d and o:
                mat[i, j] = float(np.mean(d)) - float(np.mean(o))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    abs_max = max(1.0, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-abs_max, vmax=abs_max)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels(ps)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(ns)
    ax.set_xlabel("p (features)")
    ax.set_ylabel("n (samples)")
    ax.set_title("F5 — MAE improvement: default − v3-stratified (positive = better)")
    _annotate_signed(ax, mat, abs_max)
    fig.colorbar(im, ax=ax, shrink=0.8, label="ΔMAE")
    fig.tight_layout()
    _save_fig(fig, out_dir / f"figure_v3_f5_delta_mae.{fmt}")


# ── Orchestration ────────────────────────────────────────────────


def run_comparison(
    smoke: bool = False,
    fmt: str = "png",
    output_dir: pathlib.Path | None = None,
    plot_only: bool = False,
    seed: int = 42,
    surrogate_path: pathlib.Path | None = None,
    reps_override: int | None = None,
) -> None:
    grid = SMOKE_GRID if smoke else FULL_GRID
    reps = reps_override or (REPS_SMOKE if smoke else REPS_FULL)
    maxiters = MAXITERS_SMOKE if smoke else MAXITERS
    out_dir = output_dir or V3_RESULTS_DIR / "figures" / "v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = V3_RESULTS_DIR / "v3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    trial_path = cache_dir / "compare_3way_trials.json"
    cov_path = cache_dir / "compare_3way_coverage.json"

    smallp_cfg = _load_v3_family_config(_V3_SMALLP_DIR)
    trans_cfg = _load_v3_family_config(_V3_TRANS_DIR)
    large_cfg = _load_phase31_config()

    # v4: if a surrogate JSON is available, fit and install it as the
    # gating function so v3 picks per-(regime) configs instead of using
    # the hard p-bucket cascade.
    global _SURROGATE_RECOMMEND  # noqa: PLW0603
    if surrogate_path is not None and surrogate_path.exists():
        from ._surrogate import fit_v4_surrogate, recommend_ranked

        LOGGER.info("Fitting regime surrogate from %s …", surrogate_path)
        surr = fit_v4_surrogate(surrogate_path)
        _SURROGATE_RECOMMEND = lambda regime: recommend_ranked(  # noqa: E731
            surr, regime, primary="rank_mae"
        )[0]
        LOGGER.info("Installed rank_mae regime-surrogate gate.")
    else:
        _SURROGATE_RECOMMEND = None
        if surrogate_path is not None:
            LOGGER.warning(
                "Surrogate path %s not found — falling back to bucket gating.",
                surrogate_path,
            )

    if not plot_only:
        LOGGER.info("Running selection grid (default + v3 + sklearn)…")
        trials = _run_grid(
            grid,
            reps,
            smallp_cfg,
            trans_cfg,
            large_cfg,
            seed=seed,
            maxiters=maxiters,
        )
        _save_json(trials, trial_path)

        LOGGER.info("Running coverage grid…")
        cov_results = _run_coverage_grid(
            grid,
            reps,
            smallp_cfg,
            trans_cfg,
            large_cfg,
            seed=seed,
            maxiters=maxiters,
        )
        _save_json(cov_results, cov_path)
    else:
        if not trial_path.exists():
            sys.exit(
                f"--plot-only but {trial_path} not found. Run without --plot-only first."
            )
        trials = _load_trials(trial_path)
        cov_results = _load_coverage(cov_path) if cov_path.exists() else []

    LOGGER.info("Generating figures F1–F5…")
    plot_f1_accuracy(trials, out_dir, fmt)
    plot_f2_errors(trials, out_dir, fmt)
    if cov_results:
        plot_f3_coverage(cov_results, out_dir, fmt)
        plot_f4_rmse(cov_results, out_dir, fmt)
    plot_f5_delta_mae(trials, out_dir, fmt)
    LOGGER.info("Done. Figures in %s", out_dir)


# ── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare VBPCApy default vs v3 regime-stratified vs sklearn",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Reduced grid for quick check"
    )
    parser.add_argument("--fmt", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--plot-only", action="store_true", help="Regenerate figures from saved JSON"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--surrogate-config",
        type=str,
        default=None,
        help=(
            "Path to surrogate_train.json from v4_phase_surrogate_train. "
            "When provided, replaces hard p-bucket gating with a fitted "
            "RegimeSurrogate."
        ),
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Override repetitions per grid cell (default 10 full / 1 smoke).",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    surr_path = pathlib.Path(args.surrogate_config) if args.surrogate_config else None
    print("=" * 60)
    print("V3 COMPARISON: default vs v3-stratified vs sklearn")
    if surr_path is not None:
        print(f"  v4 surrogate gating: {surr_path}")
    print("=" * 60)
    run_comparison(
        smoke=args.smoke,
        fmt=args.fmt,
        output_dir=out_dir,
        plot_only=args.plot_only,
        seed=args.seed,
        surrogate_path=surr_path,
        reps_override=args.reps,
    )


if __name__ == "__main__":
    main()
