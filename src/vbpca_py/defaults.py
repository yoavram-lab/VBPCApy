"""Regime-aware recommended configurations for VBPCA.

:func:`recommend_config` returns :class:`~vbpca_py.estimators.VBPCA` keyword
arguments tuned to maximise rank recovery (minimise ``rank_mae``) while keeping
reconstruction error within a small tolerance of the library defaults.

These configurations come from the Option A regime-surrogate trade study
(``analysis/trade_study``).  The dominant lever is a **strong ARD loadings
prior** (``hp_va`` ~ 0.7 versus the library default 0.001): it drives correct
component pruning, whereas the weak default prior under-prunes and recovers the
true rank only about a third of the time.  Recommendations are bucketed by the
feature count ``p`` — the study's primary axis of rank-recovery difficulty.

The returned dict is intended to be splatted into the estimator, e.g.::

    from vbpca_py import VBPCA, recommend_config

    cfg = recommend_config(n=200, p=100)
    model = VBPCA(n_components=10, **cfg)
"""

from __future__ import annotations

from typing import Any, Literal

__all__ = ["recommend_config"]

Priority = Literal["balanced", "accuracy", "speed"]

# Per-bucket baked recommendations (median of the surrogate's per-regime
# rank_mae-optimal configs within each p-bucket).
_BUCKET_CONFIGS: dict[str, dict[str, Any]] = {
    "smallp": {
        "hp_va": 0.70,
        "hp_vb": 0.35,
        "hp_v": 0.40,
        "va_init": 5000.0,
        "niter_broadprior": 0,
        "maxiters": 300,
        "xprobe_fraction": 0.01,
    },
    "trans": {
        "hp_va": 0.75,
        "hp_vb": 0.40,
        "hp_v": 0.35,
        "va_init": 5000.0,
        "niter_broadprior": 50,
        "maxiters": 300,
        "xprobe_fraction": 0.01,
    },
    "large": {
        "hp_va": 0.65,
        "hp_vb": 0.55,
        "hp_v": 0.20,
        "va_init": 2500.0,
        "niter_broadprior": 50,
        "maxiters": 400,
        "xprobe_fraction": 0.02,
    },
}

_SMALLP_MAX_P = 30
_TRANS_MAX_P = 70


def _bucket(p: int) -> str:
    """Return the feature-count bucket for ``p`` features."""
    if p <= _SMALLP_MAX_P:
        return "smallp"
    if p <= _TRANS_MAX_P:
        return "trans"
    return "large"


def recommend_config(
    n: int,
    p: int,
    *,
    missingness: str = "auto",  # noqa: ARG001 - reserved for future tuning
    priority: Priority = "balanced",
) -> dict[str, Any]:
    """Recommend VBPCA keyword arguments for a data regime.

    Args:
        n: Number of samples (columns of the ``p x n`` data matrix).
        p: Number of features (rows).  Selects the recommendation bucket.
        missingness: Missingness descriptor.  Reserved for future
            regime-specific tuning; accepted but not yet branched on.
        priority: Trade-off preset.  ``"balanced"`` (default) uses the
            tuned configuration as-is; ``"accuracy"`` raises the iteration
            budget; ``"speed"`` lowers it and disables the broad-prior
            warm-up.

    Returns:
        A dict of keyword arguments suitable for
        ``VBPCA(n_components=..., **cfg)`` or ``select_n_components``.

    Raises:
        ValueError: If ``n`` or ``p`` is not positive, or if ``priority``
            is not a recognised preset.
    """
    if n <= 0 or p <= 0:
        msg = f"n and p must be positive; got n={n}, p={p}"
        raise ValueError(msg)
    if priority not in {"balanced", "accuracy", "speed"}:
        msg = f"unknown priority {priority!r}"
        raise ValueError(msg)

    cfg = dict(_BUCKET_CONFIGS[_bucket(p)])

    if priority == "speed":
        cfg["maxiters"] = 100
        cfg["niter_broadprior"] = 0
    elif priority == "accuracy":
        cfg["maxiters"] = int(cfg["maxiters"]) * 2

    return cfg
