"""Regime-aware recommended configurations for VBPCA.

:func:`recommend_config` returns :class:`~vbpca_py.estimators.VBPCA` keyword
arguments tuned to maximise rank recovery (minimise ``rank_mae``) while keeping
reconstruction error within a small tolerance of the library defaults.

These configurations come from the Option A regime-surrogate trade study
(``analysis/trade_study``), which fits a random-forest surrogate over the
full joint factor space per regime and recommends the joint-optimal
config. The joint-optimal configs use a moderately strong ARD loadings
prior (``hp_va`` ~ 0.65-0.75 vs. the library default 0.001) alongside a
small xprobe fraction (~0.01-0.02); the weak default prior under-prunes
and recovers the true rank only about a third of the time.

**On "dominant lever" claims:** a marginal (univariate Spearman)
sensitivity analysis of the same trade study's data does *not* single out
``hp_va`` as dominant — ``xprobe_fraction`` is the strongest, most
significant, and most consistent per-bucket predictor of ``rank_mae`` in
that marginal view, and ``hp_va``'s marginal correlation is weak and not
statistically significant in 2 of the 3 p-buckets (though its tercile
means do show a real, nonlinear U-shape a monotonic Spearman correlation
understates). The RF surrogate optimises interactions a marginal view
can't see, so which single factor (if any) is "dominant" remains
unreconciled; treat the recommendation as a joint-optimal bundle rather
than attributing its effect to any one factor.

**Validation status (#111):** the exact configs this module ships *are*
now validated with real replication and seeded VBPCA initialization
(``analysis/trade_study/validate_shipped_defaults.py``, n_reps=8 across
training + held-out regimes) — replicated ``rank_mae`` for the shipped
config is 28-58% lower than the library default across all three
p-buckets (smallp 0.34→0.14, trans 1.20→0.74, large 1.44→1.04), at a
small cost in holdout RMSE (+0.4-3.8%). What remains unvalidated is the
*attribution* to a specific factor, not whether the shipped bundle helps.
Recommendations are bucketed by the feature count ``p`` — the study's
primary axis of rank-recovery difficulty.

**Validated range (#116):** the regime grid behind these buckets only
covers ``p`` up to 200 and ``p/n`` up to 2.0 — ``n`` itself never enters
the bucketing decision, so nothing distinguishes a balanced 100x100
matrix from a small-cohort, thousands-of-features genomics matrix
(``p/n`` of 50-1000x) once ``p`` exceeds the "large" bucket's threshold
of 70. Empirically the "large" bucket's config does not transfer to that
shape (wrong rank recovered entirely); ``recommend_config`` warns when
``p``/``p over n`` fall outside the validated region, but there's no
tuned alternative to fall back to yet.

The returned dict is intended to be splatted into the estimator, e.g.::

    from vbpca_py import VBPCA, recommend_config

    cfg = recommend_config(n=200, p=100)
    model = VBPCA(n_components=10, **cfg)
"""

from __future__ import annotations

import warnings
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

# Widest values actually present in the Option A trade study's regime
# grid, analysis/trade_study -- recommendations for anything past these
# bounds extrapolate rather than interpolate.
_MAX_VALIDATED_P = 200
_MAX_VALIDATED_P_OVER_N = 2.0


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
    missingness: str = "auto",
    priority: Priority = "balanced",
) -> dict[str, Any]:
    """Recommend VBPCA keyword arguments for a data regime.

    Args:
        n: Number of samples (columns of the ``p x n`` data matrix).
        p: Number of features (rows).  Selects the recommendation bucket.
        missingness: Missingness descriptor. **Not currently branched on** —
            recommendations are bucketed by ``p`` only. The Option A trade
            study evaluates missingness as a regime feature, but its example
            recommendations span only 4 missingness categories x 3 p-buckets
            from 23 design points total (some cells have a single point), too
            sparse to bucket on without shipping unreplicated, effectively
            arbitrary per-cell values (see #111). Passing anything other than
            the default ``"auto"`` raises a :class:`UserWarning` rather than
            silently doing nothing. Tracked in #110, blocked on trade-study
            replication support (`jcm-sci/trade-study#112
            <https://github.com/jcm-sci/trade-study/issues/112>`_).
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

    Warns:
        UserWarning: If ``p`` or the ``p/n`` aspect ratio falls outside
            what the Option A trade study's regime grid covered (``p`` up
            to 200, ``p/n`` up to 2.0). Buckets are keyed on ``p`` alone
            with no upper bound, so e.g. genomics-scale data (small
            cohorts, thousands of features -- ``p/n`` of 50-1000x) gets
            the same config as a balanced 100x100 matrix despite being
            nowhere near the validated region; empirically this can pick
            the wrong rank entirely (see #116). There's no shape-aware
            recommendation to fall back to yet -- this only flags that
            the one returned is an extrapolation, not a fix.
    """
    if n <= 0 or p <= 0:
        msg = f"n and p must be positive; got n={n}, p={p}"
        raise ValueError(msg)
    if priority not in {"balanced", "accuracy", "speed"}:
        msg = f"unknown priority {priority!r}"
        raise ValueError(msg)
    if missingness != "auto":
        warnings.warn(
            f"recommend_config() does not yet branch on missingness "
            f"(got missingness={missingness!r}); recommendations are "
            f"bucketed by p only. See "
            f"https://github.com/yoavram-lab/VBPCApy/issues/110.",
            UserWarning,
            stacklevel=2,
        )
    if p > _MAX_VALIDATED_P or p / n > _MAX_VALIDATED_P_OVER_N:
        warnings.warn(
            f"recommend_config(n={n}, p={p}) falls outside the Option A "
            f"trade study's validated region (p up to {_MAX_VALIDATED_P}, "
            f"p/n up to {_MAX_VALIDATED_P_OVER_N}); the bucketed "
            f"recommendation is an extrapolation and has been observed to "
            f"pick the wrong rank at extreme p/n ratios (e.g. small-cohort "
            f"genomics data). See "
            f"https://github.com/yoavram-lab/VBPCApy/issues/116.",
            UserWarning,
            stacklevel=2,
        )

    cfg = dict(_BUCKET_CONFIGS[_bucket(p)])

    if priority == "speed":
        cfg["maxiters"] = 100
        cfg["niter_broadprior"] = 0
    elif priority == "accuracy":
        cfg["maxiters"] = int(cfg["maxiters"]) * 2

    return cfg
