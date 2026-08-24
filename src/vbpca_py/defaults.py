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

**Validated range (#116) and extreme-aspect-ratio buckets (#120):** the
dense factorial grid behind ``smallp``/``trans``/``large`` only covers
``p`` up to 200 and ``p/n`` up to 2.0 (``n/p`` up to 15.0 in the other
direction) — outside that, e.g. small-cohort, thousands-of-features
genomics data (``p/n`` of 50-1000x) or large-cohort, few-variable
ecological/survey data (``n/p`` of 20-100x), the "large" bucket's config
does not transfer (wrong rank recovered entirely, confirmed empirically).
``recommend_config`` now routes those regimes to one of four additional
buckets (``wide_moderate``/``wide_extreme``/``tall_moderate``/
``tall_extreme``) derived by adaptive (NSGA-II) search over VBPCA's full
hyperparameter space at one representative example regime per bucket
(``analysis/trade_study/option_a_aspect_ratio.py``) rather than the dense
grid the original three buckets were tuned and replicated against — a
real, validated recommendation for that aspect ratio, just a coarser one
(warned about accordingly).

The returned dict is intended to be splatted into the estimator, e.g.::

    from vbpca_py import VBPCA, recommend_config

    cfg = recommend_config(n=200, p=100)
    model = VBPCA(n_components=10, **cfg)
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Literal

__all__ = ["recommend_config"]

Priority = Literal["balanced", "accuracy", "speed"]

# All six VBPCA convergence criteria enabled -- matches the trade study's
# "all" active_criteria preset, used by every extreme-aspect-ratio bucket
# below (#120).
_ALL_CRITERIA_TRUE: dict[str, bool] = {
    "angle": True,
    "earlystop": True,
    "rms_plateau": True,
    "cost": True,
    "composite": True,
    "slowing_down": True,
}

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
    # Extreme-aspect-ratio buckets (#116/#120): derived by NSGA-II search
    # over VBPCA's full ~15-parameter space at a handful of representative
    # example regimes (analysis/trade_study/option_a_aspect_ratio.py),
    # not the dense factorial grid smallp/trans/large were tuned and
    # validated against -- see the module docstring's "Validated range"
    # note. That richer search is also why these dicts carry more keys
    # (patience, criterion_order, convergence_criteria, rmsstop, minangle,
    # cfstop_rel) than smallp/trans/large -- VBPCA fills in its own
    # defaults for anything a bucket doesn't specify, so the differing
    # key sets don't matter for `VBPCA(n_components=k, **cfg)` usage.
    # criterion_order/convergence_criteria are VBPCA constructor kwargs in
    # their own right (a list and a dict respectively) -- the search
    # explored them via named presets
    # (analysis/trade_study/_common.py's CRITERION_ORDER_LEVELS/
    # ACTIVE_CRITERIA_PRESETS), already resolved to real values below so
    # every key here is splat-ready, not a preset name needing further
    # translation.
    "wide_extreme": {  # validated at bulk_rnaseq: n=30, p=2000, p/n=66.7
        "hp_va": 0.5031412272175019,
        "hp_vb": 0.8564912762899104,
        "hp_v": 0.6586970446826288,
        "va_init": 1637.7149265434825,
        "xprobe_fraction": 0.01764218685010746,
        "niter_broadprior": 100,
        "criterion_order": [
            "cost",
            "angle",
            "rms_plateau",
            "composite",
            "earlystop",
            "slowing_down",
        ],
        "convergence_criteria": _ALL_CRITERIA_TRUE,
        "maxiters": 500,
        "patience": 3,
        "rmsstop": [50, 0.008774842663257003, 0.007353359727600819],
        "minangle": 6.967374686269853e-05,
        "cfstop_rel": 0.0006289465573514163,
    },
    # validated at microbiome (n=50, p=300, p/n=6) and single_cell
    # (n=500, p=500, p/n=1 -- large in absolute p, not aspect ratio);
    # both independently converged to this same config during search.
    "wide_moderate": {
        "hp_va": 0.5487383020286924,
        "hp_vb": 0.6918982787407163,
        "hp_v": 0.6519647398900055,
        "va_init": 2250.4504015109924,
        "xprobe_fraction": 0.17804480533688397,
        "niter_broadprior": 200,
        "criterion_order": [
            "angle",
            "earlystop",
            "rms_plateau",
            "cost",
            "composite",
            "slowing_down",
        ],
        "convergence_criteria": _ALL_CRITERIA_TRUE,
        "maxiters": 200,
        "patience": 2,
        "rmsstop": [200, 0.003702216843854189, 0.0001644115991233856],
        "minangle": 9.539286230740105e-05,
        "cfstop_rel": 0.0009148652415765465,
    },
    "tall_extreme": {  # validated at ecological: n=3000, p=30, n/p=100
        "hp_va": 0.695977246351637,
        "hp_vb": 0.40895885488482575,
        "hp_v": 0.1733025871276451,
        "va_init": 1572.8060562841497,
        "xprobe_fraction": 0.06256072454114883,
        "niter_broadprior": 200,
        "criterion_order": [
            "angle",
            "earlystop",
            "rms_plateau",
            "cost",
            "composite",
            "slowing_down",
        ],
        "convergence_criteria": _ALL_CRITERIA_TRUE,
        "maxiters": 100,
        "patience": 2,
        "rmsstop": [200, 0.0037467515336500863, 0.006262340557985221],
        "minangle": 6.35094015773993e-05,
        "cfstop_rel": 4.53135567319468e-05,
    },
    "tall_moderate": {  # validated at cultural: n=1000, p=50, n/p=20
        "hp_va": 0.5487383020286924,
        "hp_vb": 0.6918982787407163,
        "hp_v": 0.6519647398900055,
        "va_init": 2250.4504015109924,
        "xprobe_fraction": 0.17804480533688397,
        "niter_broadprior": 200,
        "criterion_order": [
            "angle",
            "earlystop",
            "rms_plateau",
            "cost",
            "composite",
            "slowing_down",
        ],
        "convergence_criteria": _ALL_CRITERIA_TRUE,
        "maxiters": 200,
        "patience": 2,
        "rmsstop": [200, 0.003702216843854189, 0.0001644115991233856],
        "minangle": 9.539286230740105e-05,
        "cfstop_rel": 0.0009148652415765465,
    },
}

_SMALLP_MAX_P = 30
_TRANS_MAX_P = 70

# Widest values actually present in the ORIGINAL Option A trade study's
# regime grid, analysis/trade_study -- beyond these, smallp/trans/large
# extrapolate rather than interpolate, so _bucket() instead routes to a
# wide_moderate/tall_moderate bucket (#120).
_MAX_VALIDATED_P = 200
_MAX_VALIDATED_P_OVER_N = 2.0
_MAX_VALIDATED_N_OVER_P = 15.0

# Beyond these, even wide_moderate/tall_moderate are themselves
# extrapolating past their own validated example regime -- _bucket()
# routes to wide_extreme/tall_extreme instead (still just one validated
# point each, per #120, but closer than reusing a moderate-ratio bucket).
_WIDE_EXTREME_RATIO = 10.0
_TALL_EXTREME_RATIO = 50.0

# Buckets outside the dense, thoroughly-validated smallp/trans/large grid
# -- recommend_config() warns when it returns one of these (#120).
_ASPECT_RATIO_BUCKETS = frozenset({
    "wide_moderate", "wide_extreme", "tall_moderate", "tall_extreme",
})  # fmt: skip


def _p_only_bucket(p: int) -> str:
    """Return the feature-count-only bucket for ``p`` features."""
    if p <= _SMALLP_MAX_P:
        return "smallp"
    if p <= _TRANS_MAX_P:
        return "trans"
    return "large"


def _bucket(n: int, p: int) -> str:
    """Return the recommendation bucket for an ``(n, p)`` data regime."""
    p_over_n = p / n
    if p_over_n > _WIDE_EXTREME_RATIO:
        return "wide_extreme"
    if p > _MAX_VALIDATED_P or p_over_n > _MAX_VALIDATED_P_OVER_N:
        return "wide_moderate"
    n_over_p = n / p
    if n_over_p > _TALL_EXTREME_RATIO:
        return "tall_extreme"
    if n_over_p > _MAX_VALIDATED_N_OVER_P:
        return "tall_moderate"
    return _p_only_bucket(p)


def recommend_config(
    n: int,
    p: int,
    *,
    missingness: str = "auto",
    priority: Priority = "balanced",
) -> dict[str, Any]:
    """Recommend VBPCA keyword arguments for a data regime.

    Args:
        n: Number of samples (columns of the ``p x n`` data matrix). Used
            (alongside ``p``) to select an aspect-ratio-aware bucket when
            ``p/n`` or ``n/p`` falls outside the ``smallp``/``trans``/
            ``large`` buckets' validated range (#116/#120) -- otherwise
            only ``p`` selects the bucket.
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
        UserWarning: If ``(n, p)`` resolves to a ``wide_moderate``/
            ``wide_extreme``/``tall_moderate``/``tall_extreme`` bucket
            rather than ``smallp``/``trans``/``large`` -- those four are
            each derived from adaptive search at a single representative
            example regime (#120: bulk_rnaseq, microbiome + single_cell,
            ecological, cultural respectively), not the dense factorial
            grid ``smallp``/``trans``/``large`` were tuned and replicated
            against (#111). Still a real, validated recommendation for
            that specific aspect ratio -- just a coarser one.
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

    bucket = _bucket(n, p)
    if bucket in _ASPECT_RATIO_BUCKETS:
        warnings.warn(
            f"recommend_config(n={n}, p={p}) uses the {bucket!r} bucket: "
            f"derived from adaptive search at a single representative "
            f"example regime for this aspect ratio (#120), not the dense "
            f"factorial grid smallp/trans/large were tuned and replicated "
            f"against (#111). Treat as a coarser approximation. See "
            f"https://github.com/yoavram-lab/VBPCApy/issues/116.",
            UserWarning,
            stacklevel=2,
        )

    cfg = copy.deepcopy(_BUCKET_CONFIGS[bucket])

    if priority == "speed":
        cfg["maxiters"] = 100
        cfg["niter_broadprior"] = 0
    elif priority == "accuracy":
        cfg["maxiters"] = int(cfg["maxiters"]) * 2

    return cfg
