"""Translate trade-study configurations into VBPCA estimator kwargs."""

from __future__ import annotations

import copy
from typing import Any


def build_vbpca_kwargs(
    config: dict[str, Any],
    *,
    random_state: int,
    criterion_order_levels: dict[str, list[str]],
    active_criteria_presets: dict[str, dict[str, bool]],
) -> dict[str, Any]:
    """Resolve search-factor and public-estimator config representations.

    The search design stores RMS stopping and enabled criteria as separate
    factor fields, whereas :func:`vbpca_py.recommend_config` returns the
    estimator-ready ``rmsstop`` and ``convergence_criteria`` values.  Both
    representations are supported, but supplying both forms is ambiguous and
    therefore rejected.

    Returns:
        Keyword arguments accepted by ``VBPCA`` and
        ``select_n_components``.

    Raises:
        ValueError: If both representations of a compound option are present.
    """
    kwargs: dict[str, Any] = {"verbose": 0, "random_state": random_state}
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
        if key in config and config[key] is not None:
            kwargs[key] = config[key]

    has_factorized_rms = any(
        key in config for key in ("rmsstop_window", "rmsstop_atol", "rmsstop_rtol")
    )
    if "rmsstop" in config and has_factorized_rms:
        msg = "supply either rmsstop or rmsstop_window/atol/rtol, not both"
        raise ValueError(msg)
    if "rmsstop" in config:
        kwargs["rmsstop"] = copy.deepcopy(config["rmsstop"])
    elif has_factorized_rms:
        if "rmsstop_window" not in config:
            msg = "rmsstop_atol/rtol require rmsstop_window"
            raise ValueError(msg)
        kwargs["rmsstop"] = [
            config["rmsstop_window"],
            config.get("rmsstop_atol", 1e-4),
            config.get("rmsstop_rtol", 1e-3),
        ]

    if "criterion_order" in config and config["criterion_order"] is not None:
        criterion_order = config["criterion_order"]
        kwargs["criterion_order"] = copy.deepcopy(
            criterion_order_levels[criterion_order]
            if isinstance(criterion_order, str)
            and criterion_order in criterion_order_levels
            else criterion_order
        )

    if "convergence_criteria" in config and "active_criteria" in config:
        msg = "supply either convergence_criteria or active_criteria, not both"
        raise ValueError(msg)
    if "convergence_criteria" in config:
        kwargs["convergence_criteria"] = copy.deepcopy(config["convergence_criteria"])
    elif "active_criteria" in config and config["active_criteria"] is not None:
        active_criteria = config["active_criteria"]
        kwargs["convergence_criteria"] = copy.deepcopy(
            active_criteria_presets[active_criteria]
            if isinstance(active_criteria, str)
            and active_criteria in active_criteria_presets
            else active_criteria
        )

    return kwargs
