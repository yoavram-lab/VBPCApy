"""Samples-by-features component selection with an explicit legacy boundary.

Cellwise probe metrics retain their existing definitions and feature-first
numerical implementation. They are distinct from native row-wise scoring.
"""

from __future__ import annotations

from collections.abc import Iterable  # noqa: TC003 - public annotation resolution
from typing import cast

import scipy.sparse as sp

from . import _selection_core
from ._selection_core import (
    RANK_ZERO_SELECTION_SUPPORTED,
    CVConfig,
    SelectionConfig,
)
from ._types import Float64Array, Matrix  # noqa: TC001 - runtime annotation resolution
from ._validation import require_training_observations, validate_rank
from .estimators import VBPCA, _transpose_matrix
from .legacy import VBPCA as LegacyVBPCA  # noqa: N811, TC001 - runtime annotation resolution

__all__ = [
    "RANK_ZERO_SELECTION_SUPPORTED",
    "CVConfig",
    "SelectionConfig",
    "cross_validate_components",
    "select_n_components",
]


def _selection_input(
    x: object, mask: object | None, options: dict[str, object]
) -> tuple[Matrix, VBPCA, dict[str, object]]:
    """Validate native observations and adapt probe axes once.

    Returns:
        Native data, the validation estimator and feature-first fit options.

    Raises:
        ValueError: Observations, masks, probes or options are invalid.
    """
    fit_options = dict(options)
    probe = fit_options.pop("xprobe", None)
    fit_options.setdefault("compat_mode", "modern")
    constructor = dict(fit_options)
    constructor.pop("selection_verbose", None)
    template = VBPCA(0, **constructor)  # type: ignore[arg-type]
    template.get_config()
    if not 0 <= template.xprobe_fraction < 1:
        msg = "xprobe_fraction must be in [0, 1)"
        raise ValueError(msg)
    data = template._validate_observations(x, mask, reset=True)  # noqa: SLF001 - shared native boundary
    native_probe = None
    if probe is not None:
        native_probe = template._validate_observations(  # noqa: SLF001
            probe, None, reset=False
        )
        if native_probe.shape != data.shape or sp.issparse(native_probe) != sp.issparse(
            data
        ):
            msg = "xprobe must match X shape and dense/sparse representation"
            raise ValueError(msg)
        fit_options["xprobe"] = _transpose_matrix(native_probe)
    require_training_observations(data, native_probe)
    return data, template, fit_options


def _native_capacities(
    components: Iterable[int] | None, shape: tuple[int, int]
) -> list[int] | None:
    """Validate explicit capacities without guessing or truncating their values.

    Returns:
        Validated capacities, or ``None`` for the existing default grid.

    Raises:
        ValueError: Capacities are empty, nonintegral, negative or too large.
    """
    if components is None:
        return None
    values = [validate_rank(k, shape) for k in components]
    if not values:
        msg = "components must contain at least one nonnegative integer"
        raise ValueError(msg)
    return values


def _native_selected_model(source: LegacyVBPCA, template: VBPCA) -> VBPCA:
    """Publish a selected numerical fit through the native estimator contract.

    The selected fit is adopted without rerunning inference or retaining a
    compatibility estimator. Covariance blocks keep their internal axes;
    public loadings, scores, means and dense diagnostics use native axes.

    Returns:
        A fitted native estimator with the original input-name contract.

    Raises:
        RuntimeError: The selected numerical fit has incomplete loading state.
    """
    if (
        source.components_ is None
        or source.scores_ is None
        or source.mean_ is None
        or source.components_.shape[0] != template.n_features_in_
    ):
        msg = "Selected numerical fit does not cover the validated feature layout"
        raise RuntimeError(msg)
    model = VBPCA(**source.get_params(deep=False))  # type: ignore[arg-type]
    model.n_features_in_ = template.n_features_in_
    if hasattr(template, "feature_names_in_"):
        model.feature_names_in_ = template.feature_names_in_.copy()
    model.components_ = source.components_.T
    model.scores_ = source.scores_.T
    model.mean_ = source.mean_.reshape(-1)
    model.noise_variance_ = cast("float", source.noise_variance_)
    model.rms_ = cast("float", source.rms_)
    model.prms_ = cast("float", source.prms_)
    model.cost_ = cast("float", source.cost_)
    model.learning_curve_ = source.learning_curve_
    model.n_iter_ = cast("int", source.n_iter_)
    model.convergence_reason_ = cast("str", source.convergence_reason_)
    model.converged_ = bool(source.converged_)
    model.best_probe_iteration_ = source.best_probe_iteration_
    model.best_probe_rms_ = source.best_probe_rms_
    model.returned_iteration_ = source.returned_iteration_
    model.runtime_report_ = source.runtime_report_
    model.reconstruction_ = (
        source.reconstruction_.T if source.reconstruction_ is not None else None
    )
    model.variance_ = source.variance_.T if source.variance_ is not None else None
    model.predictive_variance_ = (
        source.predictive_variance_.T
        if source.predictive_variance_ is not None
        else None
    )
    model.explained_variance_ = source.explained_variance_
    model.explained_variance_ratio_ = source.explained_variance_ratio_
    model._av = source._av  # noqa: SLF001 - internal covariance axes are unchanged
    model._sv = source._sv or []  # noqa: SLF001
    model._muv = cast("Float64Array", source._muv)  # noqa: SLF001
    model._pattern_index = source._pattern_index  # noqa: SLF001
    model.prior_variances_ = cast("Float64Array", source.prior_variances_)
    model.bias_prior_variance_ = cast("float", source.bias_prior_variance_)
    model.component_relevance_ = cast("Float64Array", source.component_relevance_)
    model.prior_trace_ = source.prior_trace_
    return model


def select_n_components(
    x: object,
    *,
    mask: object | None = None,
    components: Iterable[int] | None = None,
    config: SelectionConfig | None = None,
    **opts: object,
) -> tuple[int, dict[str, object], list[dict[str, object]], VBPCA | None]:
    """Select capacity from samples-by-features observations and aligned probes.

    Args:
        x: Native samples-by-features numeric observations, including named
            tables or sparse stored-entry observations. NaNs are missing.
        mask: Optional aligned observation mask; numeric zeros remain observed.
        components: Nonnegative integral capacities within both dimensions.
            Omission retains the positive default grid; explicit zero is a
            closed-form mean-only candidate.
        config: Existing endpoint metric, stopping and diagnostic policy.
        **opts: Native fitting options. An explicit ``xprobe`` has the same
            samples-by-features layout and feature names as ``x``. Warm-start
            payloads retain their documented low-level axes. Modern mode is
            the default; ``selection_verbose`` controls sweep logging.

    Returns:
        ``(best_k, best_metrics, trace, best_model)``. A requested positive-rank
        best model is a native VBPCA with conditional new-data projection;
        rank zero retains the existing ``None`` model result. Requested dense
        diagnostics use samples-by-features axes and the existing byte budget.

    Invalid inputs, capacities, metrics or diagnostics raise ValueError.
    Cellwise PRMS/cost selection retains its existing target and fold/probe
    definitions. Row-wise sklearn scoring evaluates a different target.
    ``vbpca_py.legacy.select_n_components`` preserves feature-first callers.
    """
    data, template, fit_options = _selection_input(x, mask, opts)
    capacities = _native_capacities(components, data.shape)
    best, metrics, trace, selected = _selection_core.select_n_components(
        _transpose_matrix(data),
        mask=None,
        components=capacities,
        config=config,
        **fit_options,
    )
    model = _native_selected_model(selected, template) if selected is not None else None
    return best, metrics, trace, model


def cross_validate_components(
    x: object,
    *,
    mask: object | None = None,
    components: Iterable[int] | None = None,
    config: CVConfig | None = None,
    **opts: object,
) -> tuple[int, list[dict[str, object]]]:
    """Select capacity using existing cellwise CV on samples-by-features input.

    Args:
        x: Dense samples-by-features numeric observations; NaNs are missing.
        mask: Optional aligned observation mask.
        components: Nonnegative integral capacities within both dimensions.
        config: Cellwise fold, metric and selection policy. Feature groups and
            encoding-schema columns refer to the input feature axis.
        **opts: Fitting options, with modern mode as the default. This helper
            constructs its own probes; do not supply ``xprobe`` or a positive
            ``xprobe_fraction``. Warm starts retain their low-level axes.

    Returns:
        Selected capacity and the existing aggregate/per-fold metric records.
        Full-grid one-SE and optional first-minimum selection remain distinct.

    Raises:
        ValueError: Inputs, capacities, fold support or metric rules are invalid.

    This is cellwise capacity CV, separate from sklearn's row-wise CV and the
    estimator's marginal likelihood score. Use the explicit legacy helper for
    historical feature-first inputs and defaults.
    """
    if "xprobe" in opts or cast("float", opts.get("xprobe_fraction", 0.0)) != 0:
        msg = "cellwise CV constructs its own probes; do not supply probe options"
        raise ValueError(msg)
    data, _template, fit_options = _selection_input(x, mask, opts)
    return _selection_core.cross_validate_components(
        _transpose_matrix(data),
        mask=None,
        components=_native_capacities(components, data.shape),
        config=config,
        **fit_options,
    )
