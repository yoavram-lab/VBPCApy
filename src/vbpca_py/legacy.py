"""Explicit feature-first compatibility API for pre-0.5 callers.

This adapter preserves the historical training-score transform and result
layouts. It is not the native samples-first Pipeline estimator.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping  # noqa: TC003 - public runtime type-hint resolution
from dataclasses import replace
from pathlib import Path  # noqa: TC003 - public runtime type-hint resolution
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from vbpca_py._memory import (
    check_dense_output_budget,
    exceeds_budget,
    format_bytes,
    resolve_max_dense_bytes,
)
from vbpca_py._missing import make_xprobe_mask
from vbpca_py._native import TypedEstimator
from vbpca_py._pca_full import Matrix, pca_full
from vbpca_py._selection_core import (
    CVConfig,
    SelectionConfig,
    cross_validate_components,
    select_n_components,
)
from vbpca_py.configuration import PCAOptions
from vbpca_py.options import (
    AccessorMode,
    Algorithm,
    BiasUpdateOrder,
    CompatMode,
    CostDiagnostics,
    CovarianceWriteback,
    ExplainedVarianceSolver,
    RuntimeTuning,
    StatisticsMode,
    VarianceUpdateOrder,
)

from ._types import BoolArray, Float64Array, IndexArray, InputMask, InputMatrix  # noqa: TC001 - runtime annotation resolution

if TYPE_CHECKING:
    from vbpca_py.results import LearningCurve

__all__ = [
    "VBPCA",
    "CVConfig",
    "SelectionConfig",
    "cross_validate_components",
    "select_n_components",
]

_TOL_DEPRECATION_MESSAGE = (
    "tol is deprecated and has no effect; configure rmsstop, "
    "cfstop_rel, or minangle explicitly"
)


def _warn_deprecated_tol(tol: float | None) -> None:
    """Warn when the retained compatibility parameter is set."""
    if tol is not None:
        warnings.warn(_TOL_DEPRECATION_MESSAGE, FutureWarning, stacklevel=3)


class VBPCA(TypedEstimator):
    """Explicit feature-first adapter for the pre-0.5 interface.

    Examples:
        New native workflows use samples by features. Historical calls use
        this adapter explicitly; numerical compatibility is a separate choice.

        >>> import numpy as np
        >>> from vbpca_py import VBPCA as NativeVBPCA
        >>> from vbpca_py.legacy import VBPCA as LegacyVBPCA
        >>> x = np.random.default_rng(19).normal(size=(12, 3))
        >>> options = dict(
        ...     maxiters=4,
        ...     niter_broadprior=0,
        ...     random_state=19,
        ...     num_cpu=1,
        ...     runtime_tuning="off",
        ...     verbose=0,
        ... )
        >>> native = NativeVBPCA(1, **options).fit(x)
        >>> adapted = LegacyVBPCA(1, compat_mode="modern", **options).fit(x.T)
        >>> native.components_.shape, adapted.components_.shape
        ((1, 3), (3, 1))
        >>> np.testing.assert_allclose(native.components_, adapted.components_.T)
        >>> LegacyVBPCA(1).compat_mode
        'strict_legacy'

        This four-iteration example does not qualify convergence or old saved
        models. Old serialized native layouts require the documented migration.
    """

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        n_components: int,
        *,
        bias: bool = True,
        maxiters: int | None = None,
        tol: float | None = None,
        verbose: int | bool = 0,
        hp_va: float | None = None,
        hp_vb: float | None = None,
        hp_v: float | None = None,
        niter_broadprior: int | None = None,
        va_init: float | None = None,
        xprobe_fraction: float = 0.0,
        criterion_order: list[str] | None = None,
        convergence_criteria: dict[str, bool] | None = None,
        random_state: int | np.random.Generator | None = None,
        init: object = "random",
        max_dense_bytes: int | None = 2_000_000_000,
        uniquesv: bool = False,
        auto_pattern_masked: bool = False,
        angle_every: int = 1,
        minangle: float = 1e-8,
        algorithm: Algorithm | str = "vb",
        earlystop: bool = False,
        rmsstop: npt.ArrayLike | None = (100.0, 1e-4, 1e-3),
        cfstop: npt.ArrayLike | None = (),
        cfstop_rel: float | None = None,
        cfstop_curv: float | None = None,
        record_cost: bool = False,
        cost_diagnostics: CostDiagnostics | str = "trace",
        record_prior_trace: bool = False,
        composite_stop: Mapping[str, float] | None = None,
        patience: int = 1,
        num_cpu: int | None = None,
        num_cpu_score_update: int | None = None,
        num_cpu_loadings_update: int | None = None,
        num_cpu_noise_update: int | None = None,
        num_cpu_rms: int | None = None,
        runtime_tuning: RuntimeTuning | str = "safe",
        runtime_profile: str | Path | None = None,
        dense_sufficient_statistics: StatisticsMode | str = "auto",
        noise_sufficient_statistics: StatisticsMode | str = "auto",
        masked_batch_size: int = 0,
        cov_writeback_mode: CovarianceWriteback | str | None = None,
        log_progress_stride: int | None = None,
        accessor_mode: AccessorMode | str | None = None,
        rotate2pca: bool = True,
        display: bool = False,
        bias_update_order: BiasUpdateOrder | str = "auto",
        variance_update_order: VarianceUpdateOrder | str = "auto",
        compat_mode: CompatMode | str = "strict_legacy",
        return_diagnostics: bool = True,
        runtime_report: bool = False,
        explained_var_solver: ExplainedVarianceSolver | str = "auto",
        explained_var_gram_ratio: float = 4.0,
    ) -> None:
        """
        Initialize the estimator with common VB-PCA options.

        Args:
            n_components: Number of principal components to infer.
            bias: If True, include a bias (mean) term in the model.
            maxiters: Maximum number of iterations for the training loop.
            tol: Deprecated compatibility parameter. It has no effect; use
                an explicit convergence criterion such as ``rmsstop``,
                ``cfstop_rel``, or ``minangle`` instead.
            verbose: Verbosity level; can be an integer or a boolean.
            hp_va: Hyperprior term in the automatic relevance determination
                (ARD) update of each component's loading prior variance,
                ``Va[k] = (|a_k|^2 + tr(cov a_k) + 2 hp_va) /
                ((n_features + 2 hp_vb) / observed_fraction)``. Small
                values let unused components shrink toward zero (strong
                pruning); larger values put a floor under every ``Va[k]``
                (default 0.001). Also enters the bias prior variance.
            hp_vb: Hyperprior term in the denominator of the same update;
                larger values shrink every component's prior variance
                (default 0.001).
            hp_v: Hyperprior term in the noise variance update,
                ``V = (residual + 2 hp_v) / (n_observed + 2 hp_v)``
                (default 0.001).
            niter_broadprior: Number of initial iterations with the loading
                prior variances held at ``va_init`` before ARD updates start;
                convergence checks also wait for it (default 100).
            va_init: Initial (broad) prior variance for loadings and bias
                (default 1000).
            xprobe_fraction: Fraction of observed entries to hold out as
                probe data (default 0.0, disabled).  When positive and no
                explicit *xprobe* is passed to :meth:`fit`, a random probe
                set is generated automatically.
            criterion_order: Priority order for convergence criteria.
                Defaults to ``DEFAULT_CRITERION_ORDER`` when ``None``.
            convergence_criteria: Per-criterion enable/disable dict.  Keys
                are criterion names, values are booleans.  Criteria set to
                ``False`` are still evaluated for diagnostics but excluded
                from the stop decision. Supported criteria default to enabled.
                ``slowing_down`` is unavailable in live fitting and resolves to
                disabled; explicitly requesting it emits a warning.
            random_state: Seed (or existing ``Generator``) controlling
                internal randomness: parameter initialization when no
                explicit ``init`` value is supplied, and the automatically
                generated probe mask when ``xprobe_fraction`` is positive
                and no explicit *xprobe* is passed to :meth:`fit`.  ``None``
                (default) draws fresh entropy each call, matching prior
                behaviour; set an int for reproducible fits.
            init: Random initialization, a MAT filename or a warm-start payload.
            max_dense_bytes: Combined reconstruction/variance output limit in
                bytes; None disables it. This does not bound peak fit memory.
            uniquesv: Group identical sample missingness patterns.
            auto_pattern_masked: Enable automatic grouped-pattern handling.
            angle_every: Iteration stride for principal-angle diagnostics.
            minangle: Angle stopping threshold.
            algorithm: VB, MAP or PPCA numerical update policy.
            earlystop: Restore/stop according to held-out probe diagnostics.
            rmsstop: RMS plateau window and absolute/relative thresholds.
            cfstop: Cost plateau window and absolute/relative thresholds.
            cfstop_rel: Relative cost improvement threshold.
            cfstop_curv: Cost curvature threshold.
            record_cost: Record the per-iteration variational objective.
            cost_diagnostics: Full trace or endpoint objective evaluation.
            record_prior_trace: Record loading prior variances each iteration.
            composite_stop: Composite angle/RMS/cost stopping thresholds.
            patience: Consecutive qualifying iterations required for stopping.
            num_cpu: Global native-kernel thread hint; None inherits policy.
            num_cpu_score_update: Score-update thread override.
            num_cpu_loadings_update: Loading-update thread override.
            num_cpu_noise_update: Noise-update thread override.
            num_cpu_rms: RMS thread override.
            runtime_tuning: Off, safe or aggressive runtime tuning.
            runtime_profile: Optional runtime-profile file.
            dense_sufficient_statistics: Dense observed/complement execution.
            noise_sufficient_statistics: Noise observed/complement execution.
            masked_batch_size: Grouped-pattern batch size; zero selects policy.
            cov_writeback_mode: Covariance writeback execution mode.
            log_progress_stride: Progress logging stride.
            accessor_mode: Sparse/grouped-pattern accessor implementation.
            rotate2pca: Rotate factors to principal axes while fitting.
            display: Enable legacy progress display.
            bias_update_order: Mean-update ordering, or compatibility default.
            variance_update_order: Prior-variance ordering relative to rotation,
                or the compatibility default.
            compat_mode: Legacy or modern numerical compatibility policy.
            return_diagnostics: Include reconstruction and variance fit outputs.
            runtime_report: Retain execution-policy diagnostics after fit.
            explained_var_solver: Factor/SVD/Gram explained-variance computation.
            explained_var_gram_ratio: Gram solver aspect-ratio threshold.
        """
        self.n_components = n_components
        self.bias = bias
        self.maxiters = maxiters
        self.tol = tol
        self.verbose = verbose
        self.hp_va = hp_va
        self.hp_vb = hp_vb
        self.hp_v = hp_v
        self.niter_broadprior = niter_broadprior
        self.va_init = va_init
        self.xprobe_fraction = xprobe_fraction
        self.criterion_order = criterion_order
        self.convergence_criteria = convergence_criteria
        self.random_state = random_state
        self.init = init
        self.max_dense_bytes = max_dense_bytes
        self.uniquesv = uniquesv
        self.auto_pattern_masked = auto_pattern_masked
        self.angle_every = angle_every
        self.minangle = minangle
        self.algorithm = algorithm
        self.earlystop = earlystop
        self.rmsstop = rmsstop
        self.cfstop = cfstop
        self.cfstop_rel = cfstop_rel
        self.cfstop_curv = cfstop_curv
        self.record_cost = record_cost
        self.cost_diagnostics = cost_diagnostics
        self.record_prior_trace = record_prior_trace
        self.composite_stop = composite_stop
        self.patience = patience
        self.num_cpu = num_cpu
        self.num_cpu_score_update = num_cpu_score_update
        self.num_cpu_loadings_update = num_cpu_loadings_update
        self.num_cpu_noise_update = num_cpu_noise_update
        self.num_cpu_rms = num_cpu_rms
        self.runtime_tuning = runtime_tuning
        self.runtime_profile = runtime_profile
        self.dense_sufficient_statistics = dense_sufficient_statistics
        self.noise_sufficient_statistics = noise_sufficient_statistics
        self.masked_batch_size = masked_batch_size
        self.cov_writeback_mode = cov_writeback_mode
        self.log_progress_stride = log_progress_stride
        self.accessor_mode = accessor_mode
        self.rotate2pca = rotate2pca
        self.display = display
        self.bias_update_order = bias_update_order
        self.variance_update_order = variance_update_order
        self.compat_mode = compat_mode
        self.return_diagnostics = return_diagnostics
        self.runtime_report = runtime_report
        self.explained_var_solver = explained_var_solver
        self.explained_var_gram_ratio = explained_var_gram_ratio
        self.components_: Float64Array | None = None
        self.scores_: Float64Array | None = None
        self.mean_: Float64Array | None = None
        self.rms_: float | None = None
        self.prms_: float | None = None
        self.noise_variance_: float | None = None
        self.cost_: float | None = None
        self.n_iter_: int | None = None
        self.convergence_reason_: str | None = None
        self.converged_: bool | None = None
        self.best_probe_iteration_: int | None = None
        self.best_probe_rms_: float | None = None
        self.returned_iteration_: int | None = None
        self.learning_curve_: LearningCurve | None = None
        self.runtime_report_: dict[str, object] | None = None
        self.reconstruction_: Float64Array | None = None
        # Posterior variance of the denoised reconstruction E[AS + mu]:
        # epistemic uncertainty in the latent mean, excluding observation noise.
        self.variance_: Float64Array | None = None
        # Predictive variance for a new observed entry: variance_ + noise_variance_.
        # Use this for calibrated prediction intervals / coverage of noisy data.
        self.predictive_variance_: Float64Array | None = None
        self.explained_variance_: Float64Array | None = None
        self.explained_variance_ratio_: Float64Array | None = None
        self.n_features_in_: int | None = None
        # ARD state: final loading prior variances, the bias prior variance,
        # each returned component's share of reconstruction energy, and the
        # per-iteration prior variances when ``record_prior_trace=True``.
        self.prior_variances_: Float64Array | None = None
        self.bias_prior_variance_: float | None = None
        self.component_relevance_: Float64Array | None = None
        self.prior_trace_: Float64Array | None = None
        self._av: list[Float64Array] | None = None
        self._sv: list[Float64Array] | None = None
        self._pattern_index: IndexArray | None = None
        self._muv: Float64Array | None = None

    def _solver_kwargs(self) -> dict[str, object]:
        """Gather stored arguments without changing constructor identities.

        Returns:
            Canonical numerical option names; None-valued inherited defaults
            retain their existing estimator meaning.
        """
        result = self.get_params(deep=False)
        for name in ("n_components", "tol", "xprobe_fraction"):
            result.pop(name)
        for name in (
            "maxiters",
            "hp_va",
            "hp_vb",
            "hp_v",
            "niter_broadprior",
            "va_init",
        ):
            if result[name] is None:
                result.pop(name)
        if result.get("num_cpu") is None:
            result.pop("num_cpu")
        return result

    def get_config(self) -> PCAOptions:
        """Resolve current estimator parameters into typed solver configuration.

        Returns:
            A fresh immutable configuration with owned small policy containers.
        """
        return PCAOptions.from_kwargs(**self._solver_kwargs())

    def fit(  # noqa: C901, PLR0912, PLR0914, PLR0915
        self,
        x: InputMatrix,
        mask: InputMask | None = None,
        xprobe: Matrix | None = None,
    ) -> VBPCA:
        """
        Fit the model to data, optionally supplying a mask.

        Args:
            x: Data matrix of shape (n_features, n_samples).
            mask: Optional boolean mask of the same shape as x, where True
                indicates observed entries.
            xprobe: Optional probe matrix of the same shape as x used for
                early-stopping diagnostics.  Entries that are not ``NaN``
                (dense) or explicitly stored (sparse) are treated as held-out
                observations.  When provided, the model monitors probe-RMS
                and can stop early before ARD over-prunes components.

        Returns:
            The fitted estimator instance.

        Raises:
            ValueError: If ``mask`` shape does not match ``x`` or requested
                dense diagnostics exceed ``max_dense_bytes``.
        """
        _warn_deprecated_tol(self.tol)
        options = self.get_config()
        max_dense_bytes = resolve_max_dense_bytes(options.max_dense_bytes)
        shape = np.shape(x)
        if options.return_diagnostics and len(shape) == 2:
            check_dense_output_budget(
                (shape[0], shape[1]),
                max_dense_bytes,
                arrays=3,
                operation="Fit diagnostics",
            )
        if xprobe is not None:
            options = replace(options, xprobe=xprobe)
        elif self.xprobe_fraction > 0.0:
            x, xprobe_gen = make_xprobe_mask(
                x,
                fraction=self.xprobe_fraction,
                rng=np.random.default_rng(self.random_state),
                mask=mask,
                compat_mode=options.compat_mode,
            )
            options = replace(options, xprobe=xprobe_gen)

        if not sp.issparse(x) and mask is not None and sp.issparse(mask):
            msg = "mask must be dense when x is dense"
            raise ValueError(msg)

        if sp.issparse(x) and mask is not None and not sp.issparse(mask):
            over, est = exceeds_budget(mask.shape, np.bool_, max_dense_bytes)
            if over:
                budget = 0 if max_dense_bytes is None else max_dense_bytes
                msg = (
                    "Dense mask would exceed max_dense_bytes: "
                    f"{format_bytes(est)} > {format_bytes(int(budget))}"
                )
                raise ValueError(msg)

        mask_clean: Matrix | None = None
        if mask is not None:
            if sp.issparse(mask):
                mask_clean = sp.csr_matrix(mask)
            else:
                mask_clean = np.asarray(mask, dtype=bool)

        mask_param: Matrix | None = None
        data_for_fit: InputMatrix
        if sp.issparse(x):
            x_sparse = sp.csr_matrix(x)
            if mask_clean is not None:
                mask_csr = (
                    sp.csr_matrix(mask_clean)
                    if not sp.issparse(mask_clean)
                    else mask_clean
                )
                if mask_csr.shape != x_sparse.shape:
                    msg = "mask must have the same shape as x"
                    raise ValueError(msg)
                if options.compat_mode == CompatMode.MODERN:
                    # Construct requested zeros directly: sparse addition can
                    # erase unrelated stored zeros, including after a split.
                    mask_csr = sp.csr_matrix(mask_csr, dtype=bool, copy=True)
                    mask_csr.sum_duplicates()
                    mask_csr.eliminate_zeros()
                    rows, cols = mask_csr.nonzero()
                    values = (
                        np.asarray(x_sparse[rows, cols]).ravel()
                        if len(rows)
                        else np.empty(0, dtype=float)
                    )
                    x_sparse = sp.csr_matrix(
                        (values, (rows, cols)), shape=x_sparse.shape
                    )
                else:
                    x_pattern = x_sparse.copy()
                    x_pattern.data = np.ones_like(x_pattern.data)
                    missing_pattern = mask_csr - mask_csr.multiply(x_pattern)
                    if missing_pattern.nnz:
                        eps = np.finfo(float).eps
                        x_sparse += missing_pattern.multiply(eps)
                mask_param = mask_csr
            data_for_fit = x_sparse
        else:
            x_dense = np.array(x, copy=True)
            if mask_clean is not None:
                mask_bool = np.asarray(mask_clean, dtype=bool)
                if mask_bool.shape != x_dense.shape:
                    msg = "mask must have the same shape as x"
                    raise ValueError(msg)
                x_dense = np.where(mask_bool, x_dense, np.nan)
                mask_param = mask_bool
            data_for_fit = x_dense

        result = pca_full(
            data_for_fit, self.n_components, mask=mask_param, options=options
        )
        self.components_ = np.asarray(result["A"], dtype=float)
        self.scores_ = np.asarray(result["S"], dtype=float)
        self.mean_ = np.asarray(result["Mu"], dtype=float)
        rms_val = result.get("RMS", np.nan)
        prms_val = result.get("PRMS", np.nan)
        noise_val = result.get("V", np.nan)
        cost_val = result.get("Cost", np.nan)
        self.rms_ = (
            float(rms_val)
            if isinstance(rms_val, (float, int, np.floating))
            else float("nan")
        )
        self.prms_ = (
            float(prms_val)
            if isinstance(prms_val, (float, int, np.floating))
            else float("nan")
        )
        self.noise_variance_ = (
            float(noise_val)
            if isinstance(noise_val, (float, int, np.floating))
            else float("nan")
        )
        self.cost_ = (
            float(cost_val)
            if isinstance(cost_val, (float, int, np.floating))
            else float("nan")
        )

        # Convergence diagnostics from the learning curve.
        lc = result.get("lc")
        if lc and isinstance(lc, dict):
            rms_history = lc.get("rms", [])
            self.n_iter_ = max(0, len(rms_history) - 1)
            self.convergence_reason_ = str(lc.get("convergence_reason", "maxiters"))
            self.converged_ = self.convergence_reason_ not in {
                "maxiters",
                "earlystop",
                "slowing_down",
            }
            best_probe_iteration = lc.get("best_probe_iteration")
            self.best_probe_iteration_ = (
                int(best_probe_iteration)
                if isinstance(best_probe_iteration, (int, np.integer))
                else None
            )
            best_probe_rms = lc.get("best_probe_rms")
            self.best_probe_rms_ = (
                float(best_probe_rms)
                if isinstance(best_probe_rms, (float, int, np.floating, np.integer))
                else None
            )
            returned_iteration = lc.get("returned_iteration")
            self.returned_iteration_ = (
                int(returned_iteration)
                if isinstance(returned_iteration, (int, np.integer))
                else None
            )
            self.learning_curve_ = lc
        else:
            self.n_iter_ = 0
            self.convergence_reason_ = "maxiters"
            self.converged_ = False
            self.best_probe_iteration_ = None
            self.best_probe_rms_ = None
            self.returned_iteration_ = None
            self.learning_curve_ = None

        runtime_report = result.get("RuntimeReport")
        self.runtime_report_ = (
            runtime_report if isinstance(runtime_report, dict) else None
        )

        self.reconstruction_ = None
        if result.get("Xrec") is not None:
            self.reconstruction_ = np.asarray(result["Xrec"], dtype=float)

        self.variance_ = None
        if result.get("Vr") is not None:
            self.variance_ = np.asarray(result["Vr"], dtype=float)
        self.predictive_variance_ = None
        if result.get("Vpred") is not None:
            self.predictive_variance_ = np.asarray(result["Vpred"], dtype=float)
        self.explained_variance_ = None
        if result.get("ExplainedVar") is not None:
            self.explained_variance_ = np.asarray(result["ExplainedVar"], dtype=float)
        self.explained_variance_ratio_ = None
        if result.get("ExplainedVarRatio") is not None:
            self.explained_variance_ratio_ = np.asarray(
                result["ExplainedVarRatio"], dtype=float
            )
        self.n_features_in_ = int(self.components_.shape[0])
        av_raw = cast("list[Float64Array] | None", result.get("Av"))
        self._av = [np.asarray(a, dtype=float) for a in av_raw] if av_raw else None
        sv_raw = cast("list[Float64Array] | None", result.get("Sv"))
        self._sv = [np.asarray(s, dtype=float) for s in sv_raw] if sv_raw else None
        isv_raw = result.get("Isv")
        self._pattern_index = (
            np.asarray(isv_raw, dtype=int) if isv_raw is not None else None
        )
        muv_raw = result.get("Muv")
        self._muv = np.asarray(muv_raw, dtype=float) if muv_raw is not None else None
        self._set_pruning_state(result)
        return self

    def _set_pruning_state(self, result: Mapping[str, object]) -> None:
        va_raw = result.get("Va")
        self.prior_variances_ = (
            np.asarray(va_raw, dtype=float).ravel() if va_raw is not None else None
        )
        vmu_raw = result.get("Vmu")
        self.bias_prior_variance_ = (
            float(vmu_raw)
            if isinstance(vmu_raw, (float, int, np.floating, np.integer))
            else None
        )
        self.component_relevance_ = _component_relevance(
            cast("Float64Array", self.components_), cast("Float64Array", self.scores_)
        )
        trace = (self.learning_curve_ or {}).get("prior_trace")
        self.prior_trace_ = np.asarray(trace, dtype=float) if trace else None

    def effective_rank(self, threshold: float = 0.01) -> int:
        """Count components that survive ARD pruning.

        Args:
            threshold: Minimum share of reconstruction energy for a
                component to count.

        Returns:
            Number of components whose ``component_relevance_`` exceeds
            ``threshold``.

        Raises:
            RuntimeError: If the estimator has not been fitted.
        """
        if self.component_relevance_ is None:
            msg = "VBPCA instance is not fitted yet"
            raise RuntimeError(msg)
        return int(np.sum(self.component_relevance_ > threshold))

    def get_options(self) -> dict[str, object]:
        """Return a fresh legacy keyword export of :meth:`get_config`."""
        return self.get_config().to_kwargs()

    def transform(self, x: InputMatrix | None = None) -> Float64Array:
        """Return scores from the fitted model; new data not yet supported.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if self.scores_ is None:
            msg = "Model not fitted"
            raise RuntimeError(msg)
        if x is not None:
            msg = "Transform of new data is not supported yet"
            raise NotImplementedError(msg)
        return self.scores_

    def fit_transform(
        self, x: InputMatrix, mask: BoolArray | None = None
    ) -> Float64Array:
        """Convenience wrapper for fit + transform on the training data.

        Returns:
            Scores for the training data.
        """
        return self.fit(x, mask).transform()

    def inverse_transform(self, scores: Float64Array | None = None) -> Float64Array:
        """Reconstruct data from scores using fitted loadings and mean.

        Returns:
            Reconstructed data matrix with shape (n_features, n_samples).

        Raises:
            RuntimeError: If the model is not fitted.
            ValueError: If scores have incompatible dimensions or the single
                dense output would exceed ``max_dense_bytes``.
        """
        if self.components_ is None or self.mean_ is None:
            msg = "Model not fitted"
            raise RuntimeError(msg)
        scores_arr = self.scores_ if scores is None else np.asarray(scores, dtype=float)
        if (
            scores_arr is None
            or scores_arr.ndim != 2
            or scores_arr.shape[0] != self.components_.shape[1]
        ):
            msg = "scores must have shape (n_components, n_samples)"
            raise ValueError(msg)
        check_dense_output_budget(
            (self.components_.shape[0], scores_arr.shape[1]),
            resolve_max_dense_bytes(self.max_dense_bytes),
            operation="Inverse reconstruction",
        )
        recon = self.components_ @ scores_arr
        if self.bias:
            recon += self.mean_
        return recon

    def select_n_components(
        self,
        x: InputMatrix,
        *,
        mask: InputMask | None = None,
        components: list[int] | range | None = None,
        config: SelectionConfig | None = None,
        **opts: object,
    ) -> tuple[int, dict[str, object], list[dict[str, object]], VBPCA | None]:
        """Delegate to model selection helper using this estimator's defaults.

        Returns:
            Tuple of (best_k, best_metrics, trace, best_model) from the
            shared model-selection helper.
        """
        _warn_deprecated_tol(self.tol)
        merged_opts = self._solver_kwargs()
        merged_opts.update(opts)

        cfg = config if config is not None else SelectionConfig()
        return select_n_components(
            x,
            mask=mask,
            components=components,
            config=cfg,
            **merged_opts,
        )


def _component_relevance(loadings: Float64Array, scores: Float64Array) -> Float64Array:
    """Return each component's share of reconstruction energy.

    The share of component ``k`` is ``|a_k| |s_k|`` over the sum across
    components, in the order of the returned components. It is computed in
    the returned (rotated) basis. ``Va`` matches that basis only with
    ``variance_update_order="post_rotation"``; under the legacy order it is
    last updated before the rotation and need not align component by
    component.

    Returns:
        Shares summing to one, or zeros when the reconstruction is zero.
    """
    energy = np.linalg.norm(loadings, axis=0) * np.linalg.norm(scores, axis=1)
    total = float(energy.sum())
    if total <= 0.0:
        return np.zeros_like(energy)
    shares: Float64Array = energy / total
    return shares
