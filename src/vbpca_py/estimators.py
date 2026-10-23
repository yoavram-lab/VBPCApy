"""Native samples-by-features variational PCA estimator."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 - runtime annotation resolution
from dataclasses import replace
from pathlib import Path  # noqa: TC003 - runtime annotation resolution
from typing import Self, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from sklearn.utils import Tags  # type: ignore[import-untyped]  # noqa: TC002 - runtime annotation resolution
from sklearn.utils.validation import check_array, check_is_fitted, validate_data  # type: ignore[import-untyped]

from ._feature_names import (
    FeatureNamesMixin,
    check_named_columns,
    empty_table_array,
    named_table_input,
)
from ._full_update import _covariances_list
from ._memory import check_dense_output_budget, resolve_max_dense_bytes
from ._native import TypedEstimator
from ._pca_full import pca_full
from ._projection import ProjectionPosterior, project_rows, row_log_likelihood
from ._types import Float64Array, InputMask, InputMatrix, Matrix, ObjectArray  # noqa: TC001 - runtime annotation resolution
from ._validation import (
    check_raw_sparse_observations,
    make_native_probe,
    observed_matrix,
    require_training_observations,
    validate_rank,
)
from .configuration import PCAOptions
from .legacy import _component_relevance, _warn_deprecated_tol
from .options import (  # noqa: TC001 - runtime annotation resolution
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
from .results import LearningCurve, PCAResult  # noqa: TC001 - runtime annotation resolution

__all__ = ["VBPCA"]


class VBPCA(FeatureNamesMixin, TypedEstimator):
    """Variational PCA on samples  x  features, including missing observations.

    ``scores_`` contains joint variational training means. ``transform`` returns
    conditional new-row means under the fixed fitted state; these need not equal
    the joint means. Use ``vbpca_py.legacy.VBPCA`` for the pre-0.5 layout/API.
    """

    components_: Float64Array
    scores_: Float64Array
    mean_: Float64Array
    noise_variance_: float
    learning_curve_: LearningCurve | None
    reconstruction_: Float64Array | None
    variance_: Float64Array | None
    predictive_variance_: Float64Array | None
    explained_variance_: Float64Array | None
    explained_variance_ratio_: Float64Array | None
    _av: list[Float64Array] | None

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
        compat_mode: CompatMode | str = "modern",
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
            variance_update_order: Noise-update ordering, or compatibility default.
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

    def __getstate__(self) -> dict[str, object]:
        """Serialize an explicit layout version without modifying live state.

        Returns:
            Native estimator state with a samples-first layout marker.
        """
        state = cast("dict[str, object]", super().__getstate__()).copy()
        state["_vbpca_layout_version"] = "samples-first-1"
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Refuse unmarked pre-0.5 artifacts instead of guessing array axes.

        Raises:
            ValueError: The artifact lacks the current native layout marker.
        """
        if state.get("_vbpca_layout_version") != "samples-first-1":
            msg = (
                "Incompatible VBPCA artifact layout. Pre-0.5 estimator pickles "
                "must be opened in their original environment; refit with the "
                "native 0.5 API or the explicit vbpca_py.legacy adapter."
            )
            raise ValueError(msg)
        restored = state.copy()
        restored.pop("_vbpca_layout_version")
        super().__setstate__(restored)

    def __sklearn_tags__(self) -> Tags:  # noqa: PLW3201 - sklearn protocol
        """Advertise the implemented sparse, missing-data and float64 contracts.

        Returns:
            Native sklearn tags reflecting actual input/output behavior.
        """
        tags = cast("Tags", super().__sklearn_tags__())
        tags.input_tags.sparse = True
        tags.input_tags.allow_nan = True
        tags.transformer_tags.preserves_dtype = ["float64"]
        return tags

    def _validate_observations(
        self, X: object, mask: object | None, *, reset: bool
    ) -> Matrix:
        """Validate names/dimensions, then normalize observed values.

        Returns:
            Float64 observations with canonical missingness.

        Raises:
            ValueError: Feature names, dimensions or observations are invalid.
        """
        check_named_columns(X)
        check_raw_sparse_observations(X, mask)
        with named_table_input(self, X, reset=reset) as (validator, values):
            checked = cast(
                "InputMatrix",
                validate_data(
                    validator,
                    values,
                    reset=reset,
                    accept_sparse=["csr", "csc"],
                    dtype="numeric",
                    ensure_all_finite=False,
                ),
            )
        names = getattr(self, "feature_names_in_", None)
        if names is not None and np.unique(names).size != len(names):
            msg = "X has duplicate feature names"
            raise ValueError(msg)
        return observed_matrix(
            checked,
            cast("InputMask | None", mask),
            resolve_max_dense_bytes(self.max_dense_bytes),
        )

    def fit(
        self,
        X: object,
        y: object = None,  # noqa: ARG002 - unsupervised estimator protocol
        *,
        mask: object | None = None,
        xprobe: object | None = None,
    ) -> Self:
        """Fit samples  x  features, with optional aligned mask/probe tables.

        ``y`` is ignored. Sparse stored entries, including numeric zeros, are
        observations; absent entries and NaNs are missing. Entirely unobserved
        training features are rejected. Warm-start A/S payloads retain the
        explicitly documented low-level feature-first axes.

        Returns:
            This fitted estimator.

        Invalid data, capacity, masks or probes raise ValueError during fit.
        """
        self._clear_fit_state()
        try:
            return self._fit_observations(X, mask, xprobe)
        except Exception:
            self._clear_fit_state()
            raise

    def _clear_fit_state(self) -> None:
        """Invalidate learned state before fitting and after any fit failure."""
        for name in tuple(vars(self)):
            if name.endswith("_") or name in {"_av", "_sv", "_muv", "_pattern_index"}:
                delattr(self, name)

    def _fit_observations(
        self, X: object, mask: object | None, xprobe: object | None
    ) -> Self:
        """Fit validated observations and publish state only on success.

        Returns:
            This fitted estimator.

        Raises:
            ValueError: Probe settings or shapes are invalid.
        """
        options = self.get_config()
        shape = getattr(X, "shape", None)
        if options.return_diagnostics and shape is not None and len(shape) == 2:
            check_dense_output_budget(
                (int(shape[0]), int(shape[1])),
                resolve_max_dense_bytes(options.max_dense_bytes),
                arrays=3,
                operation="Fit diagnostics",
            )
        data = self._validate_observations(X, mask, reset=True)
        rank = validate_rank(self.n_components, data.shape)
        _warn_deprecated_tol(self.tol)
        if not 0 <= self.xprobe_fraction < 1:
            msg = "xprobe_fraction must be in [0, 1)"
            raise ValueError(msg)
        probe = None
        if xprobe is not None:
            probe = self._validate_observations(xprobe, None, reset=False)
            if probe.shape != data.shape or sp.issparse(probe) != sp.issparse(data):
                msg = "xprobe must match X shape and dense/sparse representation"
                raise ValueError(msg)
        elif self.xprobe_fraction > 0:
            data, probe = make_native_probe(
                data, self.xprobe_fraction, np.random.default_rng(self.random_state)
            )
        require_training_observations(data, probe)
        if probe is not None:
            options = replace(options, xprobe=_transpose_matrix(probe))
        result = pca_full(_transpose_matrix(data), rank, options=options)
        self._assign_result(result)
        return self

    def _assign_result(self, result: PCAResult) -> None:
        """Publish modern axes after the unchanged feature-first solver."""
        self.components_ = result["A"].T
        self.scores_ = result["S"].T
        self.mean_ = result["Mu"].reshape(-1)
        self.noise_variance_ = result["V"]
        self.rms_ = result["RMS"]
        self.prms_ = result["PRMS"]
        self.cost_ = result["Cost"]
        self.learning_curve_ = result["lc"]
        curve = self.learning_curve_ or {}
        self.n_iter_ = max(0, len(curve.get("rms", [])) - 1)
        self.convergence_reason_ = str(curve.get("convergence_reason", "maxiters"))
        self.converged_ = self.convergence_reason_ not in {
            "maxiters",
            "earlystop",
            "slowing_down",
        }
        self.best_probe_iteration_ = curve.get("best_probe_iteration")
        self.best_probe_rms_ = curve.get("best_probe_rms")
        self.returned_iteration_ = curve.get("returned_iteration")
        self.runtime_report_ = result.get("RuntimeReport")
        reconstruction = result["Xrec"]
        variance = result["Vr"]
        predictive = result["Vpred"]
        self.reconstruction_ = reconstruction.T if reconstruction is not None else None
        self.variance_ = variance.T if variance is not None else None
        self.predictive_variance_ = predictive.T if predictive is not None else None
        self.explained_variance_ = result["ExplainedVar"]
        self.explained_variance_ratio_ = result["ExplainedVarRatio"]
        self._av = _covariances_list(result["Av"]) if len(result["Av"]) else None
        self._sv = result["Sv"]
        self._muv = result["Muv"]
        self._pattern_index = result["Isv"]
        self.prior_variances_ = result["Va"]
        self.bias_prior_variance_ = result["Vmu"]
        self.component_relevance_ = _component_relevance(result["A"], result["S"])
        trace = curve.get("prior_trace")
        self.prior_trace_ = np.asarray(trace, dtype=np.float64) if trace else None

    def _projection_input(self, X: object, mask: object | None) -> Matrix:
        check_is_fitted(self, ["components_", "mean_", "noise_variance_"])
        return self._validate_observations(X, mask, reset=False)

    def _output_feature_names(self, input_features: ObjectArray) -> ObjectArray:  # noqa: ARG002
        return np.asarray(
            [f"vbpca{i}" for i in range(self.components_.shape[0])], dtype=object
        )

    def transform(self, X: object, *, mask: object | None = None) -> Float64Array:
        """Return conditional latent means for new rows without refitting."""
        data = self._projection_input(X, mask)
        means, _ = project_rows(
            data,
            self.components_.T,
            self.mean_,
            self.noise_variance_,
            self._av,
            covariance=False,
            max_dense_bytes=resolve_max_dense_bytes(self.max_dense_bytes),
        )
        return means

    def transform_posterior(
        self, X: object, *, mask: object | None = None
    ) -> ProjectionPosterior:
        """Return fresh conditional means and full latent covariance matrices."""
        data = self._projection_input(X, mask)
        means, covariance = project_rows(
            data,
            self.components_.T,
            self.mean_,
            self.noise_variance_,
            self._av,
            covariance=True,
            max_dense_bytes=resolve_max_dense_bytes(self.max_dense_bytes),
        )
        return ProjectionPosterior(means, cast("Float64Array", covariance))

    def fit_transform(
        self,
        X: object,
        y: object = None,
        *,
        mask: object | None = None,
        xprobe: object | None = None,
    ) -> Float64Array:
        """Fit and condition X; joint training variational means are in scores_.

        Returns:
            Conditional scores with one row per input sample.
        """
        return self.fit(X, y, mask=mask, xprobe=xprobe).transform(X, mask=mask)

    def inverse_transform(self, Z: object) -> Float64Array:
        """Reconstruct samples x features from samples x components.

        Returns:
            The dense reconstruction.

        Raises:
            ValueError: Latent dimensions or the dense output budget are invalid.
        """
        check_is_fitted(self, ["components_", "mean_"])
        scores = cast(
            "Float64Array",
            check_array(empty_table_array(Z), dtype=np.float64, ensure_min_features=0),
        )
        if scores.shape[1] != self.components_.shape[0]:
            msg = "Z must have shape (n_samples, n_components)"
            raise ValueError(msg)
        check_dense_output_budget(
            (scores.shape[0], self.n_features_in_),
            resolve_max_dense_bytes(self.max_dense_bytes),
            operation="Inverse reconstruction",
        )
        return scores @ self.components_ + self.mean_

    def score_samples(self, X: object, *, mask: object | None = None) -> Float64Array:
        """Return observed-row plug-in Gaussian log densities; empty rows give 0.

        Empty scoring tables are rejected; returned row storage is budgeted.
        """
        data = self._projection_input(X, mask)
        check_dense_output_budget(
            (data.shape[0], 1),
            resolve_max_dense_bytes(self.max_dense_bytes),
            operation="Row log-density output",
        )
        return self._row_densities(data)

    def _row_densities(self, data: Matrix) -> Float64Array:
        """Evaluate a scoring table without imposing a temporary-memory limit.

        Returns:
            A fresh row log-density vector.

        Raises:
            ValueError: The entire scoring table has no observations.
        """
        values, count = row_log_likelihood(
            data, self.components_.T, self.mean_, self.noise_variance_
        )
        if not count:
            msg = "X has no observed entries to score"
            raise ValueError(msg)
        return values

    def score(
        self,
        X: object,
        y: object = None,  # noqa: ARG002 - unsupervised estimator protocol
        *,
        sample_weight: object | None = None,
        mask: object | None = None,
    ) -> float:
        """Average row marginal log densities with optional scoring-only weights.

        This is distinct from the variational objective and cellwise capacity
        CV. Entry masks are separate from one-dimensional row weights.

        Returns:
            The weighted mean row log density.

        Raises:
            ValueError: Scoring weights are invalid or no entries are observed.
        """
        values = self._row_densities(self._projection_input(X, mask))
        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if (
                weights.shape != values.shape
                or not np.all(np.isfinite(weights))
                or np.any(weights < 0)
                or not np.any(weights > 0)
            ):
                msg = (
                    "sample_weight must be finite, nonnegative, "
                    "one per row, with a positive sum"
                )
                raise ValueError(msg)
            positive = weights > 0
            values = values[positive]
            weights = weights[positive]
            weights /= np.max(weights)
        return float(np.average(values, weights=weights))

    def effective_rank(self, threshold: float = 0.01) -> int:
        """Count returned components above the reconstruction-energy share.

        Returns:
            Number of components exceeding the threshold.
        """
        check_is_fitted(self, ["component_relevance_"])
        return int(np.count_nonzero(self.component_relevance_ > threshold))

    def get_options(self) -> dict[str, object]:
        """Export fresh numerical options; low-level warm-start axes are retained.

        Returns:
            A fresh legacy-key numerical option mapping.
        """
        return self.get_config().to_kwargs()


def _transpose_matrix(matrix: Matrix) -> Matrix:
    """Transpose an audited sample/feature boundary, retaining stored zeros.

    Returns:
        A transposed dense view or CSR representation retaining stored zeros.
    """
    return sp.csr_matrix(matrix.T) if sp.issparse(matrix) else matrix.T
