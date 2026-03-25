"""Scikit-learn-like estimator wrappers for VB-PCA."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from vbpca_py._memory import (
    exceeds_budget,
    format_bytes,
    resolve_max_dense_bytes,
)
from vbpca_py._pca_full import Matrix, _build_options, pca_full
from vbpca_py.model_selection import SelectionConfig, select_n_components

__all__ = ["VBPCA"]


class VBPCA:
    """Variational Bayesian PCA with a sklearn-like interface."""

    def __init__(
        self,
        n_components: int,
        *,
        bias: bool = True,
        maxiters: int | None = None,
        tol: float | None = None,
        verbose: int | bool = 0,
        **opts: object,
    ) -> None:
        """
        Initialize the estimator with common VB-PCA options.

        Args:
            n_components: Number of principal components to infer.
            bias: If True, include a bias (mean) term in the model.
            maxiters: Maximum number of iterations for the training loop.
            tol: Tolerance for convergence.
            verbose: Verbosity level; can be an integer or a boolean.
            **opts: Additional options passed to the underlying PCA_FULL implementation.
        """
        self.n_components = n_components
        self.bias = bias
        self.maxiters = maxiters
        self.tol = tol
        self.verbose = int(verbose)
        self.opts = opts
        self.components_: np.ndarray | None = None
        self.scores_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.rms_: float | None = None
        self.prms_: float | None = None
        self.noise_variance_: float | None = None
        self.cost_: float | None = None
        self.reconstruction_: np.ndarray | None = None
        self.variance_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(  # noqa: C901, PLR0912, PLR0914, PLR0915
        self, x: Matrix, mask: Matrix | None = None
    ) -> VBPCA:
        """
        Fit the model to data, optionally supplying a mask.

        Args:
            x: Data matrix of shape (n_features, n_samples).
            mask: Optional boolean mask of the same shape as x, where True
                indicates observed entries.

        Returns:
            The fitted estimator instance.

        Raises:
            ValueError: If ``mask`` shape does not match ``x``.
        """
        opts: dict[str, object] = {
            "bias": self.bias,
            "verbose": self.verbose,
        }
        if self.maxiters is not None:
            opts["maxiters"] = self.maxiters
        opts.update(self.opts)

        max_dense_bytes = resolve_max_dense_bytes(
            opts.get("max_dense_bytes", 2_000_000_000)
        )
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
        data_for_fit: Matrix
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

        result = pca_full(data_for_fit, self.n_components, mask=mask_param, **opts)
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
        self.reconstruction_ = None
        if result.get("Xrec") is not None:
            self.reconstruction_ = np.asarray(result["Xrec"], dtype=float)

        self.variance_ = None
        if result.get("Vr") is not None:
            self.variance_ = np.asarray(result["Vr"], dtype=float)
        self.explained_variance_ = None
        if result.get("ExplainedVar") is not None:
            self.explained_variance_ = np.asarray(result["ExplainedVar"], dtype=float)
        self.explained_variance_ratio_ = None
        if result.get("ExplainedVarRatio") is not None:
            self.explained_variance_ratio_ = np.asarray(
                result["ExplainedVarRatio"], dtype=float
            )
        self.n_features_in_ = int(self.components_.shape[0])
        return self

    def get_options(self) -> dict[str, object]:
        """Return the resolved pca_full options (defaults + overrides)."""
        opts: dict[str, object] = {
            "bias": self.bias,
            "verbose": self.verbose,
        }
        if self.maxiters is not None:
            opts["maxiters"] = self.maxiters
        opts.update(self.opts)
        return _build_options(opts)

    def transform(self, x: Matrix | None = None) -> np.ndarray:
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

    def fit_transform(self, x: Matrix, mask: np.ndarray | None = None) -> np.ndarray:
        """Convenience wrapper for fit + transform on the training data.

        Returns:
            Scores for the training data.
        """
        return self.fit(x, mask).transform()

    def inverse_transform(self, scores: np.ndarray | None = None) -> np.ndarray:
        """Reconstruct data from scores using fitted loadings and mean.

        Returns:
            Reconstructed data matrix with shape (n_features, n_samples).

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if self.components_ is None or self.mean_ is None:
            msg = "Model not fitted"
            raise RuntimeError(msg)
        scores_arr = self.scores_ if scores is None else np.asarray(scores, dtype=float)
        recon = self.components_ @ scores_arr
        if self.bias:
            recon += self.mean_
        return recon  # type: ignore[no-any-return]

    def select_n_components(
        self,
        x: Matrix,
        *,
        mask: Matrix | None = None,
        components: list[int] | range | None = None,
        config: SelectionConfig | None = None,
        **opts: object,
    ) -> tuple[int, dict[str, object], list[dict[str, object]], VBPCA | None]:
        """Delegate to model selection helper using this estimator's defaults.

        Returns:
            Tuple of (best_k, best_metrics, trace, best_model) from the
            shared model-selection helper.
        """
        merged_opts: dict[str, object] = {
            "bias": self.bias,
            "verbose": self.verbose,
            **self.opts,
        }
        if self.maxiters is not None:
            merged_opts["maxiters"] = self.maxiters
        if self.tol is not None:
            merged_opts["tol"] = self.tol
        merged_opts.update(opts)

        cfg = config if config is not None else SelectionConfig()
        return select_n_components(
            x,
            mask=mask,
            components=components,
            config=cfg,
            **merged_opts,
        )
