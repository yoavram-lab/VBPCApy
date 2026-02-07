"""Scikit-learn-like estimator wrappers for VB-PCA."""

from __future__ import annotations

import numpy as np

from vbpca_py._pca_full import Matrix, _build_options, pca_full


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
        """Initialize the estimator with common VB-PCA options."""
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
        self.n_features_in_: int | None = None

    def fit(self, x: Matrix, mask: np.ndarray | None = None) -> VBPCA:
        """Fit the model to data, optionally supplying a mask.

        Returns:
            Self.
        """
        opts: dict[str, object] = {
            "bias": self.bias,
            "verbose": self.verbose,
        }
        if self.maxiters is not None:
            opts["maxiters"] = self.maxiters
        opts.update(self.opts)
        x_arr = np.array(x, copy=True)
        if mask is not None:
            x_arr = np.asarray(x_arr, dtype=float)
            mask_arr = np.asarray(mask, dtype=bool)
            x_arr = np.where(mask_arr, x_arr, np.nan)
        result = pca_full(x_arr, self.n_components, **opts)
        self.components_ = np.asarray(result["A"], dtype=float)
        self.scores_ = np.asarray(result["S"], dtype=float)
        self.mean_ = np.asarray(result["Mu"], dtype=float)
        self.rms_ = float(result.get("RMS", np.nan))
        self.prms_ = float(result.get("PRMS", np.nan))
        self.noise_variance_ = float(result.get("V", np.nan))
        self.cost_ = float(result.get("Cost", np.nan))
        self.reconstruction_ = None
        if result.get("Xrec") is not None:
            self.reconstruction_ = np.asarray(result["Xrec"], dtype=float)

        self.variance_ = None
        if result.get("Vr") is not None:
            self.variance_ = np.asarray(result["Vr"], dtype=float)
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

        Raises:
            RuntimeError: If the model is not fitted.

        Returns:
            Reconstructed data matrix with shape (n_features, n_samples).
        """
        if self.components_ is None or self.mean_ is None:
            msg = "Model not fitted"
            raise RuntimeError(msg)
        scores_arr = self.scores_ if scores is None else np.asarray(scores, dtype=float)
        recon = self.components_ @ scores_arr
        if self.bias:
            recon += self.mean_
        return recon
