---
title: "VBPCApy: Variational Bayesian PCA with Missing Data Support in Python"
tags:
  - Python
  - C++
  - PCA
  - dimensionality reduction
  - Bayesian inference
  - missing data
  - variational inference
authors:
  - name: Joshua Macdonald
    orcid: 0000-0002-3643-6266
    email: jmacdo16@jh.edu
    affiliation: "1,2"
  - name: Shany Naim
    affiliation: 1
  - name: Yoav Ram
    orcid: 0000-0002-9653-4458
    corresponding: true
    affiliation: 1
affiliations:
  - name: School of Zoology, Tel Aviv University, Tel Aviv, Israel
    index: 1
  - name: Johns Hopkins University, Baltimore, MD, USA
    index: 2
date: 24 March 2026
bibliography: paper.bib
---

# Summary

VBPCApy is a Python package that implements Variational Bayesian Principal
Component Analysis (VB-PCA) following the formulation of @Ilin2010, with
native support for incomplete observations, sparse masks, and posterior
uncertainty quantification. The package provides a scikit-learn-compatible
estimator (`VBPCA`) with `fit`/`transform`/`inverse_transform` semantics,
missing-aware preprocessing utilities that preserve NaN structure through
encode–decode round-trips, and empirical model selection for the number of
latent components. Performance-critical update equations are implemented as
C++ extensions via pybind11 [@pybind11], with runtime autotuning for thread
counts and memory access patterns.

# Statement of Need

Missing values are pervasive in scientific and industrial tabular data, yet
standard dimensionality-reduction workflows typically either impute first
and then apply PCA—masking the uncertainty introduced by imputation—or
discard incomplete rows and features. This impute-then-analyze pattern is
widespread across ecology, genomics, cultural evolution, and survey research,
where incomplete observations are the norm rather than the exception.

VBPCApy addresses this gap by modelling missingness directly within the
variational inference loop, so that latent factors and noise parameters are
estimated only from observed entries. As \autoref{fig:accuracy} demonstrates,
the built-in model selection of VBPCApy recovers the true latent rank far
more reliably than the standard impute-then-PCA pipeline across every
missingness pattern tested, whereas scikit-learn's explained-variance
threshold collapses under incomplete data. The posterior covariances
produced by the variational E-step expose per-entry uncertainty in
reconstructions and scores, enabling downstream analyses—such as the
posterior predictive eigenvalue tests of @Macdonald2026—to perform
more principled dimensionality selection than the empirical cost and
probe-set metrics provided here.

Existing implementations of @Ilin2010 are available in MATLAB (the
authors' reference code) and as isolated scripts, but none provide a
pip-installable Python package with a stable API, automated model
selection, or missing-aware preprocessing. R packages such as `pcaMethods`
[@Stacklies2007] offer probabilistic PCA variants but lack the full
VB-PCA formulation with hierarchical noise and optional bias estimation.
The scikit-learn `PCA` class [@Pedregosa2011] does not handle missing
entries at all. VBPCApy fills this niche by combining the full @Ilin2010
algorithm with modern Python packaging, type-checked interfaces, and
compiled kernels.

# Key Features

**Scikit-learn-compatible estimator.** The `VBPCA` class exposes
`fit`, `transform`, and `inverse_transform` methods. Users access
reconstructions via `model.reconstruction_`, marginal variances via
`model.variance_`, and convergence diagnostics (`model.rms_`,
`model.prms_`, `model.cost_`).

**Missing-aware preprocessing.** `AutoEncoder` routes mixed-type columns
through `MissingAwareOneHotEncoder`, `MissingAwareStandardScaler`, and
`MissingAwareMinMaxScaler`, each of which operates only on observed entries
and preserves NaN masks through `inverse_transform`. A sparse variant,
`MissingAwareSparseOneHotEncoder`, keeps CSR structure end-to-end for
high-cardinality categoricals.

**Empirical model selection.** `select_n_components` sweeps candidate
component counts, fitting VBPCA at each candidate and selecting the
number of components that minimises a user-chosen convergence metric
(probe-set prediction RMS or variational cost). A
`SelectionConfig` dataclass controls patience, early stopping, and
metric-reversal detection. \autoref{fig:accuracy} shows that this
procedure substantially outperforms the standard impute-then-PCA
baseline, and remains stable across missingness patterns where the
baseline fails.

**C++ acceleration.** Six pybind11 extension modules implement the
dense, sparse, noise, and rotation update kernels, with runtime dispatch
selecting accessor and threading modes based on data shape and sparsity.

# Example

```python
import numpy as np
from vbpca_py import VBPCA, SelectionConfig, select_n_components

rng = np.random.default_rng(42)
x = rng.standard_normal((50, 200))          # features × samples
mask = rng.random(x.shape) > 0.2            # 20 % missing

cfg = SelectionConfig(metric="cost", patience=2, max_trials=10)
best_k, metrics, trace, _ = select_n_components(x, mask=mask, config=cfg)

model = VBPCA(n_components=best_k, maxiters=200)
model.fit(x, mask=mask)
print(f"Selected k={best_k}, final cost={model.cost_:.4f}")
```

# Stability of Model Selection

![Exact rank-recovery rate for VBPCApy (cost metric, top row) versus
scikit-learn PCA with 95\% explained-variance threshold (EVR95, bottom
row) across four missingness patterns (complete, MCAR, MNAR-censored,
block).  Each cell shows the recovery rate for a given combination of
sample size $n$ and feature count $p$.  VBPCApy maintains moderate
recovery across all patterns, while the impute-then-PCA baseline
degrades sharply under incomplete
data.\label{fig:accuracy}](figure_accuracy.pdf)

![Error decomposition of rank selection.  Panel A shows over- and
under-selection rates by missingness pattern.  Panel B reports the mean
absolute error (MAE) between selected and true rank, split by
missingness and metric.  Panel C plots the average selected rank
against the true rank, with error bars indicating one standard
deviation.\label{fig:errors}](figure_errors.pdf)

![Detection power — the probability of selecting at least the true
number of components — by true rank (Panel A) and by each combination
of missingness pattern and metric
(Panel B).\label{fig:power}](figure_power.pdf)

# Acknowledgements

This work was supported by the Yoav Ram Lab at Tel Aviv University.

# References
