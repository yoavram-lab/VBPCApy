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
    corresponding: true
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
latent components. The numerical backend builds on NumPy [@Harris2020] and
SciPy [@Virtanen2020], while performance-critical update equations are
implemented as C++ extensions via pybind11 [@pybind11] with runtime
autotuning for thread counts and memory access patterns.

# Statement of Need

Missing values are pervasive in scientific and industrial tabular data, yet
standard dimensionality-reduction workflows typically either impute first
and then apply PCA—masking the uncertainty introduced by imputation—or
discard incomplete rows and features. This impute-then-analyze pattern is
widespread across ecology, genomics, cultural evolution, and survey research,
where incomplete observations are the norm rather than the exception.

VBPCApy addresses this gap by modelling missingness directly within the
variational inference loop, so that latent factors and noise parameters are
estimated only from observed entries, and by exposing per-entry posterior
uncertainty on reconstructions and scores rather than the point predictions
an impute-then-PCA pipeline provides.
\autoref{fig:accuracy} illustrates the practical consequence on a factorial
stability study (16,800 trials spanning sample size, feature count, true
rank, and four missingness patterns): VBPCApy's built-in model selection
recovers the true latent rank far more reliably than scikit-learn's
explained-variance threshold applied after mean imputation, which collapses
under incomplete data. Held-out reconstruction error is correspondingly
31--56\% lower across the same grid.
As with other mean-field variational approximations, VBPCApy's posterior
intervals show a calibration gap under nominal coverage
[@Bishop1999; @Ilin2010]; extended results—error decomposition, detection
power, coverage calibration, and an accuracy/coverage tradeoff analysis
across data regimes—are documented in the project repository's `analysis/`
directory.
The posterior covariances produced by the variational E-step also enable
downstream uncertainty-aware analyses, such as the posterior predictive
eigenvalue tests of @Macdonald2024a, which use VBPCApy's posterior as a
generative engine for formal, calibrated dimensionality selection beyond
the heuristic empirical metrics provided here.

# State of the Field

@Bishop1999 introduced Bayesian PCA with automatic relevance
determination; @Ilin2010 extended this to the missing-data setting with
a full variational treatment and released a MATLAB reference
implementation. However, that code is not pip-installable, lacks a stable
API, and ships without automated model selection or missing-aware
preprocessing. The R/Bioconductor package `pcaMethods`
[@Stacklies2007] provides probabilistic PCA variants but omits the full
VB-PCA formulation with hierarchical noise, optional bias estimation, and
posterior covariances on both scores and loadings. The scikit-learn `PCA`
class [@Pedregosa2011] does not handle missing entries at all, forcing
users into impute-then-analyze workflows. VBPCApy fills this gap by
combining the complete @Ilin2010 algorithm with modern Python packaging,
type-checked interfaces, compiled C++ kernels, and an empirical
model-selection layer with early stopping.

# Key Features

**Scikit-learn-compatible estimator.** The `VBPCA` class exposes
`fit`, `transform`, and `inverse_transform` methods with access to
reconstructions (`reconstruction_`), marginal variances (`variance_`),
and convergence diagnostics (`rms_`, `prms_`, `cost_`).

**Missing-aware preprocessing.** `AutoEncoder` routes mixed-type columns
through `MissingAwareOneHotEncoder`, `MissingAwareStandardScaler`, and
`MissingAwareMinMaxScaler`, each operating only on observed entries and
preserving NaN masks through `inverse_transform`.

**Empirical model selection.** `select_n_components` sweeps candidate
component counts, selecting the rank that minimises a user-chosen
metric (probe-set RMS or variational cost). The cost criterion is
regularised by per-component automatic relevance determination (ARD)
priors [@Bishop1999]: each additional component must reduce the
data-fit term enough to offset the KL penalty from its
component-specific precision prior, preventing the monotonic cost
decrease that would otherwise make the minimum uninformative.
`SelectionConfig` controls patience, early stopping, and
metric-reversal detection.
\autoref{fig:accuracy} shows that this procedure substantially
outperforms the impute-then-PCA baseline.

**C++ acceleration.** Six pybind11 extension modules implement the
dense, sparse, noise, and rotation update kernels, with runtime dispatch
selecting accessor and threading modes based on data shape and sparsity.

# Software Design

VBPCApy follows a features × samples data convention matching the
@Ilin2010 MATLAB reference, enabling bit-for-bit parity verification
via an optional Octave bridge (`compat_mode="strict_legacy"`).
Performance-critical update equations are implemented in C++ using
pybind11 [@pybind11] and Eigen for direct access to BLAS-level matrix
operations; this provides a 5–10× speedup over equivalent pure-NumPy
loops while keeping the build portable across Linux, macOS, and Windows.
A runtime autotuning probe selects per-problem thread counts, memory
accessor modes (legacy scalar vs. buffered), and covariance writeback
strategies based on measured wall-clock time.

Preprocessing utilities (`AutoEncoder`, `MissingAwareOneHotEncoder`,
`MissingAwareStandardScaler`, `MissingAwareMinMaxScaler`) route
mixed-type columns through encode and decode paths that preserve NaN
mask structure, so that generative reconstructions can be mapped back to
the original feature space. A sparse variant,
`MissingAwareSparseOneHotEncoder`, keeps CSR structure end-to-end for
high-cardinality categoricals.

The project ships with a GitHub Actions CI pipeline (lint, type check,
test across Python 3.11–3.13), a `justfile` command runner with
benchmark and Octave-parity recipes, and a `cibuildwheel` workflow
for platform wheel publication.

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

Sparse CSR data with structural missingness can be handled directly:

```python
import scipy.sparse as sp
from vbpca_py import VBPCA

x_sparse = sp.random(80, 300, density=0.6, format="csr", random_state=0)
model = VBPCA(n_components=4, maxiters=150)
scores = model.fit_transform(x_sparse)  # mask inferred from sparsity
x_hat = model.inverse_transform()       # dense reconstruction
```

# Research Impact

The legacy MATLAB implementation of VB-PCA was used by @Macdonald2024 to
analyse cultural-transmission networks among Austronesian-speaking
peoples, where incomplete ethnographic records make standard PCA
inapplicable. VBPCApy is the Python successor to that codebase and was
developed to support the posterior predictive eigenvalue tests of
@Macdonald2024a, which require posterior covariances produced by the
variational E-step. The scikit-learn-compatible API is designed to
integrate directly into existing analysis pipelines.

![Exact rank-recovery rate for VBPCApy (cost metric, top row) versus
scikit-learn PCA with a 95\% explained-variance threshold (EVR95, bottom
row) across four missingness patterns (Complete, MCAR, MNAR-censored,
Block), from a factorial stability study (16,800 trials: 7 sample sizes
$\times$ 7 feature counts $\times$ 3 true ranks $\times$ 4 missingness
patterns $\times$ 10 replicates; full grid and methodology in the project
repository).  Each cell shows the fraction of simulations in which the
selected rank exactly matched the true rank for a given sample size $n$
and feature count $p$.  VBPCApy maintains 5–100\% recovery across all
patterns, while the impute-then-PCA baseline collapses to near-zero under
incomplete data.\label{fig:accuracy}](figure_accuracy.png)

# AI Usage Disclosure

Development of VBPCApy was assisted by GitHub Copilot, which provided code
formatting suggestions and implementation scaffolding powered by Claude
Opus 4.6 (Anthropic) and GPT-5.1/5.3 Codex (OpenAI). All generated code
was reviewed, edited, and validated by the authors, who made all core
algorithmic and architectural design decisions.

# Acknowledgements

This research was supported in part by the John Templeton Foundation (YR),
the Minerva Stiftung Center for Lab Evolution (YR), and the Zuckerman
STEM Leadership Program (JCM).

# References
