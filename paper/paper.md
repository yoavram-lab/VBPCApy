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
estimated only from observed entries. As \autoref{fig:accuracy} demonstrates,
the built-in model selection of VBPCApy recovers the true latent rank far
more reliably than the standard impute-then-PCA pipeline across every
missingness pattern tested, whereas scikit-learn's explained-variance
threshold collapses under incomplete data.
\autoref{fig:errors} further decomposes these rank-selection errors,
showing that VBPCApy's cost metric keeps both over- and under-selection
rates low with a mean absolute error roughly three times smaller than the
baseline, while \autoref{fig:power} confirms that detection power
remains roughly 65\% even at the highest true ranks.
The direction of rank-selection error matters in practice:
under-selection permanently discards true signal components, whereas
over-selection adds identifiable noise dimensions that downstream
analyses can often absorb.  EVR95's low under-selection rate (7\%) is
therefore misleading—it avoids missing components only by massively
over-selecting in 80\% of trials (mean bias $+4.4$ components; under
MNAR, 99.6\% of trials over-select).  VBPCApy's cost metric is far
better calibrated, with a mean bias of $+0.3$ components and balanced
error rates (40\% over, 17\% under), and when it does err the
magnitude is modest (median off-by-two in either direction).
\autoref{fig:mae_heatmap} maps VBPCApy's MAE across the full $(n, p)$
grid, and \autoref{fig:delta_mae} shows that this advantage over EVR95
is consistent across nearly every setting.
The posterior covariances produced by the variational E-step expose
per-entry uncertainty in reconstructions and scores, enabling downstream
analyses—such as the posterior predictive eigenvalue tests of
@Macdonald2024a—to perform more principled dimensionality selection
than the empirical cost and probe-set metrics provided here.
The layered posterior predictive bootstrap methodology of
@Macdonald2024a extends this foundation by testing whether observed
eigenvalues exceed a model-implied null envelope, offering formal
statistical calibration where the empirical sweep metrics in VBPCApy
provide only heuristic selection.
\autoref{fig:rmse} shows that VBPCApy's held-out reconstruction error
is 31--45\% lower than the impute-then-PCA baseline across all
missingness patterns, and \autoref{fig:rmse_heatmap} reveals that this
advantage is strongest in high-dimensional settings ($p \geq 50$),
where improvement reaches 41--56\%.
\autoref{fig:coverage} evaluates the posterior predictive intervals:
empirical coverage reaches roughly 63--64\% at the 95\% nominal level,
a calibration gap that is characteristic of variational approximations
whose factored posterior underestimates marginal variance
[@Bishop1999; @Ilin2010].  Because the mean-field factorisation treats
each entry independently, joint (multivariate) coverage over all
held-out entries is lower still, a limitation shared by all
fully-factored variational families.
Despite this under-coverage, VBPCApy is the
only method in this comparison that provides any uncertainty estimate;
the impute-then-PCA pipeline yields point predictions with no
accompanying variance.
\autoref{fig:pareto} makes the resulting tradeoff explicit: the
$(n, p)$ settings that achieve the best posterior coverage (low $p$,
large $n$) have lower rank-selection accuracy, while the settings that
maximise accuracy (moderate $p$, around $50$--$100$) come at the cost
of reduced coverage.
No single $(n, p)$ regime achieves both high accuracy and high coverage
simultaneously.  This accuracy--calibration tradeoff motivates the
posterior predictive bootstrap of @Macdonald2024a, which uses VBPCApy's
posterior covariances as a generative engine—sampling synthetic data
sets from the fitted model—rather than relying on the raw variational
intervals for coverage, thereby sidestepping the calibration gap in the
regimes where VBPCApy's rank selection is strongest.

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
variational E-step. The 16,800-trial stability study presented below
demonstrates that VBPCApy's built-in model selection achieves roughly
three times lower mean absolute error than the standard impute-then-PCA
pipeline. The scikit-learn-compatible API is designed to integrate
directly into existing analysis pipelines.

# Stability of Model Selection

The stability study evaluates model selection across a factorial grid of
7 sample sizes ($n \in \{20, 30, 50, 70, 100, 150, 200\}$), 7 feature counts
($p \in \{10, 20, 30, 50, 70, 100, 200\}$), 3 true latent ranks
($k_{\mathrm{true}} \in \{2, 5, 10\}$), 4 missingness patterns
(Complete, MCAR at 15\%, MNAR-censored at 15\%, and Block at 15\%),
and 10 independent replicates per setting, yielding 16,800 trials in
total.  Each trial generates a low-rank-plus-noise matrix
($\sigma_{\mathrm{noise}} = 0.5$) and compares VBPCApy's cost and
probe-set RMS metrics against scikit-learn's 95\% explained-variance
threshold (EVR95) applied after mean imputation.

![Exact rank-recovery rate for VBPCApy (cost metric, top row) versus
scikit-learn PCA with a 95\% explained-variance threshold (EVR95, bottom
row) across four missingness patterns (Complete, MCAR, MNAR-censored,
Block).  Each cell shows the fraction of simulations in which the
selected rank exactly matched the true rank for a given sample size $n$
and feature count $p$ (16,800 trials total: 7 $n$ $\times$ 7 $p$
$\times$ 3 ranks $\times$ 4 patterns $\times$ 10 replicates).
VBPCApy maintains 5–100\% recovery across all patterns, while the
impute-then-PCA baseline collapses to near-zero under incomplete
data.\label{fig:accuracy}](figure_accuracy.png)

![Error decomposition of rank selection.  (A) Over- and under-selection
rates by missingness pattern for cost, prms, and EVR95.  (B) Mean
absolute error (MAE) between selected and true rank, grouped by
missingness pattern and metric; EVR95 MAE exceeds 5 under MCAR and MNAR
while cost MAE stays below 2.  (C) Mean selected rank versus true rank
($k_{\mathrm{true}} \in \{2, 5, 10\}$) with $\pm 1$ standard deviation
bars; cost and prms track the diagonal closely, whereas EVR95
systematically over-selects.\label{fig:errors}](figure_errors.png)

![Detection power — the probability of selecting at least the true
number of components.  (A) Power versus true rank for cost and prms,
showing graceful degradation from near-unity at $k=2$ to roughly 65\%
at $k=10$.  (B) Power broken down by every combination of missingness
pattern and metric; EVR95 achieves high nominal power primarily through
systematic over-selection rather than accurate rank
recovery.\label{fig:power}](figure_power.png)

![Posterior predictive coverage on held-out entries.  (A) Empirical
coverage versus nominal level for each missingness pattern; the dashed
line marks ideal calibration.  All patterns show under-coverage
characteristic of variational approximations.  (B) Mean coverage at
the 95\% nominal level for each $(n, p)$ combination, aggregated
across ranks and replicates.  (C) Mean interval width versus nominal
level; narrow intervals confirm that under-coverage reflects
underestimated posterior variance rather than uninformative wide
bands.\label{fig:coverage}](figure_coverage.png)

![Holdout reconstruction RMSE for VBPCApy versus impute-then-PCA,
grouped by missingness pattern.  VBPCApy achieves 31--45\% lower
reconstruction error across all patterns, with the largest gains under
MNAR and block missingness where mean imputation is most
biased.\label{fig:rmse}](figure_rmse.png)

![Percentage RMSE improvement of VBPCApy over the impute-then-PCA
baseline, averaged across all missingness patterns and true ranks.
Improvement ranges from 5\% at $(n{=}20, p{=}10)$ to 56\% at
$(n{=}30, p{=}200)$, confirming that VBPCApy's reconstruction
advantage grows with feature dimensionality.\label{fig:rmse_heatmap}](figure_rmse_heatmap.png)

![Accuracy--coverage Pareto front across $(n, p)$ settings.  Each
point represents one $(n, p)$ cell, averaged over missingness patterns
and true ranks; colour indicates the number of features $p$.  The
red dashed line traces the Pareto front: no setting simultaneously
achieves both high rank-selection accuracy and high posterior coverage.
Low-$p$ settings (dark) occupy the high-coverage / low-accuracy
region, while moderate-$p$ settings ($p = 50$--$100$, teal) reach
the highest accuracy at the cost of reduced coverage.  Very high $p$
(yellow) tends to degrade both metrics, falling below the Pareto
front.\label{fig:pareto}](figure_pareto.png)

![VBPCApy rank-selection mean absolute error (MAE) by $(n, p)$,
averaged across all missingness patterns and true ranks.  While the
exact-match rate (\autoref{fig:accuracy}) penalises off-by-one
errors equally, MAE reveals that the cost metric rarely selects a
rank far from the truth.
\label{fig:mae_heatmap}](figure_mae_heatmap.png)

![MAE advantage of VBPCApy (cost metric) over the impute-then-PCA
baseline (EVR95).  Each cell shows EVR95 MAE minus VBPCApy MAE;
positive values (green) indicate settings where VBPCApy makes
smaller rank-selection errors.  The advantage is largest under
incomplete data where EVR95 systematically over-selects.
\label{fig:delta_mae}](figure_delta_mae.png)

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
