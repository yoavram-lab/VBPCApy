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
date: 22 September 2026
bibliography: paper.bib
---

# Summary

VBPCApy is an open-source Python implementation of variational Bayesian
principal component analysis (VB-PCA) for incomplete matrices. It estimates a
low-dimensional representation from observed entries while retaining
posterior uncertainty for the latent scores, loadings, and reconstructed
values. The package supports dense arrays, sparse matrices, and explicit
observation masks; supplies missing-aware preprocessing for continuous and
categorical variables; and provides held-out and cross-validated utilities for
choosing the number of components. A Python orchestration layer combines
NumPy [@Harris2020] and SciPy [@Virtanen2020] with six compiled C++ extension
modules implemented through pybind11 [@pybind11].

# Statement of need

Scientists often apply PCA to tables that contain unrecorded measurements,
inapplicable variables, or structurally absent observations. Dropping every
incomplete row can discard much of a dataset, whereas filling values before
PCA treats estimated values as if they had been observed and separates the
dimension-reduction step from its uncertainty. These problems occur in fields
including ecology, genomics, morphology, behavioural science, and comparative
cultural research.

VBPCApy is intended for researchers and developers who need a reusable
latent-factor implementation that conditions its updates on the observed set
and exposes uncertainty alongside point reconstructions. It packages the full
variational formulation described by @Ilin2010, including automatic relevance
determination [@Bishop1999], optional bias estimation, posterior covariances,
and rotation to a PCA-aligned basis. Its public interface adds data checks,
convergence diagnostics, deterministic random-state control, probe holdouts,
component-selection helpers, and preprocessing that preserves missing-value
masks through encoding and inverse transformation. The estimator follows
scikit-learn conventions for parameter inspection, cloning, fitting, and
training-data reconstruction without claiming support for out-of-sample
projection that is not yet implemented.

# State of the field

The original MATLAB implementation accompanying @Ilin2010 is the closest
algorithmic reference, but it is not distributed through the Python package
ecosystem and does not provide current testing, packaging, or estimator
interfaces. Scikit-learn's PCA implementation [@Pedregosa2011] requires a
complete input matrix, so incomplete data must be handled by a separate
imputation or row-removal step. The R/Bioconductor package `pcaMethods`
[@Stacklies2007] offers several PCA methods for incomplete biological data,
but it does not provide this full Python VB-PCA implementation and its
posterior outputs.

VBPCApy was developed as a maintained port rather than an extension to one of
those projects because its core contribution joins three requirements:
numerical continuity with the Ilin--Raiko formulation, explicit dense and
sparse missing-data contracts, and Python-native access to posterior
uncertainty. It complements, rather than replaces, conventional PCA and the
broader collection of incomplete-data methods in `pcaMethods`.

# Software design

VBPCApy deliberately retains the reference implementation's features ×
samples convention. This choice reduces ambiguity when comparing update
equations and permits optional Octave parity tests, although it differs from
the samples × features convention common in Python machine learning. A
`compat_mode` separates strict legacy behaviour from selected modern mask and
preprocessing semantics.

Observation structure is represented independently from numerical values.
Dense callers may use NaNs or an explicit Boolean mask, while sparse callers
may provide a sparse mask when stored zeros must remain observable. Probe
holdouts are removed from both the data and mask before fitting. This design
avoids interpreting every numerical zero as missing and prevents validation
entries from leaking into model updates. Memory guards reject operations that
would silently densify inputs beyond a configurable budget.

The iterative solver is decomposed into initialization, score, loading, noise,
rotation, monitoring, and convergence modules. Compiled kernels accelerate
dense and sparse update paths, while a runtime policy selects thread counts,
access modes, and covariance writeback strategies for the current workload.
Convergence criteria are configurable and emit replayable learning-curve
traces; probe-based stopping restores the state with the best observed probe
error. The package exposes both latent reconstruction variance and predictive
variance that includes observation noise, making the intended uncertainty
interpretation explicit.

The repository includes typed interfaces, property and regression tests,
optional reference-parity tests, continuous integration on Python 3.11--3.14,
binary-wheel workflows, versioned releases, tutorials, and API documentation.
The mean-field approximation and training-data-only `transform` operation are
documented limitations rather than hidden compatibility claims.

# Research impact statement

The legacy VB-PCA workflow ported by VBPCApy was used in an analysis of
incomplete comparative cultural data by @Macdonald2024. VBPCApy also provides
the posterior quantities and reproducible Python infrastructure used by the
ongoing dimensionality-selection work reported by @Macdonald2024a. These use
cases require more than a point PCA solution: they depend on observed-entry
updates, posterior covariance propagation, and consistent handling of mixed
data encodings.

The project has public releases on PyPI, cross-platform wheels, a maintained
documentation site, and external contributions merged through its public
issue and pull-request workflow. Together with reusable preprocessing,
diagnostic, and model-selection interfaces, these provide a path for adoption
outside the analyses that motivated the port.

# AI usage disclosure

GitHub Copilot and OpenAI Codex were used during software development for code
completion, refactoring suggestions, test scaffolding, and documentation.
Claude Opus 4.6 and GPT-5-family Codex models were used in that tooling;
OpenAI Codex (GPT-5) also assisted with editorial review and revision of this
manuscript. The human authors reviewed, edited, and tested all AI-assisted
outputs, verified the technical claims against the implementation and cited
sources, and made all scientific, algorithmic, and architectural decisions.

# Acknowledgements

This research was supported in part by the John Templeton Foundation (YR),
the Minerva Stiftung Center for Lab Evolution (YR), and the Zuckerman STEM
Leadership Program (JCM).

# References
