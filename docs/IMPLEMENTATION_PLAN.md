# MkDocs Documentation — Implementation Plan

## Phase 0 — Scaffold & tooling

1. Add a `docs` optional-dependency group to `pyproject.toml` (`mkdocs-material`, `mkdocstrings[python]`, `pymdown-extensions`).
2. Create `mkdocs.yml` with theme config, plugin config, full nav tree, and markdown extensions (admonitions, code highlighting, MathJax).
3. Create the `docs/` directory tree with placeholder `index.md` stubs for every nav entry.
4. Add `just docs` (build) and `just docs-serve` (live preview) recipes to the justfile.
5. Verify `mkdocs serve` boots without errors.

**Why first:** everything else depends on the build pipeline working. No content risk — just plumbing.

## Phase 1 — Landing page & Getting Started (2 pages)

1. `docs/index.md` — badges, one-paragraph pitch, install one-liner, links into tutorials and API.
2. `docs/getting-started/installation.md` — consolidate all install paths from the README (PyPI wheels, source + Eigen, extras matrix, uv sync).
3. `docs/getting-started/quickstart.md` — the two README quick-start snippets (dense, sparse) with brief annotations.

**Source material:** README sections can be migrated almost verbatim.

## Phase 2 — API Reference (5 pages, auto-generated)

1. `docs/api/vbpca.md` — `mkdocstrings` directive for `vbpca_py.VBPCA` (constructor, fit/transform/inverse_transform, all learned attributes, `get_options()`).
2. `docs/api/model-selection.md` — `select_n_components`, `cross_validate_components`, `SelectionConfig`, `CVConfig`.
3. `docs/api/preprocessing.md` — all 7 `MissingAware*` transformers + `AutoEncoder` + `check_data` + `DataReport`.
4. `docs/api/utilities.md` — `make_xprobe_mask`.
5. `docs/api/plotting.md` — `scree_plot`, `loadings_barplot`, `variance_explained_plot`.

**Why before tutorials:** tutorials will link into these pages, so they need to exist first (even if docstrings get polished later). Private modules (`_pca_full`, `_full_update`, etc.) are excluded per public API policy.

**Validation gate:** `mkdocs build --strict` must pass with zero mkdocstrings resolution errors.

## Phase 3 — Concept pages (4 pages)

1. `docs/concepts/algorithm.md` — the Ilin & Raiko generative model, E-step/M-step update structure, ARD hyperpriors (`hp_va`, `hp_vb`, `niter_broadprior`), the ELBO/cost, PCA rotation step. Cite the paper.
2. `docs/concepts/data-convention.md` — features×samples layout, dense+mask vs sparse implicit mask, `AutoEncoder` samples×features note, transpose requirement.
3. `docs/concepts/convergence.md` — enumerate all 7 criteria with option names, defaults, and an example composite rule (mostly migrated from README).
4. `docs/concepts/model-selection.md` — `select_n_components` sweep logic vs `cross_validate_components` K-fold, metrics (`prms`, `cost`), 1-SE rule, patience/max_trials.

**Source material:** README convergence section + paper Statement of Need. Original prose, not auto-generated.

## Phase 4 — Tutorials (5 narrative pages)

Each tutorial is a self-contained walkthrough with runnable code blocks and brief explanatory prose.

1. **Basic Dense PCA** (`docs/tutorials/basic-dense-pca.md`) — generate rank-5 data, fit `VBPCA`, inspect `components_`, `scores_`, `explained_variance_ratio_`, `reconstruction_`. Introduces the features×samples convention. ~50 lines of code.

2. **Missing Data & Model Selection** (`docs/tutorials/missing-data-model-selection.md`) — generate data, mask 20%, create probe set with `make_xprobe_mask`, sweep with `select_n_components(metric="cost")`, fit final model, evaluate held-out RMSE. Adapts `scripts/example_missing_pca.py`. Uses cost and/or prms (not rms) as the selection metric.

3. **Preprocessing Mixed Tabular Data** (`docs/tutorials/preprocessing-mixed-data.md`) — fabricate a dataset with categorical + continuous + missing columns. Run `check_data` for diagnostics, pipe through `AutoEncoder`, fit VBPCA, round-trip via `inverse_transform`. Highlights the samples×features → features×samples transpose.

4. **Sparse Data Workflows** (`docs/tutorials/sparse-data.md`) — build a `csr_matrix`, show how sparsity = observation mask, fit VBPCA, explain `compat_mode`. Contrast with the dense path.

5. **Cross-Validation & Diagnostics** (`docs/tutorials/cv-diagnostics.md`) — `cross_validate_components` with `CVConfig(n_splits=5, one_se_rule=True)`, plot the CV trace, inspect convergence (`prms_`, `cost_`), use `scree_plot` / `loadings_barplot`.

**Dependency:** tutorials link to API ref (Phase 2) and concepts (Phase 3).

## Phase 5 — How-To Guides (3 pages)

Task-oriented recipes for specific operational needs:

1. `docs/howto/convergence.md` — tune `rmsstop`, `cfstop`, `minangle`, `patience`, composite rules; includes a "my model won't converge" troubleshooting checklist.
2. `docs/howto/runtime-tuning.md` — `runtime_tuning` modes, `num_cpu`, env var overrides, `max_dense_bytes`, `compat_mode`.
3. `docs/howto/plotting.md` — install `[plot]` extra, call the three plotting functions, customize axes.

## Phase 6 — Remaining pages & polish

1. `docs/limitations.md` — migrate "Known limitations" from README.
2. `docs/changelog.md` — pull from `CHANGELOG.md`.
3. `docs/citation.md` — BibTeX + `CITATION.cff` content.
4. Cross-link pass: tutorials reference API pages, concept pages reference tutorials, how-tos reference both.
5. Final `mkdocs build --strict` gate — zero warnings.

## Execution order

| Step | Phases | Effort | Deliverable |
|------|--------|--------|-------------|
| 1 | 0 | Small | Working `mkdocs serve` with empty nav |
| 2 | 1 + 6 (limitations/changelog/citation) | Small | All "migrate from README" pages done |
| 3 | 2 | Medium | All API ref pages rendering from docstrings |
| 4 | 3 | Medium | All concept pages written |
| 5 | 4 | Largest | All 5 tutorials written and tested |
| 6 | 5 + final polish | Medium | How-tos + cross-links + strict build |
