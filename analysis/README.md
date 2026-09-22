# Analysis scope and provenance

This directory contains development-time validation and tuning workflows for
VBPCApy. It is not the evidence archive for the JOSS software paper, and
generated result files should not be committed here or under `paper/`.

## Results invalidated in September 2026

Do not cite or reuse the result snapshots and figures that were formerly under
`paper/`. They predated several correctness changes:

- explicit masks did not remove probe coordinates from the training set
  ([#145](https://github.com/yoavram-lab/VBPCApy/issues/145));
- model selection could substitute a different metric and had an ambiguous
  patience boundary ([#140](https://github.com/yoavram-lab/VBPCApy/issues/140));
- cross-validation did not fully seed candidate fits and could construct
  invalid folds ([#144](https://github.com/yoavram-lab/VBPCApy/issues/144));
- convergence rules could accept worsening or steep trajectories, and probe
  early stopping returned the worsened terminal state
  ([#141](https://github.com/yoavram-lab/VBPCApy/issues/141)).

These changes can alter fitted endpoints, selected ranks, and held-out metrics.
Historical files remain recoverable from git, but they are not valid evidence
for the current implementation.

## Publication boundary

The JOSS manuscript in `paper/` describes the software, its design, and
evidenced use. Comparative claims about sequential rank selection,
parallel-analysis-style alternatives, posterior-predictive methods, ensembles,
missingness mechanisms, or generative-model misspecification belong in the
`pp-eigentest` methods manuscript or a separate statistical methods paper.

Within those comparisons, wall-clock time may be reported as a resource
diagnostic but is not a methodological-quality objective. The trade-study
observable therefore retains `wall_seconds` with zero selection weight.

## Required reruns

Before changing shipped recommendations or making empirical performance
claims, rerun the following from a commit containing issues #140, #141, #144,
and #145:

1. forced-long convergence endpoints versus replayed stopping policies;
2. the routing-regime shipped-default validation, including every coarse
   shape bucket;
3. any experiment that combines an explicit training mask with a positive
   probe fraction;
4. methods-paper comparisons under multiple missingness mechanisms and
   generative-model specifications.

Record the repository commit, environment, seed schedule, job manifest, and
output checksums with every retained result set. Large replicated runs should
be sharded on Rockfish and merged only after every shard passes schema and
completeness checks.

The legacy stability workflow writes to the git-ignored
`results/stability/` directory by default. Its convenience recipes are
`just stability-analysis`, `just stability-coverage`, `just stability-plot`,
and `just stability-smoke`. Treat its current comparisons as diagnostic until
they have been reconciled with the methods-paper design and rerun from a
validated commit.
