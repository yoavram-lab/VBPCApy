# Bias and RMS update-order audit

## Question

Does the alternating RMS trace on uncentered data come from the original
mean-update order, the Python port, or both, and can the diagnostic order be
changed without materially changing rank selection or the variational
objective?

## Update sequence

The MATLAB implementation updates `Mu` from the previous residual before the
score and loading updates. It then evaluates RMS using the new factors and the
earlier mean update. Both implementations can therefore show a one-step lag on
uncentered data.

The Python audit found two additional defects:

1. An absent initial `Mu` had been replaced with zeros before the solver could
   initialize it from observed row means, as the MATLAB implementation does.
2. `_update_bias()` returns new bias and centered-data state objects, but the
   outer training loop did not retain those objects for the next iteration.
   The fitted model mean advanced while the next iteration reused stale
   centered data.

The correction restores observed-mean initialization and persists both state
objects. The `post_factor` order additionally updates the mean after the score
and loading updates, then adjusts the already computed residual algebraically
before RMS and noise-variance updates. It does not perform a second matrix
reconstruction.

## Local validation

- Date: 2026-09-28
- Shape: 40 features by 60 samples
- Generating rank: 3
- Missing fractions: 0%, 20%, and 50%
- Iterations: 15, with convergence stops disabled
- Rotation: enabled
- Initialization: fixed seed

After state persistence was corrected, the stored RMS was finite, monotone in
the tested complete and missing-data fixtures, and equal to a direct RMS
calculation from the returned `A`, `S`, and `Mu` for both update orders. Probe
RMS also matched a direct calculation from the returned model.

A separate five-rank sweep selected rank 3 under both orders. The terminal
variational costs were:

| Rank | Legacy | Post-factor |
|---:|---:|---:|
| 1 | 1305.439124 | 1305.447476 |
| 2 | 1184.022246 | 1184.022065 |
| 3 | 624.388046 | 624.388011 |
| 4 | 792.200403 | 792.308384 |
| 5 | 956.095996 | 955.999824 |

The differences were mixed in direction and below 0.02% at every rank. This
supports treating post-factor ordering as a diagnostic-coherence change, not
as a generally better objective optimizer.

The broader PCA, trajectory, cost, and model-selection regression group passed
116 tests after the correction. The two changed snapshots were strict-legacy
RMS values that had encoded the discarded Python state. Existing MATLAB
fixture comparisons continued to pass.

## Decision and remaining validation

`bias_update_order="legacy"` retains the original MATLAB sequence.
`bias_update_order="post_factor"` aligns the fitted mean, residual, RMS, probe
RMS, and noise update within an iteration. `"auto"` resolves to `legacy` in
strict-legacy mode and `post_factor` in modern mode. The runtime report records
the resolved order.

The local study is sufficient to fix Python state consistency and expose the
coherent order. It is not sufficient to claim universal convergence or rank
improvement. The preregistered Rockfish study in issue 214 should compare the
orders across matrix shapes, missingness mechanisms, seeds, convergence
decisions, selected capacities, posterior stability, and downstream
pp-eigentest decisions before any broader default change.
