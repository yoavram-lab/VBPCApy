# Convergence-detector validation

**Issue:** #186  
**Status:** replay contract implemented; Stage 1 manifest not yet frozen

This study treats convergence as a detector of a stable posterior. A rule
firing is not itself evidence that the fit is ready to stop.

## Design

Stage 1 will run stopping-disabled fits to a stable-tail endpoint across wide,
square, and tall matrices; null, low-rank, weak-factor, heteroskedastic, and
variance-misspecified generators; complete, MCAR, MAR, MNAR, and block
missingness; and independently seeded replicates. Lightweight checkpoints will
measure reconstruction-mean drift, predictive-variance and observation-noise
drift, loading-subspace and effective-rank stability, held-out predictive-score
stability, and downstream rank decisions.

`convergence_detector.py` replays candidate policies against each forced-long
learning curve. Replay uses the library's convergence implementation and
matches its criterion-specific patience and post-warmup eligibility. Policies
that differ only in `criterion_order` are collapsed before evaluation because
order changes the attributed reason when rules become eligible together, not
the stopping iteration.

Each policy will be classified by:

- premature-stop rate relative to the earliest fidelity-equivalent checkpoint;
- late/no-stop rate after a prespecified iteration margin;
- excess iterations after posterior fidelity; and
- reason-specific failures by data regime.

Stage 2 will freeze the nondominated safe policies and refit them on new seeds.
Posterior drift, predictive scores, coverage, generator-capacity selection, and
pp-eigentest PA/Seq decisions are safety gates. Iteration count and wall time
are descriptive and can choose among policies only after all safety gates pass.

## Immediate next steps

1. Freeze the fidelity margins and stable-tail rule before inspecting results.
2. Add checkpoint summaries and the immutable Stage 1 manifest.
3. Validate a small local smoke profile.
4. Pin the merge revision and manifest checksum on Rockfish.
5. Reduce Stage 1 before defining the held-out Stage 2 seed set.
