# Convergence-detector validation

**Issue:** #186  
**Status:** Stage 1 design frozen and local smoke profile validated

This study treats convergence as a detector of a stable posterior. A rule
firing is not itself evidence that the fit is ready to stop.

## Design

Stage 1 runs stopping-disabled fits to a stable-tail endpoint across wide,
square, and tall matrices; null, low-rank, weak-factor, heteroskedastic, and
variance-misspecified generators; complete, MCAR, MAR, MNAR, and block
missingness; and independently seeded replicates. Checkpoints at 50, 100, 200,
400, 800, 1200, and 1600 iterations measure reconstruction-mean drift,
predictive-variance and observation-noise drift, loading-subspace and
effective-rank stability, held-out predictive-score stability, and
active-component stability. The last two pre-endpoint checkpoints must both
pass the fidelity gate before any detector is classified.

`convergence_detector.py` replays candidate policies against each forced-long
learning curve. Replay uses the library's convergence implementation and
matches its criterion-specific patience and post-warmup eligibility. Policies
that differ only in `criterion_order` are collapsed before evaluation because
order changes the attributed reason when rules become eligible together, not
the stopping iteration.

Each policy is classified by:

- premature-stop rate relative to the earliest fidelity-equivalent checkpoint;
- late/no-stop rate after a prespecified iteration margin;
- excess iterations after posterior fidelity; and
- reason-specific failures by data regime.

The fidelity margins were frozen before the screen:

| Quantity | Margin versus the 1600-iteration endpoint |
|---|---:|
| Reconstruction relative Frobenius drift | 0.01 |
| Predictive-variance relative Frobenius drift | 0.02 |
| Observation-noise relative change | 0.02 |
| Maximum loading-subspace angle | 5 degrees |
| Held-out RMSE relative change | 0.01 |
| Active components | exact match |

The replay grid contains 208 policies. It crosses angle, RMS-plateau, relative
cost, probe-deterioration, disjunctive, and composite rules with threshold,
criterion-specific patience, and warmup values of 0, 50, 100, and 200.
`criterion_order` is not crossed.

Stage 2 will freeze the nondominated safe policies and refit them on new seeds.
Posterior drift, predictive scores, coverage, generator-capacity selection, and
pp-eigentest PA/Seq decisions are safety gates. Iteration count and wall time
are descriptive and can choose among policies only after all safety gates pass.

## Immediate next steps

1. Merge the Stage 1 runner after CI.
2. Pin the merge revision and screen-manifest checksum on Rockfish.
3. Run representative wide, square, tall, misspecified, and structured-
   missingness preflight shards.
4. Launch the screen only after every preflight shard is valid.
5. Reduce Stage 1 before defining the held-out Stage 2 seed set.

## Commands

```bash
python -m analysis.trade_study.convergence_stability_study manifest \
  --profile screen --n-reps 5 --seed 20261008 \
  --output analysis/results/convergence_detector/screen/manifest.json

python -m analysis.trade_study.convergence_stability_study run-shard \
  --manifest analysis/results/convergence_detector/screen/manifest.json \
  --shard-index 0 --retry-errors \
  --output-dir analysis/results/convergence_detector/screen

python -m analysis.trade_study.convergence_stability_study summarize \
  --manifest analysis/results/convergence_detector/screen/manifest.json \
  --output-dir analysis/results/convergence_detector/screen \
  --output analysis/results/convergence_detector/screen/summary.json
```

The screen has 75 cells and 375 shards at five replicates per cell. Rockfish
uses `analysis/rockfish/convergence_detector_shared.sbatch` with one CPU per
shard. Wall time is recorded but is not a fidelity objective.
