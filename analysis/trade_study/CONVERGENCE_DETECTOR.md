# Convergence-detector validation

**Issue:** #186  
**Status:** Stage 1 screen complete; no stopping-policy change recommended

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

The prespecified replay grid contains 208 policies. It crosses angle,
RMS-plateau, relative cost, probe-deterioration, disjunctive, and composite
rules with threshold, criterion-specific patience, and warmup values of 0, 50,
100, and 200. `criterion_order` is not crossed.

Version 2 manifests also evaluate the exact stopping policy resolved by
`recommend_config()` for each simulated matrix and a leave-one-criterion-out
ablation of that policy. Each shard retains aligned RMS, probe RMS, cost, and
angle series in a compact, versioned learning-curve record. These records allow
additional policies to be replayed without repeating the forced-long VBPCA fit.
Version 1 manifests remain readable.

## Stage 1 results

The frozen screen completed all 375 shards. The 1600-iteration endpoint passed
the stable-tail gate in 359 fits (95.7%). Among those evaluable fits, 48 of the
208 policies never stopped prematurely. The best of these still stopped late
or reached the iteration limit in 97.2% of fits. The best policy constrained to
a premature-stop rate no greater than 5% stopped prematurely in 4.7% of fits
and stopped late or not at all in 67.7%. Its premature-stop rate rose to 13.9%
under block missingness.

The version 2 follow-up repeated all 375 fits and reproduced the 95.7% stable-
endpoint rate. The exact production policy stopped prematurely in 50.4% of
the 359 evaluable fits. Its criterion ablations identify RMS plateau as the
main early-stop lever:

| Policy | Evaluable fits | Premature | Late or no stop | Median excess iterations |
|---|---:|---:|---:|---:|
| Production | 359 | 0.504 | 0.136 | 0 |
| Without RMS plateau | 359 | 0.039 | 0.833 | 494 |
| Without angle | 359 | 0.474 | 0.279 | 6 |
| Without cost | 119 | 0.160 | 0.387 | 77.5 |
| Without `slowing_down` | 359 | 0.504 | 0.136 | 0 |

Production-policy premature stopping was 69.7% for square matrices, 65.3% for
tall matrices, and 16.0% for wide matrices. Removing RMS plateau eliminated
premature stops in the square and tall groups, but every evaluable fit in those
groups then stopped late or reached the limit. The unchanged
`production_without_slowing_down` row confirms that the nominally enabled
criterion has no event input from the current training loop; issue #193 tracks
that integration defect.

Neither screen identifies a stopping policy that balances posterior-fidelity
safety and efficiency. Version 2 also rejects the current production rule as a
validated posterior-fidelity detector in this design. No replacement should be
promoted from the training screen alone. The results show that aggregate error
rates can conceal unsafe behavior by matrix shape or missingness mechanism.
Version 2 shards preserve the curves needed for denser, regime-specific policy
replay without another forced-long fit. The run used VBPCApy revision
`869eef8`, manifest SHA-256 prefix `7a688859ad59`, and summary SHA-256 prefix
`a4c8ddcb8057`.

Stage 2 will proceed only if subsequent replay identifies a nondominated safe
policy worth confirming on new seeds.
Posterior drift, predictive scores, coverage, generator-capacity selection, and
pp-eigentest PA/Seq decisions are safety gates. Iteration count and wall time
are descriptive and can choose among policies only after all safety gates pass.

## Immediate next steps

1. Do not promote any screened rule as a replacement default.
2. Quantify downstream rank and predictive effects of disabling RMS plateau,
   accepting that the safer candidate may require substantially more work.
3. Add a policy-only command that replays new registered policies from the
   retained version 2 curves.
4. Repair the unreachable `slowing_down` integration separately and keep it
   disabled until it passes the same detector safety gates.
5. Define a held-out Stage 2 seed set only if a replayed policy passes every
   safety gate.

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
