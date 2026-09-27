# Held-out convergence-policy safety confirmation

**Issue:** #195
**Status:** confirmatory Rockfish run complete; the no-RMS candidate failed the
frozen safety gate and the released production policy is retained

The convergence-detector screen found that the released production policy
often fired before its fitted posterior reached the prespecified stable-tail
region. Disabling the RMS-plateau criterion reduced the premature-stop rate
from 50.4% to 3.9%, but increased late or absent stops from 13.6% to 83.3%.
The screen compared 208 stopping-policy combinations and found no rule that
was both reliably safe and efficient. This study therefore confirms one
candidate rather than reopening policy tuning.

## Frozen comparison

Each simulated data set is analyzed twice:

1. `production`: the exact configuration returned by
   `recommend_config(n, p)`;
2. `without_rms_plateau`: the same configuration, iteration cap, criterion
   order, patience, thresholds, data, observation mask, probe entries,
   initialization seed, and posterior-predictive draws, with only
   `convergence_criteria["rms_plateau"]` set to false.

The confirmatory profile contains 75 cells. It crosses wide, square, and tall
matrices with null, strong-factor, weak-factor, heteroskedastic, and
heavy-tailed generators and complete, MCAR, MAR, MNAR, and block observation
patterns. Five held-out replicates per cell give 375 paired shards. The frozen
seed is `20261015`, distinct from the detector screen's seed `20261008`.

For each condition, VBPCA searches capacities 1 through 8 and chooses the
capacity with the lowest predictive RMS on the same support-preserving 10%
probe. Generator capacity is a predictive modeling choice, not an estimate of
scientific signal rank. Capacity is therefore assessed by its paired shift and
predictive consequences, not by equality to the simulated signal rank.

The selected posterior is passed to pp-eigentest with 200 common-random-number
draws. Posterior-predictive PA uses the Phase 4a upper quantile of 0.745, and
Seq uses the fixed-sequence level 0.055. The NumPy backend is pinned to keep
backend selection out of the estimand.

## Outcomes and decision gate

Primary safety outcomes compare `without_rms_plateau` with production using
equal-cell paired bootstrap intervals:

- held-out RMSE and interval score, each on a relative scale;
- 95% predictive coverage;
- selected generator capacity;
- PA and Seq exact-rank recovery, mean absolute rank error, and rank-zero
  positive-selection rate; and
- candidate and selected-fit convergence diagnostics.

The no-RMS candidate passes only when every 95% interval satisfies these
frozen noninferiority margins:

| Outcome | Required bound |
|---|---:|
| Relative held-out RMSE increase | upper bound <= 0.01 |
| Relative interval-score increase | upper bound <= 0.01 |
| Coverage difference | lower bound >= -0.02 |
| Mean selected-capacity increase | upper bound <= 0.25 |
| PA and Seq exact-recovery difference | lower bound >= -0.02 |
| PA and Seq MAE gain | lower bound >= -0.10 |
| PA and Seq null-positive-rate increase | upper bound <= 0.02 |

A lower selected capacity is not classified as degradation when held-out
prediction and downstream ranks pass. Agreement, absolute capacity changes,
posterior-mean and predictive-variance drift, noise-variance change, and
loading-subspace angle are reported. The posterior-drift margins from the
detector screen are retained as descriptive reference thresholds because a
safer stop may appropriately differ from a prematurely stopped production
fit. Iteration count and wall time remain descriptive after the safety gate.

Passing the gate would show that disabling RMS plateau is a safe conservative
configuration. It would not by itself make that slower policy the balanced
default. Failure retains the released default and identifies which endpoint
blocked promotion.

## Confirmatory results

Rockfish array job `31303092` completed all 375 paired shards with exit code
zero. Every record reported `status: ok`. The run pinned VBPCApy revision
`50c8cd27b6930fb3b3f2a96241578beedfe68895`, pp-eigentest revision
`4dad92307240284a1925d098a129db95444c073b`, and manifest SHA-256
`a8250e2b14301c2976d764e7af862ff0f1066a55f658b06597eabe2dd9888b28`.
The reduced `summary.json` has SHA-256
`a1ab5eb096762bacde24a273395bebcf4d650f8770ad4ea0820d69f7e1a673b5`.

| Outcome | Production | Without RMS plateau |
|---|---:|---:|
| Held-out RMSE | 0.8258 | 0.8273 |
| Predictive interval score | 4.1710 | 4.1849 |
| 95% predictive coverage | 0.9501 | 0.9503 |
| Selected capacity | 2.995 | 3.043 |
| PA exact-rank recovery | 0.781 | 0.771 |
| PA rank MAE | 0.373 | 0.384 |
| Seq exact-rank recovery | 0.781 | 0.773 |
| Seq rank MAE | 0.384 | 0.397 |
| Candidate-fit iteration-budget rate | 0.023 | 0.701 |
| Selected-fit convergence rate | 0.984 | 0.288 |
| Mean selection wall time, seconds | 8.77 | 16.71 |

The no-RMS candidate failed three frozen checks. The 95% upper bound for its
relative interval-score increase was 0.0137, above the 0.01 margin. The lower
bounds for its exact-recovery differences were -0.0293 for PA and -0.0267 for
Seq, below the -0.02 margin. The remaining seven checks passed, including
held-out RMSE, predictive coverage, selected capacity, rank MAE, and null
positive-selection rates.

Rank decisions were usually unchanged: PA agreement was 0.979 and Seq
agreement was 0.976. The candidate rescued two exact decisions for each
selector, but spoiled six PA decisions and five Seq decisions. Most spoils
occurred for wide, strong-signal matrices, where continued fitting tended to
add one selected component. Averaged over wide matrices, removing RMS plateau
increased held-out RMSE by 0.78%, increased interval score by 1.55%, and
reduced PA and Seq exact recovery by 4 percentage points. Strong Gaussian
signals accounted for the largest degradation; square and tall matrices had
negligible predictive changes.

The detector screen correctly showed that RMS plateau can stop before the
posterior reaches a stable tail under its diagnostic definition. The held-out
study shows that disabling it is not a safe general correction: it roughly
doubled fitting time, exhausted the iteration budget in most candidate fits,
and did not preserve the frozen uncertainty and downstream-rank margins. The
released production policy therefore remains the recommended default. The
no-RMS configuration may still be useful as a sensitivity analysis, but it
should not replace the default or be described as a generally safer stopping
rule.

## Local validation

The eight-cell smoke profile exercises every code path with small matrices:

```bash
python -m analysis.trade_study.convergence_policy_safety manifest \
  --profile smoke --n-reps 1 --seed 20261015 \
  --output /tmp/vbpca-policy-safety-smoke.json

for shard in 0 1 2 3 4 5 6 7; do
  python -m analysis.trade_study.convergence_policy_safety run-shard \
    --manifest /tmp/vbpca-policy-safety-smoke.json \
    --shard-index "${shard}" --retry-errors \
    --output-dir /tmp/vbpca-policy-safety-smoke-results
done

python -m analysis.trade_study.convergence_policy_safety summarize \
  --manifest /tmp/vbpca-policy-safety-smoke.json \
  --output-dir /tmp/vbpca-policy-safety-smoke-results \
  --output /tmp/vbpca-policy-safety-smoke-summary.json \
  --n-resamples 200
```

The implementation completed all eight paired smoke shards and the reducer on
the local CPU before the confirmatory manifest was frozen.

## Rockfish launch and monitoring

Use clean, pinned VBPCApy and pp-eigentest checkouts plus the same analysis
environment used for the preceding studies. The worker verifies both
revisions, import locations, and the manifest checksum before fitting.

```bash
export VBPCA_REPO_ROOT=/path/to/pinned/VBPCApy
export PP_EIGENTEST_REPO_ROOT=/path/to/pinned/pp-eigentest
export VBPCA_PYTHON=/path/to/analysis-env/bin/python
export VBPCA_POLICY_SAFETY_MANIFEST="${VBPCA_REPO_ROOT}/analysis/trade_study/manifests/convergence_policy_safety_v1.json"
export VBPCA_POLICY_SAFETY_OUTPUT_DIR=/path/to/results/convergence_policy_safety_v1
export VBPCA_REVISION="$(git -C "${VBPCA_REPO_ROOT}" rev-parse HEAD)"
export PP_EIGENTEST_REVISION="$(git -C "${PP_EIGENTEST_REPO_ROOT}" rev-parse HEAD)"
export VBPCA_POLICY_SAFETY_MANIFEST_SHA256=a8250e2b14301c2976d764e7af862ff0f1066a55f658b06597eabe2dd9888b28

sbatch --array=0-374%100 \
  --export=ALL,VBPCA_REPO_ROOT,PP_EIGENTEST_REPO_ROOT,VBPCA_PYTHON,VBPCA_POLICY_SAFETY_MANIFEST,VBPCA_POLICY_SAFETY_OUTPUT_DIR,VBPCA_REVISION,PP_EIGENTEST_REVISION,VBPCA_POLICY_SAFETY_MANIFEST_SHA256 \
  analysis/rockfish/convergence_policy_safety_shared.sbatch
```

Monitor the array with:

```bash
squeue -j <job-id>
sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
find "${VBPCA_POLICY_SAFETY_OUTPUT_DIR}" -name 'shard-*.json' | wc -l
grep -l '"status": "error"' "${VBPCA_POLICY_SAFETY_OUTPUT_DIR}"/shard-*.json
```

Reduce only after all 375 shards succeed:

```bash
"${VBPCA_PYTHON}" -m analysis.trade_study.convergence_policy_safety summarize \
  --manifest "${VBPCA_POLICY_SAFETY_MANIFEST}" \
  --output-dir "${VBPCA_POLICY_SAFETY_OUTPUT_DIR}" \
  --output "${VBPCA_POLICY_SAFETY_OUTPUT_DIR}/summary.json" \
  --n-resamples 5000
```
