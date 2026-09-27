# Held-out convergence-policy safety confirmation

**Issue:** #195
**Status:** implementation and local smoke validation complete; confirmatory
Rockfish run not yet launched

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
