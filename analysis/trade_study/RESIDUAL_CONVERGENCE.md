# Released-default residual convergence follow-up

This document preregisters the independent follow-up for
[issue #174](https://github.com/yoavram-lab/VBPCApy/issues/174). It asks
whether the six cap hits retained after the final convergence-margin
validation justify changing VBPCA 0.4's released recommendations. Cap hits are
diagnostics, not an outcome to optimize by themselves.

The study does not reconstruct or compare legacy VBPCA settings. Its reference
is the current `vbpca_py==0.4.0` release.

## Frozen design

The tracked manifest is
[`manifests/residual_convergence_v1.json`](manifests/residual_convergence_v1.json),
with SHA-256
`1b115a02119c6a33c4ebc43191f6e7c5bcd1f6e8f09e8529ecb0e9e2dd7fe43d`.
It fixes seed `20260926`, 16 paired replicates per regime, and these conditions:

1. `released`: the exact configuration returned by `recommend_config(n, p)`;
2. `double_cap`: the same configuration with `maxiters` doubled;
3. `forced_long`: the doubled cap with every convergence stop disabled, used
   only to diagnose endpoint drift.

The seven regimes include the three settings that retained cap hits in #166
and four independently shaped cases in the same two routing buckets:

| Regime | Shape | Missingness | Rank | Noise | Bucket |
|---|---:|---|---:|---:|---|
| `wide_complete_anchor` | 40 x 200 | complete | 5 | 0.3 | wide moderate |
| `wide_mcar_anchor` | 50 x 300 | MCAR | 2 | 0.5 | wide moderate |
| `wide_complete_holdout` | 64 x 320 | complete | 5 | 0.3 | wide moderate |
| `wide_mnar_holdout` | 80 x 400 | censored MNAR | 8 | 0.7 | wide moderate |
| `tall_extreme_mnar_anchor` | 1020 x 20 | censored MNAR | 3 | 1.0 | tall extreme |
| `tall_extreme_mnar_holdout` | 1275 x 25 | censored MNAR | 4 | 1.0 | tall extreme |
| `tall_extreme_complete_holdout` | 1800 x 30 | complete | 3 | 0.5 | tall extreme |

Each condition/regime pair is one Rockfish shard, giving 21 independent,
resumable array tasks and 336 fits overall. Data and initialization remain
paired across conditions by regime and replicate.

## Endpoints and decision rule

Primary quality endpoints are exact-rank recovery, directional rank error,
rank MAE, held-out RMSE, interval score, and 95% predictive coverage. Selected
and candidate convergence, iterations, and cap hits are diagnostics. Wall-clock
time is not a quality objective.

Paired differences use a regime-stratified bootstrap with 10,000 resamples.
The doubled cap advances only if all of the following hold:

1. at least one meaningful quality improvement has a 95% interval excluding
   zero: exact recovery improves by at least 0.05, rank MAE improves by at
   least 0.10, held-out RMSE improves by at least 1% of the released mean, or
   interval score improves by at least 1% of the released mean;
2. the coverage point difference is no worse than -0.02 and its lower 95%
   bound is above -0.04;
3. the two wide-complete regimes are reported separately, and the candidate
   is not promoted if both show an exact-recovery decline of at least 0.10 or
   a rank-MAE worsening of at least 0.25.

Eliminating cap hits without passing those gates does not justify a defaults
change. `forced_long` is never a release candidate; disagreement with it is a
diagnostic for endpoint drift.

## Local validation

Generate a disposable smoke manifest and run its six shards before submitting
the registered study:

```bash
python -m analysis.trade_study.residual_convergence_followup manifest \
  --profile smoke --n-reps 1 --seed 20260926 \
  --output /tmp/vbpca-residual-smoke.json

for shard in 0 1 2 3 4 5; do
  python -m analysis.trade_study.residual_convergence_followup run-shard \
    --manifest /tmp/vbpca-residual-smoke.json \
    --shard-index "${shard}" \
    --output-dir /tmp/vbpca-residual-smoke-results \
    --n-jobs 1
done

python -m analysis.trade_study.residual_convergence_followup summarize \
  --manifest /tmp/vbpca-residual-smoke.json \
  --output-dir /tmp/vbpca-residual-smoke-results \
  --output /tmp/vbpca-residual-smoke-summary.json \
  --n-resamples 100
```

## Rockfish launch

Use clean, pinned VBPCApy and trade-study checkouts. The worker verifies both
revisions, package import locations, release version, and the manifest hash
before fitting anything.

```bash
export VBPCA_REPO_ROOT=/path/to/pinned/VBPCApy
export VBPCA_TRADE_STUDY_ROOT=/path/to/pinned/trade-study
export VBPCA_PYTHON=/path/to/analysis-env/bin/python
export VBPCA_RESIDUAL_MANIFEST="${VBPCA_REPO_ROOT}/analysis/trade_study/manifests/residual_convergence_v1.json"
export VBPCA_RESIDUAL_OUTPUT_DIR=/path/to/results/residual_convergence_v1
export VBPCA_REVISION="$(git -C "${VBPCA_REPO_ROOT}" rev-parse HEAD)"
export VBPCA_TRADE_STUDY_REVISION="$(git -C "${VBPCA_TRADE_STUDY_ROOT}" rev-parse HEAD)"
export VBPCA_RESIDUAL_MANIFEST_SHA256=1b115a02119c6a33c4ebc43191f6e7c5bcd1f6e8f09e8529ecb0e9e2dd7fe43d
export VBPCA_RELEASE=0.4.0

sbatch --array=0-20%12 \
  --export=ALL,VBPCA_REPO_ROOT,VBPCA_TRADE_STUDY_ROOT,VBPCA_PYTHON,VBPCA_RESIDUAL_MANIFEST,VBPCA_RESIDUAL_OUTPUT_DIR,VBPCA_REVISION,VBPCA_TRADE_STUDY_REVISION,VBPCA_RESIDUAL_MANIFEST_SHA256,VBPCA_RELEASE \
  analysis/rockfish/residual_convergence_shared.sbatch
```

After every shard completes, reduce once:

```bash
"${VBPCA_PYTHON}" -m analysis.trade_study.residual_convergence_followup \
  summarize \
  --manifest "${VBPCA_RESIDUAL_MANIFEST}" \
  --output-dir "${VBPCA_RESIDUAL_OUTPUT_DIR}" \
  --output "${VBPCA_RESIDUAL_OUTPUT_DIR}/summary.json" \
  --n-resamples 10000
```

Record the Slurm job ID and retained output checksum here only after reduction
passes completeness and provenance validation. Any resulting defaults change
requires a separate release PR and a new patch release.
