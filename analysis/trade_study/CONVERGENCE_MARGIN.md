# Post-warmup convergence-margin validation

This study closes the evidence gate in
[issue #133](https://github.com/yoavram-lab/VBPCApy/issues/133). Three shipped
routing buckets currently have no eligible post-warmup convergence window:
`wide_moderate`, `tall_moderate`, and `tall_extreme`.

The paired design compares:

- the exact shipped configuration;
- the shipped warmup with caps of 400 and 800 iterations;
- 50-iteration and zero-iteration warmups with a 400-iteration cap; and
- a forced 800-iteration endpoint with all numerical stop criteria disabled.

`cap800` is the practical reference. The forced endpoint diagnoses whether
accepted numerical stops differ from a long-run endpoint; it is not presumed
to be better. Runtime and iteration counts are secondary diagnostics and do
not compensate for worse rank recovery, held-out prediction, or coverage.

The `smoke` profile uses smaller shapes that route to all three affected
buckets. `screen` uses the exact microbiome, cultural, and ecological shapes
that produced the shipped configurations. `confirm` adds held-out shapes and
complete, MCAR, MNAR-censored, and block-missingness settings.

## Local smoke test

Install the local `trade-study` package in the analysis environment, then run:

```bash
python -m analysis.trade_study.validate_convergence_margins manifest \
  --profile smoke --n-reps 1 --seed 20260922 \
  --output analysis/results/convergence_margin/smoke/manifest.json

for condition_index in 0 1 2 3 4 5; do
  python -m analysis.trade_study.validate_convergence_margins run-condition \
    --manifest analysis/results/convergence_margin/smoke/manifest.json \
    --condition-index "${condition_index}" \
    --output-dir analysis/results/convergence_margin/smoke/conditions \
    --n-jobs 4
done

python -m analysis.trade_study.validate_convergence_margins summarize \
  --manifest analysis/results/convergence_margin/smoke/manifest.json \
  --output-dir analysis/results/convergence_margin/smoke/conditions \
  --output analysis/results/convergence_margin/smoke/summary.json
```

Each condition is an atomic checkpoint. Re-running a complete condition
validates and reuses it; a checkpoint with a different config, replicate set,
observable schema, or non-finite score fails closed.

## Rockfish shared-array screen

Create the `screen` manifest once in a clean, pinned checkout and record its
checksum. Export cluster-specific paths rather than committing them:

```bash
export VBPCA_REPO_ROOT=/path/to/clean/VBPCApy
export VBPCA_PYTHON=/path/to/analysis/python
export VBPCA_MARGIN_MANIFEST=/path/to/results/screen/manifest.json
export VBPCA_MARGIN_OUTPUT_DIR=/path/to/results/screen/conditions
export VBPCA_REVISION="$(git -C "${VBPCA_REPO_ROOT}" rev-parse HEAD)"

"${VBPCA_PYTHON}" -m analysis.trade_study.validate_convergence_margins \
  manifest --profile screen --n-reps 3 --seed 20260922 \
  --output "${VBPCA_MARGIN_MANIFEST}"
export VBPCA_MARGIN_MANIFEST_SHA256="$(sha256sum "${VBPCA_MARGIN_MANIFEST}" | cut -d ' ' -f 1)"

sbatch --array=0-5%6 \
  --export=ALL,VBPCA_REPO_ROOT,VBPCA_PYTHON,VBPCA_MARGIN_MANIFEST,VBPCA_MARGIN_OUTPUT_DIR,VBPCA_REVISION,VBPCA_MARGIN_MANIFEST_SHA256 \
  analysis/rockfish/convergence_margin_shared.sbatch
```

Monitor with `squeue -u "$USER"`, then inspect completed jobs with
`sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode`. After all six
array tasks succeed, run `summarize` as above. Advance only nondominated
candidates to a new `confirm` manifest with distinct seeds; do not edit or
reuse the screen manifest.
