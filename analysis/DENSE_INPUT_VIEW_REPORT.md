# Dense input-view benchmark

## Question

Can the masked dense kernels avoid repeated mask conversion and hidden
C/Fortran layout copies without changing the VB-PCA updates?

## Local comparison

- Date: 2026-09-28
- Baseline: `c005011`
- Candidate: `2879bb7`
- CPU: Intel Core Ultra 9 275HX, 24 allocated cores
- Components: 10
- Missing rate: 0.18
- Iterations: 3, convergence stops disabled
- Replicates: 3 paired fits
- Diagnostics and runtime autotuning: disabled

Representative command:

```bash
python analysis/benchmark_backend_phases.py \
  --features 5846 --samples 2504 --components 10 \
  --missing-rate 0.18 --iterations 3 --threads 24 --repeats 3 \
  --order C --runtime-tuning off --skip-diagnostics
```

## Results

| Shape and layout | Revision | Wall time | Score phase | Loading phase | Terminal RMS |
|---|---|---:|---:|---:|---:|
| 1,000 × 1,500, C | baseline | 0.481 s | 0.102 s | 0.085 s | 0.2976462579 |
| 1,000 × 1,500, C | candidate | 0.446 s | 0.082 s | 0.059 s | 0.2976462579 |
| 1,000 × 1,500, F | baseline | 0.489 s | 0.121 s | 0.091 s | 0.2976462579 |
| 1,000 × 1,500, F | candidate | 0.510 s | 0.104 s | 0.074 s | 0.2976462579 |
| 5,846 × 2,504, C | baseline | 3.429 s | 0.787 s | 0.867 s | 0.2993456203 |
| 5,846 × 2,504, C | candidate | 2.719 s | 0.432 s | 0.467 s | 0.2993456203 |

At the target shape, median wall time fell by 20.7%. The score and loading
phases fell by 45.1% and 46.2%, respectively. Process maximum RSS was
1,327,468 KiB for the baseline and 1,327,156 KiB for the candidate, so this
short in-process measurement did not resolve a peak-memory reduction.

The medium F-order wall result was noisy even though both update phases became
faster. The target-scale result and exact terminal RMS support retaining the
change. Equation-level and fixed-trajectory tests separately cover both C and F
layouts, boolean and uint8 masks, and multiple native thread counts.

## Decision

The masked dense native interface now reads C- and Fortran-contiguous float64
arrays through their original strides and reads boolean or uint8 masks without
promoting them to float64. Unsupported non-contiguous layouts and ambiguous
mask dtypes raise an error instead of triggering a hidden copy. The remaining
per-observation vector allocations and covariance staging are tracked
separately in issue 209.
