# Dense kernel workspace benchmark

Issue #209 targets two allocation costs in the masked dense score and loading
updates:

- a dynamic component vector constructed for every observed matrix cell; and
- posterior covariance blocks staged in an Eigen matrix and copied into NumPy
  output after the threaded kernel.

The implementation reuses one component vector and right-hand-side vector per
worker. It allocates the final C-contiguous NumPy covariance arrays before
releasing the GIL, then lets each worker write its disjoint output blocks
directly. The Cholesky solves, posterior equations, float64 arithmetic, and
summation order are unchanged.

## Reproduction

The comparison used commit `d474b47` plus the merged #208 input-view changes
as the baseline. Run each command three times within one process:

```bash
python analysis/benchmark_backend_phases.py \
  --features 1000 --samples 1500 --components <5|10|25> \
  --true-rank 5 --missing-rate 0.18 --iterations 3 \
  --threads 24 --repeats 3 --runtime-tuning off
```

The study-shaped comparison replaced the dimensions with
`--features 5846 --samples 2504 --components 10`. Tests ran on the local
24-core CPU. Values below are medians of three repeats.

## Results

| Shape | Rank | Measure | Baseline (s) | Optimized (s) | Change |
|---|---:|---|---:|---:|---:|
| 1,000 × 1,500 | 5 | wall | 0.5540 | 0.4490 | -19.0% |
| 1,000 × 1,500 | 5 | scores | 0.0685 | 0.0364 | -46.9% |
| 1,000 × 1,500 | 5 | loadings | 0.0643 | 0.0368 | -42.8% |
| 1,000 × 1,500 | 10 | wall | 0.8418 | 0.7734 | -8.1% |
| 1,000 × 1,500 | 10 | scores | 0.1242 | 0.0840 | -32.4% |
| 1,000 × 1,500 | 10 | loadings | 0.0659 | 0.0539 | -18.2% |
| 1,000 × 1,500 | 25 | wall | 3.2181 | 2.7873 | -13.4% |
| 1,000 × 1,500 | 25 | scores | 0.2930 | 0.2510 | -14.3% |
| 1,000 × 1,500 | 25 | loadings | 0.2661 | 0.2321 | -12.9% |
| 5,846 × 2,504 | 10 | wall | 5.5680 | 5.5681 | 0.0% |
| 5,846 × 2,504 | 10 | scores | 0.4619 | 0.4014 | -13.1% |
| 5,846 × 2,504 | 10 | loadings | 0.4527 | 0.4370 | -3.5% |

The terminal RMS was identical at every rank and shape. Study-shaped peak RSS
changed from 1,441,724 KiB to 1,440,584 KiB, which is effectively unchanged.
The optimized kernel phases were faster at the study shape, while total wall
time was unchanged within the resolution of this short local benchmark.
Consequently, this change removes avoidable allocation and copy work, but it
does not support a claim of end-to-end acceleration for that workload by
itself.

## Fidelity checks

The focused suite covers independent update equations, fixed seeded
trajectories, C- and Fortran-contiguous inputs, compact masks, covariance
writeback modes, and serial versus multithreaded execution:

```bash
pytest -q \
  tests/test_dense_input_views.py \
  tests/test_dense_masked_parallelism.py \
  tests/test_update_equations.py \
  tests/test_update_trajectory.py -m "not perf"
```

All 34 selected tests passed; four performance-marked cases were deselected.
