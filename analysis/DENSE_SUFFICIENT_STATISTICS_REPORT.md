# Dense sufficient-statistics crossover

Issue #210 evaluates an exact alternative to observed-cell accumulation for
masked dense VBPCA updates. The complement formulation computes complete-data
sufficient statistics with matrix multiplication and subtracts contributions
from missing cells.

## Correctness

Direct score and loading equation tests cover C- and Fortran-contiguous arrays
with one, two, and four threads. Fixed-iteration trajectory tests cover
complete data, MCAR, MAR, and structured missingness. Observed-cell and
complement modes agree to the test tolerance, and a component-selection
regression selects the same rank. In the study-shaped benchmark, the largest
absolute kernel-output difference was 1.9e-14.

## Study-shaped crossover

We benchmarked a 5,846 by 2,504 dense matrix at rank 10 with 24 threads. Each
entry is the median of five paired measurements after one warmup. Ratios below
one favor complement accumulation.

| Observed fraction | Score ratio | Loading ratio |
|---:|---:|---:|
| 0.50 | 1.644 | 2.432 |
| 0.65 | 1.382 | 1.695 |
| 0.75 | 1.262 | 1.457 |
| 0.82 | 1.075 | 1.148 |
| 0.90 | 0.975 | 0.839 |
| 1.00 | 0.570 | 0.921 |

The complement kernels are slower at the approximately 82% observed density
used by the genomics study. Repeated three-iteration full fits at that density
also showed no stable end-to-end advantage after accounting for run order.

## Decision

The default `"auto"` policy therefore retains observed-cell accumulation for
the study workload. It selects complement accumulation only for matrices with
at least 100,000 cells and observed fraction at or above
`min(0.99, max(0.75, 1.02 - 0.01 * rank))`. The margin is intentionally more
conservative than a single kernel crossover. Users can request `"observed"`
or `"complement"` explicitly for controlled comparisons. The resolved choice,
density, and threshold are recorded in the runtime report.

Reproduce the study-shaped kernel benchmark with:

```bash
python analysis/benchmark_dense_sufficient_statistics.py \
  --features 5846 --samples 2504 --components 10 --threads 24 \
  --densities 0.50 0.65 0.75 0.82 0.90 1.00 --repeats 5
```
