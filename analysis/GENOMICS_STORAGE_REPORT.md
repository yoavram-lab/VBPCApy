# Genomics storage benchmark

## Question

Does explicitly storing every observed genotype dosage, including zeros, make
the sparse VBPCA path worthwhile at the matrix shape planned for the
pp-eigentest example?

## Frozen local comparison

- Date: 2026-09-28
- Code: `30055beacea8073ede319a02c4587573bb690d72`
- VBPCApy: 0.4.2 development tree
- NumPy: 2.5.3
- SciPy: 1.18.1
- CPU: Intel Core Ultra 9 275HX, 24 allocated cores
- Shape: 5,846 variants by 2,504 samples
- Observed fraction: 0.82013
- Observed-zero fraction: 0.44370
- Components: 10
- Iterations: 3, with convergence stops disabled
- Replicates: 3 paired fits from the same data and initialization

Command:

```bash
just bench-genomics-storage --features 5846 --samples 2504 \
  --components 10 --iterations 3 --repeats 3 --threads 24
```

## Results

| Representation | Input storage | Median fit time | RMS |
|---|---:|---:|---:|
| Dense float64 + boolean mask | 131,745,456 bytes | 3.356 s | 0.6051845578 |
| CSR with every observed zero stored | 144,088,260 bytes | 2.419 s | 0.6051845578 |

Explicit-zero CSR used 1.094 times the input storage and 0.721 times the
dense-path runtime. RMS agreed to displayed precision in every paired fit.

## Decision

Dense data plus an explicit mask remain the default recommendation because the
representation directly preserves the observation model and ordinary sparse
conversion silently drops observed genotype zeros. Explicit-zero CSR is a
supported opt-in when runtime matters and the caller verifies that the sparse
structure contains every observed position. On this workload it traded about
9% more input storage for about 28% lower short-run fit time.

This comparison measures input arrays and three update iterations. It does not
replace process-level peak-memory measurement or the longer density crossover
study planned under issues 208–210. Re-run each representation separately
under `/usr/bin/time -v` before making a peak-memory claim.
