# Genomics dosage data

Genotype dosage matrices contain many zeros, and those zeros are measured
values: a sample carries no copy of the alternate allele. VBPCApy's sparse
input treats an unstored entry as missing, so ordinary 0/1/2 dosage data
should be supplied as a dense floating-point array with `NaN` (or an explicit
mask) for missing calls.

## Simulate a dosage matrix

The example has 300 samples (rows) from two populations and 400 variants
(columns). Allele frequencies differ slightly between the populations, and 5%
of genotype calls are missing.

```python
import numpy as np

rng = np.random.default_rng(7)

n_per_population, n_variants = 150, 400
base = rng.uniform(0.05, 0.5, size=n_variants)
shift = rng.normal(0.0, 0.1, size=n_variants)
frequencies = np.clip(np.vstack([base + shift, base - shift]), 0.01, 0.99)
population = np.repeat([0, 1], n_per_population)

dosage = rng.binomial(2, frequencies[population]).astype(np.float64)
dosage[rng.random(dosage.shape) < 0.05] = np.nan
```

With real data, convert every missing-genotype code to `np.nan` first. The
array has samples on rows and variants on columns.

## Inspect the matrix

```python
from vbpca_py import check_data

observed = np.isfinite(dosage)
report = check_data(dosage, mask=observed)
print(f"{len(report.warnings)} warnings, for example:")
for message in report.warnings[:3]:
    print(" ", message)
```

`check_data` and the missing-aware scalers use observed entries only. An
explicit mask is combined with the finite values, so a `NaN` is never treated
as observed. Rare variants have skewed dosage distributions, so skewness
warnings are expected here; they suggest a transform for continuous
measurements but are not a reason to transform dosages.

## Scale and fit

```python
from vbpca_py import VBPCA, MissingAwareStandardScaler

scaler = MissingAwareStandardScaler()
scaled = scaler.fit_transform(dosage, mask=observed)

model = VBPCA(n_components=5, maxiters=300, random_state=0)
model.fit(scaled, mask=observed)

print("Explained variance ratio:", model.explained_variance_ratio_.round(3))
first = model.scores_[:, 0]
print("Mean first score by population:", [
    round(float(first[population == group].mean()), 2) for group in (0, 1)
])
```

The first component separates the two populations. The other components are
shrunk to near zero by automatic relevance determination, because the
simulation has a single population contrast.

C- and Fortran-ordered dense arrays give the same fit; Fortran order can
avoid a layout conversion in some native operations.

## Hold out probes without preprocessing leakage

To assess prediction of withheld genotypes, choose the probe entries *before*
estimating the scaling statistics. Fit the scaler to the remaining training
entries and apply it to the probe values.

```python
from vbpca_py import make_xprobe_mask

probe_rng = np.random.default_rng(2026)
train_raw, probe_raw = make_xprobe_mask(
    dosage, fraction=0.10, rng=probe_rng, mask=observed
)
train_mask = np.isfinite(train_raw)
probe_mask = np.isfinite(probe_raw)

probe_scaler = MissingAwareStandardScaler().fit(train_raw, mask=train_mask)
x_train = probe_scaler.transform(train_raw, mask=train_mask)
x_probe = probe_scaler.transform(probe_raw, mask=probe_mask)

probe_model = VBPCA(n_components=5, maxiters=300, random_state=0)
probe_model.fit(x_train, xprobe=x_probe)
print(f"Held-out RMS error (scaled units): {probe_model.prms_:.3f}")
```

The finite entries of `x_probe` are the held-out targets. Natural missingness,
the withheld probes and the scaling statistics stay aligned, and no probe value
influences the scaler or the fit.

## Why ordinary CSR conversion is unsafe

```python
from scipy import sparse

unsafe = sparse.csr_matrix(np.nan_to_num(dosage))
print(f"Stored entries: {unsafe.nnz} of {int(observed.sum())} observed calls")
```

Converting to CSR drops the zero dosages, and VBPCApy would then treat those
calls as missing. Sparse input is valid only when every observed zero is
stored explicitly, or when every unstored position really is missing. Storing
the common zero genotype removes most of the memory advantage of CSR. The
sparse kernels can still be faster, so explicit-zero CSR is a reasonable
opt-in after measurement, but a dense array with a mask is the safer default.

The [genomics storage report](https://github.com/yoavram-lab/VBPCApy/blob/main/analysis/GENOMICS_STORAGE_REPORT.md)
compares storage and run time for the two representations.
