# Genomics dosage data

Genotype dosage matrices contain many zeros, but those zeros are measured
values. VBPCApy's sparse solver assigns a different meaning to an unstored
entry: it is missing. Use a dense floating-point array and an explicit
observation mask for ordinary 0/1/2 dosage data.

The preprocessing utilities use the scikit-learn convention of samples by
features. The VBPCA estimator uses features by samples, so transpose only after
preprocessing.

## Prepare and inspect the matrix

Convert every missing-value code to `np.nan` before constructing the mask.
The mask remains the source of truth when additional entries are hidden for an
evaluation.

```python
import numpy as np

from vbpca_py import MissingAwareStandardScaler, VBPCA, check_data, make_xprobe_mask

# dosage has shape (n_samples, n_variants) and contains 0, 1, 2, or NaN.
dosage = np.asarray(dosage, dtype=np.float64)
observed = np.isfinite(dosage)

report = check_data(dosage, mask=observed)
for message in report.warnings:
    print(message)
```

`check_data` and `MissingAwareStandardScaler` use observed entries only.
An explicit mask is intersected with finite values, so a `NaN` is never used
as an observed value even when its mask entry is accidentally true.

## Scale and fit a production model

```python
scaler = MissingAwareStandardScaler()
scaled = scaler.fit_transform(dosage, mask=observed)

# VBPCA uses features by samples.
x = np.asfortranarray(scaled.T)
mask = np.asfortranarray(observed.T)

model = VBPCA(n_components=10, maxiters=500, random_state=0)
model.fit(x, mask=mask)
```

C- and Fortran-contiguous dense inputs have the same statistical semantics.
Fortran order can reduce layout conversion in some native operations, but it
does not change the fitted model.

## Create a probe set without preprocessing leakage

When probe entries are used to assess prediction, choose them before estimating
the scaling statistics. Fit the scaler on the remaining training entries, then
apply those statistics to the probe values.

```python
rng = np.random.default_rng(2026)
train_raw_t, probe_raw_t = make_xprobe_mask(
    dosage.T,
    fraction=0.10,
    rng=rng,
    mask=observed.T,
)

train_raw = train_raw_t.T
train_mask = np.isfinite(train_raw)
probe_raw = probe_raw_t.T
probe_mask = np.isfinite(probe_raw)

scaler = MissingAwareStandardScaler().fit(train_raw, mask=train_mask)
x_train = scaler.transform(train_raw, mask=train_mask).T
x_probe = scaler.transform(probe_raw, mask=probe_mask).T

model = VBPCA(n_components=10, maxiters=500, random_state=0)
model.fit(x_train, mask=np.isfinite(x_train))
```

The finite entries of `x_probe` are the held-out targets. This construction
keeps natural missingness, injected probe missingness, and scaling aligned.

## Why ordinary CSR conversion is unsafe

```python
from scipy import sparse

unsafe = sparse.csr_matrix(dosage)
```

This conversion normally drops dosage zeros. VBPCApy would consequently treat
those positions as unobserved. A sparse representation is valid only when
every observed zero is explicitly stored, or when every unstored position
really is missing. Explicitly storing the common zero genotype usually removes
the memory advantage of CSR. The sparse kernels can still be faster, so an
explicit-zero CSR representation is a reasonable measured opt-in. Dense data
plus a mask remain the safer default because their meaning is unambiguous.

The reproducible storage and runtime comparison is in the
[`genomics storage report`](https://github.com/yoavram-lab/VBPCApy/blob/main/analysis/GENOMICS_STORAGE_REPORT.md).
