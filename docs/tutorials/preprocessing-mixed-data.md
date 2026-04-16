# Preprocessing Mixed Tabular Data

Real-world datasets often mix categorical and continuous columns, contain
missing values, and have skewed or outlier-heavy distributions. This tutorial
shows how to use VBPCApy's missing-aware preprocessing pipeline to handle all
of these before fitting VB-PCA.

## Create a mixed dataset

```python
import numpy as np

rng = np.random.default_rng(42)
n_samples = 300

# Continuous columns (samples × features)
age = rng.normal(40, 12, n_samples)
income = rng.lognormal(10.5, 0.8, n_samples)

# Categorical column (3 categories encoded as integers)
region = rng.choice([0, 1, 2], size=n_samples)

# Stack into a single array
X = np.column_stack([age, income, region]).astype(float)

# Sprinkle 10% missing values
missing = rng.random(X.shape) < 0.10
X[missing] = np.nan

print(f"Shape: {X.shape}  ({n_samples} samples × 3 features)")
print(f"Missing: {np.isnan(X).sum()} entries ({100 * np.isnan(X).mean():.1f}%)")
```

## Preflight diagnostics

Before encoding, run `check_data` to flag potential issues:

```python
from vbpca_py import check_data

report = check_data(X, column_names=["age", "income", "region"])

print(f"Passed: {report.passed}")
for w in report.warnings:
    print(f"  ⚠ {w}")
print(f"\nSuggested transforms: {report.suggested_pretransforms}")
```

`check_data` inspects each column for skewness, outliers (by MAD), near-zero
variance, and high missing fractions. The `DataReport` suggests per-column
transforms such as log or winsorization.

## Encode with AutoEncoder

`AutoEncoder` routes each column through the appropriate transformer based on
its cardinality:

- Columns with ≤ `cardinality_threshold` unique values → one-hot encoded.
- Remaining columns → scaled (standard or min-max).

```python
from vbpca_py import AutoEncoder

auto = AutoEncoder(
    cardinality_threshold=10,
    continuous_scaler="standard",
)
Z = auto.fit_transform(X)

print(f"Encoded shape: {Z.shape}")
print(f"Feature names: {auto.feature_names_out_}")
```

!!! note "Convention mismatch"
    `AutoEncoder` expects **samples × features** (scikit-learn convention), but
    `VBPCA` expects **features × samples**. Transpose when passing to the model.

## Fit VB-PCA

```python
from vbpca_py import VBPCA

# Transpose: features × samples
Z_t = Z.T
mask_t = (~np.isnan(Z_t)).astype(float)

model = VBPCA(n_components=3, maxiters=200, verbose=1)
model.fit(Z_t, mask=mask_t)

print(f"RMS: {model.rms_:.4f}")
print(f"Cost: {model.cost_:.2f}")
```

## Round-trip reconstruction

```python
Z_recon_t = model.inverse_transform()    # features × samples
Z_recon = Z_recon_t.T                     # back to samples × features
X_recon = auto.inverse_transform(Z_recon)

print(f"Reconstructed shape: {X_recon.shape}")
```

The `inverse_transform` of `AutoEncoder` decodes one-hot columns back to
integer categories and inverts the scaling on continuous columns.

## Individual preprocessors

You can also use the individual transformers directly for finer control:

```python
from vbpca_py import (
    MissingAwareStandardScaler,
    MissingAwareOneHotEncoder,
    MissingAwareLogTransformer,
    MissingAwarePowerTransformer,
    MissingAwareWinsorizer,
)

# Log-transform a skewed column
log_tf = MissingAwareLogTransformer()
income_log = log_tf.fit_transform(income.reshape(-1, 1))

# Winsorize outliers to 1st–99th percentile
winsor = MissingAwareWinsorizer(lower_percentile=1, upper_percentile=99)
income_clipped = winsor.fit_transform(income.reshape(-1, 1))

# Yeo-Johnson power transform
power = MissingAwarePowerTransformer()
income_power = power.fit_transform(income.reshape(-1, 1))
```

All transformers preserve NaN entries through the round-trip and support
`inverse_transform` for reconstruction (except `MissingAwareWinsorizer`, which
is lossy).

## Next steps

- [Sparse Data Workflows](sparse-data.md) — work with sparse matrices.
- [Preprocessing API reference](../api/preprocessing.md) — full documentation for
  all transformers.
