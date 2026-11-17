# Sparse Data Workflows

This tutorial uses SciPy sparse matrices whose **stored entries are the
observed values** and whose unstored positions are missing. Sparse storage is
appropriate only when that pattern matches what was measured.

## Construct sparse data

The example has 500 samples and 100 features with a rank-four signal; 20% of
entries are observed and stored.

```python
import numpy as np
import scipy.sparse as sp

rng = np.random.default_rng(42)

n_samples, n_features, true_rank = 500, 100, 4

signal = rng.standard_normal((n_samples, true_rank)) @ rng.standard_normal(
    (true_rank, n_features)
)
X_full = signal + 0.3 * rng.standard_normal((n_samples, n_features))

observed = rng.random((n_samples, n_features)) < 0.20
rows, cols = np.nonzero(observed)
X_sparse = sp.csr_matrix((X_full[rows, cols], (rows, cols)), shape=X_full.shape)

print(f"Shape: {X_sparse.shape}")
print(f"Stored entries: {X_sparse.nnz}")
print(f"Density: {X_sparse.nnz / np.prod(X_sparse.shape):.1%}")
```

## Stored entries define the observations

For sparse input, the stored entries, including any explicitly stored zeros,
are the observed values. Positions that are not stored are treated as
missing.

!!! warning "Observed zeros must be stored explicitly"
    Converting a dense array with `sp.csr_matrix(dense)` drops its zeros. Do
    not use that conversion for genotype dosages, counts, one-hot indicators
    or other data in which zero is a measured value, unless every observed
    zero is kept in the sparse structure. Dense data with an explicit mask are
    usually clearer in that case; see the
    [genomics dosage tutorial](genomics-dosage.md).

Because the storage pattern is the observation pattern, a separate mask is
usually unnecessary:

```python
from vbpca_py import VBPCA

model = VBPCA(n_components=4, maxiters=200, random_state=0)
model.fit(X_sparse)
print(f"Training RMS: {model.rms_:.4f}")
```

An explicit sparse mask must have exactly the same stored positions as the
data:

```python
mask = X_sparse.copy()
mask.data[:] = 1.0

masked_model = VBPCA(n_components=4, maxiters=200, random_state=0)
masked_model.fit(X_sparse, mask=mask)
```

## Share covariance across repeated observation patterns

Samples with the same set of observed features can share one score-covariance
matrix. `uniquesv=True` groups such samples using the sparse structure:

```python
grouped = VBPCA(
    n_components=4,
    maxiters=200,
    uniquesv=True,
    return_diagnostics=False,
    random_state=0,
)
grouped.fit(X_sparse)
```

Whether this saves time depends on how often observation patterns repeat.
`return_diagnostics=False` skips the full posterior variance outputs, which
are dense samples x features arrays.

## Sparse preprocessing

`MissingAwareSparseOneHotEncoder` one-hot encodes categorical columns stored
as sparse numeric codes:

```python
from vbpca_py import MissingAwareSparseOneHotEncoder

codes = rng.choice([0, 1, 2, 3], size=100).astype(float)
column = sp.csr_matrix(
    (codes, (np.arange(100), np.zeros(100, dtype=int))), shape=(100, 1)
)

encoder = MissingAwareSparseOneHotEncoder()
encoded = encoder.fit_transform(column)

print(f"Encoded shape: {encoded.shape}")
print(f"Encoded type: {type(encoded).__name__}")
```

!!! warning "Numeric categories only"
    The sparse encoder requires numeric category codes; strings cannot be
    stored in a SciPy sparse matrix.

## Compatibility modes

`compat_mode` selects how observations and edge cases are handled:

```python
modern = VBPCA(n_components=4, compat_mode="modern")  # default
historical = VBPCA(n_components=4, compat_mode="strict_legacy")
```

The default `"modern"` mode uses the stored entries (including stored zeros)
as the observations throughout fitting, probe handling and noise updates.
`"strict_legacy"` reproduces the original MATLAB implementation's handling.
The two can give slightly different fits; record the mode with your analysis
settings. See [Compatibility Modes](../howto/compatibility-modes.md) for the
full comparison.

## Dense and sparse fits agree

The same observations supplied as a dense array with `NaN` for missing entries
give the same fit:

```python
X_dense = np.where(observed, X_full, np.nan)

dense_model = VBPCA(n_components=4, maxiters=200, random_state=0)
dense_model.fit(X_dense)

print(f"Dense RMS:  {dense_model.rms_:.6f}")
print(f"Sparse RMS: {model.rms_:.6f}")
print(
    "Largest component difference:",
    float(np.abs(dense_model.components_ - model.components_).max()),
)
```

Fitting uses sparse kernels, but the full reconstruction and its posterior
variance are dense arrays.

!!! note "`inverse_transform` returns dense arrays"
    `inverse_transform` and `reconstruction_` are always dense `numpy`
    arrays, even for sparse input.

## Next steps

- [Cross-Validation & Diagnostics](cv-diagnostics.md): choose the number of
  components with K-fold cross-validation.
- [Data Convention](../concepts/data-convention.md): choosing between sparse
  and dense input.
