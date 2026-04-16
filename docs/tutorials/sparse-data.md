# Sparse Data Workflows

This tutorial shows how to use VBPCApy with sparse matrices — the natural
format for high-dimensional data with structural zeros such as genomics count
matrices, term-document matrices, or one-hot-encoded survey data.

## Construct sparse data

```python
import numpy as np
import scipy.sparse as sp

rng = np.random.default_rng(42)

n_features, n_samples, true_rank = 100, 500, 4

# Low-rank signal
W = rng.standard_normal((n_features, true_rank))
S = rng.standard_normal((true_rank, n_samples))
X_dense = W @ S + 0.3 * rng.standard_normal((n_features, n_samples))

# Zero out 80% of entries to simulate structural sparsity
sparsity_mask = rng.random((n_features, n_samples)) < 0.20
X_dense[~sparsity_mask] = 0.0

X_sparse = sp.csr_matrix(X_dense)

print(f"Shape: {X_sparse.shape}")
print(f"Stored entries (nnz): {X_sparse.nnz}")
print(f"Density: {X_sparse.nnz / np.prod(X_sparse.shape):.2%}")
```

## Key concept: sparsity = observation mask

For sparse inputs, the **stored entries** (including stored zeros) define the
observation set. Entries not stored in the CSR/CSC structure are treated as
**unobserved** (missing).

This means you generally do **not** need to pass a separate mask:

```python
from vbpca_py import VBPCA

model = VBPCA(n_components=4, maxiters=200, verbose=1)
scores = model.fit_transform(X_sparse)
```

If you do pass an explicit mask, it must match `spones(X)` exactly:

```python
mask = X_sparse.copy()
mask.data[:] = 1.0

model = VBPCA(n_components=4, maxiters=200)
model.fit(X_sparse, mask=mask)
```

## Sparse preprocessing

For categorical data stored in sparse format, use `MissingAwareSparseOneHotEncoder`:

```python
from vbpca_py import MissingAwareSparseOneHotEncoder

# Single categorical column as a sparse vector
cat_col = sp.csr_matrix(rng.choice([0, 1, 2, 3], size=(100, 1)).astype(float))

enc = MissingAwareSparseOneHotEncoder()
encoded = enc.fit_transform(cat_col)

print(f"Encoded shape: {encoded.shape}")
print(f"Encoded type: {type(encoded)}")  # csr_matrix
```

!!! warning "Numeric categories only"
    `MissingAwareSparseOneHotEncoder` requires numeric category values. String
    categories cannot survive the CSR round-trip.

## Compatibility modes

The `compat_mode` option controls edge-case handling for sparse data:

```python
# Default: matches original MATLAB reference behaviour
model = VBPCA(n_components=4, compat_mode="strict_legacy")

# Updated semantics for empty-row/column detection
model = VBPCA(n_components=4, compat_mode="modern")
```

The difference affects how entirely-empty rows or columns are detected and
removed before fitting. For most datasets the results are identical.

## Dense vs sparse comparison

```python
# Densify for comparison (only for small matrices!)
X_dense_full = X_sparse.toarray()
mask_dense = (X_dense_full != 0).astype(float)

model_dense = VBPCA(n_components=4, maxiters=200)
model_dense.fit(X_dense_full, mask=mask_dense)

model_sparse = VBPCA(n_components=4, maxiters=200)
model_sparse.fit(X_sparse)

print(f"Dense  RMS: {model_dense.rms_:.6f}")
print(f"Sparse RMS: {model_sparse.rms_:.6f}")
```

The results should be numerically close. The sparse path avoids materialising
the full matrix and uses specialised C++ kernels.

!!! note "`inverse_transform` returns dense"
    Even when the input is sparse, `inverse_transform()` always returns a dense
    `np.ndarray`. This is a known limitation.

## Next steps

- [Cross-Validation & Diagnostics](cv-diagnostics.md) — select $k$ with K-fold
  CV and inspect convergence.
- [Data Convention](../concepts/data-convention.md) — when to choose sparse vs dense.
