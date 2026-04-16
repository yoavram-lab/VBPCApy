# Data Convention

## Matrix layout

VBPCApy follows the convention from the original MATLAB implementation:

$$
X \in \mathbb{R}^{D \times N}
$$

where $D$ = number of features (rows) and $N$ = number of samples (columns).

!!! warning "This differs from scikit-learn"
    scikit-learn uses samples × features (each *row* is an observation). If your
    data is in sklearn layout, **transpose before passing to VBPCA**:

    ```python
    model.fit(X.T)
    ```

### AutoEncoder convention

`AutoEncoder` (and the other `MissingAware*` preprocessors) expect
**samples × features** — the sklearn convention. This means a typical pipeline
looks like:

```python
auto = AutoEncoder(cardinality_threshold=10)
z = auto.fit_transform(x_samples_by_features)

model = VBPCA(n_components=5)
model.fit(z.T)  # transpose to features × samples

z_recon = model.inverse_transform().T  # back to samples × features
x_recon = auto.inverse_transform(z_recon)
```

## Dense data with missing entries

For dense `np.ndarray` input, missing entries are indicated by an explicit boolean mask:

```python
x = np.array([[1.0, np.nan, 3.0],
              [4.0, 5.0,    np.nan]])

mask = ~np.isnan(x)  # True = observed, False = missing

model = VBPCA(n_components=2)
model.fit(x, mask=mask)
```

- `mask` must have the same shape as `x`.
- `True` (or 1) means **observed**; `False` (or 0) means **missing**.
- If all entries are observed, you can omit the mask entirely.

## Sparse data

For `scipy.sparse.csr_matrix` or `csc_matrix` input, the **stored entries**
(including stored zeros) define the observation set:

```python
import scipy.sparse as sp

x_sparse = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
# Stored entries: (0,0)=1, (0,1)=0, (1,0)=0, (1,1)=2
# All four entries are "observed"
```

If you pass a mask for sparse data, it must match `spones(X)` exactly — same
sparsity pattern with all stored values set to 1.

### When to use sparse vs dense

| Scenario | Format | Why |
|----------|--------|-----|
| High-dimensional data with structural zeros (genomics counts, one-hot surveys) | **Sparse CSR/CSC** | Implicit observation mask; sparse kernels avoid materialising the full matrix |
| Moderate dimensions with random missingness (NaN-masked tabular data) | **Dense + explicit mask** | Dense kernels benefit from BLAS; mask is straightforward |

## Compatibility modes

The `compat_mode` option controls how empty rows/columns and sparse masks are
handled:

- `"strict_legacy"` (default): matches the original MATLAB reference behaviour.
- `"modern"`: updated semantics for edge cases in sparse empty-row/column detection.
