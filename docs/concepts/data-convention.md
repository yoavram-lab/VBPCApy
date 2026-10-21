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

Dense `np.ndarray` inputs can use NaN for missing entries. An optional Boolean
mask marks which values are observed:

```python
x = np.array([[1.0, np.nan, 3.0],
              [4.0, 5.0,    np.nan]])

mask = ~np.isnan(x)  # True = observed, False = missing

model = VBPCA(n_components=2)
model.fit(x, mask=mask)
```

- `mask` must have the same shape as `x`.
- `True` (or 1) means **observed**; `False` (or 0) means **missing**.
- Without a mask, NaNs are missing and other values, including zero, are observed.

## Sparse data

For `scipy.sparse.csr_matrix` or `csc_matrix` input, the **stored entries**
(including stored zeros) define the observation set:

```python
--8<-- "scripts/snippets/stored_observations.py"
```

Constructing CSR directly from a dense array would discard its zeros and store
only two observations in this example. Explicit coordinate construction keeps
all four. The example source above is also executed in the test suite.

If you pass a mask for sparse data, it must match `spones(X)` exactly — same
sparsity pattern with all stored values set to 1.

### When to use sparse vs dense

| Scenario | Format | Why |
|----------|--------|-----|
| Unstored positions are genuinely unobserved | **Sparse CSR/CSC** | The sparse structure is the observation mask |
| Zero is an observed value, including genotype dosage and count data | **Dense + explicit mask** | Ordinary CSR conversion drops observed zeros and changes the likelihood |
| Moderate dimensions with random missingness (NaN-masked tabular data) | **Dense + explicit mask** | Dense kernels benefit from BLAS; mask is straightforward |

Sparse storage is not determined by the numerical fraction of zeros alone. If
zero carries information, every observed zero would need to be stored
explicitly in CSR/CSC. At ordinary genotype density this usually costs at least
as much memory as a dense float64 array plus a boolean mask. See the
[genomics dosage tutorial](../tutorials/genomics-dosage.md) for the recommended
workflow.

## Compatibility modes

The `compat_mode` option controls how empty rows/columns and sparse masks are
handled:

- `"strict_legacy"` (default): matches the original MATLAB reference behaviour.
- `"modern"`: emptiness follows observed entries. Stored numeric zeros are
  observations, stored NaNs are missing, and an observed constant-zero feature
  is retained.

An explicit observation mask uses Boolean values in either mode. A false mask
entry is unobserved even when it occupies a stored sparse position.

Compatibility mode also resolves automatic mean/noise update orders. It is a
numerical policy choice, not a matrix-layout switch. Axis detection avoids a
full-shape dense mask; subsequent SciPy slicing can still allocate indexing
metadata proportional to an axis length.
