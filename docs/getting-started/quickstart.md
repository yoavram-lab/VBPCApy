# Quick Start

## Dense data

VBPCApy expects data in **features × samples** layout (each column is one observation).

```python
import numpy as np
from vbpca_py import VBPCA

# 50 features, 200 samples
x = np.random.randn(50, 200)

model = VBPCA(n_components=5, maxiters=100)
scores = model.fit_transform(x)
recon = model.inverse_transform()

# Learned attributes
print("Components shape:", model.components_.shape)   # (50, 5)
print("Scores shape:", model.scores_.shape)            # (5, 200)
print("RMS:", model.rms_)
print("Final cost:", model.cost_)
```

### With missing entries

Pass a boolean mask where `True` = observed, `False` = missing:

```python
mask = np.ones_like(x, dtype=bool)
mask[x < -2] = False  # mask some entries

model = VBPCA(n_components=5, maxiters=100)
scores = model.fit_transform(x, mask=mask)

# Reconstruction and marginal variance
recon = model.reconstruction_
var = model.variance_
```

### Preprocessing pipeline

For mixed categorical + continuous data, use `AutoEncoder` to encode before fitting:

```python
from vbpca_py import AutoEncoder

auto = AutoEncoder(cardinality_threshold=10, continuous_scaler="standard")
z = auto.fit_transform(x, mask=mask)

model = VBPCA(n_components=5, maxiters=100)
scores = model.fit_transform(z, mask=np.ones_like(z, dtype=bool))

# Round-trip back to original space
z_recon = model.inverse_transform()
x_recon = auto.inverse_transform(z_recon)
```

## Sparse data

Sparse inputs must be CSR or CSC. The stored entries define the observation set (including stored zeros):

```python
import scipy.sparse as sp
from vbpca_py import VBPCA

x_sparse = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])

# Mask must match spones(X); omit to infer from X
mask = x_sparse.copy()
mask.data[:] = 1.0

model = VBPCA(n_components=2, maxiters=100)
scores = model.fit_transform(x_sparse, mask=mask)
```

!!! note "Dense vs sparse masks"
    - **Dense:** pass a boolean mask of 0/1 with the same shape.
    - **Sparse:** the observation set is the stored entries of `X` (including stored zeros). If you pass a mask it must match `spones(X)` exactly.

## Key options

| Option | Description | Default |
|--------|-------------|---------|
| `n_components` | Number of latent components | *required* |
| `bias` | Estimate per-feature mean | `True` |
| `maxiters` | Maximum EM iterations | `1000` |
| `tol` | Convergence tolerance | `1e-4` |
| `verbose` | Logging verbosity (0, 1, or 2) | `0` |
| `xprobe_fraction` | Fraction of entries to hold out as probe | `0.0` |

See [VBPCA API reference](../api/vbpca.md) for the complete list.
