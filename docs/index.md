# VBPCApy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/877331914.svg)](https://doi.org/10.5281/zenodo.19389250)

**Variational Bayesian PCA with missing data support.**

VBPCApy implements Variational Bayesian Principal Component Analysis
(Ilin & Raiko, 2010) with native support for incomplete observations,
sparse masks, and posterior uncertainty quantification. It provides a
scikit-learn-compatible estimator, missing-aware preprocessing utilities,
and empirical model selection for the number of latent components.

## Quick install

```bash
pip install vbpca-py
```

## Minimal example

```python
import numpy as np
from vbpca_py import VBPCA

x = np.random.randn(50, 200)  # 50 features × 200 samples
model = VBPCA(n_components=5, maxiters=100)
scores = model.fit_transform(x)
recon = model.inverse_transform()
```

## Next steps

- [Installation](getting-started/installation.md) — full install guide with extras and build-from-source instructions.
- [Quick Start](getting-started/quickstart.md) — dense and sparse examples with annotations.
- [Tutorials](tutorials/basic-dense-pca.md) — narrative walkthroughs for common workflows.
- [API Reference](api/vbpca.md) — complete public API documentation.
