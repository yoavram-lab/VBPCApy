# VBPCApy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/877331914.svg)](https://doi.org/10.5281/zenodo.19389250)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://yoavram-lab.github.io/VBPCApy/)

Variational Bayesian PCA (Ilin & Raiko, 2010) with support for missing data, sparse masks, optional bias terms, and an orthogonal post-rotation to a PCA basis. The implementation follows the original MATLAB reference while adding Python-native APIs, fast C++ extensions, and runtime autotuning.

**[Documentation](https://yoavram-lab.github.io/VBPCApy/)** · **[API Reference](https://yoavram-lab.github.io/VBPCApy/api/vbpca/)** · **[Tutorials](https://yoavram-lab.github.io/VBPCApy/tutorials/basic-dense-pca/)**

## Statement of need

Missing values are common in scientific and industrial tabular datasets, but many analysis pipelines either impute first (masking uncertainty) or drop incomplete samples. VBPCApy models missingness directly and exposes posterior uncertainty outputs alongside reconstructions, enabling uncertainty-aware latent-factor analysis in a single reproducible Python API.

## Installation

**From PyPI** (pre-built wheels for Python 3.11–3.14, Linux/macOS/Windows):
```bash
pip install vbpca-py
```

**With plotting support:**
```bash
pip install vbpca-py[plot]
```

See the [installation guide](https://yoavram-lab.github.io/VBPCApy/getting-started/installation/) for building from source and Eigen setup.

## Quick start

```python
import numpy as np
from vbpca_py import VBPCA

# 50 features, 200 samples
x = np.random.randn(50, 200)
mask = np.ones_like(x)  # 1 = observed, 0 = missing

model = VBPCA(n_components=5, maxiters=100)
scores = model.fit_transform(x, mask=mask)
recon = model.reconstruction_
var = model.variance_
```

More examples: [quickstart](https://yoavram-lab.github.io/VBPCApy/getting-started/quickstart/), [dense PCA tutorial](https://yoavram-lab.github.io/VBPCApy/tutorials/basic-dense-pca/), [missing data & model selection](https://yoavram-lab.github.io/VBPCApy/tutorials/missing-data-model-selection/), [sparse data](https://yoavram-lab.github.io/VBPCApy/tutorials/sparse-data/).

## Features

- Dense or sparse data with explicit missing-entry masks
- Optional bias estimation and rotation to PCA-aligned solution
- Posterior covariances for scores and loadings; held-out probe RMS
- C++ extensions with runtime autotune for threading and memory
- Missing-aware preprocessing: one-hot, standard/minmax scaling, log, power, winsorize, auto-routing (`AutoEncoder`)
- Preflight data diagnostics via `check_data()` / `DataReport`
- scikit-learn-compatible estimator (`fit`/`transform`/`inverse_transform`)
- Model selection via `select_n_components` and `cross_validate_components`
- Configurable convergence: subspace angle, RMS/cost plateau, ELBO, curvature, composite rules, patience

See the [concept guides](https://yoavram-lab.github.io/VBPCApy/concepts/algorithm/) and [API reference](https://yoavram-lab.github.io/VBPCApy/api/vbpca/) for full details.

## Development

```bash
git clone https://github.com/yoavram-lab/VBPCApy.git
cd VBPCApy
uv sync --extra dev --extra plot
just ci          # lint + typecheck + test
just docs-serve  # local docs preview
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{vbpca_py2026,
  author = {Macdonald, Joshua and Naim, Shany and Ram, Yoav},
  title = {{VBPCApy}: Variational Bayesian PCA with Missing Data Support},
  year = {2026},
  url = {https://github.com/yoavram-lab/VBPCApy},
  version = {0.2.0},
}
```

```bibtex
@article{ilin2010practical,
  title={Practical Approaches to Principal Component Analysis in the Presence of Missing Values},
  author={Ilin, Alexander and Raiko, Tapani},
  journal={Journal of Machine Learning Research},
  volume={11},
  pages={1957--2000},
  year={2010}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable metadata.

## License

MIT — see [LICENSE](LICENSE).
