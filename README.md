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

More examples: [quickstart](https://yoavram-lab.github.io/VBPCApy/getting-started/quickstart/), [dense PCA tutorial](https://yoavram-lab.github.io/VBPCApy/tutorials/basic-dense-pca/), [missing data & model selection](https://yoavram-lab.github.io/VBPCApy/tutorials/missing-data-model-selection/), [genomics dosage data](https://yoavram-lab.github.io/VBPCApy/tutorials/genomics-dosage/), and [sparse data](https://yoavram-lab.github.io/VBPCApy/tutorials/sparse-data/).

## Features

For sparse observations whose values include zero, use `compat_mode="modern"`
and retain those zeros explicitly in storage. Modern fits use the mask-defined
observation set through noise updates, probe exclusion and NaN cleanup. See the
[sparse tutorial](docs/tutorials/sparse-data.md) for storage and mode choices.

- Dense or sparse data with explicit missing-entry masks
- Optional bias estimation and rotation to PCA-aligned solution
- Posterior covariances for scores and loadings; held-out probe RMS
- C++ extensions with runtime autotune for threading and memory
- Missing-aware preprocessing: one-hot, standard/minmax scaling, log, power, winsorize, auto-routing (`AutoEncoder`)
- Preflight data diagnostics via `check_data()` / `DataReport`
- scikit-learn-compatible estimator (`fit`/`transform`/`inverse_transform`, `get_params`/`set_params`, cloning) — scikit-learn is an optional dependency
- Model selection via `select_n_components` and `cross_validate_components`
- Configurable convergence: subspace angle, RMS/cost plateau, ELBO, curvature, composite rules, patience, with per-criterion enable/disable and custom ordering
- Convergence diagnostics (`n_iter_`, `converged_`, `convergence_reason_`, `learning_curve_`, best-probe state metadata) and `predictive_variance_` (includes observation noise)
- `recommend_config(n, p, priority)` — regime-aware default hyperparameters from a surrogate trade study

See the [concept guides](https://yoavram-lab.github.io/VBPCApy/concepts/algorithm/) and [API reference](https://yoavram-lab.github.io/VBPCApy/api/vbpca/) for full details.

## Typed fit results

`PCAResult` and `LearningCurve` are exported from `vbpca_py`. They describe
ordinary result dictionaries, retaining existing keys and array ownership.
Learning curves contain numeric histories and metadata with different types:

```python
from vbpca_py import LearningCurve

curve: LearningCurve | None = model.learning_curve_
if curve is not None:
    print(curve.get("convergence_reason"))
    print(curve.get("returned_iteration"))
```

The low-level `pca_full()` return annotation is `PCAResult`. See the
[result reference](docs/api/results.md) for array shapes, optional diagnostic
outputs and shared covariance blocks. Type annotations specify dtypes; they
do not validate array shapes or make arrays immutable.

## Categorical component selection

Dense `AutoEncoder` and `MissingAwareOneHotEncoder` fits expose an
`encoding_schema_` snapshot (`EncodingSchema` / `EncodedVariable`, exported
from `vbpca_py`). Pass it to CV to hold out whole variable cells and decode
single-binary, dropped-reference, centered and weighted indicators before
categorical scoring:

```python
from vbpca_py import AutoEncoder, CVConfig, cross_validate_components

encoder = AutoEncoder(
    column_types=["categorical", "categorical", "continuous"],
    drop="first",
    block_weighting="equal_variance",
    handle_unknown="raise",
)
Z = encoder.fit_transform(X)  # samples x encoded features
best_k, results = cross_validate_components(
    Z.T,
    components=range(5),
    config=CVConfig(encoding_schema=encoder.encoding_schema_, metric="brier"),
)
```

Continuous and ordinal variables contribute to encoded probe RMS, but are
excluded from nominal categorical scores. Gaussian reconstructions are
decoded, clipped at `1e-6` and normalized as approximate category
probabilities; this is not a calibrated categorical likelihood. Equivalent
decoded predictions have equivalent scores; refitting different encodings
can yield different predictions. Schema-free categorical selection requires
full unweighted one-hot blocks and rejects singleton groups and transformed
targets. `feature_groups` alone cannot identify binary versus numeric columns.

This API splits an already encoded matrix. A schema describes the transform;
use the raw-table API below when learning preprocessing within CV folds.
Unknown categories cannot be distinguished from a dropped reference when
`handle_unknown="ignore"`; use `"raise"` when scoring new raw data. See the
[model-selection guide](docs/concepts/model-selection.md) and
[benchmark validation note](scripts/benchmark_study.md).

Live fitting disables the legacy `slowing_down` criterion because no event
counter is supplied. Explicit requests warn, and the learning curve records its
unavailability. Stopping flags must be booleans. Accepted heuristic stops do not
certify posterior stability; supported thresholds and frozen study policies are
unchanged. See the [convergence guide](docs/concepts/convergence.md).

## Raw-table component selection

`cross_validate_raw_components` accepts **samples × original variables**, with
NaNs or an explicit observation mask. It holds out original variable cells
before fitting means, scales, categorical centering and block weights. A fresh
`AutoEncoder` is fitted once per fold and reused across every candidate rank,
including first-minimum early stopping.

```python
from vbpca_py import AutoEncoder, CVConfig, cross_validate_raw_components

def make_encoder():
    return AutoEncoder(
        column_types=["categorical", "continuous", "ordinal"],
        # External codebook; do not derive this from held-out values.
        column_levels=[["no", "yes"], None, ["low", "medium", "high"]],
        mean_center_ohe=True,
        block_weighting="equal_variance",
        handle_unknown="raise",
    )

best_k, results, folds = cross_validate_raw_components(
    X, encoder_factory=make_encoder, components=range(4),
    config=CVConfig(n_splits=3, metric="brier", seed=11),
    maxiters=200,
)
```

Declare column kinds explicitly. `column_levels=None` (or a `None` entry) learns
levels from training cells; a held-only unknown level raises an error. External
nominal vocabularies and ordinal orders support declared levels absent from
training. Each returned `PreprocessedFold` exposes original training/holdout
masks, encoded matrices, the fitted encoder and its schema;
`prepare_raw_cv_folds` constructs these without fitting candidate models.

Results include nominal Brier/log/accuracy scores and `rmse_variable_j` for each
numeric input column: continuous values in input units, ordinal values in
unrounded level-index units. Numeric summaries report contributing fold counts;
encoded `prms` is also retained. This predicts cells of the same samples;
sample-wise sklearn CV evaluates new samples. Run the complete mixed-data
example with `just example-raw-cv`, or
`python scripts/example_raw_cell_cv.py` after installing the package.

Native update kernels reject nonpositive or nonfinite noise variances and
malformed sparse pointers with `ValueError`. Worker factorization failures
raise `RuntimeError` after workers join. See the [runtime guide](docs/howto/runtime-tuning.md#native-failures)
for input and failed-fit semantics.

## Development

The [0.4.7 validation](docs/releases/0.4.7.md) covers native failure repairs
and numerical references. The [0.4.6 release validation](docs/releases/0.4.6.md)
describes the paired fitting-compatibility check and package/adoption gates. General scoring and
benchmark validation commands are in [scripts/benchmark_study.md](scripts/benchmark_study.md).

```bash
git clone https://github.com/yoavram-lab/VBPCApy.git
cd VBPCApy
uv sync --extra dev --extra plot
just ci          # lint + typecheck + numerical references + tests
just numerical-reference  # bounded mathematical/reference checks
just docs-serve  # local docs preview
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

The [numerical validation guide](docs/howto/numerical-validation.md) describes
the always-run reference checks, compatibility scope and tolerances. Extended
Octave comparisons remain separate from the required numerical gate.

The [October 2026 backend review](analysis/BACKEND_REVIEW_2026_10_20.md)
records remaining measured costs, reference-paper checks and hardware-tuning
requirements. Its diagnostic probes and algebraic prototypes are analysis
tools; they do not change the installed solver. Reproduction commands are in
[scripts/benchmark_study.md](scripts/benchmark_study.md#backend-review-probes).

The `trade-a-*` analysis recipes require a local
[trade-study](https://github.com/jcm-sci/trade-study) checkout. After syncing
the environment, install its full extras (including adaptive optimisation,
surrogate fitting and sensitivity analysis):

```bash
# Default: trade-study is checked out alongside VBPCApy.
just trade-a-install-adaptive

# Override the checkout location, including paths containing spaces.
TRADE_STUDY_PATH="/path/to/trade study" just trade-a-install-adaptive
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{vbpca_py2026,
  author = {Macdonald, Joshua and Naim, Shany and Ram, Yoav},
  title = {{VBPCApy}: Variational Bayesian PCA with Missing Data Support},
  year = {2026},
  url = {https://github.com/yoavram-lab/VBPCApy},
  version = {0.4.11},
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
