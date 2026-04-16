# Model Selection

VBPCApy provides two strategies for choosing the number of latent components $k$:
a sequential sweep and K-fold cross-validation.

## `select_n_components` — sequential sweep

Fits VB-PCA for each candidate $k$ and selects the best according to a
chosen metric.

```python
from vbpca_py import select_n_components, SelectionConfig

cfg = SelectionConfig(metric="cost", patience=2, max_trials=12)
best_k, metrics, trace, best_model = select_n_components(
    x, mask=mask, components=range(1, 15), config=cfg, maxiters=200
)
```

### Available metrics

| Metric | Description |
|--------|-------------|
| `"cost"` | Variational free energy (negative ELBO). Lower is better. |
| `"prms"` | Probe-set RMS — reconstruction error on held-out entries. Requires a probe set via `xprobe` or `xprobe_fraction`. |

### `SelectionConfig` fields

| Field | Default | Description |
|-------|---------|-------------|
| `metric` | `"prms"` | Selection metric |
| `stop_on_metric_reversal` | `True` | Stop sweeping when the metric worsens |
| `patience` | `None` | Consecutive worsening trials before stopping |
| `max_trials` | `None` | Cap on the number of $k$ values tried |
| `compute_explained_variance` | `True` | Compute explained variance for the best model |
| `return_best_model` | `False` | Include the fitted `VBPCA` object in the return |

### Return value

`select_n_components` returns a 4-tuple:

1. `best_k` — the selected number of components.
2. `best_metrics` — endpoint metrics dict for the winning $k$.
3. `trace` — list of per-$k$ metric dicts.
4. `best_model` — the fitted `VBPCA` instance (if `return_best_model=True`, else `None`).

## `cross_validate_components` — K-fold CV

Partitions the *observed entries* (not full rows) into folds, fits on each
training fold, and evaluates on the held-out fold.

```python
from vbpca_py import cross_validate_components, CVConfig

cfg = CVConfig(n_splits=5, metric="prms", one_se_rule=True)
best_k, results = cross_validate_components(
    x, mask=mask, components=range(1, 10), config=cfg, maxiters=200
)
```

### `CVConfig` fields

| Field | Default | Description |
|-------|---------|-------------|
| `n_splits` | `5` | Number of CV folds |
| `metric` | `"prms"` | Metric to evaluate on held-out entries |
| `one_se_rule` | `True` | Select the simplest model within 1 SE of the best |
| `seed` | `0` | Random seed for fold assignment |

### 1-SE rule

When `one_se_rule=True`, the selected $k$ is the smallest value whose mean
CV metric is within one standard error of the overall best. This favours
simpler models — fewer components — when the improvement from additional
components is not statistically significant.

## Choosing between the two

| | `select_n_components` | `cross_validate_components` |
|---|---|---|
| **Speed** | Faster — one fit per $k$ | Slower — $k \times \text{n\_splits}$ fits |
| **Reliability** | Good with probe set | More robust variance estimate |
| **Best for** | Quick exploration, large data | Publication-quality model selection |
