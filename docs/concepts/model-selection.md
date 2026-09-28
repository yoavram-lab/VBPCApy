# Model Selection

VBPCApy provides two strategies for choosing the number of latent components $k$:
a sequential sweep and K-fold cross-validation.

## `select_n_components` — sequential sweep

Fits VB-PCA for each positive candidate $k$ and selects the best according to a
chosen metric. An explicitly requested rank zero is evaluated as a closed-form
mean-only model.

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
| `"rms"` | Training-set reconstruction RMS. Lower is better. |

The requested metric is the objective: selection raises a clear error if that
metric is unavailable instead of silently substituting another metric. For a
cost sweep, VBPCApy records cost without enabling a cost-based convergence
stop, so choosing the objective does not change when candidate fits terminate.

### Rank zero

Include `0` in `components` to compare against a mean-only model, for example
`components=range(0, 10)`. The mean is estimated separately for each feature
from the training entries. Rank zero can be selected with held-out probe RMS or
training RMS. It has no variational free energy, so a candidate set containing
zero cannot use `metric="cost"`. Positive ranks remain the default when
`components` is omitted. Rank-zero selection currently supports dense input.

### `SelectionConfig` fields

| Field | Default | Description |
|-------|---------|-------------|
| `metric` | `"prms"` | Selection metric |
| `stop_on_metric_reversal` | `False` | Stop sweeping when the metric worsens |
| `patience` | `None` | Stop after exactly this many consecutive non-improving candidates; `0` stops on the first miss, and `None` disables the stop |
| `max_trials` | `None` | Cap on the number of $k$ values tried |
| `compute_explained_variance` | `True` | Compute explained variance for the best model |
| `return_best_model` | `False` | Include the fitted `VBPCA` object in the return |
| `reuse_runtime_policy` | `True` | Reuse measured execution settings within compatible component-count regimes |

With runtime tuning enabled, a sweep measures execution settings once per
power-of-two component range, such as 4–7 or 8–15, and reuses those settings
for the remaining candidates in that range. Shape, observed-entry count, dense
or sparse representation, mask representation, and runtime options are part of
the compatibility key. The fitted posterior is never reused. Set
`reuse_runtime_policy=False` to tune every candidate independently. Trace
entries report `runtime_policy_source`, `runtime_policy_regime`, and
`runtime_tuning_sec`.

### Return value

`select_n_components` returns a 4-tuple:

1. `best_k` — the selected number of components.
2. `best_metrics` — endpoint metrics dict for the winning $k$.
3. `trace` — list of per-$k$ metric dicts.
4. `best_model` — the fitted `VBPCA` instance when requested and a positive rank wins; otherwise `None`.

## `cross_validate_components` — K-fold CV

Partitions the *observed entries* (not full rows) into folds, fits on each
training fold, and evaluates on the held-out fold.

Cross-validation currently accepts dense matrices only. It uses held-out probe
RMS (`"prms"`) as its selection objective; variational cost is still reported
as a training diagnostic but is not presented as a held-out CV metric. Fold
construction preserves at least one training observation in every non-empty
row and column.

```python
from vbpca_py import cross_validate_components, CVConfig

cfg = CVConfig(n_splits=5, metric="prms", one_se_rule=True)
best_k, results = cross_validate_components(
    x, mask=mask, components=range(0, 10), config=cfg, maxiters=200
)
```

### `CVConfig` fields

| Field | Default | Description |
|-------|---------|-------------|
| `n_splits` | `5` | Number of CV folds |
| `metric` | `"prms"` | Held-out probe RMS (the only supported CV objective) |
| `one_se_rule` | `True` | Select the simplest model within 1 SE of the best |
| `seed` | `0` | Random seed for fold assignment and candidate fits |

### 1-SE rule

When `one_se_rule=True`, the selected $k$ is the smallest value whose mean
CV metric is within one standard error of the overall best. This favours
simpler models with fewer components when cross-validation uncertainty does
not clearly support the additional complexity. The rule is a selection heuristic,
not a hypothesis test.

## Choosing between the two

| | `select_n_components` | `cross_validate_components` |
|---|---|---|
| **Speed** | Faster — one fit per $k$ | Slower — $k \times \text{n\_splits}$ fits |
| **Reliability** | Good with probe set | More robust variance estimate |
| **Best for** | Quick exploration, large data | Repeated-fold comparisons and uncertainty summaries |
