# Model Selection

## Cost diagnostic modes

Selection preserves full per-iteration cost traces by default, including when
the selection metric is PRMS. Cost-driven selection and stopping, and requested
replay evidence, retain those traces.

For an explicitly fixed PRMS stopping policy that does not consume cost, opt in
to a single cost evaluation per fitted candidate:

```python
from vbpca_py import SelectionConfig, select_n_components

best_k, metrics, trace, model = select_n_components(
    X,
    components=[0, 2, 4],
    config=SelectionConfig(metric="prms", cost_diagnostics="endpoint"),
    record_cost=False,
    convergence_criteria={"cost": False, "composite": False},
)
```

Endpoint mode preserves fitted factors, the stopping policy, selected capacity
and the historical returned-iteration cost. If probe stopping restores an
earlier iteration, its cost is evaluated using that saved state. Earlier cost
trace entries remain `NaN`; endpoint mode cannot supply a replay trace. An
initialization-only return and the mean-only rank-zero candidate have no
iterative variational cost, so their cost remains `NaN`.

The fitter rejects endpoint mode with `record_cost=True` or any enabled
cost-consuming event, including relative/curvature cost stops and composite
`elbo_rel`. Cost-based capacity selection requires trace mode. No stopping
criterion is silently disabled. `SelectionConfig.cost_diagnostics` controls
the selection sweep; standalone `VBPCA`/`pca_full` also accept the fit option
`cost_diagnostics="endpoint"`. Cross-validation accepts that fit option and
records the mode in its result table. Candidate traces and requested runtime
reports retain diagnostic-mode provenance.

The [paired sweep benchmark](../../scripts/benchmark_study.md#selection-cost-diagnostics)
compares total sweep time, endpoint metrics and cost-evaluation counts.

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
RMS (`"prms"`) or categorical Brier/log score as its selection objective;
variational cost is still reported
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
| `metric` | `"prms"` | Held-out probe RMS; with `feature_groups`, also `"brier"` or `"log_score"` |
| `one_se_rule` | `True` | Select the simplest model within 1 SE of the best |
| `seed` | `0` | Random seed for fold assignment and candidate fits |
| `feature_groups` | `None` | Variable label per feature; folds hold out whole (variable, sample) cells |
| `encoding_schema` | `None` | Fitted dense encoding metadata; derives groups and decodes categorical scores |
| `selection_rule` | `None` | `"one_se"`, `"minimum"` or `"first_minimum"`; `None` follows `one_se_rule` |
| `early_stop` | `False` | With `"first_minimum"`, stop the sweep once the rule triggers |

### 1-SE rule

When `one_se_rule=True`, the selected $k$ is the smallest value whose mean
CV metric is within one standard error of the overall best. This favours
simpler models with fewer components when cross-validation uncertainty does
not clearly support the additional complexity. The rule is a selection heuristic,
not a hypothesis test.

### First-minimum rule

`selection_rule="first_minimum"` walks the candidates in increasing order and
stops at the first $k$ whose successor does not lower the mean CV metric by
more than the successor's standard error. Unlike the 1-SE rule it never looks
past that point, so it is not drawn to a second descent at large $k$ (seen
when automatic relevance determination shrinks large models). With
`early_stop=True` the candidates above the selected one are never fitted.

### One-hot encoded data

Entrywise folds are not a valid holdout for one-hot encoded variables: the
other indicators of a held-out cell stay in training and, because they sum to
one, nearly determine the held-out value. When the data have real structure,
held-out error then keeps falling with rank and capacity is overestimated.
Pass the encoder's `encoding_schema_` so every indicator of a cell is held out
together. Each result then also reports categorical scores of the held-out
cells, reading the reconstructed block as level probabilities: `brier`,
`log_score` and `accuracy`. `metric="brier"` selects on the Brier score, which
is bounded and usually the most useful of the three; the log score is dominated
by cells whose observed level receives almost no predicted mass, because a
Gaussian low-rank reconstruction is not a calibrated categorical model.

```python
from vbpca_py import AutoEncoder, CVConfig, cross_validate_components

encoder = AutoEncoder(binary="both", handle_unknown="raise")
Z = encoder.fit_transform(X)  # samples x features
best_k, cv = cross_validate_components(
    Z.T,
    components=range(10),
    config=CVConfig(
        encoding_schema=encoder.encoding_schema_,
        selection_rule="first_minimum",
        metric="brier",
    ),
)
```

The fitted schema records variable kind, category order, encoded columns,
reference category, centering offsets and block weights. Scores first undo
the affine encoding, then restore a reference level as one minus the sum of
the retained indicators. This includes single-indicator binaries. Values
are clipped at `1e-6` and normalized before scoring. Continuous and ordinal
columns are excluded from nominal scores; `prms` remains in encoded units.
The schema also identifies a one-level categorical variable correctly.

`feature_groups` without a schema retains the full unweighted one-hot
convention. Categorical selection rejects singleton groups (their kind is
ambiguous) and targets that are not full one-hot blocks; pass fitted metadata
for mixed, centered, weighted or dropped encodings. An explicitly supplied
group list must match the schema. Partial categorical cells, invalid targets
and nonfinite held-out predictions raise errors instead of producing a
misleading score. Unknown levels coded as all zeros are rejected for full
blocks; a dropped-reference encoding cannot distinguish them from its
reference level, so use `handle_unknown="raise"` on new raw tables.

### Preprocessing and the CV target

This API accepts an already encoded matrix. A schema describes its transform,
but does not fit preprocessing separately inside each fold. Fixed or externally
fitted transforms can use it directly. For learned centering, scaling, category
frequencies or weights, use the raw-table path below. Centering in the dense
one-hot encoder is learned at fit time and remains fixed on transform.

Variable-cell reconstruction CV predicts missing cells of the same samples;
sample-wise sklearn CV evaluates held-out samples. These answer different
prediction questions.

## `cross_validate_raw_components` — training-only preprocessing

Pass a dense **samples-by-original-variables** table and a factory returning a
fresh `AutoEncoder` with explicit `column_types` and `handle_unknown="raise"`.
The raw mask has that same shape. Holdouts are assigned using only this
observation pattern and `CVConfig.seed`; changing held-out values cannot affect
fold assignments or the fitted preprocessing in that fold.

```python
from vbpca_py import AutoEncoder, CVConfig, cross_validate_raw_components

def make_encoder():
    return AutoEncoder(
        column_types=["categorical", "continuous", "ordinal"],
        column_levels=[["no", "yes"], None, ["low", "medium", "high"]],
        binary="single", mean_center_ohe=True,
        block_weighting="equal_variance", handle_unknown="raise",
    )

best_k, results, folds = cross_validate_raw_components(
    X, encoder_factory=make_encoder, components=range(4),
    config=CVConfig(n_splits=3, metric="brier", seed=11,
                    selection_rule="first_minimum", early_stop=True),
    maxiters=200,
)
```

All nontraining raw cells are replaced with NaNs before fitting each encoder.
Its fitted means, scales, category order and weights then transform both
training and held-out cells. Every encoded indicator belonging to a raw cell
shares that cell's observation mask. The same fitted encoder is reused for all
candidate ranks, including early stopping. The three return values are selected
capacity, aggregate records and `PreprocessedFold` audit objects. To inspect
preprocessing alone, call `prepare_raw_cv_folds` with the same inputs.

Use `column_levels` for vocabularies and ordinal order declared independently
of these held-out data, such as a survey codebook. A `None` entry learns levels
from training observations (ordinals use sorted training levels). Unknown
held-out levels raise an actionable error instead of becoming a dropped
reference. Declared but training-absent categories retain their indicator and
zero training frequency. A one-level categorical variable retains one indicator.

The training-support backbone remains in every training fold: not every observed
cell is validation-eligible. Every original variable/sample and encoded
feature/sample must retain training support. Too few eligible cells, nonfinite
encoded observations, or a categorical objective with no categorical holdouts
in a fold raise errors. Candidate capacities must fit the smallest encoded fold
dimension. Raw CV derives groups and schemas; leave `CVConfig.feature_groups`
and `encoding_schema` unset and do not supply an external `xprobe`.

Nominal scores decode each fold's schema. `rmse_variable_j` measures continuous
error in input units or ordinal error in unrounded level-index units; it is
reported per numeric variable, avoiding a combined score across incompatible
units. These metrics have `mean_`, `std_`, `se_`, per-fold values and
`n_folds_rmse_variable_j` counts. Folds without held-out cells of a variable have
NaN for that score and do not contribute to its numeric summary. If no fold can
hold out that variable, the mean is NaN and the contributing count is zero.
Numeric means average the contributing fold RMSEs, rather than pooling cells.
Selection still uses `prms` in encoded units or nominal Brier/log score.

This is reconstruction CV for missing cells of the same samples. For
generalization to new samples, use a sample-wise sklearn split and fit the
entire preprocessing/model pipeline on each training sample set. See the
installed-package example in `scripts/example_raw_cell_cv.py`.

## Choosing between the two

| | `select_n_components` | `cross_validate_components` |
|---|---|---|
| **Speed** | Faster — one fit per $k$ | Slower — $k \times \text{n\_splits}$ fits |
| **Reliability** | Good with probe set | More robust variance estimate |
| **Best for** | Quick exploration, large data | Repeated-fold comparisons and uncertainty summaries |
