# Executed examples

These standalone scripts run during each documentation build. Every page shows
the executed source, its output and plots derived from the returned fitted
state. All data are synthetic. The short fits demonstrate the API; they do not
establish posterior convergence, calibration or a recommended component count.

All model inputs use samples × features. Numeric outputs use float64 arrays
unless an output container is selected explicitly. Training `scores_` contains
joint variational means; `transform(new_rows)` computes conditional projections
under the fixed fitted state. Missingness and scoring targets are described
with each example.

| Example | What it demonstrates |
| --- | --- |
| [Missing dense observations](dense/index.md) | Reconstruction, independent new rows and full latent covariance |
| [Sparse observations](sparse/index.md) | Stored observed zeros, probes and an equivalent dense observation pattern |
| [Mixed-data Pipeline](pipeline/index.md) | Raw masks, schema, row CV/GridSearchCV and decoding new rows in input units |
| [Cost versus probe RMS](selection/index.md) | Common held-out cells, separate metric units and explicit capacities |
| [Original-variable cell validation](raw-cv/index.md) | Training-only preprocessing, categorical scores and numeric units |
| [Plotting utilities](plots/index.md) | Scree, component loadings and explained variance from the fitted model |

These examples use the 0.5 API. From its matching source checkout, install
`python -m pip install '.[plot]'` to run the scripts. Published 0.4 packages use
the earlier estimator interface. Optional table output also needs
the `data` extra for pandas or the `polars` extra for Polars. The
[complete Pipeline tutorial](../tutorials/missing-aware-pipeline.md) executes
both supported table containers and describes metadata routing and scoring.

The [migration guide](../api/migration-05.md) explains the explicit feature-first
legacy adapter and compatibility choices. The source-collected examples in
the [estimator](../api/vbpca.md), [options](../api/options.md) and
[preprocessing](../api/preprocessing.md) references are executed in CI too.
