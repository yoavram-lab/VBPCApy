# A missing-aware Pipeline from fit to reconstruction

This example uses the 0.5 API. Run its standalone script from a matching checkout
using the [installation instructions](../getting-started/installation.md).
The input has 36 samples and two original variables: a categorical region and
a continuous size. NaNs are missing. A Boolean mask also excludes arbitrary
placeholder values, including strings in the continuous column.

```bash
python scripts/example_masked_pipeline.py
```

The Pipeline consumes samples × variables throughout. `ObservationMasker`
replaces excluded dense values with NaNs before `AutoEncoder` learns categories
and scaling. The encoder expands the region into indicator features and
standardizes size. `VBPCA` receives that samples × encoded-features table.
No transpose or raw mask is passed across feature expansion.

## Row validation and capacity search

Enable sklearn metadata routing with
`config_context(enable_metadata_routing=True)`. Pass the original-variable
mask as `observation_mask=mask` for Pipeline/GridSearchCV fitting, or through
`params={"observation_mask": mask}` for `cross_validate`. Each validation split
slices data and mask by sample, then learns categories and scaling on training
rows alone. The search compares capacities zero and one and refits the selected
Pipeline on all training rows.

The default unsupervised scorer averages observed-entry Gaussian marginal
log densities across validation rows; larger values are preferred. It uses
fitted loading means and noise, integrating the latent scores. It does not
integrate posterior loading uncertainty. This rowwise target differs from
the cellwise PRMS/Brier targets in the [component-selection guide](../concepts/model-selection.md).

A budget of eight fitting iterations keeps this an API example. The result does not establish
posterior convergence, predictive calibration or a recommended capacity for
other data.

## New rows and inverse reconstruction

The four new rows are separate from the fitting table and have their own mask.
Call `search.best_estimator_.transform(new_data, observation_mask=new_mask)`:
the search wrapper's `transform` method does not route mask metadata. These
are conditional projections under fixed fitted parameters; joint training
variational means remain in the model's `scores_`. A completely missing new
row receives zero latent mean.

`fitted[1:].inverse_transform(projected)` reverses the fitted model and encoder,
returning region labels and continuous sizes in original units. The leading
masker has no inverse, so this explicitly uses the numerical/encoding portion
of the fitted Pipeline. Reconstruction predicts all original variables; it
does not restore hidden placeholders, recover the input mask or guarantee
equality with observed values. No fitting occurs during this operation.

## Names and output tables

The default run needs NumPy, SciPy and sklearn, without table dependencies.
For named input and pandas output, install the `data` extra; for this mixed
Pipeline with Polars output, install both `data` and `polars` extras:

```bash
python scripts/example_masked_pipeline.py --output pandas
python scripts/example_masked_pipeline.py --output polars
```

Pandas retains the new-row index. Polars preserves row order; the raw object
masking step stays in pandas mode and subsequent numeric outputs use Polars.
Use `get_feature_names_out()` to inspect encoded features and latent output
names. Inverse reconstruction returns an array in original variable order,
regardless of the configured transform output container.

## Executed source

The source below is the same standalone script exercised on the minimum and
current supported sklearn versions, including both optional table modes.

```python
--8<-- "scripts/example_masked_pipeline.py"
```
