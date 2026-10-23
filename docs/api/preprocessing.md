# Preprocessing

Transformers validate nonempty 2-D input at fit and inference.
Transforms check fitted feature counts and supplied string feature names/order;
inverse transforms check encoded output width. Numeric transformers reject
non-numeric input, infinity in observed entries and complex data. Dense
categorical encoders also accept string/object labels. NaNs remain missing.
See the [0.5 migration guide](migration-05.md) for changed errors, fitted-state
handling and mask boundaries.

Sparse inputs use stored-entry observations, including stored zeros. Supported
CSR/CSC/COO matrices and arrays are validated before conversion; stored NaNs
are removed and duplicate coordinates are summed without modifying the input.
Sparse encoders and scalers reject extra masks. Sparse one-hot encoding supports
independent category blocks for multiple input columns, with flat
`categories_` and output-to-input `feature_groups_` metadata. Dense numeric
input to this encoder observes every non-NaN value, including zero; it returns
the same sparse indicator representation.

Use a leading `ObservationMasker` to consume explicit Boolean masks before
these transformers. It produces dense NaNs or sparse stored observations and
leaves downstream steps to infer missingness after any feature expansion.
Direct calls accept `mask=...`; Pipeline metadata routing uses the dedicated
`observation_mask=...` alias. Enable routing with
`sklearn.config_context(enable_metadata_routing=True)`. Cross-validation and
GridSearchCV then slice masks for both fitting and scoring. Masks align by row
and column position, never by labels. Leave `Pipeline.transform_input` unset;
do not route original-variable masks to width-changing downstream steps.

Sparse-data masks may be dense or sparse Boolean arrays/matrices. A true mask
entry declares an observation, including a zero absent from data storage;
duplicate true entries form a union. Stored NaNs remain missing. Dense data
require dense Boolean masks. Raw sparse buffers are validated before conversion,
and observed infinity is rejected before duplicate aggregation. Inputs and
masks are not modified. The masker stores feature metadata only and requires a
fresh mask at inference when placeholders need exclusion. Masked dense outputs
own their arrays; unmasked outputs may borrow validated input. Sparse
normalization never creates a dense data/mask copy.
Dense validation and mask application use bounded scratch blocks alongside the
returned array, rather than another full observed-value or inverse-mask table.

The [executable example](https://github.com/yoavram-lab/VBPCApy/blob/main/scripts/example_masked_pipeline.py)
covers mixed string/numeric input, feature expansion, row CV, GridSearchCV and
new-row projection. This is ordinary row-wise scoring; original-cell holdouts
use the separate raw-table component-selection API. See
[scikit-learn's routing guide](https://scikit-learn.org/stable/metadata_routing.html)
for the opt-in and alias mechanism.

For masked projection after a grid search, call
`search.best_estimator_.transform(X_new, observation_mask=new_mask)`.
The supported sklearn search wrapper's `transform` does not accept metadata;
fit and score route it, and the selected Pipeline handles new-row transforms.

::: vbpca_py.ObservationMasker

The standard and min-max scalers use each finite, nonzero computed standard
deviation or range, even when its magnitude is small. A change of measurement
units therefore does not by itself classify a nonconstant column as constant.
Exactly constant observed columns use a unit denominator; wholly missing
columns retain NaN mean/variance or extrema and also use a unit denominator.
Zero or nonfinite computed scales use the same fallback. These rules apply to
dense observed entries and stored sparse observations, including stored zeros.
Sparse standardization computes centered moments in bounded chunks of stored
entries without making a dense copy of the data.

::: vbpca_py.AutoEncoder

::: vbpca_py.MissingAwareOneHotEncoder

::: vbpca_py.MissingAwareSparseOneHotEncoder

::: vbpca_py.preprocessing.MissingAwareOrdinalEncoder

::: vbpca_py.MissingAwareStandardScaler

::: vbpca_py.MissingAwareMinMaxScaler

::: vbpca_py.MissingAwareLogTransformer

::: vbpca_py.MissingAwarePowerTransformer

::: vbpca_py.MissingAwareWinsorizer

::: vbpca_py.check_data

::: vbpca_py.DataReport

## Feature names and output containers

All ten preprocessing transformers and native VBPCA implement
`get_feature_names_out(input_features=None)`. String input names are captured
at fit and checked in order. Unnamed inputs and tables with wholly non-string
column labels use positional `x0`, `x1`, ... names. Mixed string/non-string
labels and duplicate table columns are rejected. Supplied `input_features`
must be a unique one-dimensional string sequence matching fitted names when
present. Returned arrays own their name membership.

Scalers, ordinal encoding and the masker preserve the validated input names.
One-hot names follow fitted category order and reference drops; AutoEncoder
uses the original variable's name for each fitted plan. Generated collisions
receive stable `__<output-position>` suffixes, incremented if necessary.
Existing encoder `feature_names_out_` lists agree with default method output
and no longer change during transform. For unnamed fits, the older `col0`
display prefix becomes `x0`. VBPCA names use fitted rank, including zero,
rather than later parameter changes. Refit after changing encoding parameters.

Use native sklearn `set_output(transform="pandas")` or global
`config_context(transform_output="pandas")` for dense table output. Install
`vbpca_py[data]` for pandas. Polars 2.x is qualified through
`vbpca_py[polars]` and `transform="polars"`. Pandas preserves the input index,
including duplicate index labels; Polars preserves row order without adding
an index column. Zero-column dense outputs retain their row count in both
qualified containers. Default NumPy/SciPy outputs and all numerical operations
remain unchanged. Type annotations describe the default numerical return;
configured table output is applied by sklearn's wrapping layer. Inverse
transforms and `transform_posterior` retain their existing array/record output.

Sparse output cannot be wrapped as a dense pandas/Polars table and raises a
clear error without densification. This applies to the sparse one-hot encoder
and sparse paths of scalers, AutoEncoder and ObservationMasker. Explicit complex
indicator storage retains its real-indicator meaning; Polars can store those
values as object columns. Inverse validation still refuses nonzero imaginary
components rather than enabling complex-valued model inputs.

For raw mixed string/NaN data, keep the leading masker in pandas mode when
the remaining Pipeline uses Polars:

```python
pipeline.set_output(transform="polars")
pipeline.named_steps["observe"].set_output(transform="pandas")
```

ObservationMasker explicitly refuses raw object output in Polars mode because
the native adapter's inference cannot safely represent those buffers. Numeric
masker output and subsequent encoded/projection outputs support Polars.
Polars input nulls in object columns are converted to canonical NaNs before
encoding; this can require an owned input copy. It does not reinterpret
ordinary NumPy/Pandas object-category values. AutoEncoder's private nested
encoders stay in default mode, avoiding repeated internal table wrapping.
The [executed example](https://github.com/yoavram-lab/VBPCApy/blob/main/scripts/example_feature_output.py)
demonstrates names, indices and matching numerical results with both outputs.
These behaviors use the
[native sklearn output API](https://scikit-learn.org/stable/developers/develop.html#developer-api-for-set-output).

## Fitted encoding metadata

For a complete fit/search/new-data/reconstruction workflow, see the
[missing-aware Pipeline tutorial](../tutorials/missing-aware-pipeline.md).
Its inverse uses the fitted model/encoder portion of the Pipeline; the leading
masker cannot reconstruct excluded input values or an original observation mask.

`schema.validate(n_features)` checks feature coverage and categorical decoding
metadata. For a mixed schema, use
`schema.validate(n_features, strict_numeric=True)` to also check numeric inverse
maps. Each continuous or ordinal variable must have exactly one encoded column,
one finite real scalar fitted offset and a positive finite real scalar weight.
`decode_numeric` applies
the same numeric checks and returns continuous input units or unrounded ordinal
level-index units. Raw-cell CV validates mixed schemas before model fitting.

A wholly missing numeric training column retains a NaN offset in its fitted
schema. It cannot pass strict validation or numeric decoding; declaring ordinal
levels does not supply the missing scaling observations. The default validation
mode still permits numeric descriptors without affine metadata when only
categorical scoring is required.

::: vbpca_py.encoding.EncodingSchema

::: vbpca_py.encoding.EncodedVariable
