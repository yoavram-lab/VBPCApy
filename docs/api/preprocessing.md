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
Sparse transformers reject extra masks. Sparse one-hot encoding supports
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

## Fitted encoding metadata

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
