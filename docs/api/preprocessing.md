# Preprocessing

All nine transformers validate nonempty 2-D input at fit and inference.
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
