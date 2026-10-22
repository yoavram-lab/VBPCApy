# Preprocessing

The standard and min-max scalers use each finite, nonzero computed standard
deviation or range, even when its magnitude is small. A change of measurement
units therefore does not by itself classify a nonconstant column as constant.
Exactly constant observed columns use a unit denominator; wholly missing
columns retain NaN mean/variance or extrema and also use a unit denominator.
Zero or nonfinite computed scales use the same fallback. These rules apply to
dense observed entries and stored sparse observations, including stored zeros.
Sparse standardization computes centered column moments without making a dense
copy of the data.

::: vbpca_py.AutoEncoder

::: vbpca_py.MissingAwareOneHotEncoder

::: vbpca_py.MissingAwareSparseOneHotEncoder

::: vbpca_py.MissingAwareStandardScaler

::: vbpca_py.MissingAwareMinMaxScaler

::: vbpca_py.MissingAwareLogTransformer

::: vbpca_py.MissingAwarePowerTransformer

::: vbpca_py.MissingAwareWinsorizer

::: vbpca_py.check_data

::: vbpca_py.DataReport

## Fitted encoding metadata

::: vbpca_py.encoding.EncodingSchema

::: vbpca_py.encoding.EncodedVariable
