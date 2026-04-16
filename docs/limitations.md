# Known Limitations

- **`transform(new_data)` is not implemented.** Only training scores are returned. To project new data, refit on the combined dataset.

- **`inverse_transform()` always returns dense output**, even when the input was sparse CSR/CSC.

- **`MissingAwareSparseOneHotEncoder` requires numeric categories.** String categories cannot survive the CSR round-trip.

- **Data convention.** `AutoEncoder` expects samples × features; `VBPCA` expects features × samples. Transpose as needed.

- **RMS oscillation with uncentered data.** When `bias=True` (the default) and the input data has non-zero feature means, the RMS convergence trace can exhibit a stable period-2 oscillation caused by a one-iteration lag between the mean update and the reconstruction error.

    **Workaround:** center your data before fitting — use `MissingAwareStandardScaler` (or `AutoEncoder`) as a preprocessing step. Pre-centered data eliminates the oscillation entirely, even with `bias=True`.
