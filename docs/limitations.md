# Known Limitations

- **`transform(new_data)` is not implemented.** Only training scores are returned. To project new data, refit on the combined dataset.

- **`inverse_transform()` always returns dense output**, even when the input was sparse CSR/CSC.

- **`MissingAwareSparseOneHotEncoder` requires numeric categories.** String categories cannot survive the CSR round-trip.

- **Data convention.** `AutoEncoder` expects samples × features; `VBPCA` expects features × samples. Transpose as needed.

- **RMS oscillation with uncentered data.** When `bias=True` (the default) and the input data has non-zero feature means, the RMS convergence trace can exhibit a stable period-2 oscillation caused by a one-iteration lag between the mean update and the reconstruction error.

    **Workaround:** center your data before fitting — use `MissingAwareStandardScaler` (or `AutoEncoder`) as a preprocessing step. Pre-centered data eliminates the oscillation entirely, even with `bias=True`.

- **Fits are non-reproducible unless seeded.** `VBPCA(random_state=None)` (the default) draws fresh entropy for parameter initialization and any auto-generated xprobe mask on every call to `fit()`, so repeated fits on the same data can converge to different results. Pass an `int` or `np.random.Generator` via `random_state` for reproducible runs. Prior to #109, the default initialization was silently seeded with a fixed value; this is no longer the case.
