# Known Limitations

- **`transform(new_data)` is not implemented.** Only training scores are returned. To project new data, refit on the combined dataset.

- **`inverse_transform()` always returns dense output**, even when the input was sparse CSR/CSC.

- **`MissingAwareSparseOneHotEncoder` requires numeric categories.** String categories cannot survive the CSR round-trip.

- **Data convention.** `AutoEncoder` expects samples × features; `VBPCA` expects features × samples. Transpose as needed.

- **Legacy RMS ordering with uncentered data.** The original MATLAB update
  order estimates the mean before updating the latent factors. With
  `bias=True` and non-zero feature means, this one-step lag can produce a
  period-2 RMS trace. It remains available as
  `bias_update_order="legacy"` and is selected automatically by
  `compat_mode="strict_legacy"`.

    Use `bias_update_order="post_factor"` to estimate the mean and evaluate
    RMS from the same factor state. `recommend_config()` and
    `compat_mode="modern"` select that order automatically. The registered
    defaults study retained prediction and calibration and found equal or lower
    rank error with post-factor ordering. Pre-centering with
    `MissingAwareStandardScaler` remains a useful conditioning step, but it is
    no longer required solely to align the mean and RMS diagnostics.

- **Legacy variance ordering with `rotate2pca`.** The MATLAB order updates the
  ARD prior variances $V_a$ before each iteration's PCA rotation, so the
  loadings update can apply a component's prior variance to a different
  column, and the returned $V_a$ need not match the returned components. It
  remains the `compat_mode="strict_legacy"` default
  (`variance_update_order="legacy"`); `variance_update_order="post_rotation"`
  re-estimates the prior variances after the rotation. A paired comparison of
  the two orders is planned before any change to `recommend_config()` (#214).

- **Fits are non-reproducible unless seeded.** `VBPCA(random_state=None)` (the default) draws fresh entropy for parameter initialization and any auto-generated xprobe mask on every call to `fit()`, so repeated fits on the same data can converge to different results. Pass an `int` or `np.random.Generator` via `random_state` for reproducible runs. Prior to #109, the default initialization was silently seeded with a fixed value; this is no longer the case.
