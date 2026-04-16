# Convergence Tuning

Recipes for adjusting stopping criteria when the defaults don't fit your
dataset.

## Tighten convergence for publication results

Use both RMS and cost plateau criteria with a patience window:

```python
from vbpca_py import VBPCA

model = VBPCA(
    n_components=5,
    maxiters=2000,
    rmsstop=[200, 1e-6, 1e-5],
    cfstop=[200, 1e-6, 1e-5],
    patience=5,
    verbose=1,
)
model.fit(X, mask=mask)
```

## Speed up exploratory fits

Relax the criteria for quick iteration:

```python
model = VBPCA(
    n_components=5,
    maxiters=200,
    rmsstop=[50, 1e-3, 1e-2],
    verbose=2,  # coarse progress bar
)
model.fit(X, mask=mask)
```

## Use composite stopping

Require multiple criteria to trigger simultaneously:

```python
model = VBPCA(
    n_components=5,
    composite_stop={
        "rmsstop": [100, 1e-4, 1e-3],
        "cfstop": [100, 1e-3, 1e-2],
    },
    patience=3,
)
model.fit(X, mask=mask)
```

## Enable probe-based early stopping

Hold out entries and stop when probe RMS starts increasing:

```python
from vbpca_py import make_xprobe_mask

X_train, X_probe = make_xprobe_mask(X, fraction=0.10)

model = VBPCA(
    n_components=5,
    maxiters=1000,
    earlystop=True,
)
model.fit(X_train, mask=~np.isnan(X_train), xprobe=X_probe)
```

## Troubleshooting: model won't converge

1. **Check the data scale.** Very large or very small values can cause numerical
   issues. Use `MissingAwareStandardScaler` or `AutoEncoder` to normalise.

2. **Center the data.** Uncentered data with `bias=True` can cause RMS
   oscillation. Pre-center with `MissingAwareStandardScaler`.

3. **Increase `maxiters`.** The default (1000) may not be enough for large or
   noisy data.

4. **Relax `minangle`.** The subspace-angle criterion can trigger prematurely on
   near-singular problems. Try `minangle=1e-10` or disable it.

5. **Inspect the learning curve.** Set `verbose=1` to watch RMS and cost per
   iteration. Oscillation suggests a data-conditioning issue.

## Troubleshooting: model converges too slowly

1. **Reduce `niter_broadprior`.** The default (100) delays ARD pruning. Set to
   50 or 25 for faster warmup.

2. **Lower `maxiters`** and accept a rougher fit for exploration.

3. **Use `runtime_tuning="safe"`** to enable thread autotuning (see
   [Runtime & Threading](runtime-tuning.md)).

See [Convergence Criteria](../concepts/convergence.md) for a reference of all
options and their defaults.
