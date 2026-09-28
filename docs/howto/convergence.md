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
        "angle": 1e-4,
        "rms": 1e-3,
        "elbo_rel": 1e-4,
    },
    patience=3,
)
model.fit(X, mask=mask)
```

## Enable probe-based early stopping

Hold out entries and stop when probe RMS starts increasing:

```python
model = VBPCA(
    n_components=5,
    maxiters=1000,
    earlystop=True,
    xprobe_fraction=0.10,
    random_state=42,
)
model.fit(X, mask=mask)
```

When probe RMS first worsens, the returned factors are restored to the best
probe iteration. Inspect `best_probe_iteration_`, `best_probe_rms_`, and
`returned_iteration_`; `learning_curve_` retains the later trigger iteration.

## Troubleshooting: model won't converge

1. **Check the data scale.** Very large or very small values can cause numerical
   issues. Use `MissingAwareStandardScaler` or `AutoEncoder` to normalise.

2. **Check the bias update order.** The MATLAB-compatible order can produce an
   alternating RMS trace on uncentered data. Use
   `bias_update_order="post_factor"`, or `compat_mode="modern"`, to update the
   mean after the current factor updates. Pre-centering with
   `MissingAwareStandardScaler` can still improve conditioning.

3. **Increase `maxiters`.** The default (1000) may not be enough for large or
   noisy data.

4. **Relax `minangle`.** The subspace-angle criterion can trigger prematurely on
   near-singular problems. Try `minangle=1e-10` or disable it.

5. **Inspect the learning curve.** Set `verbose=1` to watch RMS and cost per
   iteration. Alternation can indicate legacy bias ordering as well as poor
   data conditioning.

## Troubleshooting: model converges too slowly

1. **Reduce `niter_broadprior`.** The default (100) delays ARD pruning. Set to
   50 or 25 for faster warmup.

2. **Lower `maxiters`** and accept a rougher fit for exploration.

3. **Use `runtime_tuning="safe"`** to enable thread autotuning (see
   [Runtime & Threading](runtime-tuning.md)).

See [Convergence Criteria](../concepts/convergence.md) for a reference of all
options and their defaults.

## Use ELBO as the stopping gate

If the subspace-angle criterion fires too early (for example under heavy
missingness), disable it and the other individual criteria so ELBO is the only
eligible numerical-convergence stop. Ordering alone is only a tie-breaker when
multiple OR criteria are ready on the same iteration.

```python
model = VBPCA(
    n_components=5,
    criterion_order=["cost", "composite", "rms_plateau", "angle", "earlystop", "slowing_down"],
    convergence_criteria={
        "angle": False,
        "earlystop": False,
        "rms_plateau": False,
        "cost": True,
        "composite": False,
        "slowing_down": False,
    },
    cfstop=[200, 1e-6, 1e-5],
)
model.fit(X, mask=mask)
```

## Disable angle-based stopping

Disable angle without setting `minangle=0` — this way diagnostic logging
still records the subspace angle each iteration:

```python
model = VBPCA(
    n_components=5,
    convergence_criteria={"angle": False},
)
model.fit(X, mask=mask)
```

## Combine ordering and disabling

You can use both options together.  For example, run only ELBO and
composite criteria, with ELBO checked first:

```python
model = VBPCA(
    n_components=5,
    criterion_order=["cost", "composite", "rms_plateau", "angle", "earlystop", "slowing_down"],
    convergence_criteria={
        "angle": False,
        "earlystop": False,
        "rms_plateau": False,
        "slowing_down": False,
    },
    cfstop=[200, 1e-6, 1e-5],
)
model.fit(X, mask=mask)
```
