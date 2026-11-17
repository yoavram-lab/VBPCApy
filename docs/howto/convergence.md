# Convergence Tuning

Recipes for adjusting when a fit stops, and for checking that it has
stabilized. The examples share this incomplete matrix of 300 samples and 40
features:

```python
import numpy as np

from vbpca_py import VBPCA

rng = np.random.default_rng(0)
signal = rng.standard_normal((300, 5)) @ rng.standard_normal((5, 40))
X = signal + 0.5 * rng.standard_normal(signal.shape)
X[rng.random(X.shape) < 0.2] = np.nan
```

A fit stops at `maxiters` or when an enabled criterion is met: the
subspace-angle change (`angle`), a plateau in training RMS (`rms_plateau`),
worsening probe error (`earlystop`), the relative change in cost (`cost`), or a
combination (`composite`). [Convergence Criteria](../concepts/convergence.md)
defines each one.

## Compare with a fixed-iteration reference

A criterion can stop a fit while other quantities are still changing. To check
a fit, refit with every stopping criterion disabled and a larger iteration cap,
then compare the quantities your analysis uses:

```python
no_stopping = dict.fromkeys(
    ["angle", "earlystop", "rms_plateau", "cost", "composite", "slowing_down"],
    False,
)

default_fit = VBPCA(n_components=5, random_state=1).fit(X)
reference = VBPCA(
    n_components=5,
    maxiters=600,
    convergence_criteria=no_stopping,
    random_state=1,
).fit(X)


def relative_change(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


print(f"Default fit stopped after {default_fit.n_iter_} iterations "
      f"({default_fit.convergence_reason_})")
print("Reconstruction change:",
      relative_change(default_fit.reconstruction_, reference.reconstruction_))
print("Predictive variance change:",
      relative_change(default_fit.predictive_variance_, reference.predictive_variance_))
```

Repeat with a still larger cap to confirm that the reference itself has
stopped changing. A small change in cost alone does not show that the
posterior mean and variance have stabilized.

## Cost-change stopping in `recommend_config`

`recommend_config()` leaves the relative cost-change and composite criteria
disabled. In a
[rank-one regression test](https://github.com/yoavram-lab/VBPCApy/blob/main/tests/test_recommended_stopping.py),
the relative cost-change rule stopped fits before the reconstruction had
stabilized, compared with a reference fit with stopping disabled. That test is
one numerical check, not a general study of stopping rules. Cost-based
stopping can still be requested explicitly, and `record_cost=True` records the
cost without using it to stop.

The recommended settings keep the angle and RMS-plateau criteria. A
[study of stopping rules](https://github.com/yoavram-lab/VBPCApy/blob/main/analysis/trade_study/CONVERGENCE_DETECTOR.md)
found that these can also stop before the posterior has stabilized, most often
through the RMS plateau, and no alternative rule worked across all the settings
studied. When an analysis depends on the posterior mean or variance, compare
it with a fixed-iteration reference as shown above.

## Speed up exploratory fits

Relax the RMS-plateau criterion for quick iterations:

```python
model = VBPCA(n_components=5, maxiters=200, rmsstop=[50, 1e-3, 1e-2], random_state=0)
model.fit(X)
```

`rmsstop=[window, absolute, relative]` stops when the training RMS has changed
by less than the absolute or relative tolerance over the last `window`
iterations. `verbose=1` prints progress for each iteration and `verbose=2` a
coarse progress bar.

## Use composite stopping

Require several conditions to hold at once:

```python
model = VBPCA(
    n_components=5,
    convergence_criteria={
        "angle": False,
        "earlystop": False,
        "rms_plateau": False,
        "cost": False,
        "composite": True,
        "slowing_down": False,
    },
    composite_stop={"angle": 1e-4, "rms": 1e-3, "elbo_rel": 1e-4},
    patience=3,
    random_state=0,
)
model.fit(X)
```

`patience=3` requires the condition to hold on three consecutive iterations.

## Stop when the probe error starts to rise

Hold out a fraction of the observed entries and stop when the error on them
increases:

```python
model = VBPCA(
    n_components=5,
    maxiters=1000,
    earlystop=True,
    xprobe_fraction=0.10,
    random_state=42,
)
model.fit(X)
print("Best probe iteration:", model.best_probe_iteration_)
print("Returned iteration:", model.returned_iteration_)
```

When the probe error first worsens, the fit returns the parameters from the
iteration with the lowest probe error. `best_probe_rms_` gives that error;
`learning_curve_` still records the later iterations.

## Troubleshooting: the fit does not converge

1. **Check the data scale.** Features with very different spreads, or values far
   from unit scale, slow convergence. Standardize with
   `MissingAwareStandardScaler` or `AutoEncoder`.
2. **Check the bias update order.** The original MATLAB update order can give
   an alternating RMS trace on uncentered data. The default
   `compat_mode="modern"` updates the mean after the factors
   (`bias_update_order="post_factor"`).
3. **Increase `maxiters`.** The default of 1000 iterations may not be enough for
   large or noisy data.
4. **Tighten or disable the angle criterion.** On nearly singular problems the
   subspace angle can settle before other quantities do. Try `minangle=1e-10`
   or disable it (below).
5. **Inspect the learning curve.** `learning_curve_` holds RMS, probe RMS and
   (with `record_cost=True`) cost for every iteration.

## Troubleshooting: the fit converges too slowly

1. **Reduce `niter_broadprior`.** During the first `niter_broadprior` iterations
   (default 100) the priors stay broad and no stopping criterion applies.
   Values of 25 to 50 shorten this warm-up.
2. **Lower `maxiters`** and accept a rougher fit for exploration.
3. **Use runtime tuning** to choose thread counts automatically; see
   [Runtime & Threading](runtime-tuning.md).

## Stop on the cost alone

To make the relative cost change the only stopping criterion, disable the
others and set its tolerance with `cfstop=[window, absolute, relative]`:

```python
model = VBPCA(
    n_components=5,
    convergence_criteria={
        "angle": False,
        "earlystop": False,
        "rms_plateau": False,
        "cost": True,
        "composite": False,
        "slowing_down": False,
    },
    cfstop=[200, 1e-6, 1e-5],
    random_state=0,
)
model.fit(X)
```

Because cost-change stopping can end a fit early (see above), compare such a
fit with a fixed-iteration reference.

## Disable angle-based stopping

Disabling the criterion, rather than setting `minangle=0`, keeps the subspace
angle in the learning curve:

```python
model = VBPCA(n_components=5, convergence_criteria={"angle": False}, random_state=0)
model.fit(X)
```

## Order criteria that trigger together

All enabled criteria are evaluated every iteration. `criterion_order` only
decides which one is reported in `convergence_reason_` when several are met on
the same iteration; it does not change when the fit stops.

```python
model = VBPCA(
    n_components=5,
    criterion_order=["cost", "composite", "rms_plateau", "angle", "earlystop", "slowing_down"],
    convergence_criteria={"angle": False, "earlystop": False, "slowing_down": False},
    cfstop=[200, 1e-6, 1e-5],
    random_state=0,
)
model.fit(X)
print(model.convergence_reason_)
```

## The `slowing_down` criterion

`slowing_down` comes from the MATLAB implementation, where it relies on an
event counter that the Python fitter does not have. It is always disabled:
requesting it emits a warning, and
`learning_curve_["criterion_available_slowing_down"]` records that it was
unavailable. Because the rule never fired, this does not change any fit.
Every `convergence_criteria` value must be a boolean; strings such as
`"False"` are rejected.

## Record the stopping policy of an analysis

State every stopping flag explicitly, even where it matches the default, so
the policy is clear from the analysis code:

```python
from vbpca_py import recommend_config

options = recommend_config(n=300, p=40)
flags = dict(options.get("convergence_criteria", {}))
flags.update(cost=False, composite=False)
options.update(convergence_criteria=flags, cfstop_rel=None, maxiters=1200)
model = VBPCA(n_components=5, random_state=0, **options).fit(X)
print(model.get_options()["convergence_criteria"])
```

Record the VBPCApy version, the resolved options from `get_options()` and the
random seeds with the results. A larger `maxiters` allows more iterations but
does not stop another enabled criterion from ending the fit. To reproduce an
earlier analysis, use the version and settings recorded with it.
