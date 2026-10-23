# Convergence Tuning

These feature-first examples use the explicit pre-0.5 compatibility adapter.
For the native samples-first estimator, see the [0.5 migration guide](../api/migration-05.md).

Recipes for adjusting stopping criteria when the defaults don't fit your
dataset.

## Record a fixed-iteration reference

An active criterion can stop a fit even when another criterion is still moving.
For a comparison reference, disable numerical and probe stopping explicitly:

```python
from vbpca_py.legacy import VBPCA

model = VBPCA(
    n_components=5,
    maxiters=2000,
    convergence_criteria={
        "angle": False, "earlystop": False, "rms_plateau": False,
        "cost": False, "composite": False, "slowing_down": False,
    },
    record_cost=True,
    verbose=1,
)
model.fit(X, mask=mask)
```

Compare reconstruction mean, predictive variance, noise and subspace with a
longer cap whose tail is itself stable. A small relative cost change alone
does not establish posterior stability.

## Recommended cost-stop correction

`recommend_config()` no longer enables relative cost-change or composite
stopping in its coarse shape buckets. In the public
[rank-one regression](https://github.com/yoavram-lab/VBPCApy/blob/main/tests/test_recommended_stopping.py),
the historical relative-cost rule stops before the reconstruction is stable.
The corrected recommendation is compared with a stopping-disabled reference
whose endpoint is checked under a larger iteration cap. This is a bounded
numerical regression, not a calibration study or a general convergence proof.
The fitter still supports explicitly requested cost criteria, and
`record_cost=True` records cost without enabling a cost stop.

This correction retains recommended priors, probes, caps, angle and RMS rules.
It does not qualify a universal convergence detector. The
[broader posterior-stability study](https://github.com/yoavram-lab/VBPCApy/blob/main/analysis/trade_study/CONVERGENCE_DETECTOR.md)
found remaining premature stops, chiefly from RMS plateau, and its held-out
replacement candidate failed qualification. Inspect posterior stability for
the intended use; existing pinned scientific policies stay explicit and are
not migrated by this recommendation change.

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
    convergence_criteria={
        "angle": False, "earlystop": False, "rms_plateau": False,
        "cost": False, "composite": True, "slowing_down": False,
    },
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

4. **Tighten `minangle`.** The subspace-angle criterion can trigger prematurely on
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

## Configure a cost-only experiment

If the subspace-angle criterion fires too early (for example under heavy
missingness), disable it and the other individual criteria so ELBO is the only
eligible numerical stopping rule. This is an explicit experimental policy;
the relative-cost failure above is a reason to check it against stable longer
fits. Ordering alone is a tie-breaker when multiple OR criteria are ready on
the same iteration.

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

## Unavailable legacy criterion

`slowing_down` is disabled in live fitting because no event counter is supplied.
Requesting it explicitly emits a warning and resolves to disabled. The returned
`learning_curve_["criterion_available_slowing_down"]` values are zero. Existing
configurations keep their fitted endpoints because this rule was already inert.
Use boolean values for every `convergence_criteria` entry; strings such as
`"False"` are rejected rather than interpreted through Python truthiness.


## Specify a study's stopping policy

For a study that disables cost and composite stopping, record those flags
explicitly even though current recommendations now leave them disabled:

```python
from vbpca_py import recommend_config
from vbpca_py.legacy import VBPCA

options = recommend_config(n=60, p=180)
flags = dict(options.get("convergence_criteria", {}))
flags.update(cost=False, composite=False)
options.update(convergence_criteria=flags, cfstop_rel=None, maxiters=1200)
model = VBPCA(n_components=3, **options)
```

Record the library version, full resolved options, variance-update order and
random seeds alongside the results. A longer cap permits more iterations but
does not override another active stopping rule.

This recipe changes which rules may stop a fit. It does not establish posterior
stability or predictive calibration. Compare the chosen policy with longer fits
and the outcomes needed by the study. Existing pinned studies should retain
their recorded settings; adopting another policy or release requires an explicit
new scientific configuration.
