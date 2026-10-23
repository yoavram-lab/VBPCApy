# Native estimator migration in 0.5 development

The native `VBPCA` boundary consumes samples x features. It requires
scikit-learn >=1.8, stores constructor parameters unchanged and validates data
and fitted state through its native validation helpers. This follows the
[scikit-learn estimator contract](https://scikit-learn.org/stable/developers/develop.html)
and [`validate_data`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.validate_data.html).
The package is still in unreleased 0.5 development. Native preprocessing,
selection and sklearn composition use the contracts described here.

| API | Data | Components | Joint training scores | Mean |
| --- | --- | --- | --- | --- |
| `vbpca_py.VBPCA` | samples x features | `(q, p)` | `(n, q)` | `(p,)` |
| `vbpca_py.legacy.VBPCA` | features x samples | `(p, q)` | `(q, n)` | historical column layout |
| Low-level `pca_full` | features x samples | result `A: (p, q)` | result `S: (q, n)` | historical result keys |

The maintainer approved **modern mode as the native estimator default**.
The explicit legacy adapter and low-level solver retain their historical
defaults. Explicit `compat_mode="strict_legacy"` still requests historical
numerical/observation semantics; use the legacy adapter when reproducing the
pre-0.5 interface. Do not silently update a frozen study installation.
The [mode comparison](../howto/compatibility-modes.md) details observation,
epsilon and update-order differences and provides an executed migration example.

## Fit, project and reconstruct

```python
import numpy as np
from vbpca_py import VBPCA

X = np.random.default_rng(23).normal(size=(12, 4))
X[0, 0] = np.nan
model = VBPCA(2, maxiters=4, niter_broadprior=0, verbose=0,
              random_state=23, num_cpu=1, runtime_tuning="off")
Z = model.fit_transform(X)
assert Z.shape == (12, 2)
assert model.components_.shape == (2, 4)
assert model.scores_.shape == (12, 2)
assert model.mean_.shape == (4,)
np.testing.assert_allclose(Z, model.transform(X))
new_scores = model.transform(X[:3])
assert model.inverse_transform(new_scores).shape == (3, 4)
posterior = model.transform_posterior(X[:3])
assert posterior.mean.shape == (3, 2)
assert posterior.covariance.shape == (3, 2, 2)
```

`fit_transform(X)` means `fit(X).transform(X)`. These are conditional score
means under the fixed fitted loading, mean and noise state. VB projection
retains the full loading-covariance contribution to score precision. The joint
variational means from training remain separately available in `scores_` and
can differ from conditional projections. Projection does not refit or mutate
the fitted factors. An entirely missing new row has zero latent mean and unit
prior covariance. Rank zero remains a supported mean-only model.

Projection returns float64 arrays. Ordinary `transform` allocates its `(n,q)`
output without a full `(n,q,q)` covariance bank. `transform_posterior` exposes
that bank explicitly. Its frozen/slotted `ProjectionPosterior` holds fresh,
writable output arrays; field immutability does not freeze the arrays. Direct
record construction borrows validated float64 arrays. Output budgets bound the
combined returned mean/covariance storage, not peak process memory.

## Observations, probes and validation

Dense NaNs are missing and numeric zeros are observed in modern mode. Direct
calls can supply aligned entry masks using keyword arguments. An explicit
mask is consumed before fitting or projecting; it is not a row weight.
NaNs remain missing even if an explicit mask marks their positions as observed.
The legacy dense explicit-mask path could instead count such NaNs as zero
values; the adapter preserves that behavior for compatibility.
Masks and explicit `xprobe` tables follow the same sample-first layout as X.
Automatic probes preserve the exact remaining sparse observation set.
The standalone `make_xprobe_mask` helper also defaults to modern splitting;
older helper-based reproductions must explicitly request
`compat_mode="strict_legacy"`. That historical path can discard unselected
stored zeros and sample stored NaNs. Legacy automatic fits pass their own
configured mode. See the [probe policy comparison](../howto/compatibility-modes.md#split-automatic-probes-without-losing-observed-zeros).
Rows with probe entries must retain at least one training observation. Native
fitting rejects probe-only rows because the solver removes empty training
samples and cannot evaluate their probe diagnostics. Low-level and legacy
calls also reject any observed probe entries that would be discarded by
training-axis removal, including probe-only features and samples removed by
historical sparse sum-based handling. This applies in both compatibility modes;
it replaces silently incomplete PRMS diagnostics with a clear error. Empty
training axes without probe observations can still be removed as before.

Sparse stored entries are observed, including stored zeros; absent entries
are missing. Conversions, row slicing and probe generation must retain this
storage structure. Constructing a sparse matrix from a dense table can discard
numeric zeros: construct coordinates explicitly when those zeros are observed,
or use dense NaNs/masks. Sparse explicit masks can declare observed zero values
that are absent from the input's storage. Duplicate mask entries are combined
as Boolean observations, never as repeated data values.
Native validation checks raw sparse buffers and observed infinity before
conversion can coalesce duplicate coordinates. Explicitly excluded infinities
remain unobserved; duplicate `+inf`/`-inf` cannot silently become missing data
when their location is observed. This also applies to new-row projection,
scoring, explicit probes and public component-selection inputs.

Empty training tables, entirely unobserved tables/features, observed infinity,
invalid capacities and stale feature counts are rejected. Constant observed
features and entirely missing sample rows remain supported. A failed fit/refit
clears fitted state. String feature names and order are validated when supplied
through a supported named table. Warm-start mappings retain the low-level
`A: (p,q)`, `S: (q,n)` axes; the observation/probe tables use the public axes.

Unsupervised preprocessing now accepts `y=None` and keyword-only `mask` on
`fit`/`fit_transform`; pass `mask=...` explicitly rather than as a second
positional argument. All nine preprocessor constructors store parameters
unchanged and create learned attributes during fitting. Invalid log offsets
and winsorization quantiles are rejected by `fit`, so construction and cloning
can retain requested settings. `transform` and `inverse_transform` require a
fitted estimator and use scikit-learn's `NotFittedError`.

Refitting discards old learned arrays/plans before fitting. A failed fit,
including cancellation, clears partial state; refit successfully before using
the transformer again. Successful pickle/deepcopy round trips retain fitted
state. Fitted-state detection recognizes populated feature metadata in
compatible older preprocessor artifacts and treats old unfitted `None`
defaults as unfitted. Native `VBPCA`'s separate serialized-layout rules are
described below.

All nine preprocessors now validate nonempty 2-D input, appropriate numeric or
categorical dtypes, fitted feature counts and supplied string feature names/order.
Duplicate names, observed infinity and raw complex input are rejected. Inverse
transforms validate the fitted encoded width before decoding. Wholly missing
columns retain their existing behavior, including zero-width categorical
blocks. Named inputs capture `feature_names_in_`; generated output names and
pandas/Polars wrapping now follow the contract below.

Sparse preprocessing accepts CSR, CSC and COO matrices/arrays with valid
structure. Stored NaNs become missing entries without eliminating stored
zeros. Duplicate coordinates are summed on owned storage. Explicit masks on
sparse encoders/scalers are rejected: scalers previously ignored them silently.
Use a leading `ObservationMasker` to normalize sparse observations before these
transformers. Dense inverse masks for one-hot and automatic encoding have the
original variable width, not the expanded indicator width.
Automatic encoders retain the plans learned for their fitted dense/sparse
representation; cross-representation categorical composition is not qualified
by this increment.

`MissingAwareSparseOneHotEncoder` now supports multiple numeric categorical
columns. Each has an independent, sorted, full-category block; `categories_`
keeps its flat output-column list and `feature_groups_` identifies each input
variable. Single-column artifacts retain the same metadata meaning. Dense
numeric input is also supported, with every non-NaN value observed, including
zeros. Ignored unknown categories have no stored indicator. Centering changes
stored indicators without filling absent entries.
The existing configurable complex indicator dtype remains supported for
real-valued indicators with finite zero imaginary components; the encoder's
NaN-plus-NaN-imaginary missing sentinel is also retained. Inverse validation
uses real views and decoder scratch blocks; it does not enable complex-valued
raw categories or model inputs.

The default preprocessors pass all applicable generic estimator checks on the
minimum/current supported sklearn versions without skips or expected failures.
For explicit masks in a Pipeline, put `ObservationMasker()` before learned
preprocessing and enable sklearn metadata routing. Pass
`observation_mask=mask` to Pipeline/GridSearchCV, or
`params={"observation_mask": mask}` to `cross_validate`. The leading step
consumes this alias; downstream encoders and VBPCA receive dense NaNs or sparse
stored observations. CV slices training and test masks by row before feature
expansion. Do not set `transform_input` for entry masks or forward the raw mask
to a wider encoded table. Direct masker calls use `mask=mask`; every new table
needs its own aligned mask when it contains placeholders. Without explicit
masks, ordinary NaN/stored-entry Pipelines need no routing opt-in. See the
[preprocessing API](preprocessing.md) for sparse zero handling and the runnable
example. Output names and containers are described below.

`EncodingSchema.validate(..., strict_numeric=True)` now checks mixed numeric
inverse maps explicitly; default categorical-only validation remains compatible
with numeric descriptors that omit affine metadata.

Every native transformer now provides `get_feature_names_out` and sklearn's
`set_output`. Default outputs remain arrays/sparse matrices. Named pandas
output preserves the index, and qualified Polars 2.x output preserves row order,
including zero-column dense outputs. Sparse output is refused in table mode
without densification. Unnamed encoder display metadata now uses `x0`, `x1`,
... rather than `col0`, `col1`, ...; AutoEncoder names use original-variable
positions rather than repeated nested-column labels. Generated collisions
receive deterministic suffixes. In a mixed raw-data Pipeline using Polars for
numerical outputs, keep the leading masker in pandas mode. Polars input nulls
are missing observations. See the [preprocessing API](preprocessing.md) for the
optional extras, configuration and supported dtype boundaries. Inverse and
posterior-record output remain unchanged. The complete
[Pipeline example](../tutorials/missing-aware-pipeline.md) demonstrates new-row
projection and inverse reconstruction through the fitted model/encoder portion.

## Scoring and cellwise selection

`score_samples(X)` gives each row's observed-entry Gaussian marginal log density
under fitted loading means and noise, integrating unit-normal latent scores.
This plug-in score excludes posterior loading uncertainty; it is neither the
variational objective nor a fully uncertainty-integrated VB predictive score.
The implementation uses rank-sized solves without an observed-features-squared
covariance allocation. Empty rows contribute zero; a table without observations
is rejected. `score(X)` averages row scores; optional finite, nonnegative
`sample_weight` values weight rows for scoring only and require a positive sum.
They do not alter training. With metadata routing enabled, request scoring
weights explicitly through `set_score_request(sample_weight=True)`.

Cellwise PRMS/Brier capacity selection remains a distinct target. Public
`select_n_components` and `cross_validate_components` consume samples x
features, including masks and explicit selection probes, and default to modern
mode. Requested positive-rank selected models are native estimators with
`components_: (q,p)`, `scores_: (n,q)` and conditional `transform(X_new)`;
they adopt the selected fit without refitting. Rank-zero selection retains the
existing `None` best-model result. Diagnostics keep their byte budgets and
use native axes. Warm starts retain their explicitly documented low-level axes.

Native capacities reject negative, Boolean, nonintegral and oversized values.
Cellwise CV constructs its own probes and refuses external probe options.
Feature groups/schema columns describe the input feature axis. Raw CV still
fits preprocessing separately within each cell fold; its public prepared
training/probe arrays and `numeric_scores` inputs now use samples x encoded
features. Rebuild older prepared-fold pickles, whose layout is explicitly
refused. The [selection guide](../concepts/model-selection.md) explains the
unchanged target/fold rules and includes the executed public example.

To preserve historical feature-first helper inputs, defaults and returned
compatibility models, import `select_n_components` and
`cross_validate_components` from `vbpca_py.legacy`. Existing numerical/study
characterizations remain on that explicit boundary; archived registrations
and scientific results are not rewritten.

Grouped sklearn CV and multi-metric GridSearchCV support explicitly requested
raw mask metadata and scoring-only row weights. Custom scorers must declare
their metadata requests through sklearn's public routing API. Training remains
unweighted. On supported sklearn versions, `GridSearchCV.transform` does not
accept metadata: use
`search.best_estimator_.transform(X_new, observation_mask=new_mask)` for masked
inference after search. `search.score` supports requested scoring metadata.

## Explicit legacy API and serialized models

```python
from vbpca_py.legacy import VBPCA as LegacyVBPCA

old_api = LegacyVBPCA(2, maxiters=4, niter_broadprior=0, verbose=0,
                     random_state=23, num_cpu=1, runtime_tuning="off")
old_scores = old_api.fit_transform(X.T)
assert old_scores.shape == (2, 12)
assert old_api.inverse_transform().shape == (4, 12)
assert old_api.compat_mode == "strict_legacy"
```

The adapter preserves the old training-score `transform()` and no-argument
inverse reconstruction. It is an explicit compatibility interface and does not
claim native Pipeline transformer semantics. Legacy characterization tests and
low-level independent numerical references remain separate from the native
estimator tests. Historical engineering examples import this adapter explicitly.

New native serialized state includes a layout version. Pre-0.5 pickles named
`vbpca_py.estimators.VBPCA` lack that marker and are explicitly refused by the
native class. Open those artifacts in their original environment and refit using
the native or explicit legacy API; do not guess orientation from array shapes.
New native pickle/deepcopy round trips preserve the declared layout and fitted
state without putting a serialization marker into constructor parameters.
