# Encoding-aware selection validation

Result-contract checks run with the regular test suite and strict consumer
typing:

```bash
just typecheck
uv run pytest -q tests/test_result_contracts.py tests/test_sparse_pattern_groups.py
```

These checks cover the existing dictionary keys and heterogeneous learning
curves, plus sparse covariance groups that partition sample columns. The
sparse repair preserves observed zeros and compares shared/unshared fits,
cost histories and uncertainty after consistent component-sign alignment.
It changes a previously failing sparse-sharing path; adopting it in a study
requires recording the new software pin. No frozen benchmark is regenerated.

Encoding-aware selection affects categorical scoring and the selected capacity for
mixed encodings. It adds a fitted schema for dense encoders, preserves fitted
centering on later transforms, and decodes nominal categorical predictions
before scoring. Probe RMS remains in encoded units. Fitting a different
encoding can still change the Gaussian model; scoring equivalence refers to
the same decoded predictions, not identical fitted posteriors.

Reproduce the validation from a clean checkout:

```bash
uv sync --extra dev --extra data --extra plot
just ci
uv run pytest -q tests/test_encoding_aware_scores.py
uv run pytest -q tests/test_raw_model_selection.py
just example-raw-cv
just build-check
```

Publication studies already pinned to 0.4.5 retain their registered fitting
policy and full-one-hot scoring contract. Adopting these changes requires a
new software pin and an explicit encoding schema; no production results or
locked benchmark artifacts are regenerated for this change. The tests compare
decoded scores, original-variable masks and rank-zero CV across encoding
modes, and exercise positive-rank candidates separately.

The raw-table path constructs original-variable cell holdouts before
learning means, scales, category frequencies or block weights. Each fold fits
one encoder and reuses its schema for all capacities. External codebooks can
declare categorical/ordinal levels; an undeclared held-only level raises an
error. The diagnostic tests change held-out values with the training mask fixed
and compare fitted statistics, then check decoded scores and original-unit
numeric RMSE. `scripts/example_raw_cell_cv.py` runs against an installed package
and prints the fold-specific numeric training statistics. The new path leaves
the encoded-array API and locked 0.4.5 publication settings in place; adopting
fold-local preprocessing is a separately registered methodology change.

The `just bench-study-full`, `just bench-study-summary`, and
`just bench-study-paper` sequence referenced in CONTRIBUTING.md has no recipes
in the current justfile. Current benchmark recipes are listed by `just --list`;
use the commands above for this scoring change.

For the 0.4.6 patch, [release validation](../docs/releases/0.4.6.md) adds the
paired fixed-policy fitting check against 0.4.5. Both snapshots must use the
same registered policy module, numerical environment and controlled native
build. This checks the numerical implementation on fixed matrices; adopting
raw-table CV remains a separately registered preprocessing-method change.

## Native failure checks

Runtime failure handling rejects invalid noise variances and sparse pointers,
and propagates native worker errors as Python exceptions. It does not change
update equations, numerical defaults or locked publication settings. Do not
regenerate study artifacts for this repair. Reproduce the process-survival and
numerical checks with:

```bash
uv sync --extra dev --extra plot
uv run pytest -q tests/test_native_failures.py
just numerical-reference
just ci
just build-check
```

## Unavailable stopping option and flag validation

Live fitting reports the legacy `slowing_down` event stream as unavailable and
resolves its flag to disabled. Explicit requests warn; removing the inert flag
from recommendations does not change their computed endpoints. Convergence
flags now require booleans, with NumPy boolean scalars normalized into a copied
mapping. Invalid strings/numbers cannot silently enable a stop.

These changes do not restore a new stopping event, retune supported thresholds,
change optimization updates, or alter locked publication software pins. The
existing posterior-stability screen did not qualify a replacement default;
the held-out no-RMS candidate also failed its safety gate. Further policy
confirmation remains separate. Reproduce this option check:

```bash
uv sync --extra dev --extra data --extra plot
uv run pytest -q tests/test_unavailable_slowing_down.py tests/test_converge.py tests/test_defaults.py
just ci
```


## Recommendation cost-stop correction

Recommendations disable relative cost-change/composite stopping. The public
regression below compares the historical early-cost policy with the corrected
recommendation and stable longer fits. The library's broader detector
confirmation remains negative; this correction does not qualify a universal
posterior-stability detector, tune priors or migrate frozen publication policies.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q tests/test_recommended_stopping.py tests/test_defaults.py
just numerical-reference
just ci
```

The bounded rank-one regression compares the historical early-cost policy and
the corrected recommendation on the same 400 × 400 matrix and initialization,
then checks both against a stopping-disabled fit whose endpoint is stable
under a larger cap. It is a numerical regression, not a new calibration study.
All recommendation buckets and priority presets are checked for inactive
relative-cost stopping. Existing retained-curve replay still uses its recorded
options and immutable generating identities.

## Retained-curve policy replay

The [convergence-detector analysis](../analysis/trade_study/CONVERGENCE_DETECTOR.md)
now provides a policy-only command for the retained v2 screen. It reproduces
selected baseline stops before evaluating exploratory policies, requires a
complete consistent shard set, reports unavailable policies and unstable
endpoints separately, and writes a new provenance-bearing report. No fits,
frozen policies or publication pins change. The completed held-out confirmation
remains negative; replay results alone cannot qualify a replacement default.

## Backend review probes

The [backend review](../analysis/BACKEND_REVIEW_2026_10_20.md) separates
measured implementation costs from proposed statistical changes. Its
prototypes retain full covariance terms and are not installed solver paths.
Run from a clean source checkout with the package installed:

```bash
uv sync --extra dev --extra plot
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VBPCA_NUM_THREADS=1 \
  uv run python analysis/review_backend_costs.py --output /tmp/vbpca-backend-review.json
uv run pytest -q tests/test_backend_rearrangements.py
just numerical-reference
```

The probe writes source/native hashes, numerical versions, platform, process
affinity, thread environment and NumPy build details alongside timings and
allocation/conversion counters. It uses 512 features, 256 samples, ranks 8 and
32, four fixed iterations and 1% missingness. It alternates reference/prototype
timing order after warmup and includes covariance packing. Keep generated JSON
and benchmark outputs outside tracked source. Record actual BLAS pools and
scheduler/node information separately for cluster qualification.

The [algebra-check instructions](../analysis/proof_checks/README.md) pin Lean,
mathlib and SymPy and state the proof boundaries. Follow numerical comparisons
with representative fixed-policy full fits before adopting a rewrite; the
local phase ratios do not establish whole-fit or cross-machine speedups.

## Posterior variance contractions

Compare the historical einsum, optimized einsum, full GEMM and bounded GEMM
implementations, including covariance packing:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --extra dev python scripts/benchmark_posterior_variance.py \
  --repeats 5 --include-fits --output /tmp/vbpca-variance.json
uv run pytest -q tests/test_posterior_variance.py tests/test_diagnostic_memory.py
just numerical-reference
```

The command uses deterministic full, off-diagonal posterior covariances and
alternating measurement order after warmup. It records shape/rank, individual
timings, NumPy allocation peaks, covariance-packing scope, software versions,
source hash, affinity and actual BLAS pools. NumPy peaks exclude native BLAS
workspace. Whole-fit comparisons use the same fixed four-iteration policy and
compare factors, reconstruction, noise and both uncertainty outputs. Timing on
a shared host is advisory; pin CPU allocation and repeat on target hardware
before interpreting throughput. The rewrite changes diagnostic evaluation,
and leaves numerical stopping, statistical defaults and publication pins alone.

## Sparse allocation and view lifetimes

```bash
uv run pytest -q tests/test_sparse_views.py tests/test_sparse_observed_zero_indices.py
just numerical-reference
uv run --extra dev python scripts/check_sparse_allocations.py \
  --output /tmp/vbpca-sparse-allocations.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --extra dev python analysis/benchmark_backend_phases.py \
  --features 193 --samples 257 --components 32 --iterations 4 --threads 1 \
  --sparse --skip-rotation --skip-diagnostics \
  --output /tmp/vbpca-sparse-phases.json
```

The allocation command requires Linux/glibc and a C compiler. It builds a
temporary allocator interposer, launches a fresh interpreter, warms both sparse
kernels, and records three allocation counts at ranks 8 and 32 for fixed
193 × 257 axes with 2,480 versus 39,681 stored observations. Explicitly stored
observed zeros remain in both workloads. Single native/BLAS threads isolate the
count. Whole-call setup/output costs are included; the gate rejects allocation
growth with observation count. The helper is never linked into production
wheels. Numerical, thread-error and sparse-pointer checks remain independent.

The view-lifetime tests count one CSC conversion in a four-iteration fit without
bias changes and four with bias changes, instead of the previous eight. They
check exact trajectories against uncached conversions, pattern grouping, both
rotation/bias-update settings and explicit in-place invalidation. Conversion
and packing remain included in phase/whole-fit measurements. Timing comparisons
should still use the fresh-process procedure below on the target allocation;
these allocation counts alone do not establish a whole-fit speedup.
The phase profiler records sparse-view conversion counts and time separately,
inside the score/loading and total-fit measurements. Add `--skip-bias` to
exercise reuse across iterations; remove `--skip-rotation` to include rotation
and its invalidation. The defaults of existing dense profiling commands are
unchanged. Preserve snapshots and identities for each paired revision run.

## Selection cost diagnostics

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --extra dev python scripts/benchmark_selection_cost.py \
  --repeats 5 --output /tmp/vbpca-selection-cost.json
uv run pytest -q tests/test_endpoint_cost.py
just numerical-reference
```

This paired probe measures complete PRMS capacity sweeps with identical probes,
seeds and fixed iterations, including a mean-only candidate. It alternates
trace/endpoint order after warmup, compares selected capacity and endpoint
metrics exactly, and records full sweep times, cost-call counts, versions,
affinity and BLAS pools. Full cost traces remain the default and are required
for cost-based selection, cost-consuming stopping and requested replay evidence.
The endpoint mode is an explicit diagnostic option, rather than a change to
stopping criteria, scientific defaults or frozen publication profiles.

## Source-revision memory and thread profiles

`analysis/compare_backend_revisions.py` compares a baseline and candidate source
tree using the same interpreter and numerical environment. Build/install each
tree's intended native extensions first. For a Python-only repair with unchanged
native sources, the same extension binaries can be used in both trees; retain
their hashes. Confirm the loaded package paths rather than relying on an
editable installation's version metadata.

Run on an allocated Linux compute node or an otherwise suitable workstation.
Inspect CPU topology with `lscpu -p=CPU,CORE,SOCKET` and supply four CPU IDs from
distinct physical cores within the process allocation. These IDs are an example;
choose valid IDs for the machine:

```bash
uv sync --extra dev --extra plot
uv run python analysis/compare_backend_revisions.py \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --cpus 4 5 6 7 --repeats 3 --output /tmp/vbpca-revision-profile
```

The driver runs children sequentially, alternating revision order between
repeats. It compares a 1,536-by-1,024 rank-8 fit, a 512-by-256 rank-32 fit and
RMS on 3,000-by-1,500 arrays. Fits use four fixed iterations with stopping and
tuning disabled. Each workload uses `(allocated cores, native threads, BLAS
limit)` settings `(1,1,1)`, `(2,2,1)`, `(4,4,1)`, `(4,1,4)` and `(4,4,4)`.
Inspect actual BLAS pools separately, for example with
`python -m threadpoolctl -i numpy scipy.linalg` under the same environment and
affinity. Thread limits do not prove that small operations use every thread.

The comparison requires equality of saved factors, covariance arrays, noise,
RMS trajectories, available cost arrays and requested diagnostics within each
execution setting. These profiling fits suppress cost recording; separate
fixed-policy integration tests exercise nonempty cost curves.
It checks loaded source paths and actual affinity. Reports retain source/native
hashes, environment limits, numerical versions and snapshot comparison receipts.
Successful dense snapshots are deleted to bound artifact storage; failures leave
their snapshots and logs for diagnosis. Generated reports remain outside git.

Fit timings exclude imports, input generation and snapshot writes. Fit peak RSS
is the fresh process's high-water mark, including imports and input generation;
it is not an incremental allocation measurement. The RMS probe additionally
records NumPy-visible traced allocation peaks and median timings after warmup.
Use repeated paired results and variability, not a single peak or timing, when
deciding whether a patch improves performance. Record CPU model/topology,
software builds, BLAS pools and other competing work for each machine.

## Sparse observed-zero validation

Run `uv run pytest -q tests/test_sparse_observed_zero_indices.py` to compare
modern dense and sparse representations with explicitly stored zeros. The
checks cover fixed-iteration factors, noise, covariance, reconstruction,
uncertainty and cost/RMS histories, including probes and NaN cleanup. They
also retain explicit strict-legacy coordinate and probe behavior. Record the
compatibility mode and software pin with sparse benchmarks. This correctness
repair does not upgrade or rerun a frozen scientific study automatically.

## Masked noise statistics

The full-covariance identity replaces the observed-cell covariance sum with
complete Gram contractions minus missing-cell terms. It preserves the mean
residual, bias uncertainty, priors, floors and full off-diagonal cross terms.
The implementation screens subtraction and falls back to observed accumulation.
The real-arithmetic proof in `analysis/proof_checks/lean/` establishes finite-sum identities;
it does not certify the floating-point screen or statistical convergence.

```bash
uv run pytest -q tests/test_noise_statistics.py tests/test_noise_trajectories.py
uv run python scripts/check_numerical_reference.py
uv run python analysis/compare_backend_revisions.py --help
```

The `masked-noise` comparison suite uses alternating observed/automatic fresh
processes, records cost traces, and checks full posterior outputs at the same
`rtol=1e-8, atol=1e-10`. Its optional joint orthogonal alignment transforms both
factors and both covariance families; physical predictions, means, noise and
priors are compared directly. Raw factor failures and alignment errors remain
in the receipt. Rotation tests reject changed uncertainty and physical outputs.
This accounts for equivalent weak latent coordinates without relaxing tolerances.

On a shared x86 host, 45 paired runs covered ranks 8/32, 1% missingness, a 30%
missing fallback, and five native/BLAS/affinity combinations. Every pair passed.
At one native and one BLAS thread the noise phase improved about 15x/33x and
whole fixed-iteration fits including cost traces about 1.5x/3.6x. These are
descriptive timings for those matrices, not portable performance thresholds.
No frozen publication run or software pin is regenerated.

## Sequential capacity qualification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python analysis/benchmark_capacity_prefix.py --output /tmp/vbpca-capacity-prefix.json
uv run pytest -q tests/test_variable_cell_cv.py
```

The script registers three fixed seeds and three Gaussian fixtures (null,
strong rank one and weak rank two), entrywise folds, capacities 0–6, fixed
80-iteration fitting and independent held-out predictions before execution.
It compares full first minimum, sequential first minimum and full one-SE,
records the full curves, positive/rank-zero fit counts, CPU/wall time, input
hashes and predictive 95% interval coverage. Candidate prefixes must match
full first-minimum selection; later-dip and exhausted-grid unit examples
separately demonstrate why the target differs from global one-SE. The
fixture measures capacity and predictive calibration. It explicitly does not
estimate a downstream significance rank or qualify a rank-test calibration.

The October 21 run agreed on first-minimum capacity in all nine fixtures.
Sequential positive-rank fit counts were 3–6 versus 18 for a full sweep, with
about 0.26–0.76 s versus 1.87–2.31 s on this shared host. Independent predictive
95% interval coverage ranged about 0.927–0.955. These nine small fixtures do
not establish general coverage, type-I error or signal-rank performance; the
weak rank-two fixtures selected capacity one, and one null fixture selected
capacity one. All one-SE capacities happened to agree in these fixtures;
the explicit later-dip test supplies the counterexample to universal agreement.
The default remains the full sweep, and frozen publication policies are unchanged.
## Representative runtime probes and allocation throughput

```bash
uv run pytest -q tests/test_runtime_evidence.py tests/test_runtime_policy.py tests/test_runtime_policy_dense.py tests/test_cov_writeback_autotune.py
uv run python scripts/check_numerical_reference.py
# Supply distinct physical cores from the actual process allocation.
uv run python analysis/benchmark_allocation_throughput.py --cpus 0 2 4 5 --components 8 --output /tmp/vbpca-allocation-q8
uv run python analysis/benchmark_allocation_throughput.py --cpus 0 2 4 5 --components 32 --output /tmp/vbpca-allocation-q32
```

The allocation experiment compares 1/2/4 workers and reverses configuration
order between repeats, with equal total fits, data and initializations. Kernel
statistics, priors, stopping rules and target rank remain fixed; generated
profiles/snapshots stay outside tracked source. BLAS uses one thread per worker.
The pilot screens a 2 GiB combined-worker budget by default; each result records
summed per-process peak RSS, fit CPU time, startup-inclusive wall time, full
posterior numerical receipts and source/native/BLAS/allocation provenance.
This is an advisory shared-host measurement. Rebenchmark on the intended compute
node and allocation rather than inferring universal thread or worker defaults.
Portable package build flags and frozen publication pins remain unchanged.


The October 21 shared-host allocation run used four allocated physical cores,
three alternating-order repeats per configuration and four completed fits per
batch (512 features, 256 samples, four fixed iterations). All posterior
comparisons passed at ranks 8 and 32. Weak latent coordinates are jointly
orthogonally aligned with both covariance families, retaining raw failures in
the receipt; means, priors, noise and full prediction/variance outputs are
compared directly. Timings include imports/startup and therefore need not favor
more workers on these small fits. Summed process peaks stayed below the 2 GiB
pilot-screened budget. These measurements qualify the benchmark's execution and
provenance, not a universal outer-worker or statistical default.
## Advisory performance CI

The manual **Advisory performance comparison** workflow builds two immutable
compatible revisions in one numerical environment. Default `ubuntu-latest`
timings are advisory. A dedicated Linux runner label can be supplied for a
controlled comparison; the label itself does not establish hardware stability.
The baseline and candidate are resolved to SHAs before either is built. A
custom runner must already provide Eigen and a portable C++ compiler; its
allocation and build environment are recorded with the artifacts.

```bash
uv run python scripts/run_performance_comparison.py --baseline BASE_SHA --candidate CANDIDATE_SHA --output /tmp/vbpca-perf-ci
```

The driver chooses up to four distinct allocated physical cores, records
unavailable larger profiles, builds both portable native wheels with the same
interpreter/compiler environment, and runs their deterministic allocation gates.
It then alternates baseline/candidate order over at least three paired repeats
for dense C/F layouts, ranks 8/32, missingness, sparse conversions, frozen sparse
views, diagnostic covariance contractions and safe-tuning startup. Full fit,
phase, packing/conversion, tuning and memory records accompany source/native
hashes, software/BLAS identities and requested/effective pools when supported
by the compared library. Use revisions at or after `16df5d7`.

Numerical and allocation failures fail the workflow. Physical posterior outputs
and cost traces retain required comparisons; equivalent weak latent coordinates
use the existing orthogonal posterior qualifier with raw failures retained.
Timing does not fail the workflow. Reports include paired speed ratios, their
range, median and median absolute deviation. An advisory signal needs a ratio
effect above both 5% and twice its MAD; that is a descriptive triage rule, not a
calibrated statistical test or an adopted hard timing threshold. A controlled
runner needs repeated baseline variability measurements before a hard gate is
chosen. Normal CI retains numerical, output-memory and allocation contracts.
Generated builds/reports stay outside tracked source; workflow artifacts retain
provenance and failed-comparison snapshots for 30 days.
