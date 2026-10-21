# Python API and typing modernization

This work accompanies the backend repairs. Keep numerical changes separate
from API migration so that their evidence and compatibility costs can be
reviewed independently. Preserve the frozen pp-eigentest environments.

The maintainer approved completing the full modernization and native estimator
stack in one 0.5 release on 21 October 2026. The current execution order,
Pipeline/missingness decisions, migration and release gates are in
[V05_IMPLEMENTATION_PLAN.md](V05_IMPLEMENTATION_PLAN.md). That plan supersedes
the earlier suggestion to ship this work through incremental 0.4 patches.

## Patterns reviewed

- flepimop2 uses dtype-specific array aliases, protocols and validated
  configuration boundaries, with executable examples collected by CI.
- op_system uses typed result records and frozen/slotted configurations in
  its qualification helpers, with explicit ownership and tolerance contracts.
- op_engine uses typed callable interfaces and validated problem/configuration
  records. Backend-specific array types remain explicit.

Apply these patterns within VBPCApy's Python 3.11 minimum. This is an independent
implementation; no source is copied between repositories. A frozen dataclass
does not make an array immutable, and a type annotation does not enforce an
array's shape or dtype at runtime. Keep those contracts explicit and tested.

## First change: typed results

`vbpca_py.results` describes the existing dictionary keys, float64 arrays,
integer pattern indices and covariance representations. `pca_full()` returns
`PCAResult`; the estimator exposes `LearningCurve`. Learning curves contain
numeric histories, scalar iteration indices, strings, patience counts and
optional prior vectors. Replace the inaccurate all-numeric annotation and
remove the casts and assignment suppressions it required.

Runtime return keys, covariance sharing and array ownership are retained.
Renaming the historical keys is deferred; descriptive documentation provides
their meaning without forcing every downstream caller to migrate. Callers
needing an abstract read-only result mapping can annotate `Mapping[str, object]`.

Validation includes strict typing of example consumers, rejection of unknown
keys and incorrectly typed metadata, actual result representations for
dense/sparse and VB/PPCA/MAP fits, diagnostics on/off, covariance sharing and
list/stacked covariance writeback. The test suite executes the result-module
and `pca_full` docstring examples. These checks supplement the existing full
regressions and independent numerical references.

## Next changes

1. Replace solver option dictionaries at the numerical boundary with a typed
   validated configuration. Coordinate categorical literals and validation
   with the estimator API. Keep public constructor parameters discoverable by
   `get_params`, cloning and grid search; store constructor inputs unchanged
   and validate them at fit time. Define the compatibility path for low-level
   keyword callers before changing their handling.
2. Extend dtype/shape contracts through the numerical helpers as each backend
   repair lands. Keep dense numeric masks distinct from boolean observation
   masks and sparse observed zeros distinct from missing entries. Avoid broad
   `Any` substitutions that hide an uncertain contract.
3. Use frozen/slotted records for new immutable configuration and receipt
   structures. Retain mutable training state where the algorithm updates it.
   Test ownership and serialization where the change affects callers.
4. Add bounded examples for each updated public interface and execute them in
   CI. Extend documentation example collection after auditing existing examples
   and optional-dependency requirements; examples should explain stopping and
   diagnostic choices without implying convergence from a short demonstration.

The dense-copy repair remains the first performance change. Diagnostic memory
guards and exact noise-update acceleration follow their separate correctness
and resource checks. Hardware-policy retuning follows representative backend
profiles; do not select a new statistical or stopping default during this
modernization. Type/documentation-only work can accompany a qualified patch
release, but does not by itself justify a release or a new scientific run.
