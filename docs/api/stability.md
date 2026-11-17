# API stability and deprecations

This page describes what counts as VBPCApy's public interface, how changes to
it are announced, and how to find the documentation for the release you have
installed.

## Documentation for each release

Use the version selector at the top of each page to choose a release. `latest`
is the newest release, each `major.minor` entry documents that release line,
and `dev` follows the development branch. Pages for an earlier release show a
banner that links to the latest documentation. Check your installed version
with:

```python
import vbpca_py

print(vbpca_py.__version__)
```

## What is public

The public interface consists of

- the names exported from `vbpca_py` (listed in `vbpca_py.__all__`);
- the `vbpca_py.legacy`, `vbpca_py.options` and `vbpca_py.plotting` modules;
- the constructor parameters, defaults, fitted attributes, return values,
  warnings and exceptions documented in the [API reference](vbpca.md).

Modules and names that begin with an underscore are internal. They can change
in any release without notice.

## Versioning

VBPCApy follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0, a minor release (for example 0.5) can contain breaking changes.
From 1.0 onwards, breaking changes are made only in major releases. Patch
releases contain fixes that do not change the public interface.

A change to a default that alters fitted results is treated like an interface
change: it is listed in the changelog with the setting that restores the
earlier behaviour, where one exists. Results are reproducible for a fixed
version, input, random state and thread configuration. They can differ in the
last digits across platforms, BLAS libraries or thread counts.

## How changes are announced

- Every release lists breaking changes under a **Breaking changes** heading in
  the [changelog](../changelog.md), together with migration steps. Larger
  changes also get a migration guide, such as the
  [0.5 migration guide](migration-05.md).
- A deprecated parameter, function or behaviour emits a `FutureWarning` that
  names its replacement and the release in which it will be removed. It
  remains available for at least one minor release after the warning first
  appears, and from 1.0 onwards it is removed only in a major release.
- Each pull request is checked against the latest release with
  [`griffe check`](https://mkdocstrings.github.io/griffe/guide/users/checking/),
  which reports removed or renamed public objects and changed signatures.
  An intended breaking change must carry the `breaking-change` label and a
  changelog entry.

## The legacy adapter

`vbpca_py.legacy` provides the pre-0.5 feature-first interface and its
historical numerical defaults. It is a supported compatibility interface for
reproducing earlier analyses and comparing results with the original MATLAB
implementation. It is not deprecated. If that changes, the deprecation follows
the rules above.
