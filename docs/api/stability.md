# API stability and deprecations

This page describes what counts as VBPCApy's public interface, how changes to
it are announced, and how to find the documentation for the release you have
installed.

## Documentation for each release

Use the version selector at the top of each page to choose a release. `latest`
is the newest release, each `major.minor` entry documents that release line,
and `dev` follows the development branch. Pages for any version other than
`latest` show a banner that links to the latest documentation. Check your
installed version with:

```python
import vbpca_py

print(vbpca_py.__version__)
```

## What is public

The public interface consists of

- the modules whose names do not begin with an underscore: `vbpca_py`,
  `vbpca_py.configuration`, `vbpca_py.defaults`, `vbpca_py.encoding`,
  `vbpca_py.estimators`, `vbpca_py.legacy`, `vbpca_py.masking`,
  `vbpca_py.model_selection`, `vbpca_py.options`, `vbpca_py.plotting`,
  `vbpca_py.preprocessing`, `vbpca_py.raw_model_selection` and
  `vbpca_py.results`, and the names in them that do not begin with an
  underscore;
- the low-level `pca_full` function documented in the API reference, which is
  currently imported from `vbpca_py._pca_full`;
- the constructor parameters, defaults, fitted attributes, return values,
  warnings and exceptions documented in the [API reference](vbpca.md).

Other modules and names that begin with an underscore are internal. They can
change in any release without notice.

## Versioning

VBPCApy follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0, a minor release (for example 0.5) can contain breaking changes.
From 1.0 onwards, breaking changes are made only in major releases. Patch
releases contain fixes that do not change the public interface.

A change to a default that alters fitted results is treated like an interface
change: it is listed in the changelog with the setting that restores the
earlier behaviour, where one exists. A fit is reproducible for a fixed version,
input, random state, `runtime_tuning="off"` and explicit thread counts
(`num_cpu`). The default runtime tuning chooses thread counts by timing, and
results can differ slightly between thread counts, platforms and BLAS
libraries.

## How changes are announced

- From release 0.5.1 onwards, the [changelog](../changelog.md) lists breaking
  changes under a **Breaking changes** heading in each release, together with
  migration steps. Release 0.5.0 describes its breaking changes in its
  introduction and in the [0.5 migration guide](migration-05.md); larger
  changes will continue to get a migration guide.
- A deprecated parameter, function or behaviour emits a `FutureWarning` that
  names its replacement. Deprecations introduced from 0.5.1 onwards also name
  the release in which they will be removed. A deprecated feature remains
  available for at least one minor release after the warning first appears,
  and from 1.0 onwards it is removed only in a major release.
- Each pull request is checked against the latest release with
  [`griffe check`](https://mkdocstrings.github.io/griffe/guide/users/checking/),
  which reports removed or renamed public objects and changed signatures. An
  intended breaking change must carry the `breaking-change` label and a
  changelog entry.

## The legacy adapter

`vbpca_py.legacy` provides the pre-0.5 feature-first interface and its
historical numerical defaults. It is a supported compatibility interface for
reproducing earlier analyses and comparing results with the original MATLAB
implementation. It is not deprecated. If that changes, the deprecation follows
the rules above.
