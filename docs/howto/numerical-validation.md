# Numerical reference checks

Run the bounded numerical checks before changing the fit loop, update equations
or native kernels:

```bash
uv sync --extra dev --extra plot
just numerical-reference
```

`just ci` includes the same checks. The `numerical-reference` CI job runs on
every pull request and main-branch push, without a path filter or an allowed
failure. Package/native-helper build errors fail the job. Missing imports,
collection errors, skipped checks, expected failures and an empty selection
cannot produce a passing reference run.

## What the checks establish

- Dense score and loading means/covariances are compared with direct NumPy
  solutions of the Gaussian conditional equations. Both observed-cell and
  complement sufficient statistics are exercised, using C/F layouts and one,
  two and four workers.
- Sparse CSC/CSR updates use the same independent equations, including an
  explicitly stored observed zero. Noise corrections are checked against
  independent second-moment calculations, with and without loading covariance.
- Modern-mode fixed-iteration trajectories exercise the actual requested native
  path and compare layouts, worker counts and observed/complement updates.
- Fit-loop RMS and noise updates have independent residual/formula checks.
  Centering, Gaussian data cost and rotation also have direct algebraic checks.
  A seeded strict-legacy end-to-end RMS regression retains its pre-existing
  reference value from revision `5abfeb6761b00c3af00304a375a050ee5bcde4f2`.

The equation checks use `rtol=1e-12`, `atol=1e-13`. Modern trajectory comparisons
use their documented tolerances in `tests/test_update_trajectory.py` (up to
`rtol=5e-9`, `atol=5e-11`); the frozen legacy RMS uses `1e-12` relative and
absolute tolerance. These are small float64 fixtures, not tolerances for all
data, ill-conditioned inputs or arbitrary floating-point backends.

The legacy scalar reference is a compatibility regression, not an independently
established MATLAB oracle. Modern-mode differences in masking and update order
are checked under their own contracts, rather than forcing modern fits to match
legacy endpoints. These checks do not establish posterior calibration,
convergence-policy safety or performance on a scientific study.

## Extended Octave comparisons

`just ci-all` also runs the Octave tests. They require local Octave/MEX tooling
and remain separate from the required bounded reference gate. Optional Octave
availability or helper skips cannot replace or disable the required checks.
The extended CI job is diagnostic; inspect its output when changing the legacy
reference implementations.
