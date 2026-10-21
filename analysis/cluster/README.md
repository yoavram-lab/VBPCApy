# Backend comparison on allocated cluster cores

This qualification repeats the paired source comparison on Rockfish and
Skipjack. It uses synthetic matrices, not the SNP analysis, and does not
qualify a stopping policy or change a statistical default.

Each host receives one job with four physical cores, 4 GiB and a ten-minute
wall limit. Allow up to eight accounted logical CPUs per profile for SMT
rounding. Skipjack also receives one environment preparation job requesting
one CPU, 2 GiB and a ten-minute limit; allow two accounted CPUs for that job.
The total allocation ceiling is **three CPU-hours**, including preparation.
There are no automatic retries; failed
or preempted jobs require a new recorded decision within the remaining budget.
Existing jobs and scientific environments are left in place.

The comparison has three repeats of each of three workloads at five native
and BLAS thread settings: 45 pairs per host. Children run sequentially. See
[`../BACKEND_COPY_VALIDATION_2026_10_20.md`](../BACKEND_COPY_VALIDATION_2026_10_20.md)
for the workload sizes, fixed iterations, numerical checks and timing scope.

Before submission, stage two Git source archives in a new directory with
`baseline/` and `candidate/` prefixes, plus `archives.sha256` and an external
`registration.json`. The registration records complete source revisions,
per-file SHA256 identities, archive hashes, environment pins, limits and the
absence of automatic retries. Supply its checksum independently to the job.
Verify archives before extracting them; retain the registration and scheduler
receipt alongside the results.

Export `VBPCA_PROFILE_ROOT`, `VBPCA_PROFILE_PYTHON` and
`VBPCA_PROFILE_REGISTRATION_SHA256`, then submit `backend_comparison.sbatch`
with the site's partition/account options. Skipjack's noninteractive SSH
sessions require `/etc/profile.d/00-slurm.sh`. Prepare its separate environment
with `prepare_profile_environment.sbatch` and `VBPCA_PROFILE_ENV`; that job
installs only pinned public wheels. Rockfish reuses its existing isolated
0.4.10 interpreter without modifying it.

The preflight validates both source manifests before linking the same installed
native extensions into their staged trees. Do not build or install into those
trees afterward. It rejects insufficient physical cores and records the host,
allocation, versions and native binary hashes. Each child records its actual
BLAS pools and must match the requested limits. The comparator requires exact
equality of all saved numerical arrays and retains comparison receipts.

Report within-host paired timing and memory changes. Hardware, Python and
wheel builds can differ across clusters, so absolute times alone do not
establish hardware scaling. Preserve incomplete receipts on failure and keep
generated matrices and snapshots out of Git.

The first registration and its allocation amendment are retained under
`registrations/2026-10-20/`. The frozen source archive has a 20-minute launcher
default. Its submissions use the amended ten-minute limit: an explicit
override on Skipjack and a reduction of the pending Rockfish job's time limit.
The newer launcher defaults to ten minutes. Numerical sources are identical.
