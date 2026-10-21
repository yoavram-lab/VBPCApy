# Dense mask and ownership repair validation

The first repair from the backend review preserves boolean masks, removes
unnecessary fully observed scans when covariances rule out the fast path,
uses the newly owned RMS residual as its output buffer, and removes one
duplicate preparation copy. A shared mask normalizer retains the historical
float coercion for numeric masks. Preparation retains the former C layout;
caller arrays and probe isolation remain protected.

Baseline source is `07ae33fef8257c5241f403bedbb905d9d1f79de9` (0.4.10).
Profiling used identical native extension binaries in both trees. Loaded source hashes
remain constant across all profiles, with differences limited to
`_full_update.py`, `_rms.py` and `_sparsity.py`. Statistical options, stopping
events and frozen publication pins are unchanged.

## Numerical checks

The new storage tests cover C/F layouts, positive and negative strides,
read-only arrays, integer/float inputs, numeric masks, observed zeros,
caller ownership and the single preparation allocation. Fixed six-iteration
fits compare boolean and float mask handling across VB, PPCA and MAP, modern
and legacy compatibility, shared covariance patterns, explicit probes and
one/two/four native threads. These tests retain cost traces and require exact
array equality, including factors, covariances and diagnostics.
Additional integration checks change a held-out value from zero to one million
and require identical training endpoints. Selection checks retain the same
capacity, scores and candidate metrics at one/two/four native threads.

The independent source-revision profiles below also require equality of their
saved numerical arrays. All **45 pairs passed**. Their fit workloads suppress
cost recording; the separate integration tests exercise nonempty cost curves.
The comparator rejects changed values/dtypes, incomplete snapshot fields and
missing or additional snapshots. Success receipts retain array and file
identities while successful dense snapshots are removed to limit disk use.

## Runtime and memory profiles

Measurements used Linux x86_64, Python 3.12, NumPy 2.5.4 and SciPy 1.18.1 on an
Intel Core Ultra 7 265U. CPU IDs 4–7 were verified as four distinct physical
cores with no SMT siblings in those sets. Affinity restricted each child to
one, two or four of those CPUs. Separate inspection under each environment
confirmed the NumPy and SciPy OpenBLAS pools had the requested one/four thread
limits (reported builds 0.3.34.106.0 and 0.3.31.dev respectively).

There were three fresh-process pairs for each workload/execution setting,
alternating which revision ran first. Children ran sequentially. Fit timings
exclude imports, input generation and output snapshots. Peak RSS includes the
entire fresh child. Workstation timing has interference and thermal variation;
three pairs establish a bounded screen, not a cross-machine speed guarantee.

The rank-8 fit uses 1,536 features, 1,024 samples, 1% missingness and no full
diagnostics. The rank-32 fit uses 512 features, 256 samples, 20% missingness
and diagnostics. Both use four fixed iterations and complement score/loading
statistics. The final column gives the median of paired percentage changes;
it need not equal the change between the two independently computed medians.

| Cores | Native / BLAS limit | Rank-8 fit seconds, before → after | Median paired change |
| --- | --- | --- | --- |
| 1 | 1 / 1 | 1.357 → 1.354 | −1.9% |
| 2 | 2 / 1 | 1.004 → 0.937 | −6.7% |
| 4 | 4 / 1 | 0.792 → 0.672 | −9.8% |
| 4 | 1 / 4 | 1.330 → 1.282 | −2.2% |
| 4 | 4 / 4 | 0.848 → 0.805 | −4.3% |

Rank-8 median peak RSS fell from 278–290 MiB to 267 MiB across most settings
(the one-core baseline was 281 MiB). Within a setting, median reductions were
about 11–22 MiB. Individual timing changes ranged from −15.9% to +1.9%.

| Cores | Native / BLAS limit | Rank-32 fit seconds, before → after | Median paired change |
| --- | --- | --- | --- |
| 1 | 1 / 1 | 1.717 → 1.731 | +0.8% |
| 2 | 2 / 1 | 1.056 → 1.062 | +0.5% |
| 4 | 4 / 1 | 0.781 → 0.771 | −1.6% |
| 4 | 1 / 4 | 1.644 → 1.631 | −0.3% |
| 4 | 4 / 4 | 1.119 → 1.131 | −1.3% |

Rank-32 memory was about 175 MiB for both revisions. Individual timing changes
ranged from −7.0% to +9.0%; this screen does not establish a useful rank-32
whole-fit speedup. Noise/covariance costs still dominate that regime.

The separate rank-8 RMS probe uses 3,000-by-1,500 arrays and 20% missingness.
Seven timed calls follow warmup, then one additional call records traced peak
allocation. Its array values and RMS are equal between revisions.

| Cores | Native / BLAS limit | Median RMS milliseconds, before → after | Median paired change |
| --- | --- | --- | --- |
| 1 | 1 / 1 | 51.2 → 34.1 | −32.4% |
| 2 | 2 / 1 | 53.2 → 36.9 | −30.6% |
| 4 | 4 / 1 | 49.3 → 35.4 | −28.2% |
| 4 | 1 / 4 | 46.3 → 29.9 | −35.2% |
| 4 | 4 / 4 | 47.3 → 29.7 | −36.8% |

The last two settings differ in native-thread configuration, which RMS dense
matrix multiplication does not use. Traced peaks fell from 144,001,216 to
72,001,024 bytes. Median fresh-process peak RSS fell from about 317 to
248 MiB. These are distinct measurements: traced allocation concerns the RMS
call, while RSS also includes imports, inputs, warmup and allocator behavior.

Four native threads with a one-thread BLAS limit were fastest among the tested
fit configurations on this machine. This does not choose SNP settings or a
new default. The completed SNP attempt stopped in its native fixture before
any SNP-data fit, so there is no completed SNP runtime to extrapolate from.

## Reproduction and adoption

Use the [fresh-process procedure](../scripts/benchmark_study.md#source-revision-memory-and-thread-profiles),
with baseline and candidate trees in one numerical environment. Retain actual
pool/topology metadata separately. Generated JSON/logs remain outside git.
Run the full suite, numerical-reference gate, strict source type checks and
lint checks before merging. Optional Octave parity remains a separate gate.

The local comparison report SHA256 is
`5db2d86e387ed406fb6fa05f40286e704b443844926bb1f9dd742183cc3cf716`.
Its recorded driver and benchmark-source hashes match the committed files.
The separately rebuilt candidate passed all 87 focused tests and all 48
required numerical references, with no unavailable references. All 1,181 tests
collect from the source archive against the candidate package. Full-suite and
platform-matrix results are recorded in the repair PR's validation and CI.

This is eligible for inclusion in a patch release after CI passes. It does
not adopt a new algorithm or retune statistical defaults. Hardware tuning,
diagnostic memory limits, exact noise statistics and sparse overhead remain
separate repairs with their own qualification.

## Rockfish qualification

Rockfish job `32325436` completed all 45 pairs on node `sr07`, an Intel Xeon
Gold 6448Y, using four distinct physical cores (CPU IDs 0–3). It used Python
3.11.7, NumPy 2.2.6, SciPy 1.17.1 and the isolated public 0.4.10 native build.
All saved numerical arrays matched exactly. Loaded source signatures were
constant across fits; only the same three Python files differed between
revisions. All six native binary hashes matched between revisions and the
hardware receipt. Each child checked its requested BLAS pools.

| Workload | Cores | Native / BLAS | Seconds, before → after | Median paired change | Child peak RSS MiB, before → after |
| --- | --- | --- | --- | --- | --- |
| dense-q8 | 1 | 1 / 1 | 1.3080 → 1.2088 | -7.6% | 257.3 → 245.3 |
| dense-q8 | 2 | 2 / 1 | 1.1575 → 1.0082 | -12.6% | 257.4 → 245.0 |
| dense-q8 | 4 | 4 / 1 | 0.9573 → 0.8543 | -10.8% | 257.4 → 245.4 |
| dense-q8 | 4 | 1 / 4 | 1.3708 → 1.2268 | -9.1% | 263.2 → 250.7 |
| dense-q8 | 4 | 4 / 4 | 1.1364 → 1.1114 | -2.2% | 263.2 → 251.0 |
| dense-q32 | 1 | 1 / 1 | 1.6674 → 1.6587 | -0.5% | 152.7 → 150.9 |
| dense-q32 | 2 | 2 / 1 | 1.1724 → 1.1714 | -0.1% | 154.0 → 151.1 |
| dense-q32 | 4 | 4 / 1 | 0.7934 → 0.7843 | -0.8% | 153.0 → 153.0 |
| dense-q32 | 4 | 1 / 4 | 1.7134 → 1.7645 | +3.6% | 159.7 → 156.9 |
| dense-q32 | 4 | 4 / 4 | 1.5868 → 1.6085 | +1.4% | 160.1 → 157.2 |
| rms | 1 | 1 / 1 | 0.0917 → 0.0617 | -32.9% | 299.7 → 229.4 |
| rms | 2 | 2 / 1 | 0.0915 → 0.0602 | -32.9% | 299.9 → 229.5 |
| rms | 4 | 4 / 1 | 0.0891 → 0.0592 | -33.6% | 299.9 → 229.5 |
| rms | 4 | 1 / 4 | 0.0805 → 0.0506 | -37.5% | 312.0 → 241.2 |
| rms | 4 | 4 / 4 | 0.0796 → 0.0506 | -37.0% | 311.9 → 241.4 |

The rank-8 fit used about 12 MiB less peak process memory. Its median paired
time reductions ranged from 2.2% to 12.6% across these settings. Rank-32
changes were small and included slowdowns; no useful whole-fit speedup is
established there. RMS traced peaks fell from 144,067,452 to 72,067,260 bytes,
with median paired time reductions of 32.9–37.5%. Four native threads and
one BLAS thread were fastest among the tested fit settings on this host.

The job took 277 seconds on four allocated CPUs (0.3078 allocated CPU-hours;
396.297 seconds of accounted CPU time). Slurm sampled RSS is separate from
the fresh-child peaks in the table. Setup plus Rockfish consumed about 0.3233
allocated CPU-hours of the three-hour ceiling; Skipjack is still queued.
These are synthetic paired comparisons at pinned revisions, not SNP fits or
a cross-hardware scaling claim.

Compact identities and aggregates are retained in
[`cluster/registrations/2026-10-20/rockfish-summary.json`](cluster/registrations/2026-10-20/rockfish-summary.json).
Raw `comparison.json`, `hardware.json` and scheduler output remain under
`/scratch4/struelo1/jcmacdon/vbpca-backend-292-aca9189/` and in the local
qualification artifact directory. The comparison SHA256 is
`846d00b3accd5a6b31506211d6e29deb26138245039957c97337c7489a8b01be`.
