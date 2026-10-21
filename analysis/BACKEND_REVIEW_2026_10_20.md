# VBPCApy backend review, 20 October 2026

Baseline: released 0.4.10, source
`606cbc96c943d80f7d2e922dc7c9daacfd391ad2`. The review covers open issues,
Python orchestration, native update/residual/rotation modules, runtime tuning,
the original reference software and relevant mathematical literature.

The largest measured remaining costs are masked noise updates and full
uncertainty diagnostics. Exact rearrangements appear promising. Adopt them
only after numerical and end-to-end qualification; no runtime update,
statistical default, stopping rule or publication software pin changes in
this review.

## Priority and issue disposition

| Order | Work | Evidence and boundary | Tracking |
| --- | --- | --- | --- |
| 1 | Correct the memory contract | Sparse fit diagnostics exceed the documented limit; guide corrected in this pass | [291](https://github.com/yoavram-lab/VBPCApy/issues/291) |
| 2 | Density-adaptive noise statistics | Dominant iteration phase at near-complete q=32; retain observed-cell fallback | [289](https://github.com/yoavram-lab/VBPCApy/issues/289) |
| 3 | Remove mask/input copies | Three boolean-to-float mask conversions per iteration, plus initialization; redundant preparation copy | [292](https://github.com/yoavram-lab/VBPCApy/issues/292) |
| 4 | Faster, bounded diagnostic contractions | Exact full-covariance prototype; dense outputs and temporaries need a separate memory policy | [290](https://github.com/yoavram-lab/VBPCApy/issues/290) |
| 5 | Representative hardware tuning | Current probe differs from full-VB/complement training; persistent profiles lack hardware identity | [293](https://github.com/yoavram-lab/VBPCApy/issues/293) |
| 6 | Sparse allocation and format reuse | Remaining native per-cell temporaries; two CSR-to-CSC conversions per iteration with final PCA rotation disabled | [294](https://github.com/yoavram-lab/VBPCApy/issues/294) |
| 7 | Optional endpoint cost diagnostics | PRMS selection forces full cost recording even when the caller disables it | [295](https://github.com/yoavram-lab/VBPCApy/issues/295) |
| Separate correctness work | Clean the MATLAB reference debug block | Static empty-covariance access and shared temporary output; Octave execution still needed | [296](https://github.com/yoavram-lab/VBPCApy/issues/296) |

Issues 207–213 already addressed substantial backend work: equation/phase
tests, native mask copies, dense scratch allocation, score/loading complement
statistics, policy reuse, explained variance and sparse observation semantics.
The issues above describe remaining work rather than reopening those repairs.

Keep [247](https://github.com/yoavram-lab/VBPCApy/issues/247) and
[186](https://github.com/yoavram-lab/VBPCApy/issues/186) as scientific safeguards.
Premature stopping can produce a fast but unsuitable result. Previous
stopping candidates failed their qualification; this review does not reverse
that evidence. Frozen 0.4.5/0.4.6 analyses remain unchanged.

[248](https://github.com/yoavram-lab/VBPCApy/issues/248) needed a scope update:
CV already implements `selection_rule="first_minimum", early_stop=True`.
Qualification remains. Prefix patience does not in general reproduce a
full-grid one-SE selection because an unseen later minimum can change the
target. [28](https://github.com/yoavram-lab/VBPCApy/issues/28) now calls for
controlled timing and deterministic allocation checks rather than treating
an `ubuntu-latest` label as fixed hardware.

After the draft, return to
[214](https://github.com/yoavram-lab/VBPCApy/issues/214) for statistical default
retuning, [16](https://github.com/yoavram-lab/VBPCApy/issues/16) for a separately
qualified factorial posterior, and the v0.5 API work. Out-of-sample projection
and sklearn semantics are useful features, but mixing their implementation
with a deadline backend rewrite makes numerical attribution harder.

## Measurements and reproducibility

Local measurements used the public 0.4.10 wheel, Python 3.12.14, NumPy 2.0.0
(OpenBLAS 0.3.27 build) and SciPy 1.18.1 on Linux x86_64. Native, OMP, MKL
and OpenBLAS thread environment limits were one. Six installed Python source
hashes matched the baseline checkout. These are bounded workstation probes,
not measurements on Rockfish or Skipjack.

The diagnostic script is
[`review_backend_costs.py`](review_backend_costs.py); reproduction commands
are in [the benchmark note](../scripts/benchmark_study.md#backend-review-probes).
Matrix shape is 512 features by 256 samples, 1% missingness, four iterations,
fixed seed and all convergence events disabled. The prototypes require
pattern-free full-VB covariances. Timing alternates reference/candidate order
after warmup, includes covariance packing, and reports medians of five noise
pairs or three variance pairs.

| Rank | Current noise covariance sum | Prototype | Phase ratio | Absolute difference |
| --- | --- | --- | --- | --- |
| 8 | 8.40 ms | 0.658 ms | 12.8 | 1.46e-11 |
| 32 | 164 ms | 4.31 ms | 38.0 | 1.56e-10 |

| Rank | Current marginal variance | Prototype | Phase ratio | Maximum absolute difference |
| --- | --- | --- | --- | --- |
| 8 | 10.9 ms | 1.39 ms | 7.85 | 2.78e-17 |
| 32 | 176 ms | 18.5 ms | 9.52 | 3.89e-16 |

The noise timings compare the covariance/mean-variance sum with RMS set to
zero. Separate six-iteration tests retain RMS, the current hyperprior/floor
and cost calculation and compare full-VB factors, covariance and cost/RMS
trajectories for both score/loading accumulation modes. Neither result
qualifies a new production path at arbitrary dimensions or scales.

A separate three-repeat phase probe using
`benchmark_backend_phases.py` found that the q=32, 1%-missing noise phase
took about 0.664 s of 0.837 s total iteration time for four iterations
(about 79%). At q=8 it was about 58%. Diagnostics were disabled in that
probe. These proportions explain the priority; phase ratios above cannot be
applied to total runtime without accounting for the other work.

Small orchestration reproductions found:

- Four iterations caused four float64 mask conversions in each score/loading
  updater and five in RMS including initialization. At 5,846 by 2,504 this
  represents about 351 MB of mask-write traffic per iteration, not that much
  additional resident memory accumulated on every iteration.
- A sparse 24 by 36 fit with `max_dense_bytes=1` returned three dense diagnostic
  arrays of 6,912 bytes each. The current guard applies to dense observation
  masks, not these outputs. Three float64 outputs at 5,846 by 2,504 alone
  occupy about 335 MiB before temporary arrays and covariance storage.
- Four sparse iterations with `rotate2pca=False` caused eight CSR-to-CSC
  conversions. They are outside the score/loading subphase timers but inside
  total iteration time. Value-changing updates need explicit invalidation
  before caching views.
- Two four-iteration PRMS candidates made eight cost evaluations despite
  `record_cost=False`. A separate fixed-iteration 18%-missing probe measured
  about 27% extra time at q=8 and 4% at q=32 with cost recording; factors and
  noise were identical in that probe. Replay/stopping diagnostics still need
  full curves when requested.

Current CV already omits full fit diagnostics for many candidate paths;
do not count the full diagnostic allocation against every CV candidate.
Likewise, some ownership and probe-isolation copies are necessary. Remove
only copies whose ownership and observation contracts are preserved.

## Paper, reference code and algebra

[Ilin and Raiko (2010)](https://jmlr.org/papers/volume11/ilin10a/ilin10a.pdf),
Appendix C, supplies the full-covariance update equations. Section 5's
reconstruction uncertainty retains score, loading, covariance-cross and bias
terms; predictive measurement uncertainty adds noise. Section 4.3 uses
posterior second moments in rotations. Section 6.3 and Appendix D describe
the factorial VBPCAd family, which is a distinct approximation. Numerical
agreement under artificial missingness does not establish the model's
missing-data assumptions for an application.

The proposed scalar noise contraction is the complete-data covariance sum

```text
<A.T A, sum_j Sv[j]>_F
+ <S S.T, sum_i Av[i]>_F
+ <sum_i Av[i], sum_j Sv[j]>_F
```

minus the same contributions over missing cells, plus observed-count-weighted
bias variance and `RMS² * n_observed`. The prototype retains all covariance
cross terms. The marginal-variance prototype reshapes outer products and full
covariances for three matrix products; it is not a diagonal approximation.

[Proof sources and assumptions](proof_checks/README.md) include six
kernel-checked Lean identities with arbitrary finite dimensions and two SymPy
checks. The algebra needs no covariance positivity or symmetry assumption;
interpreting the arrays as covariances does. The SymPy moment calculation
assumes independent centered errors with finite second moments. These checks
do not prove floating-point safety, inference convergence or scientific
calibration. Complete-minus-missing subtraction needs a cancellation-aware
eligibility rule and fallback before production use.

The [author's toolbox](https://users.ics.aalto.fi/alexilin/software/) was
downloaded independently. Provenance:

| Artifact | SHA256 |
| --- | --- |
| `pcamv.zip` | `3e3b8a5e1e25437e4d4928ac6574620edd1d0437ee33de4616bea74959f9a898` |
| Original `pca_full.m` | `2f821f2efe56fcdaab34db8675cea3c1036811d76f92b8686a7e157c7034ef62` |
| Original `cf_full.m` | `bc938a3aa875f81e82e34b3aa74c454b114ff7191d566c9cee6165024b2e863d` |

The vendored MATLAB file contains local trace instrumentation. Its early
debug block accesses `Av{1}` unconditionally and writes a shared `/tmp`
filename; the downloaded original does not. This is a static finding, not
an executed Octave failure. Preserve useful trace fields while removing the
side effect and checking covariance-free algorithms. The original software
also has small hyperprior terms in its noise/prior updates. Preserve those
terms when comparing code; matching an unregularized displayed formula alone
would silently change the implemented method.

For possible later algorithmic acceleration,
[Luttinen, Ilin and Raiko (2009)](https://www.esann.org/sites/default/files/proceedings/legacy/es2009-113.pdf)
derive posterior-aware centering/whitening from the variational objective.
This supports treating rotations as part of the optimization method. It does
not justify removing their uncertainty terms or disabling them solely for a
shorter timing. A new acceleration needs its own objective and trajectory
checks, beyond the exact sum rearrangements considered here.

## Hardware tuning and the flepiMoP precedent

The public historical flepiMoP revision referenced by the legacy application
is `be0f220ab46ee330ddf2b05e3b3e295132b01182`. Its
[batch resource estimator](https://github.com/HopkinsIDD/flepiMoP/blob/be0f220ab46ee330ddf2b05e3b3e295132b01182/flepimop/gempyor_pkg/src/gempyor/batch/_estimate.py)
uses bounded smaller job sizes, scheduled measurements and prediction bounds
to produce reusable CPU/memory/time estimates. Its
[inference notebook](https://github.com/HopkinsIDD/flepiMoP/blob/be0f220ab46ee330ddf2b05e3b3e295132b01182/examples/simple_usa_statelevel/inference_benchmark.ipynb)
also separates outer processes from linear-algebra threading. Adapt the
measurement workflow, rather than copying `cpu_count()-2` or extrapolating
VBPCA cost linearly in rank. In VBPCA the covariance work includes quadratic
and solve costs; an unvalidated small-rank extrapolation can be misleading.
Set thread variables before importing numerical libraries and respect the
actual scheduler/affinity allocation.

The current VBPCA tuner measures C-order covariance-free observed-cell kernels,
while full-VB fitting can use covariances and complement accumulation. Optional
persistent profiles key shape/rank/observed count but omit hardware, allocation
and build identity. `bulk` and `kernel` currently follow the same writeback
branch, so benchmarking both does not compare two implementations. These
limitations make per-host fixed-policy pilot fits more useful than trusting
one autotuned count without validation.

| Dimension | Qualification settings and records |
| --- | --- |
| Machine/allocation | CPU model and ISA, OS, affinity, scheduler CPUs, cgroup quota, SMT; record node and job allocation |
| Numerical build | VBPCA source/native hashes, compiler flags, Python/NumPy/SciPy, BLAS vendor/version and actual thread pools |
| Workload | Shape, rank, layout, observed fraction and missing-pattern geometry, covariance/pattern mode, compatibility/update order |
| Execution | Native versus BLAS threads; observed versus complement; diagnostics; measured versus reused policy |
| Measurement | Separate cold/tuning and warm fit time; repeated paired timings; phase and total time; peak RSS; CPU-hours and concurrent-fit throughput |
| Eligibility | Fixed iterations and identical statistical settings first; equations/trajectories/selection checks before accepting a speed result |

Use allocated compute nodes, not login nodes. Begin with the bounded q=8
genotype pilot already registered in pp-eigentest. Examine its receipts before
expanding rank or concurrency within the remaining resource ceiling. Compare
single-fit latency with outer-worker throughput; the fastest individual fit
need not minimize study CPU-hours. Keep portable distribution wheels, and
qualify any node-specific ISA build explicitly.

[NumPy documents](https://numpy.org/doc/stable/reference/global_state.html#number-of-threads-used-for-linear-algebra)
that BLAS can use an independent thread pool. Native `num_cpu` is therefore
not a total-process thread limit. Linux
[cgroup CPU quotas](https://docs.kernel.org/admin-guide/cgroup-v2.html)
can further constrain a process beyond its affinity mask. Report requested
and effective counts separately; do not reinterpret a kernel's internal
clamp as proof that the requested setting ran.

## Adoption and release decision

The baseline changelog has an empty Unreleased section after 0.4.10.
Land review scripts, bounded checks and factual documentation first. They
do not warrant a PyPI release on their own. Exact backend repairs can justify
a patch release once numerical references, representative trajectories,
memory behavior and target-hardware checks pass. State measured scope in
release notes and retain portable builds.

Posterior-family, stopping-policy, rank-target or broad default changes need
separate scientific qualification and an explicit study-adoption decision.
Do not bundle them into an urgent performance patch or silently rerun locked
publication analyses. Hosted CI is currently blocked by organization billing;
run relevant checks locally and retain the results, without representing
unstarted hosted jobs as green.
