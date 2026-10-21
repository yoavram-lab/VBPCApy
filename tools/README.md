# MATLAB/Octave reference

The reference software is Alexander Ilin and Tapani Raiko's
[PCA with missing values](https://users.ics.aalto.fi/alexilin/software/pcamv.zip),
accompanying [Ilin and Raiko (2010), Appendix C](https://jmlr.org/papers/v11/ilin10a.html).
The source archive reviewed in October 2026 has SHA256
`3e3b8a5e1e25437e4d4928ac6574620edd1d0437ee33de4616bea74959f9a898`;
its original `pca_full.m` has SHA256
`2f821f2efe56fcdaab34db8675cea3c1036811d76f92b8686a7e157c7034ef62`.
The previously omitted `miscomb.m` pattern-grouping helper is copied unchanged
from that archive so `uniquesv=1` can execute.

Our `pca_full.m` retains local, in-memory learning-curve instrumentation:
noise variance, the individual noise-update contributions, and covariance-sum
traces. These fields support numerical reference checks and are additional to
the upstream learning curve. No covariance debug files are exported. The
reference's documented autosave options remain available.

The score/loading/mean updates, rotations and small hyperprior terms remain
those of the upstream software. In particular the noise update retains
`(sXv + 2*hpV) / (ndata + 2*hpV)` with `hpV=0.001`; the displayed paper equations
do not justify silently dropping those software regularizers. Point-estimate
PPCA/MAP runs have empty loading covariance stores; their covariance trace
is recorded as `NaN` and their loading-covariance noise terms are zero.

Run the required Python numerical gate with `just numerical-reference`.
With Octave and the optional Python bridge installed, run
`uv run pytest -q -m octave`. The reference algorithm smoke tests cover VB,
PPCA and MAP, both covariance-pattern settings and both rotation settings.
Interpreter absence skips the local interpreter-dependent tests. Once Octave
is installed, an evaluation failure fails the check and retains its status,
stdout and stderr. CI installs Octave when the path filter selects this job;
the job's failures are no longer advisory. Helper-control regressions also run
in the standard Python suite without requiring Octave.
