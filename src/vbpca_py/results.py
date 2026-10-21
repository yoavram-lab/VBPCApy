"""Typed dictionaries for the existing PCA_FULL results and learning curves.

These annotations describe ordinary dictionaries; they do not copy arrays,
change mutability or rename result keys. Matrix orientation remains features
by samples. Array dimensions below use ``p`` features, ``n`` samples and ``q``
fitted components. NumPy typing specifies dtype but does not check shapes.

Examples:
    >>> import numpy as np
    >>> from vbpca_py._pca_full import pca_full
    >>> data = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    >>> result = pca_full(
    ...     data,
    ...     1,
    ...     maxiters=2,
    ...     niter_broadprior=0,
    ...     return_diagnostics=False,
    ...     random_state=4,
    ... )
    >>> result["A"].shape, result["S"].shape
    ((2, 1), (1, 3))
    >>> result["Xrec"] is None
    True
    >>> isinstance(result["lc"]["returned_iteration"], int)
    True
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

Float64Array: TypeAlias = npt.NDArray[np.float64]
IndexArray: TypeAlias = npt.NDArray[np.int_]
PosteriorCovariances: TypeAlias = MutableSequence[Float64Array] | Float64Array
"""Covariance blocks as a sequence or stacked array; possibly empty.

Score blocks may be shared by missingness pattern. Use ``Isv`` to look up
the block for a sample when that index is present.
Disabled covariance terms have an empty representation.
"""

PhaseTimingKey: TypeAlias = Literal[
    "phase_scores_sec",
    "phase_loadings_sec",
    "phase_rms_sec",
    "phase_noise_sec",
    "phase_converge_sec",
    "phase_total_sec",
]
NumericTraceKey: TypeAlias = Literal[
    "rms",
    "prms",
    "time",
    "cost",
    "angle",
    "_stop",
    "phase_scores_sec",
    "phase_loadings_sec",
    "phase_rms_sec",
    "phase_noise_sec",
    "phase_converge_sec",
    "phase_total_sec",
    "criterion_satisfied_angle",
    "criterion_satisfied_earlystop",
    "criterion_satisfied_rms_plateau",
    "criterion_satisfied_cost",
    "criterion_satisfied_composite",
    "criterion_satisfied_slowing_down",
    "criterion_satisfied_cost_plateau",
    "criterion_satisfied_cfstop_rel",
    "criterion_satisfied_cfstop_curv",
    "criterion_available_slowing_down",
]


class LearningCurve(TypedDict, total=False):
    """Metric histories and metadata from a fit, including optional fields.

    Numeric histories include the initial state at index zero. Metadata are
    scalars, strings or mappings, rather than numeric histories. ``prior_trace``
    contains one vector per recorded update when prior tracing is requested.
    Fields beginning with an underscore support the live training loop.
    """

    rms: list[float]
    prms: list[float]
    time: list[float]
    cost: list[float]
    angle: list[float]
    phase_scores_sec: list[float]
    phase_loadings_sec: list[float]
    phase_rms_sec: list[float]
    phase_noise_sec: list[float]
    phase_converge_sec: list[float]
    phase_total_sec: list[float]
    criterion_satisfied_angle: list[float]
    criterion_satisfied_earlystop: list[float]
    criterion_satisfied_rms_plateau: list[float]
    criterion_satisfied_cost: list[float]
    criterion_satisfied_composite: list[float]
    criterion_satisfied_slowing_down: list[float]
    criterion_satisfied_cost_plateau: list[float]
    criterion_satisfied_cfstop_rel: list[float]
    criterion_satisfied_cfstop_curv: list[float]
    criterion_available_slowing_down: list[float]
    convergence_reason: str
    terminal_iteration: int
    returned_iteration: int
    best_probe_iteration: int
    best_probe_rms: float
    prior_trace: list[list[float]]
    _stop: list[float]
    _criterion_patience: dict[str, float]
    _candidate_convergence_reason: str
    _convergence_reason: str


class CovarianceResult(TypedDict):
    """Posterior covariance blocks and optional sample-to-pattern index."""

    A: PosteriorCovariances
    S: PosteriorCovariances
    Isv: IndexArray | None
    Mu: Float64Array


class HyperparameterResult(TypedDict):
    """Prior variances for loadings (``Va``) and the mean (``Vmu``)."""

    Va: Float64Array
    Vmu: float


class PCAResult(TypedDict):
    """Full-covariance PCA result with the historical key names.

    ``A`` and ``S`` have shapes ``(p, q)`` and ``(q, n)``. ``Mu`` holds the
    feature means and ``V`` the observation-noise variance. ``Av``/``Sv`` are
    covariance blocks, also available through ``cv``. They are not elementwise
    variance matrices. ``Muv`` contains posterior mean variances; ``Va``/``Vmu``
    are prior variances, also available through ``hp``.

    With diagnostics enabled, ``Xrec``, ``Vr`` and ``Vpred`` have shape
    ``(p, n)``: reconstruction, latent reconstruction variance and predictive
    variance for a new observed entry, respectively. ``Vpred`` adds ``V`` to
    ``Vr``. These arrays and the explained-variance arrays are ``None`` when
    diagnostics are disabled. ``RuntimeReport`` is optional runtime metadata.

    ``RMS``, ``PRMS`` and ``Cost`` describe the returned model, which may be an
    earlier probe-selected state. Missing metrics are NaN. Probe iteration and
    RMS fields are ``None`` when no finite probe optimum was recorded.
    """

    A: Float64Array
    S: Float64Array
    Mu: Float64Array
    V: float
    Av: PosteriorCovariances
    Sv: PosteriorCovariances
    Isv: IndexArray | None
    Muv: Float64Array
    Va: Float64Array
    Vmu: float
    lc: LearningCurve
    Xrec: Float64Array | None
    Vr: Float64Array | None
    Vpred: Float64Array | None
    ExplainedVar: Float64Array | None
    ExplainedVarRatio: Float64Array | None
    RMS: float
    PRMS: float
    Cost: float
    BestProbeIteration: int | None
    BestProbeRMS: float | None
    ReturnedIteration: int | None
    cv: CovarianceResult
    hp: HyperparameterResult
    RuntimeReport: dict[str, object] | None
