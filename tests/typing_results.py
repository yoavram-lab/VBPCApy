"""Static contracts checked by the strict type-checking CI step."""

from typing import assert_type

import numpy as np

from vbpca_py import Float64Array, LearningCurve, PCAResult
from vbpca_py._pca_full import pca_full
from vbpca_py.estimators import VBPCA


def check_result_types(result: PCAResult, estimator: VBPCA) -> None:
    assert_type(pca_full(np.ones((2, 3)), 1), PCAResult)
    assert_type(result["A"], Float64Array)
    assert_type(result["Xrec"], Float64Array | None)
    assert_type(result["V"], float)
    assert_type(result["lc"], LearningCurve)
    assert_type(result["lc"].get("returned_iteration"), int | None)
    assert_type(result["lc"].get("best_probe_rms"), float | None)
    assert_type(result["lc"].get("rms"), list[float] | None)
    assert_type(result["lc"].get("prior_trace"), list[list[float]] | None)
    assert_type(estimator.learning_curve_, LearningCurve | None)
