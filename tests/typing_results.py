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


# Negative checks also run with warn_unused_ignores: losing the option types
# would make these suppressions unused and fail strict consumer typing.
def check_categorical_options() -> None:
    from vbpca_py.options import Algorithm, CompatMode

    value: Algorithm = Algorithm.parse("vb")
    assert_type(value, Algorithm)
    value = CompatMode.MODERN  # type: ignore[assignment]
    value = "vb"  # type: ignore[assignment]
    Algorithm.NOT_AN_ALGORITHM  # type: ignore[attr-defined]  # noqa: B018


def check_configuration_types(estimator: VBPCA) -> None:
    from vbpca_py import PCAOptions
    from vbpca_py.options import Algorithm

    config = PCAOptions(algorithm=Algorithm.VB, maxiters=3)
    assert_type(config.algorithm, Algorithm)
    assert_type(config.maxiters, int)
    assert_type(config.num_cpu, int | None)
    assert_type(config.to_kwargs(), dict[str, object])
    assert_type(PCAOptions.from_kwargs(maxiters=3), PCAOptions)
    assert_type(estimator.get_config(), PCAOptions)
    assert_type(estimator.get_params(), dict[str, object])
    assert_type(estimator.set_params(maxiters=3), VBPCA)
    assert_type(pca_full(np.ones((2, 3)), 1, options=config), PCAResult)
    PCAOptions(maxiters="three")  # type: ignore[arg-type]
    PCAOptions(maxiter=3)  # type: ignore[call-arg]
    PCAOptions(algorithm="vb")  # type: ignore[arg-type]
    VBPCA(1, maxiter=3)  # type: ignore[call-arg]
    pca_full(np.ones((2, 3)), 1, options={})  # type: ignore[arg-type]
