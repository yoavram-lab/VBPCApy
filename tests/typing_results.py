"""Static contracts checked by the strict type-checking CI step."""

from typing import assert_type

import numpy as np

from vbpca_py import Float64Array, LearningCurve, PCAResult, ProjectionPosterior
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


def check_preprocessor_parameter_types() -> None:
    import scipy.sparse as sp

    from vbpca_py import MissingAwareLogTransformer, MissingAwareStandardScaler
    from vbpca_py.preprocessing import MissingAwareSparseOneHotEncoder

    scaler = MissingAwareStandardScaler()
    assert_type(scaler.get_params(), dict[str, object])
    assert_type(scaler.set_params(), MissingAwareStandardScaler)
    transform = MissingAwareLogTransformer()
    assert_type(transform.get_params(), dict[str, object])
    assert_type(transform.set_params(offset=2.0), MissingAwareLogTransformer)
    MissingAwareLogTransformer(offset="two")  # type: ignore[arg-type]
    MissingAwareStandardScaler(misspelled=True)  # type: ignore[call-arg]
    encoder = MissingAwareSparseOneHotEncoder()
    assert_type(encoder.fit([[0.0], [1.0]]), MissingAwareSparseOneHotEncoder)
    assert_type(encoder.fit_transform([[0.0], [1.0]]), sp.csr_matrix)
    assert_type(encoder.transform(np.ones((2, 1))), sp.csr_matrix)
    assert_type(encoder.inverse_transform(sp.csr_matrix((2, 1))), sp.csr_matrix)


def check_numerical_helper_contracts() -> None:
    import numpy.typing as npt
    import scipy.sparse as sp

    from vbpca_py._categorical_scores import _cell_scores
    from vbpca_py._full_update import CovarianceStore, ScoreState
    from vbpca_py._monitoring import InitResult
    from vbpca_py._noise_statistics import NoiseComplement
    from vbpca_py._pca_full import ModelState, PreparedProblem
    from vbpca_py._types import (
        BoolArray,
        CountArray,
        IndexArray,
        InputArray,
        SparseIndexArray,
    )
    from vbpca_py.encoding import EncodedVariable
    from vbpca_py.errpca_pt import errpca_pt
    from vbpca_py.preprocessing import (
        DataReport,
        MissingAwareOneHotEncoder,
        _check_missing,
        _check_outliers,
        _check_skewness,
        _check_variance,
        _CheckDataConfig,
        _sparse_col_counts,
    )

    diagnostic_values: Float64Array = np.array([0.0, 2.0, 4.0])
    report = DataReport()
    cfg = _CheckDataConfig(2.0, 5.0, 1e-10, 0.5, 3, emit_warnings=False)
    assert_type(_check_missing(diagnostic_values, "0", {}, cfg, report), None)
    assert_type(_check_variance(diagnostic_values, "0", {}, cfg, report), bool)
    assert_type(_check_skewness(diagnostic_values, "0", {}, cfg, report), None)
    assert_type(_check_outliers(diagnostic_values, "0", {}, cfg, report), None)
    assert_type(
        _cell_scores(
            np.eye(2),
            np.eye(2),
            EncodedVariable("categorical", (0, 1), ("a", "b"), offset=(0.0, 0.0)),
            validate=True,
        ),
        tuple[Float64Array, Float64Array, BoolArray] | None,
    )

    def records(
        prepared: PreparedProblem,
        state: ModelState,
        score: ScoreState,
        initial: InitResult,
        plan: NoiseComplement,
    ) -> None:
        assert_type(prepared.ix_obs, IndexArray)
        assert_type(prepared.n_obs_row, CountArray)
        assert_type(state.a, Float64Array)
        assert_type(state.av, CovarianceStore)
        assert_type(score.scores, Float64Array)
        assert_type(initial.a, Float64Array)
        assert_type(plan.rows, npt.NDArray[np.int32])
        integer_values: npt.NDArray[np.int32] = np.ones((2, 1), dtype=np.int32)
        state.a = integer_values  # type: ignore[assignment]
        floating_counts: npt.NDArray[np.float32] = np.ones(2, dtype=np.float32)
        unsigned_counts: npt.NDArray[np.uint64] = np.ones(2, dtype=np.uint64)
        extended_counts: npt.NDArray[np.longdouble] = np.ones(2, dtype=np.longdouble)
        prepared.n_obs_row = floating_counts
        prepared.n_obs_row = unsigned_counts
        prepared.n_obs_row = extended_counts
        complex_counts: npt.NDArray[np.complex128] = np.ones(2, dtype=np.complex128)
        prepared.n_obs_row = complex_counts  # type: ignore[assignment]
        plan.observed_count = 3  # type: ignore[misc]

    del records
    data = np.ones(3, dtype=np.float64)
    indices = np.array([0, 2, 1], dtype=np.int32)
    indptr = np.array([0, 2, 3], dtype=np.int32)
    residual = errpca_pt(data, indices, indptr, np.ones((2, 1)), np.ones((1, 3)))
    assert_type(residual["data"], Float64Array)
    assert_type(residual["indices"], npt.NDArray[np.int32])
    assert_type(residual["shape"], tuple[int, int])
    residual["shape"] = data  # type: ignore[typeddict-item]
    residual["indices"] = data  # type: ignore[typeddict-item]
    assert_type(
        _sparse_col_counts(sp.csc_matrix((2, 3))),
        SparseIndexArray,
    )
    assert_type(
        MissingAwareOneHotEncoder(dtype=np.complex64).transform(np.ones((3, 1))),
        InputArray,
    )
    assert_type(pca_full(np.ones((2, 3), dtype=np.float32), 1), PCAResult)
    assert_type(
        pca_full(
            np.ones((2, 3), dtype=np.int32), 1, mask=np.ones((2, 3), dtype=np.int16)
        ),
        PCAResult,
    )
    observed: BoolArray = np.ones((2, 3), dtype=bool)
    floating_values: Float64Array = np.ones((2, 3), dtype=np.float64)
    observed = floating_values  # type: ignore[assignment]
    assert_type(observed, BoolArray)


def check_native_projection_types(estimator: VBPCA) -> None:
    assert_type(estimator.transform(np.ones((3, 4))), Float64Array)
    assert_type(estimator.inverse_transform(np.ones((3, 1))), Float64Array)
    assert_type(estimator.transform_posterior(np.ones((3, 4))), ProjectionPosterior)
    posterior = estimator.transform_posterior(np.ones((3, 4)))
    assert_type(posterior.mean, Float64Array)
    assert_type(posterior.covariance, Float64Array)
    assert_type(estimator.score_samples(np.ones((3, 4))), Float64Array)
    assert_type(estimator.score(np.ones((3, 4))), float)
    posterior.mean = np.zeros((3, 1))  # type: ignore[misc]


def check_encoding_validation_types() -> None:
    from vbpca_py import EncodedVariable, EncodingSchema

    variable = EncodedVariable("continuous", (0,), offset=(2.0,), weight=0.5)
    schema = EncodingSchema((variable,))
    assert_type(schema.validate(1), None)
    assert_type(schema.validate(1, strict_numeric=True), None)
    assert_type(variable.decode_numeric(np.zeros((1, 2))), Float64Array)
    schema.validate(1, strict_numeric="yes")  # type: ignore[arg-type]
