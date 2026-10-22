"""Core package for the variational Bayesian PCA implementation."""

from importlib.metadata import version

from vbpca_py._missing import make_xprobe_mask
from vbpca_py.configuration import PCAOptions
from vbpca_py.defaults import recommend_config
from vbpca_py.encoding import EncodedVariable, EncodingSchema
from vbpca_py.estimators import VBPCA
from vbpca_py.model_selection import (
    RANK_ZERO_SELECTION_SUPPORTED,
    CVConfig,
    SelectionConfig,
    cross_validate_components,
    select_n_components,
)
from vbpca_py.preprocessing import (
    AutoEncoder,
    DataReport,
    MissingAwareLogTransformer,
    MissingAwareMinMaxScaler,
    MissingAwareOneHotEncoder,
    MissingAwareOrdinalEncoder,
    MissingAwarePowerTransformer,
    MissingAwareSparseOneHotEncoder,
    MissingAwareStandardScaler,
    MissingAwareWinsorizer,
    check_data,
)
from vbpca_py.raw_model_selection import (
    PreprocessedFold,
    cross_validate_raw_components,
    prepare_raw_cv_folds,
)
from vbpca_py.results import (
    CovarianceResult,
    Float64Array,
    HyperparameterResult,
    IndexArray,
    LearningCurve,
    NumericTraceKey,
    PCAResult,
    PhaseTimingKey,
    PosteriorCovariances,
)

__all__ = [
    "RANK_ZERO_SELECTION_SUPPORTED",
    "VBPCA",
    "AutoEncoder",
    "CVConfig",
    "CovarianceResult",
    "DataReport",
    "EncodedVariable",
    "EncodingSchema",
    "Float64Array",
    "HyperparameterResult",
    "IndexArray",
    "LearningCurve",
    "MissingAwareLogTransformer",
    "MissingAwareMinMaxScaler",
    "MissingAwareOneHotEncoder",
    "MissingAwareOrdinalEncoder",
    "MissingAwarePowerTransformer",
    "MissingAwareSparseOneHotEncoder",
    "MissingAwareStandardScaler",
    "MissingAwareWinsorizer",
    "NumericTraceKey",
    "PCAOptions",
    "PCAResult",
    "PhaseTimingKey",
    "PosteriorCovariances",
    "PreprocessedFold",
    "SelectionConfig",
    "check_data",
    "cross_validate_components",
    "cross_validate_raw_components",
    "make_xprobe_mask",
    "prepare_raw_cv_folds",
    "recommend_config",
    "select_n_components",
]

__version__ = version("vbpca_py")
