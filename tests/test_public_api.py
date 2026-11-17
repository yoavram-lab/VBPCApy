"""Public API surface tests.

The expected ``__all__`` of every public module is listed here, so any change
to the public surface has to be made deliberately. See docs/api/stability.md.
"""

from __future__ import annotations

import importlib
import warnings
from importlib.metadata import version

import pytest

import vbpca_py
from vbpca_py import _pca_full, results

PUBLIC_MODULES: dict[str, set[str]] = {
    "vbpca_py": {
        "RANK_ZERO_SELECTION_SUPPORTED",
        "VBPCA",
        "AutoEncoder",
        "CVConfig",
        "CovarianceResult",
        "DataReport",
        "EncodedVariable",
        "EncodingSchema",
        "HyperparameterResult",
        "LearningCurve",
        "MissingAwareLogTransformer",
        "MissingAwareMinMaxScaler",
        "MissingAwareOneHotEncoder",
        "MissingAwareOrdinalEncoder",
        "MissingAwarePowerTransformer",
        "MissingAwareSparseOneHotEncoder",
        "MissingAwareStandardScaler",
        "MissingAwareWinsorizer",
        "ObservationMasker",
        "PCAOptions",
        "PCAResult",
        "PosteriorCovariances",
        "PreprocessedFold",
        "ProjectionPosterior",
        "SelectionConfig",
        "check_data",
        "cross_validate_components",
        "cross_validate_raw_components",
        "make_xprobe_mask",
        "prepare_raw_cv_folds",
        "recommend_config",
        "select_n_components",
    },
    "vbpca_py.configuration": {"PCAOptions"},
    "vbpca_py.defaults": {"recommend_config"},
    "vbpca_py.encoding": {"EncodedVariable", "EncodingSchema"},
    "vbpca_py.estimators": {"VBPCA"},
    "vbpca_py.legacy": {
        "VBPCA",
        "CVConfig",
        "SelectionConfig",
        "cross_validate_components",
        "pca_full",
        "select_n_components",
    },
    "vbpca_py.masking": {"ObservationMasker"},
    "vbpca_py.model_selection": {
        "RANK_ZERO_SELECTION_SUPPORTED",
        "CVConfig",
        "SelectionConfig",
        "cross_validate_components",
        "select_n_components",
    },
    "vbpca_py.options": {
        "AccessorMode",
        "Algorithm",
        "BiasUpdateOrder",
        "CompatMode",
        "CostDiagnostics",
        "CovarianceWriteback",
        "Criterion",
        "ExplainedVarianceSolver",
        "RuntimeTuning",
        "StatisticsMode",
        "VarianceUpdateOrder",
    },
    "vbpca_py.plotting": {"loadings_barplot", "scree_plot", "variance_explained_plot"},
    "vbpca_py.preprocessing": {
        "AutoEncoder",
        "DataReport",
        "MissingAwareLogTransformer",
        "MissingAwareMinMaxScaler",
        "MissingAwareOneHotEncoder",
        "MissingAwareOrdinalEncoder",
        "MissingAwarePowerTransformer",
        "MissingAwareSparseOneHotEncoder",
        "MissingAwareStandardScaler",
        "MissingAwareWinsorizer",
        "ObservationMasker",
        "check_data",
    },
    "vbpca_py.raw_model_selection": {
        "PreprocessedFold",
        "cross_validate_raw_components",
        "prepare_raw_cv_folds",
    },
    "vbpca_py.results": {
        "CovarianceResult",
        "Float64Array",
        "HyperparameterResult",
        "IndexArray",
        "LearningCurve",
        "NumericTraceKey",
        "PCAResult",
        "PhaseTimingKey",
        "PosteriorCovariances",
    },
}

DEPRECATED_ROOT_ALIASES = (
    "Float64Array",
    "IndexArray",
    "NumericTraceKey",
    "PhaseTimingKey",
)


@pytest.mark.parametrize("module_name", sorted(PUBLIC_MODULES))
def test_public_module_exports_are_deliberate(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert set(module.__all__) == PUBLIC_MODULES[module_name]
    assert len(module.__all__) == len(set(module.__all__))
    for name in module.__all__:
        assert hasattr(module, name), f"{module_name}.{name} is listed but missing"


def test_rank_zero_capability_flag() -> None:
    assert vbpca_py.RANK_ZERO_SELECTION_SUPPORTED is True


def test_pca_full_is_public_through_legacy() -> None:
    from vbpca_py.legacy import pca_full

    assert pca_full is _pca_full.pca_full


@pytest.mark.parametrize("name", DEPRECATED_ROOT_ALIASES)
def test_root_typing_aliases_are_deprecated(name: str) -> None:
    with pytest.warns(FutureWarning, match=rf"vbpca_py\.{name} .*vbpca_py\.results"):
        alias = getattr(vbpca_py, name)
    assert alias is getattr(results, name)
    assert name not in vbpca_py.__all__


def test_public_root_names_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in vbpca_py.__all__:
            getattr(vbpca_py, name)


def test_unknown_root_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="no attribute 'not_a_name'"):
        _ = vbpca_py.not_a_name


def test_package_version_matches_distribution_metadata() -> None:
    assert isinstance(vbpca_py.__version__, str)
    assert vbpca_py.__version__ == version("vbpca_py")
