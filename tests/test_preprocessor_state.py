"""Native construction, fitted-state cleanup, refitting and artifact ownership."""

import copy
import gc
import pickle
import weakref

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import (
    check_get_params_invariance,
    check_no_attributes_set_in_init,
)
from sklearn.utils.validation import check_is_fitted

from vbpca_py import preprocessing as p

CLASSES = (
    p.MissingAwareStandardScaler,
    p.MissingAwareMinMaxScaler,
    p.MissingAwareOneHotEncoder,
    p.MissingAwareSparseOneHotEncoder,
    p.MissingAwareOrdinalEncoder,
    p.AutoEncoder,
    p.MissingAwareLogTransformer,
    p.MissingAwareWinsorizer,
    p.MissingAwarePowerTransformer,
)


def _case(cls):
    x = np.array([[0.0], [1.0], [np.nan], [2.0], [1.0], [2.0]])
    if cls is p.MissingAwareSparseOneHotEncoder:
        rows, columns = np.nonzero(~np.isnan(x))
        x = sp.csr_matrix((x[rows, columns], (rows, columns)), shape=x.shape)
    return cls(), x


def _assert_output(actual, expected):
    if sp.issparse(expected):
        assert sp.issparse(actual)
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual.indptr, expected.indptr)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_allclose(actual.data, expected.data, rtol=2e-12, atol=2e-12)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("cls", CLASSES)
def test_real_constructor_and_parameter_checks(cls):
    model, _ = _case(cls)
    check_no_attributes_set_in_init(cls.__name__, model)
    check_get_params_invariance(cls.__name__, model)
    assert vars(model) == model.get_params(deep=False)
    assert clone(model).get_params() == model.get_params()
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(misspelled=1)


@pytest.mark.parametrize("cls", CLASSES)
def test_all_public_inference_requires_fit(cls):
    model, data = _case(cls)
    for operation in (
        lambda: check_is_fitted(model),
        lambda: model.transform(data),
        lambda: model.inverse_transform(data),
    ):
        with pytest.raises(NotFittedError):
            operation()
    assert not hasattr(model, "n_features_in_")


@pytest.mark.parametrize("cls", CLASSES)
def test_failed_refit_clears_partial_state_releases_old_arrays_and_can_recover(cls):
    model, data = _case(cls)
    original_params = model.get_params(deep=False)
    model.fit(data)
    expected = model.transform(data)
    refs = [
        weakref.ref(value)
        for value in vars(model).values()
        if isinstance(value, np.ndarray | BaseEstimator)
    ]
    invalid_mask = np.zeros((1, 2), dtype=bool)
    if cls is p.MissingAwareLogTransformer:
        model.set_params(offset=0.0)
        invalid_mask = None
    elif cls is p.MissingAwareWinsorizer:
        model.set_params(lower_quantile=0.9, upper_quantile=0.1)
        invalid_mask = None
    with pytest.raises(ValueError):
        model.fit(data, mask=invalid_mask)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert not hasattr(model, "n_features_in_")
    assert not any(name.endswith("_") for name in vars(model))
    gc.collect()
    assert all(ref() is None for ref in refs)
    with pytest.raises(NotFittedError):
        model.transform(data)
    with pytest.raises(NotFittedError):
        model.inverse_transform(expected)
    model.set_params(**original_params).fit(data)
    check_is_fitted(model)
    _assert_output(model.transform(data), expected)


@pytest.mark.parametrize("cls", CLASSES)
def test_refit_matches_fresh_fit_and_pickle_deepcopy(cls):
    model, data = _case(cls)
    model.fit(data)
    changed = data.copy()
    if sp.issparse(changed):
        changed.data *= 2.0
    else:
        changed *= 2.0
    model.fit(changed)
    expected = clone(model).fit_transform(changed)
    _assert_output(model.transform(changed), expected)
    for restored in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        check_is_fitted(restored)
        _assert_output(restored.transform(changed), expected)
        restored.fit(data)
        _assert_output(model.transform(changed), expected)


@pytest.mark.parametrize("cls", CLASSES)
def test_old_unfitted_none_metadata_is_not_mistaken_for_a_fit(cls):
    model, data = _case(cls)
    # Pre-lifecycle artifacts stored this attribute even before fitting.
    model.n_features_in_ = None
    with pytest.raises(NotFittedError):
        check_is_fitted(pickle.loads(pickle.dumps(model)))
    with pytest.raises(NotFittedError):
        model.transform(data)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_mid_fit_cancellation_cleans_state_and_reraises_same_exception(
    monkeypatch, exception_type
):
    model, data = _case(p.MissingAwarePowerTransformer)
    model.fit(data)
    failure = exception_type("cancel synthetic fit")

    def cancel(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr("scipy.optimize.minimize_scalar", cancel)
    with pytest.raises(exception_type) as raised:
        model.fit(data)
    assert raised.value is failure
    assert not hasattr(model, "n_features_in_")
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_constructor_preserves_mutable_parameter_identity_until_fit():
    levels = [[0.0, 1.0]]
    encoder = p.MissingAwareOrdinalEncoder(levels=levels)
    assert encoder.levels is levels
    cloned = clone(encoder)
    assert cloned.levels == levels
    assert cloned.levels is not levels
    encoder.fit(np.array([[0.0], [1.0]]))
    encoder.levels_[0].append(2.0)
    assert levels == [[0.0, 1.0]]


@pytest.mark.parametrize(
    ("cls", "params"),
    [
        (p.MissingAwareLogTransformer, {"offset": 0.0}),
        (
            p.MissingAwareWinsorizer,
            {"lower_quantile": 0.9, "upper_quantile": 0.1},
        ),
    ],
)
def test_invalid_constructor_policy_is_checked_during_fit(cls, params):
    model = cls(**params)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    with pytest.raises(ValueError):
        cloned.fit(np.ones((2, 1)))
    with pytest.raises(NotFittedError):
        check_is_fitted(cloned)
