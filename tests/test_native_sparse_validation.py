"""Native sparse observations are validated before duplicate aggregation."""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError

from vbpca_py import VBPCA, CVConfig, SelectionConfig, estimators
from vbpca_py import cross_validate_components as cross_validate
from vbpca_py import select_n_components as select

SPARSE_TYPES = (
    sp.csr_matrix,
    sp.csc_matrix,
    sp.coo_matrix,
    sp.csr_array,
    sp.csc_array,
    sp.coo_array,
)
OPTIONS = {
    "maxiters": 3,
    "niter_broadprior": 0,
    "random_state": 346,
    "runtime_tuning": "off",
    "num_cpu": 1,
    "verbose": 0,
}


def _raw(factory, duplicates):
    dense = np.random.default_rng(346).normal(size=(8, 3))
    dense[0, 1], dense[2, 1] = 0, np.nan
    rows, cols = np.indices(dense.shape)
    keep = ((rows != 0) | (cols != 0)) & ((rows != 1) | (cols != 0))
    values = np.r_[dense[keep], duplicates]
    rows = np.r_[rows[keep], np.zeros(len(duplicates), dtype=int)]
    cols = np.r_[cols[keep], np.zeros(len(duplicates), dtype=int)]
    if factory in {sp.coo_matrix, sp.coo_array}:
        raw = factory((values, (rows, cols)), shape=dense.shape)
    else:
        csc = factory in {sp.csc_matrix, sp.csc_array}
        major, minor = (cols, rows) if csc else (rows, cols)
        order = np.lexsort((minor, major))
        size = dense.shape[1] if csc else dense.shape[0]
        pointer = np.r_[0, np.cumsum(np.bincount(major, minlength=size))]
        raw = factory((values[order], minor[order], pointer), shape=dense.shape)
    return raw, dense


def _freeze(raw):
    fields = (
        ["data", "row", "col"] if raw.format == "coo" else ["data", "indices", "indptr"]
    )
    before = {name: getattr(raw, name).copy() for name in fields}
    for name in fields:
        getattr(raw, name).flags.writeable = False
    return before


def _assert_unchanged(raw, before):
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(raw, name), values)


def _call(method, model, x, mask=None):
    if method == "select":
        return select(x, mask=mask, components=[1], **OPTIONS)
    if method == "cv":
        return cross_validate(
            x, mask=mask, components=[1], config=CVConfig(n_splits=2), **OPTIONS
        )
    return getattr(model, method)(x, mask=mask)


@pytest.mark.parametrize("factory", SPARSE_TYPES)
@pytest.mark.parametrize("values", [[np.inf], [-np.inf], [np.inf, -np.inf]])
@pytest.mark.parametrize("explicit_mask", [False, True])
@pytest.mark.parametrize(
    "method",
    [
        "fit",
        "fit_transform",
        "transform",
        "transform_posterior",
        "score_samples",
        "score",
        "select",
        "cv",
    ],
)
def test_raw_observed_infinity_is_rejected_before_aggregation(
    factory, values, explicit_mask, method
):
    raw, dense = _raw(factory, values)
    before = _freeze(raw)
    model = VBPCA(1, **OPTIONS).fit(dense)
    components = model.components_.copy()
    mask = factory(np.ones(raw.shape, dtype=bool)) if explicit_mask else None
    with pytest.raises(ValueError, match="infinity"):
        _call(method, model, raw, mask)
    if method in {"fit", "fit_transform"}:
        with pytest.raises(NotFittedError):
            model.transform(dense)
    else:
        np.testing.assert_array_equal(model.components_, components)
    _assert_unchanged(raw, before)


@pytest.mark.parametrize("factory", SPARSE_TYPES)
@pytest.mark.parametrize("method", ["fit", "select"])
def test_explicit_probe_raw_infinity_is_rejected(factory, method):
    probe, dense = _raw(factory, [np.inf, -np.inf])
    before = _freeze(probe)
    rows, cols = np.nonzero(np.isfinite(dense))
    training = factory((dense[rows, cols], (rows, cols)), shape=dense.shape)
    call = VBPCA(1, **OPTIONS).fit if method == "fit" else select
    options = {} if method == "fit" else {"components": [1], **OPTIONS}
    with pytest.raises(ValueError, match="infinity"):
        call(training, xprobe=probe, **options)
    _assert_unchanged(probe, before)


@pytest.mark.parametrize("factory", SPARSE_TYPES)
@pytest.mark.parametrize("mask_kind", ["dense", "sparse", "numeric"])
@pytest.mark.parametrize("values", [[np.inf, -np.inf], [2.0, -2.0], [np.nan]])
def test_exclusions_preserve_missingness_and_requested_stored_zeros(
    factory, mask_kind, values, monkeypatch
):
    raw, expected = _raw(factory, values)
    before = _freeze(raw)
    observed = np.ones(raw.shape, dtype=bool)
    observed[0, 0] = False
    # A true mask requests an observed zero at this absent coordinate.
    expected[1, 0], expected[0, 0] = 0, np.nan
    rows, cols = np.nonzero(np.isfinite(expected))
    reference = sp.csr_matrix(
        (expected[rows, cols], (rows, cols)), shape=expected.shape
    )
    mask = (
        factory(observed)
        if mask_kind == "sparse"
        else observed.astype(np.uint8)
        if mask_kind == "numeric"
        else observed
    )
    mask_before = mask.copy()

    def forbidden(*args, **kwargs):
        pytest.fail("raw sparse validation must not densify observations or masks")

    for cls in SPARSE_TYPES:
        monkeypatch.setattr(cls, "toarray", forbidden)
        monkeypatch.setattr(cls, "todense", forbidden)
    actual = VBPCA(1, **OPTIONS).fit(raw, mask=mask)
    expected_model = VBPCA(1, **OPTIONS).fit(reference)
    np.testing.assert_array_equal(actual.components_, expected_model.components_)
    np.testing.assert_array_equal(actual.scores_, expected_model.scores_)
    np.testing.assert_array_equal(actual.mean_, expected_model.mean_)
    np.testing.assert_allclose(
        actual.transform(raw, mask=mask), expected_model.transform(reference)
    )
    np.testing.assert_allclose(
        actual.score_samples(raw, mask=mask), expected_model.score_samples(reference)
    )
    config = SelectionConfig(metric="rms", return_best_model=True)
    selected = select(raw, mask=mask, components=[1], config=config, **OPTIONS)[-1]
    np.testing.assert_array_equal(selected.components_, actual.components_)
    _assert_unchanged(raw, before)
    if sp.issparse(mask):
        np.testing.assert_array_equal(mask.data, mask_before.data)
    else:
        np.testing.assert_array_equal(mask, mask_before)


@pytest.mark.parametrize("factory", SPARSE_TYPES)
@pytest.mark.parametrize("where", ["data", "mask", "probe"])
def test_invalid_raw_sparse_indices_fail_before_sklearn_conversion(
    factory, where, monkeypatch
):
    malformed, dense = _raw(factory, [0.0])
    if malformed.format == "coo":
        malformed.row[0] = dense.shape[0]
    else:
        malformed.indices[0] = max(dense.shape)
    validate = estimators.validate_data

    def forbidden(estimator, values, **kwargs):
        if values is malformed:
            pytest.fail("malformed sparse input reached sklearn conversion")
        return validate(estimator, values, **kwargs)

    monkeypatch.setattr(estimators, "validate_data", forbidden)
    model = VBPCA(1, **OPTIONS)
    data = malformed if where == "data" else dense
    options = (
        {"mask": malformed}
        if where == "mask"
        else {"xprobe": malformed}
        if where == "probe"
        else {}
    )
    with pytest.raises(ValueError, match="Invalid sparse"):
        model.fit(data, **options)
    assert not hasattr(model, "n_features_in_")


@pytest.mark.parametrize(
    "factory",
    [
        sp.bsr_matrix,
        sp.bsr_array,
        sp.dia_matrix,
        sp.dia_array,
        sp.lil_matrix,
        sp.lil_array,
        sp.dok_matrix,
        sp.dok_array,
    ],
)
def test_other_sklearn_sparse_formats_keep_finite_support_and_refuse_infinity(factory):
    dense = np.random.default_rng(346).normal(size=(8, 3))
    raw = factory(dense)
    actual = VBPCA(1, **OPTIONS).fit(raw)
    reference = VBPCA(1, **OPTIONS).fit(dense)
    np.testing.assert_allclose(actual.components_, reference.components_)
    dense[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        actual.fit(factory(dense))


@pytest.mark.parametrize("factory", SPARSE_TYPES)
@pytest.mark.parametrize("selected", [False, True])
def test_duplicate_readonly_mask_entries_are_boolean_union_without_mutation(
    factory, selected
):
    raw, dense = _raw(factory, [np.inf, -np.inf])
    mask, _ = _raw(factory, [False, selected])
    # Keep raw mask duplicates; copying via COO/CSR conversion can aggregate.
    mask.data = mask.data.astype(bool)
    before = _freeze(mask)
    model = VBPCA(1, **OPTIONS).fit(dense)
    if selected:
        with pytest.raises(ValueError, match="infinity"):
            model.transform(raw, mask=mask)
    else:
        model.transform(raw, mask=mask)
    _assert_unchanged(mask, before)
