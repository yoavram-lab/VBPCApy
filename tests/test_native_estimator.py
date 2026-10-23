"""Native axes and inference agree with independent conditional/Gaussian algebra."""

import copy
import inspect
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from vbpca_py import VBPCA
from vbpca_py._pca_full import pca_full
from vbpca_py.legacy import VBPCA as LegacyVBPCA  # noqa: N811 - estimator class alias


def _settings():
    return {
        "maxiters": 4,
        "niter_broadprior": 0,
        "rmsstop": None,
        "minangle": 0.0,
        "verbose": 0,
        "num_cpu": 1,
        "runtime_tuning": "off",
        "random_state": 23,
    }


def _stored(values, pattern=None):
    pattern = np.ones(values.shape, dtype=bool) if pattern is None else pattern
    rows, cols = np.nonzero(pattern)
    return sp.csr_matrix((values[rows, cols], (rows, cols)), shape=values.shape)


@pytest.mark.parametrize("rank", [0, 2])
@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
@pytest.mark.parametrize("kind", ["dense", "missing", "masked", "sparse"])
def test_native_fit_axes_match_direct_low_level_solver(rank, algorithm, kind):
    x = np.random.default_rng(23).normal(size=(11, 5))
    x[0, 0] = 0.0
    mask = None
    if kind == "missing":
        x[::3, 1] = np.nan
    elif kind == "masked":
        mask = np.ones(x.shape, dtype=bool)
        mask[::3, 1] = False
    elif kind == "sparse":
        x = _stored(x)
    before = x.copy()
    model = VBPCA(rank, algorithm=algorithm, **_settings()).fit(x, mask=mask)
    canonical = np.where(mask, x, np.nan) if mask is not None else x
    solver_input = sp.csr_matrix(canonical.T) if sp.issparse(x) else canonical.T
    expected = pca_full(
        solver_input, rank, algorithm=algorithm, compat_mode="modern", **_settings()
    )
    np.testing.assert_allclose(
        model.components_, expected["A"].T, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(model.scores_, expected["S"].T, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        model.mean_, expected["Mu"].ravel(), rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        model.reconstruction_, expected["Xrec"].T, rtol=2e-12, atol=2e-12
    )
    assert model.n_features_in_ == 5
    assert model.inverse_transform(model.scores_).shape == (11, 5)
    np.testing.assert_array_equal(
        x.toarray() if sp.issparse(x) else x,
        before.toarray() if sp.issparse(before) else before,
    )


def _known_model(q=2, covariance=True):
    model = VBPCA(q, **_settings())
    model.components_ = np.array([[1.0, 2.0, -1.0], [0.5, -0.3, 1.2]])[:q]
    model.mean_ = np.array([0.3, -0.2, 0.8])
    model.noise_variance_ = 0.7
    model.n_features_in_ = 3
    model._av = (
        [np.array([[0.2, 0.03], [0.03, 0.4]])[:q, :q] * (j + 1) for j in range(3)]
        if covariance
        else None
    )
    return model


@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("covariance", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_projection_full_second_moments_and_missing_row_prior(q, covariance, sparse):
    model = _known_model(q, covariance)
    x = np.array([[0.0, np.nan, 1.4], [np.nan, np.nan, np.nan], [2.0, 3.0, 1.0]])
    data = _stored(np.nan_to_num(x), ~np.isnan(x)) if sparse else x
    before = copy.deepcopy(vars(model))
    posterior = model.transform_posterior(data)
    for i, row in enumerate(x):
        observed = np.flatnonzero(~np.isnan(row))
        a = model.components_[:, observed].T
        precision = np.eye(q) + a.T @ a / model.noise_variance_
        if model._av is not None:
            for j in observed:
                precision += model._av[j] / model.noise_variance_
        expected_cov = np.linalg.inv(precision)
        expected_mean = (
            expected_cov
            @ a.T
            @ (row[observed] - model.mean_[observed])
            / model.noise_variance_
        )
        np.testing.assert_allclose(posterior.mean[i], expected_mean, atol=2e-14)
        np.testing.assert_allclose(posterior.covariance[i], expected_cov, atol=2e-14)
    np.testing.assert_allclose(model.transform(data), posterior.mean)
    np.testing.assert_array_equal(model.components_, before["components_"])
    assert not np.shares_memory(model.components_, posterior.mean)
    assert not np.shares_memory(model.components_, posterior.covariance)
    restored = pickle.loads(pickle.dumps(posterior))
    np.testing.assert_array_equal(restored.mean, posterior.mean)


@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("sparse", [False, True])
def test_score_matches_independent_observed_gaussian_marginals(q, sparse):
    model = _known_model(q)
    x = np.array([[0.0, np.nan, 1.4], [np.nan, np.nan, np.nan], [2.0, 3.0, 1.0]])
    data = _stored(np.nan_to_num(x), ~np.isnan(x)) if sparse else x
    expected = []
    for row in x:
        observed = np.flatnonzero(~np.isnan(row))
        a = model.components_[:, observed].T
        cov = a @ a.T + model.noise_variance_ * np.eye(observed.size)
        expected.append(
            multivariate_normal.logpdf(row[observed], model.mean_[observed], cov)
            if observed.size
            else 0.0
        )
    np.testing.assert_allclose(model.score_samples(data), expected, atol=2e-13)
    weights = np.array([2.0, 0.0, 1.0])
    assert model.score(data, sample_weight=weights) == pytest.approx(
        np.average(expected, weights=weights)
    )


@pytest.mark.parametrize(
    "weights", [[1, 2], [-1, 0, 1], [0, 0, 0], [1, np.nan, 1], [[1], [2], [3]]]
)
def test_invalid_scoring_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="sample_weight"):
        _known_model().score(np.ones((3, 3)), sample_weight=weights)


def test_constructor_clone_defaults_and_independent_legacy_convention():
    model = VBPCA(2, **_settings())
    assert isinstance(model, (BaseEstimator, TransformerMixin))
    assert set(vars(model)) == set(inspect.signature(VBPCA).parameters)
    assert not hasattr(model, "components_")
    assert model.compat_mode == "modern"
    assert LegacyVBPCA(2).compat_mode == "strict_legacy"
    assert clone(model).get_params() == model.get_params()
    x = np.random.default_rng(1).normal(size=(9, 4))
    modern = model.fit(x)
    old = LegacyVBPCA(2, compat_mode="modern", **_settings()).fit(x.T)
    np.testing.assert_allclose(modern.components_, old.components_.T, atol=2e-12)
    np.testing.assert_allclose(modern.scores_, old.scores_.T, atol=2e-12)


def test_fit_transform_is_conditional_projection_and_pipeline_preserves_samples():
    x = np.random.default_rng(1).normal(size=(9, 4))
    model = VBPCA(2, **_settings())
    z = model.fit_transform(x)
    np.testing.assert_allclose(z, model.transform(x))
    assert z.shape == (9, 2)
    assert not np.allclose(z, model.scores_)
    pipeline = Pipeline([("pca", clone(model))])
    np.testing.assert_allclose(pipeline.fit_transform(x), z)
    assert pipeline.transform(x[:3]).shape == (3, 2)


@pytest.mark.parametrize("rank", [-1, 1.5, True, 5])
def test_invalid_capacity_rejected(rank):
    with pytest.raises(ValueError, match="n_components"):
        VBPCA(rank, **_settings()).fit(np.ones((7, 4)))


@pytest.mark.parametrize(
    "data",
    [
        np.ones(3),
        np.empty((0, 3)),
        np.empty((3, 0)),
        np.full((3, 4), np.nan),
        np.array([[1.0, np.nan], [2.0, np.nan]]),
    ],
)
def test_invalid_training_data_and_failed_refit_clear_model(data):
    model = VBPCA(1, **_settings()).fit(np.arange(24.0).reshape(8, 3))
    with pytest.raises(ValueError):
        model.fit(data)
    with pytest.raises(NotFittedError):
        model.transform(np.ones((2, 3)))


def test_sparse_explicit_mask_observes_absent_zero_without_epsilon():
    model = _known_model()
    x = sp.csr_matrix([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]])
    mask = _stored(np.ones((2, 3), dtype=bool))
    expected = model.transform(x.toarray())
    np.testing.assert_allclose(model.transform(x, mask=mask), expected, atol=2e-14)
    assert x.nnz == 3


def test_readonly_strided_input_and_masks_do_not_mutate_callers():
    x = np.arange(72.0, dtype=np.float32).reshape(12, 6)[::2, ::2]
    mask = np.ones(x.shape, dtype=bool)
    mask[0, 0] = False
    x.setflags(write=False)
    mask.setflags(write=False)
    before = x.copy()
    model = VBPCA(1, **_settings()).fit(x, mask=mask)
    model.transform(x, mask=mask)
    np.testing.assert_array_equal(x, before)


def test_output_budgets_and_plain_transform_avoid_covariance_bank():
    model = _known_model()
    model.max_dense_bytes = 3 * 2 * 8
    assert model.transform(np.ones((3, 3))).shape == (3, 2)
    with pytest.raises(ValueError, match="max_dense_bytes"):
        model.transform_posterior(np.ones((3, 3)))
    with pytest.raises(ValueError, match="max_dense_bytes"):
        model.inverse_transform(np.ones((3, 2)))


def test_duplicate_sparse_mask_coordinates_do_not_double_values():
    from vbpca_py._validation import observed_matrix

    data = sp.csr_matrix([[2.0, 3.0]])
    duplicate = sp.csr_matrix((np.ones(3, dtype=bool), [0, 0, 1], [0, 3]), shape=(1, 2))
    canonical = sp.csr_matrix([[True, True]])
    normalized = observed_matrix(data, duplicate, None)
    np.testing.assert_array_equal(normalized.toarray(), [[2.0, 3.0]])
    model = _known_model()
    x = sp.csr_matrix([[2.0, 3.0, 0.0], [1.0, 2.0, 0.0]])
    duplicate = sp.csr_matrix(
        (np.ones(8, dtype=bool), [0, 0, 1, 2, 0, 1, 1, 2], [0, 4, 8]), shape=(2, 3)
    )
    canonical = _stored(np.ones((2, 3), dtype=bool))
    np.testing.assert_allclose(
        model.transform(x, mask=duplicate), model.transform(x, mask=canonical)
    )
    left = VBPCA(1, **_settings()).fit(x, mask=duplicate)
    right = VBPCA(1, **_settings()).fit(x, mask=canonical)
    np.testing.assert_allclose(left.components_, right.components_, atol=2e-12)
    np.testing.assert_allclose(left.scores_, right.scores_, atol=2e-12)


def test_automatic_sparse_probe_partitions_stored_zeros_exactly():
    from vbpca_py._validation import make_native_probe

    x = _stored(np.zeros((10, 4)))
    train, probe = make_native_probe(x, 0.1, np.random.default_rng(23))
    assert train.nnz == 36
    assert probe.nnz == 4
    assert x.nnz == 40
    assert not np.any(train.data)
    assert not np.any(probe.data)
    # nonzero() excludes zeros: inspect the storage coordinates themselves.
    train_coo, probe_coo = train.tocoo(), probe.tocoo()
    train_coords = set(zip(train_coo.row, train_coo.col, strict=True))
    probe_coords = set(zip(probe_coo.row, probe_coo.col, strict=True))
    assert train_coords.isdisjoint(probe_coords)
    assert len(train_coords | probe_coords) == 40
    model = VBPCA(0, xprobe_fraction=0.1, **_settings()).fit(x)
    assert model.n_features_in_ == 4
    assert np.isfinite(model.prms_)


def test_failed_fit_is_not_fitted_to_native_validation():
    from sklearn.utils.validation import check_is_fitted

    model = VBPCA(1, **_settings())
    with pytest.raises(ValueError):
        model.fit(np.array([[1.0, np.nan], [2.0, np.nan]]))
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert not hasattr(model, "n_features_in_")


def test_sparse_nested_list_mask_uses_validated_shape():
    model = _known_model()
    x = sp.csr_matrix([[1.0, 0.0, 2.0]])
    np.testing.assert_allclose(
        model.transform(x, mask=[[True, True, True]]), model.transform(x.toarray())
    )


@pytest.mark.parametrize(
    "kind",
    [
        "standard",
        "minmax",
        "auto",
        "log",
        "winsor",
        "power",
        "onehot",
        "ordinal",
        "sparse_onehot",
    ],
)
def test_preprocessing_pipeline_accepts_y_as_targets_not_an_entry_mask(kind):
    from vbpca_py import preprocessing as prep

    numeric = np.random.default_rng(1).uniform(0.1, 2.0, size=(12, 3))
    labels = np.arange(12)
    transformers = {
        "standard": prep.MissingAwareStandardScaler(),
        "minmax": prep.MissingAwareMinMaxScaler(),
        "auto": prep.AutoEncoder(column_types=["continuous"] * 3),
        "log": prep.MissingAwareLogTransformer(),
        "winsor": prep.MissingAwareWinsorizer(),
        "power": prep.MissingAwarePowerTransformer(),
        "onehot": prep.MissingAwareOneHotEncoder(),
        "ordinal": prep.MissingAwareOrdinalEncoder(),
        "sparse_onehot": prep.MissingAwareSparseOneHotEncoder(),
    }
    x = numeric
    if kind in {"onehot", "ordinal", "sparse_onehot"}:
        x = np.tile([[0.0], [1.0], [2.0]], (4, 1))
        if kind == "sparse_onehot":
            x = _stored(x)
    pipe = Pipeline([("pre", transformers[kind]), ("pca", VBPCA(1, **_settings()))])
    assert pipe.fit(x, labels).transform(x).shape == (12, 1)


def test_native_tags_advertise_supported_representations():
    from sklearn.utils import get_tags

    tags = get_tags(VBPCA(1))
    assert tags.input_tags.sparse
    assert tags.input_tags.allow_nan
    assert tags.transformer_tags.preserves_dtype == ["float64"]


def test_loadings_plot_uses_declared_native_and_legacy_axes():
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib import pyplot as plt

    from vbpca_py.plotting import loadings_barplot

    native = _known_model()
    legacy = LegacyVBPCA(2)
    legacy.components_ = native.components_.T
    for model in (native, legacy):
        figure = loadings_barplot(model, component=1)
        np.testing.assert_allclose(
            [patch.get_height() for patch in figure.axes[0].patches],
            native.components_[1],
        )
        plt.close(figure)


@pytest.mark.parametrize("fitted", [False, True])
def test_native_pickle_and_deepcopy_preserve_versioned_layout(fitted):
    model = VBPCA(2, **_settings())
    x = np.random.default_rng(23).normal(size=(8, 3))
    if fitted:
        model.fit(x)
    for restored in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert restored.get_params() == model.get_params()
        assert "_vbpca_layout_version" not in vars(restored)
        if fitted:
            np.testing.assert_allclose(restored.transform(x), model.transform(x))
            np.testing.assert_allclose(restored.scores_, model.scores_)


def test_unmarked_old_estimator_state_is_explicitly_rejected():
    model = object.__new__(VBPCA)
    legacy = LegacyVBPCA(1)
    for state in (vars(legacy), {**vars(legacy), "_vbpca_layout_version": "unknown"}):
        with pytest.raises(ValueError, match=r"artifact layout.*Pre-0.5"):
            model.__setstate__(state)


@pytest.mark.parametrize("data_kind", [sp.csr_matrix, sp.csr_array])
@pytest.mark.parametrize("mask_kind", [sp.csr_matrix, sp.csr_array, np.asarray])
def test_empty_sparse_mask_returns_latent_prior_and_refuses_empty_fit_score(
    data_kind, mask_kind
):
    data = data_kind(np.ones((2, 3)))
    mask = mask_kind(np.zeros((2, 3), dtype=bool))
    model = _known_model()
    result = model.transform_posterior(data, mask=mask)
    np.testing.assert_array_equal(result.mean, np.zeros((2, 2)))
    np.testing.assert_allclose(result.covariance, np.broadcast_to(np.eye(2), (2, 2, 2)))
    with pytest.raises(ValueError, match="no observed"):
        model.score(data, mask=mask)
    with pytest.raises(ValueError, match="no training observations"):
        VBPCA(1, **_settings()).fit(data, mask=mask)


@pytest.mark.parametrize("data_kind", [sp.csr_matrix, sp.csr_array])
@pytest.mark.parametrize("mask_kind", [sp.csr_matrix, sp.csr_array])
def test_mixed_sparse_array_and_matrix_observation_containers(data_kind, mask_kind):
    x = data_kind(np.ones((4, 3)))
    mask = mask_kind(np.ones((4, 3), dtype=bool))
    model = VBPCA(1, **_settings()).fit(x, mask=mask)
    assert model.transform(x, mask=mask).shape == (4, 1)


@pytest.mark.parametrize("rank", [0, 1])
def test_native_estimator_passes_full_sklearn_checks(rank):
    # SciPy reads its Array API setting at import. Use a fresh CPU-only process
    # so the current sklearn NumPy dispatch check executes regardless of the
    # parent test runner's import order. Refuse all skips and expected failures.
    code = (
        "from sklearn.utils.estimator_checks import check_estimator\n"
        "from vbpca_py import VBPCA\n"
        f"results = check_estimator(VBPCA({rank}, **{_settings()!r}))\n"
        "unpassed = [r for r in results if r['status'] != 'passed']\n"
        "assert results and not unpassed, unpassed\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "SCIPY_ARRAY_API": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("sparse", [False, True])
def test_nan_remains_missing_when_explicit_mask_marks_it_true(sparse):
    x = np.random.default_rng(23).normal(size=(8, 3))
    x[0, 0] = np.nan
    mask = np.ones(x.shape, dtype=bool)
    mask[1, 1] = False
    canonical = np.where(mask, x, np.nan)
    observed = _stored(x) if sparse else x
    explicit = VBPCA(1, **_settings()).fit(observed, mask=mask)
    implicit_data = (
        _stored(np.nan_to_num(canonical), ~np.isnan(canonical)) if sparse else canonical
    )
    implicit = VBPCA(1, **_settings()).fit(implicit_data)
    np.testing.assert_allclose(explicit.mean_, implicit.mean_, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        explicit.components_, implicit.components_, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        explicit.scores_, implicit.scores_, rtol=2e-12, atol=2e-12
    )
    assert explicit.noise_variance_ == pytest.approx(implicit.noise_variance_)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_probe_only_rows_are_rejected_before_silent_diagnostic_omission(
    sparse, automatic, rank
):
    x = np.array([[10.0, np.nan], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    probe = np.full(x.shape, np.nan)
    probe[0, 0] = x[0, 0]
    if sparse:
        x = _stored(np.nan_to_num(x), ~np.isnan(x))
        probe = _stored(np.nan_to_num(probe), ~np.isnan(probe))
    model = VBPCA(
        rank,
        **(_settings() | {"random_state": 11}),
        xprobe_fraction=1 / 7 if automatic else 0,
    )
    with pytest.raises(ValueError, match="probe entries must retain"):
        model.fit(x, xprobe=None if automatic else probe)
    assert not hasattr(model, "components_")


def test_fit_diagnostic_budget_rejects_before_observation_copy(monkeypatch):
    from vbpca_py import estimators

    monkeypatch.setattr(
        estimators, "observed_matrix", lambda *_: pytest.fail("unnecessary data copy")
    )
    with pytest.raises(ValueError, match=r"Fit diagnostics.*max_dense_bytes"):
        VBPCA(1, max_dense_bytes=1, **_settings()).fit(np.ones((4, 3)))


def test_projection_record_rejects_wrong_dtype_and_shapes():
    from vbpca_py import ProjectionPosterior

    with pytest.raises(TypeError, match="float64"):
        ProjectionPosterior(np.zeros((3, 2), dtype=np.float32), np.zeros((3, 2, 2)))
    with pytest.raises(ValueError, match="shapes"):
        ProjectionPosterior(np.zeros((3, 2)), np.zeros((3, 3, 3)))


def test_native_guide_examples_execute():
    from pathlib import Path

    docs = Path(__file__).resolve().parents[1] / "docs"
    for relative in (
        "api/migration-05.md",
        "concepts/data-convention.md",
        "getting-started/quickstart.md",
    ):
        guide = docs / relative
        namespace = {}
        for block in guide.read_text().split("```python\n")[1:]:
            exec(compile(block.split("```", 1)[0], str(guide), "exec"), namespace)  # noqa: S102 - execute owned documentation


def test_large_finite_weights_have_stable_scoring_average():
    model = _known_model()
    data = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    assert model.score(data, sample_weight=[1e308, 1e308]) == pytest.approx(
        model.score(data)
    )
    assert model.score(data, sample_weight=[0.0, 1e308]) == pytest.approx(
        model.score(data[1:])
    )


def test_scalar_scoring_budget_excludes_temporary_row_vector():
    model = _known_model()
    model.max_dense_bytes = 0
    assert np.isfinite(model.score(np.ones((2, 3))))
    with pytest.raises(ValueError, match="max_dense_bytes"):
        model.score_samples(np.ones((2, 3)))


def test_named_tables_enforce_feature_names_order_and_refit_reset():
    import pandas as pd

    table = pd.DataFrame(
        np.random.default_rng(23).normal(size=(8, 3)), columns=["a", "b", "c"]
    )
    model = VBPCA(1, **_settings()).fit(table)
    np.testing.assert_array_equal(model.feature_names_in_, table.columns)
    assert model.transform(table.iloc[:2]).shape == (2, 1)
    for reordered in (table[["b", "a", "c"]], table.rename(columns={"a": "other"})):
        with pytest.raises(ValueError, match="feature names"):
            model.transform(reordered)
    with pytest.raises(ValueError, match="feature names"):
        model.fit(table, xprobe=table[["b", "a", "c"]])
    assert not hasattr(model, "components_")
    with pytest.raises(
        ValueError, match=r"duplicate feature names|unique column names"
    ):
        model.fit(table.set_axis(["a", "a", "b"], axis=1))
    model.fit(table.to_numpy())
    assert not hasattr(model, "feature_names_in_")


def test_new_data_feature_count_must_match_fitted_state():
    model = VBPCA(1, **_settings()).fit(np.ones((6, 3)))
    for operation in (model.transform, model.transform_posterior, model.score_samples):
        with pytest.raises(ValueError, match="features"):
            operation(np.ones((2, 2)))


def test_weight_normalization_retains_readonly_caller_values():
    model = _known_model()
    weights = np.array([1e308, 1e308])
    weights.setflags(write=False)
    model.score(np.ones((2, 3)), sample_weight=weights)
    np.testing.assert_array_equal(weights, [1e308, 1e308])
