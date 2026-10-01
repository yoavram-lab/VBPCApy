"""Tests for encoder variable groups and binary coding (#250)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py.preprocessing import (
    AutoEncoder,
    MissingAwareOneHotEncoder,
    MissingAwareSparseOneHotEncoder,
)

X = np.array(
    [[0.0, 1.0], [1.0, 2.0], [np.nan, 3.0], [1.0, np.nan], [0.0, 2.0]],
)


def test_single_binary_coding_reports_one_column_per_binary_variable() -> None:
    encoder = MissingAwareOneHotEncoder().fit(X)

    assert encoder.feature_groups_ is not None
    np.testing.assert_array_equal(encoder.feature_groups_, [0, 1, 1, 1])
    assert encoder.feature_kinds_ == ["categorical", "categorical"]


def test_both_binary_coding_one_hot_encodes_every_level() -> None:
    encoder = MissingAwareOneHotEncoder(binary="both")

    z = encoder.fit_transform(X)

    assert encoder.feature_groups_ is not None
    np.testing.assert_array_equal(encoder.feature_groups_, [0, 0, 1, 1, 1])
    np.testing.assert_allclose(z[[0, 1, 3, 4]][:, :2].sum(axis=1), 1.0)
    assert np.isnan(z[2, :2]).all()
    decoded = encoder.inverse_transform(z)
    assert decoded[1, 0] == pytest.approx(1.0)
    assert decoded[4, 0] == pytest.approx(0.0)


def test_invalid_binary_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="binary must be"):
        MissingAwareOneHotEncoder(binary="pairs").fit(X)  # type: ignore[arg-type]


def test_binary_is_a_constructor_parameter() -> None:
    assert MissingAwareOneHotEncoder(binary="both").get_params()["binary"] == "both"


def test_auto_encoder_reports_groups_and_kinds_for_mixed_columns() -> None:
    rng = np.random.default_rng(0)
    mixed = np.column_stack([
        rng.normal(size=30),
        rng.integers(0, 3, size=30).astype(float),
        rng.integers(0, 2, size=30).astype(float),
    ])

    encoder = AutoEncoder(
        column_types=["continuous", "categorical", "categorical"], binary="both"
    ).fit(mixed)

    assert encoder.feature_groups_ is not None
    np.testing.assert_array_equal(encoder.feature_groups_, [0, 1, 1, 1, 2, 2])
    assert encoder.feature_kinds_ == ["continuous", "categorical", "categorical"]
    assert encoder.transform(mixed).shape[1] == len(encoder.feature_groups_)


def test_sparse_encoder_groups_cover_its_columns() -> None:
    column = sp.csr_matrix(np.array([[1.0], [0.0], [2.0], [3.0]]))

    encoder = MissingAwareSparseOneHotEncoder().fit(column)

    assert encoder.feature_groups_ is not None
    assert encoder.feature_groups_.shape == (len(encoder.feature_names_out_),)
    assert set(encoder.feature_groups_.tolist()) == {0}
