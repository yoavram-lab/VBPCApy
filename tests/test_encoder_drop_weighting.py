"""Tests for reference-level drop and block weighting in one-hot encoders (#249)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vbpca_py.preprocessing import AutoEncoder, MissingAwareOneHotEncoder

X = np.array(
    [
        [0.0, 1.0, 5.0],
        [1.0, 2.0, 6.0],
        [2.0, 0.0, 5.0],
        [1.0, 1.0, 6.0],
        [0.0, 2.0, 5.0],
    ],
)
MISSING = np.array(
    [
        [True, True, True],
        [True, False, True],
        [True, True, True],
        [False, True, True],
        [True, True, False],
    ],
)


def _block_variance(z: np.ndarray, groups: np.ndarray, j: int) -> float:
    return float(np.nansum(np.nanvar(z[:, groups == j], axis=0)))


def test_drop_first_removes_one_indicator_per_variable() -> None:
    encoder = MissingAwareOneHotEncoder(drop="first", binary="both")

    z = encoder.fit_transform(X)

    assert z.shape == (5, 2 + 2 + 1)
    assert encoder.feature_groups_.tolist() == [0, 0, 1, 1, 2]
    assert encoder.feature_names_out_ == [
        "col0_1.0",
        "col0_2.0",
        "col1_2.0",
        "col1_0.0",
        "col2_6.0",
    ]
    assert encoder.categories_[1] == [1.0, 2.0, 0.0]


def test_drop_first_breaks_the_sum_to_one_dependence() -> None:
    codes = np.random.default_rng(0).integers(0, 3, size=(40, 2)).astype(float)
    full = MissingAwareOneHotEncoder().fit_transform(codes)
    dropped = MissingAwareOneHotEncoder(drop="first").fit_transform(codes)

    with_intercept = np.column_stack([np.ones(40), full])
    assert np.linalg.matrix_rank(with_intercept) == with_intercept.shape[1] - 2
    with_intercept = np.column_stack([np.ones(40), dropped])
    assert np.linalg.matrix_rank(with_intercept) == with_intercept.shape[1]


def test_drop_matches_single_coding_for_binary_variables() -> None:
    single = MissingAwareOneHotEncoder(binary="single").fit_transform(X[:, [2]])
    dropped = MissingAwareOneHotEncoder(binary="both", drop="first").fit_transform(
        X[:, [2]]
    )

    np.testing.assert_array_equal(single, dropped)


@pytest.mark.parametrize("drop", [None, "first"])
def test_equal_variance_gives_every_block_unit_variance(drop: Any) -> None:
    encoder = MissingAwareOneHotEncoder(
        binary="both", drop=drop, block_weighting="equal_variance"
    )

    z = encoder.fit_transform(X, mask=MISSING)

    assert encoder.block_weights_ is not None
    for j in range(3):
        assert _block_variance(z, encoder.feature_groups_, j) == pytest.approx(1.0)


def test_unweighted_blocks_keep_their_raw_variance() -> None:
    encoder = MissingAwareOneHotEncoder(binary="both")

    z = encoder.fit_transform(X)

    np.testing.assert_array_equal(encoder.block_weights_, np.ones(3))
    assert _block_variance(z, encoder.feature_groups_, 0) == pytest.approx(
        0.24 + 0.24 + 0.16
    )


@pytest.mark.parametrize(
    "options",
    [
        {"drop": "first"},
        {"block_weighting": "equal_variance"},
        {"drop": "first", "block_weighting": "equal_variance", "mean_center": True},
        {"binary": "both", "drop": "first", "mean_center": True},
    ],
)
def test_inverse_transform_round_trips(options: dict[str, Any]) -> None:
    encoder = MissingAwareOneHotEncoder(**options)

    z = encoder.fit_transform(X, mask=MISSING)
    back = encoder.inverse_transform(z, mask=MISSING)

    observed = back[MISSING].astype(float)
    np.testing.assert_array_equal(observed, X[MISSING])


@pytest.mark.parametrize(
    ("options", "message"),
    [({"drop": "last"}, "drop"), ({"block_weighting": "mca"}, "block_weighting")],
)
def test_invalid_options_are_rejected(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MissingAwareOneHotEncoder(**options).fit(X)


def test_auto_encoder_passes_drop_and_weighting_through() -> None:
    encoder = AutoEncoder(
        column_types=["categorical", "categorical", "categorical"],
        binary="both",
        drop="first",
        block_weighting="equal_variance",
    )

    z = encoder.fit_transform(X)

    assert z.shape == (5, 5)
    assert encoder.feature_groups_.tolist() == [0, 0, 1, 1, 2]
    for j in range(3):
        assert _block_variance(z, encoder.feature_groups_, j) == pytest.approx(1.0)
