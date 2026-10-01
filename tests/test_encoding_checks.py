"""Tests for categorical encoding detection in check_data (#250)."""

from __future__ import annotations

import numpy as np

from vbpca_py._encoding_checks import detect_one_hot_blocks
from vbpca_py.preprocessing import MissingAwareOneHotEncoder, check_data


def _codes(n: int = 60) -> np.ndarray:
    rng = np.random.default_rng(0)
    codes = np.column_stack([
        rng.integers(0, 3, n),
        rng.integers(0, 2, n),
        rng.integers(0, 4, n),
    ]).astype(float)
    codes[rng.random(codes.shape) < 0.1] = np.nan
    return codes


def test_detected_blocks_match_the_encoder_groups() -> None:
    codes = _codes()
    encoder = MissingAwareOneHotEncoder(binary="both")
    z = encoder.fit_transform(codes, mask=~np.isnan(codes))

    groups = detect_one_hot_blocks(z, ~np.isnan(z))

    assert encoder.feature_groups_ is not None
    np.testing.assert_array_equal(groups, encoder.feature_groups_)


def test_continuous_columns_stay_singletons() -> None:
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.normal(size=30), rng.normal(size=30)])

    groups = detect_one_hot_blocks(x, np.ones_like(x, dtype=bool))

    np.testing.assert_array_equal(groups, [0, 1])


def test_check_data_suggests_feature_groups_and_warns() -> None:
    codes = _codes()
    z = MissingAwareOneHotEncoder(binary="both").fit_transform(
        codes, mask=~np.isnan(codes)
    )

    report = check_data(z)

    assert report.suggested_feature_groups == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert any("one-hot block" in w for w in report.warnings)


def test_rare_levels_and_single_indicator_binaries_are_flagged() -> None:
    x = np.zeros((100, 4))
    x[:2, 0] = 1.0
    x[2:, 1] = 1.0
    x[::2, 2] = 1.0
    x[:, 3] = np.arange(100) % 5

    report = check_data(x, column_names=["a_rare", "a_common", "flag", "score"])

    assert report.suggested_feature_groups == [0, 0, 1, 2]
    assert any(w.startswith("a_rare: rare level (2 of 100") for w in report.warnings)
    assert any(w.startswith("flag: binary variable") for w in report.warnings)
    assert any(w.startswith("score: 5 integer codes") for w in report.warnings)


def test_no_blocks_leaves_groups_unset() -> None:
    rng = np.random.default_rng(2)

    report = check_data(rng.normal(size=(40, 3)))

    assert report.suggested_feature_groups is None
