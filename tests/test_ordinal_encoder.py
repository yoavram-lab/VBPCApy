"""Tests for the ordinal column type (#249)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py import AutoEncoder, MissingAwareOrdinalEncoder

GRADES = np.array(
    [["low"], ["high"], ["mid"], ["high"], ["low"], ["mid"]], dtype=object
)
ORDER = [["low", "mid", "high"]]


def test_levels_map_to_equally_spaced_standardized_scores() -> None:
    encoder = MissingAwareOrdinalEncoder(levels=ORDER)

    z = encoder.fit_transform(GRADES)

    expected = np.array([0.0, 2.0, 1.0, 2.0, 0.0, 1.0])
    expected = (expected - expected.mean()) / expected.std()
    np.testing.assert_allclose(z.ravel(), expected)
    assert encoder.feature_kinds_ == ["ordinal"]
    assert encoder.feature_groups_.tolist() == [0]


def test_default_order_sorts_the_observed_values() -> None:
    codes = np.array([[3.0], [1.0], [2.0], [3.0]])

    encoder = MissingAwareOrdinalEncoder().fit(codes)

    assert encoder.levels_ == [[1.0, 2.0, 3.0]]


def test_minmax_scaler_maps_the_extremes_to_zero_and_one() -> None:
    z = MissingAwareOrdinalEncoder(levels=ORDER, scaler="minmax").fit_transform(GRADES)

    np.testing.assert_allclose(z.ravel(), [0.0, 1.0, 0.5, 1.0, 0.0, 0.5])


def test_missing_entries_stay_missing_and_round_trip() -> None:
    mask = np.array([[True], [True], [False], [True], [True], [True]])
    encoder = MissingAwareOrdinalEncoder(levels=ORDER)

    z = encoder.fit_transform(GRADES, mask=mask)
    back = encoder.inverse_transform(z)

    assert np.isnan(z[2, 0])
    assert back[mask].tolist() == GRADES[mask].tolist()


def test_inverse_rounds_to_the_nearest_level_and_clips() -> None:
    encoder = MissingAwareOrdinalEncoder(levels=ORDER, scaler="minmax").fit(GRADES)

    back = encoder.inverse_transform(np.array([[0.2], [0.3], [0.8], [1.7], [-0.4]]))

    assert back.ravel().tolist() == ["low", "mid", "high", "high", "low"]


def test_unknown_levels_raise_or_become_missing() -> None:
    encoder = MissingAwareOrdinalEncoder(levels=ORDER).fit(GRADES)
    unseen = np.array([["mid"], ["top"]], dtype=object)

    with pytest.raises(ValueError, match="Unknown level"):
        encoder.transform(unseen)
    ignoring = MissingAwareOrdinalEncoder(levels=ORDER, handle_unknown="ignore")
    z = ignoring.fit(GRADES).transform(unseen)
    assert np.isfinite(z[0, 0])
    assert np.isnan(z[1, 0])


@pytest.mark.parametrize(
    ("options", "message"),
    [({"scaler": "robust"}, "scaler"), ({"levels": [["a"], ["b"]]}, "levels")],
)
def test_invalid_options_are_rejected(options: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MissingAwareOrdinalEncoder(**options).fit(GRADES)


def test_auto_encoder_routes_ordinal_columns() -> None:
    x = np.column_stack([
        np.array([1, 3, 2, 3, 1, 2], dtype=object),
        np.array(["a", "b", "a", "c", "b", "c"], dtype=object),
    ])
    encoder = AutoEncoder(column_types=["ordinal", "categorical"], binary="both")

    z = encoder.fit_transform(x)

    assert z.shape == (6, 1 + 3)
    assert encoder.feature_kinds_ == ["ordinal", "categorical"]
    assert encoder.feature_groups_.tolist() == [0, 1, 1, 1]
    assert encoder.inverse_transform(z).tolist() == x.tolist()


def test_auto_encoder_rejects_ordinal_sparse_columns() -> None:
    x = sp.csr_matrix(np.array([[1.0], [2.0], [3.0]]))

    with pytest.raises(ValueError, match="ordinal"):
        AutoEncoder(column_types=["ordinal"]).fit(x)
