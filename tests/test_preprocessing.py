"""Tests for missing-aware preprocessing utilities."""

import numpy as np
import pytest

from vbpca_py.preprocessing import (
    AutoEncoder,
    MissingAwareMinMaxScaler,
    MissingAwareOneHotEncoder,
    MissingAwareStandardScaler,
)


def test_missing_aware_one_hot_preserves_missing_and_inverse() -> None:
    x = np.array(
        [
            ["a", "x"],
            ["b", "y"],
            [np.nan, "x"],
        ],
        dtype=object,
    )
    enc = MissingAwareOneHotEncoder(mean_center=True)
    z = enc.fit_transform(x)
    # Missing row should be all NaN for the first column's block.
    assert np.isnan(z[2, 0:2]).all()
    x_inv = enc.inverse_transform(z)
    assert x_inv.shape == x.shape
    assert x_inv[0, 0] == "a"
    assert x_inv[1, 0] == "b"
    assert np.isnan(x_inv[2, 0])


def test_missing_aware_standard_scaler_round_trip_with_missing() -> None:
    x = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
    mask = ~np.isnan(x)
    scaler = MissingAwareStandardScaler()
    z = scaler.fit_transform(x, mask=mask)
    assert np.isnan(z[1, 0])
    x_back = scaler.inverse_transform(z, mask=mask)
    np.testing.assert_allclose(x_back[mask], x[mask])


def test_missing_aware_minmax_scaler_round_trip() -> None:
    x = np.array([[0.0], [5.0], [np.nan]])
    scaler = MissingAwareMinMaxScaler()
    z = scaler.fit_transform(x)
    assert 0.0 <= z[0, 0] <= 1.0
    x_back = scaler.inverse_transform(z)
    np.testing.assert_allclose(x_back[0:2, 0], x[0:2, 0])


def test_autoencoder_mixed_schema_and_inverse() -> None:
    x = np.array(
        [
            ["red", 1.0, 10.0],
            ["blue", 2.0, 20.0],
            [np.nan, np.nan, 30.0],
        ],
        dtype=object,
    )
    mask = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    )
    auto = AutoEncoder(cardinality_threshold=5, continuous_scaler="standard")
    z = auto.fit_transform(x, mask=mask)
    x_inv = auto.inverse_transform(z)
    assert x_inv.shape == x.shape
    # Observed categorical values round-trip
    assert x_inv[0, 0] == "red"
    assert x_inv[1, 0] == "blue"
    # Missing entries remain NaN
    assert str(x_inv[2, 0]) == "nan"
    assert str(x_inv[2, 1]) == "nan"


def test_one_hot_encoder_unknown_category_raise() -> None:
    x_train = np.array([["a"], ["b"]], dtype=object)
    x_test = np.array([["c"]], dtype=object)

    enc = MissingAwareOneHotEncoder(handle_unknown="raise")
    enc.fit(x_train)
    with pytest.raises(ValueError, match="Unknown category"):
        enc.transform(x_test)


def test_one_hot_encoder_unknown_category_ignore() -> None:
    x_train = np.array([["a"], ["b"]], dtype=object)
    x_test = np.array([["c"]], dtype=object)

    enc = MissingAwareOneHotEncoder(handle_unknown="ignore")
    z = enc.fit_transform(x_train)
    assert z.shape == (2, 2)

    z_test = enc.transform(x_test)
    assert z_test.shape == (1, 2)
    assert np.allclose(z_test, 0.0)


def test_autoencoder_column_types_override() -> None:
    x = np.array(
        [
            [1, 10.0],
            [2, 20.0],
            [3, 30.0],
        ],
        dtype=object,
    )

    auto = AutoEncoder(
        column_types=["categorical", "continuous"],
        cardinality_threshold=1,
        continuous_scaler="minmax",
    )
    z = auto.fit_transform(x)
    x_back = auto.inverse_transform(z)

    assert z.shape[1] >= 2
    assert x_back.shape == x.shape
    assert x_back[0, 0] == 1


def test_autoencoder_transform_before_fit_raises() -> None:
    auto = AutoEncoder()
    with pytest.raises(RuntimeError, match="not fitted"):
        auto.transform(np.array([[1.0], [2.0]]))


def test_autoencoder_inverse_transform_before_fit_raises() -> None:
    auto = AutoEncoder()
    with pytest.raises(RuntimeError, match="not fitted"):
        auto.inverse_transform(np.array([[0.0], [1.0]]))
