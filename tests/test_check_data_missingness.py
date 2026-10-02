"""Tests for missingness, scale and MCAR diagnostics in check_data (#249)."""

from __future__ import annotations

import numpy as np
import pytest

from vbpca_py.preprocessing import check_data


def _standard(n: int = 200, p: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, p))


def test_missingness_summary_counts_rows_and_patterns() -> None:
    x = _standard(6, 3)
    mask = np.ones(x.shape, dtype=bool)
    mask[0, :2] = False
    mask[1, 0] = False
    mask[2, 0] = False

    report = check_data(x, mask=mask, missing_fraction_warn=0.5)

    assert report.missingness == {
        "fraction_missing": pytest.approx(4 / 18),
        "complete_rows": 3,
        "n_patterns": 3,
        "max_row_missing_fraction": pytest.approx(2 / 3),
        "rows_above_threshold": 1,
    }
    assert any("1 of 6 rows" in message for message in report.warnings)


def test_unit_scale_data_raise_no_scale_warning() -> None:
    report = check_data(_standard())

    assert not any("scale" in message for message in report.warnings)
    assert report.passed


def test_unequal_feature_scales_are_flagged() -> None:
    x = _standard() * np.array([1.0, 1.0, 1.0, 50.0])

    report = check_data(x)

    assert any("differ by a factor of" in message for message in report.warnings)


def test_tiny_absolute_scale_is_flagged() -> None:
    report = check_data(_standard() * 0.025)

    assert any("roughly unit-scale" in message for message in report.warnings)


def test_indicator_columns_are_left_out_of_the_scale_check() -> None:
    rng = np.random.default_rng(1)
    levels = rng.integers(0, 3, 200)
    one_hot = np.eye(3)[levels]
    binary = rng.integers(0, 2, (200, 1)).astype(float)
    x = np.column_stack([one_hot, binary, _standard(200, 2) * 3.0])

    report = check_data(x)

    assert not any("scale" in message for message in report.warnings)


def test_one_hot_message_points_to_the_reference_drop() -> None:
    levels = np.random.default_rng(2).integers(0, 3, 100)
    report = check_data(np.eye(3)[levels])

    assert any("drop='first'" in message for message in report.warnings)


def test_mcar_screen_is_off_by_default() -> None:
    report = check_data(_standard())

    assert "mcar_pairs_tested" not in report.missingness


def test_mcar_screen_flags_value_dependent_missingness() -> None:
    x = _standard(400, 3, seed=3)
    mask = np.ones(x.shape, dtype=bool)
    mask[:, 0] = x[:, 1] < 0.3

    report = check_data(x, mask=mask, mcar_test=True)

    assert report.missingness["mcar_pairs_flagged"] >= 1
    assert any("not completely at random" in message for message in report.warnings)


def test_mcar_screen_passes_random_missingness() -> None:
    rng = np.random.default_rng(4)
    x = _standard(400, 3, seed=5)
    mask = rng.random(x.shape) > 0.2

    report = check_data(x, mask=mask, mcar_test=True)

    assert report.missingness["mcar_pairs_tested"] == 6
    assert report.missingness["mcar_pairs_flagged"] == 0
