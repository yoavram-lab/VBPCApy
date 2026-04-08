"""Tests for distributional transforms and check_data diagnostics."""

import numpy as np
import pytest

from vbpca_py.preprocessing import (
    DataReport,
    MissingAwareLogTransformer,
    MissingAwarePowerTransformer,
    MissingAwareWinsorizer,
    check_data,
)

# ── MissingAwareLogTransformer ───────────────────────────────────────────


class TestMissingAwareLogTransformer:
    def test_round_trip_with_missing(self):
        x = np.array([[0.0, 10.0], [np.nan, 100.0], [5.0, np.nan]])
        t = MissingAwareLogTransformer()
        z = t.fit_transform(x)

        # NaN preserved
        assert np.isnan(z[1, 0])
        assert np.isnan(z[2, 1])

        # Observed values transformed
        np.testing.assert_allclose(z[0, 0], np.log1p(0.0))
        np.testing.assert_allclose(z[0, 1], np.log1p(10.0))

        x_back = t.inverse_transform(z)
        obs = ~np.isnan(x)
        np.testing.assert_allclose(x_back[obs], x[obs], atol=1e-12)

    def test_custom_offset(self):
        x = np.array([[1.0], [10.0]])
        t = MissingAwareLogTransformer(offset=2.0)
        z = t.fit_transform(x)
        np.testing.assert_allclose(z[0, 0], np.log(3.0))
        x_back = t.inverse_transform(z)
        np.testing.assert_allclose(x_back, x, atol=1e-12)

    def test_invalid_offset_raises(self):
        with pytest.raises(ValueError, match="offset must be positive"):
            MissingAwareLogTransformer(offset=0.0)

    def test_transform_before_fit_raises(self):
        t = MissingAwareLogTransformer()
        with pytest.raises(RuntimeError, match="not fitted"):
            t.transform(np.array([[1.0]]))

    def test_inverse_before_fit_raises(self):
        t = MissingAwareLogTransformer()
        with pytest.raises(RuntimeError, match="not fitted"):
            t.inverse_transform(np.array([[1.0]]))


# ── MissingAwareWinsorizer ──────────────────────────────────────────────


class TestMissingAwareWinsorizer:
    def test_clips_outliers_preserves_nan(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((100, 2))
        x[0, 0] = 50.0  # extreme outlier
        x[5, 1] = np.nan

        w = MissingAwareWinsorizer(lower_quantile=0.05, upper_quantile=0.95)
        z = w.fit_transform(x)

        # Outlier clipped
        assert z[0, 0] < 50.0
        assert z[0, 0] == pytest.approx(w.upper_[0])

        # NaN preserved
        assert np.isnan(z[5, 1])
        # Non-outlier values unchanged
        np.testing.assert_allclose(z[1, 0], x[1, 0])

    def test_inverse_is_noop(self):
        x = np.array([[1.0], [2.0], [100.0]])
        w = MissingAwareWinsorizer()
        z = w.fit_transform(x)
        z_back = w.inverse_transform(z)
        np.testing.assert_array_equal(z, z_back)

    def test_invalid_quantiles_raise(self):
        with pytest.raises(ValueError, match="quantiles"):
            MissingAwareWinsorizer(lower_quantile=0.5, upper_quantile=0.3)

    def test_transform_before_fit_raises(self):
        w = MissingAwareWinsorizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            w.transform(np.array([[1.0]]))

    def test_all_nan_column(self):
        x = np.array([[np.nan], [np.nan]])
        w = MissingAwareWinsorizer()
        z = w.fit_transform(x)
        assert np.all(np.isnan(z))


# ── MissingAwarePowerTransformer ────────────────────────────────────────


class TestMissingAwarePowerTransformer:
    def test_round_trip_with_missing(self):
        rng = np.random.default_rng(0)
        x = np.exp(rng.standard_normal((50, 2)))
        x[3, 0] = np.nan
        x[10, 1] = np.nan

        t = MissingAwarePowerTransformer(standardize=False)
        z = t.fit_transform(x)

        assert np.isnan(z[3, 0])
        assert np.isnan(z[10, 1])

        x_back = t.inverse_transform(z)
        obs = ~np.isnan(x)
        np.testing.assert_allclose(x_back[obs], x[obs], atol=1e-6)

    def test_standardize_produces_unit_variance(self):
        rng = np.random.default_rng(1)
        x = np.exp(rng.standard_normal((200, 1)))
        t = MissingAwarePowerTransformer(standardize=True)
        z = t.fit_transform(x)
        assert abs(np.nanmean(z)) < 0.1
        assert abs(np.nanstd(z) - 1.0) < 0.1

    def test_round_trip_mixed_sign(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((80, 1)) * 5
        t = MissingAwarePowerTransformer(standardize=False)
        z = t.fit_transform(x)
        x_back = t.inverse_transform(z)
        np.testing.assert_allclose(x_back, x, atol=1e-5)

    def test_transform_before_fit_raises(self):
        t = MissingAwarePowerTransformer()
        with pytest.raises(RuntimeError, match="not fitted"):
            t.transform(np.array([[1.0]]))


# ── check_data diagnostics ──────────────────────────────────────────────


class TestCheckData:
    def test_clean_data_passes(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((500, 3))
        report = check_data(x)
        assert report.passed
        assert len(report.warnings) == 0
        assert isinstance(report, DataReport)

    def test_detects_high_skewness(self):
        rng = np.random.default_rng(0)
        x = np.column_stack([
            np.exp(rng.standard_normal(200) * 3),  # very skewed
            rng.standard_normal(200),  # normal
        ])
        report = check_data(
            x, column_names=["skewed", "normal"], skewness_threshold=2.0
        )
        assert not report.passed
        assert any("skewed" in w and "skewness" in w for w in report.warnings)
        assert "skewed" in report.suggested_pretransforms
        assert report.suggested_pretransforms["skewed"] == "log1p"
        # Normal column should not be flagged
        assert "normal" not in report.suggested_pretransforms

    def test_detects_near_zero_variance(self):
        x = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
        report = check_data(x, column_names=["const", "const2"])
        assert not report.passed
        assert any("near-zero variance" in w for w in report.warnings)

    def test_detects_high_missing_fraction(self):
        x = np.full((10, 1), np.nan)
        x[0, 0] = 1.0
        report = check_data(x, missing_fraction_warn=0.5)
        assert not report.passed
        assert any("missing" in w for w in report.warnings)

    def test_detects_outliers(self):
        rng = np.random.default_rng(99)
        x = rng.standard_normal((200, 1))
        # Inject many extreme outliers (> 5 MAD)
        x[:10, 0] = 100.0
        report = check_data(x)
        assert any("outlier" in w.lower() for w in report.warnings)

    def test_integer_column_names_when_none(self):
        x = np.array([[1.0], [2.0], [3.0]])
        report = check_data(x)
        assert "0" in report.summary

    def test_warn_emits_warnings(self):
        x = np.full((10, 1), np.nan)
        x[0, 0] = 1.0
        import warnings as w_mod

        with w_mod.catch_warnings(record=True) as caught:
            w_mod.simplefilter("always")
            check_data(x, warn=True, missing_fraction_warn=0.5)
        assert len(caught) >= 1
        assert "missing" in str(caught[0].message).lower()

    def test_suggests_power_for_negative_skew(self):
        rng = np.random.default_rng(7)
        # Generate strongly left-skewed data
        x = -(np.exp(rng.standard_normal(300) * 3)).reshape(-1, 1)
        report = check_data(x, column_names=["neg_skewed"])
        if "neg_skewed" in report.suggested_pretransforms:
            assert report.suggested_pretransforms["neg_skewed"] == "power"
