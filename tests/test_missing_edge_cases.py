"""Edge-case tests for missing data scenarios.

Covers extreme missingness patterns that are important to validate:
- >50% missing entries
- Entire rows or columns missing
- Single observation per row
- Diagonal-only observations (extreme sparsity)
"""

from __future__ import annotations

import numpy as np

from vbpca_py import VBPCA


def _make_low_rank(
    n_features: int = 30,
    n_samples: int = 60,
    rank: int = 3,
    noise_std: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Generate a low-rank features × samples matrix with noise."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n_features, rank))
    s = rng.standard_normal((rank, n_samples))
    return w @ s + noise_std * rng.standard_normal((n_features, n_samples))


class TestHighMissingness:
    """Tests with >50% missing data."""

    def test_sixty_percent_missing_converges(self) -> None:
        """60% missing — model should converge and produce finite output."""
        x = _make_low_rank(seed=1)
        rng = np.random.default_rng(1)
        mask = rng.random(x.shape) > 0.6  # 60% missing
        x_obs = np.where(mask, x, np.nan)

        model = VBPCA(n_components=3, maxiters=200, bias=True)
        model.fit(x_obs, mask=mask)

        assert model.rms_ is not None
        assert np.isfinite(model.rms_)
        recon = model.inverse_transform()
        assert np.all(np.isfinite(recon))

    def test_seventy_percent_missing_finite(self) -> None:
        """70% missing — should still produce finite reconstruction."""
        x = _make_low_rank(n_features=40, n_samples=80, seed=2)
        rng = np.random.default_rng(2)
        mask = rng.random(x.shape) > 0.7
        x_obs = np.where(mask, x, np.nan)

        model = VBPCA(n_components=3, maxiters=200, bias=True)
        model.fit(x_obs, mask=mask)
        recon = model.inverse_transform()
        assert np.all(np.isfinite(recon))


class TestAllMissingRowCol:
    """Tests where entire rows or columns have no observations."""

    def test_all_missing_row_handled(self) -> None:
        """One row entirely unobserved — should not crash."""
        x = _make_low_rank(seed=10)
        mask = np.ones_like(x, dtype=bool)
        mask[0, :] = False  # Row 0 all missing
        x_obs = np.where(mask, x, np.nan)

        model = VBPCA(n_components=3, maxiters=100, bias=True)
        # The model may handle this gracefully or raise — either is valid
        try:
            model.fit(x_obs, mask=mask)
            recon = model.inverse_transform()
            # If it succeeds, reconstruction for observed entries should be finite
            assert np.all(np.isfinite(recon[mask]))
        except (ValueError, np.linalg.LinAlgError):
            pass  # Acceptable to reject this input

    def test_all_missing_column_handled(self) -> None:
        """One column entirely unobserved — should not crash."""
        x = _make_low_rank(seed=11)
        mask = np.ones_like(x, dtype=bool)
        mask[:, 0] = False  # Column 0 all missing
        x_obs = np.where(mask, x, np.nan)

        model = VBPCA(n_components=3, maxiters=100, bias=True)
        try:
            model.fit(x_obs, mask=mask)
            recon = model.inverse_transform()
            assert np.all(np.isfinite(recon[mask]))
        except (ValueError, np.linalg.LinAlgError):
            pass  # Acceptable to reject this input

    def test_multiple_missing_rows(self) -> None:
        """Three rows entirely unobserved."""
        x = _make_low_rank(n_features=40, seed=12)
        mask = np.ones_like(x, dtype=bool)
        mask[:3, :] = False  # First 3 rows all missing
        x_obs = np.where(mask, x, np.nan)

        model = VBPCA(n_components=3, maxiters=100, bias=True)
        try:
            model.fit(x_obs, mask=mask)
            recon = model.inverse_transform()
            assert np.all(np.isfinite(recon[mask]))
        except (ValueError, np.linalg.LinAlgError):
            pass


class TestSparseObservationPatterns:
    """Tests with very sparse observation patterns."""

    def test_single_observation_per_row(self) -> None:
        """Each row has exactly 1 observed entry."""
        x = _make_low_rank(n_features=20, n_samples=30, seed=20)
        rng = np.random.default_rng(20)
        mask = np.zeros_like(x, dtype=bool)
        for i in range(x.shape[0]):
            j = rng.integers(0, x.shape[1])
            mask[i, j] = True
        x_obs = np.where(mask, x, np.nan)

        model = VBPCA(n_components=2, maxiters=100, bias=True)
        try:
            model.fit(x_obs, mask=mask)
            recon = model.inverse_transform()
            # At least the observed entries should be finite
            assert np.all(np.isfinite(recon[mask]))
        except (ValueError, np.linalg.LinAlgError):
            pass  # Acceptable

    def test_two_observations_per_row(self) -> None:
        """Each row has exactly 2 observed entries — minimal but usable."""
        x = _make_low_rank(n_features=20, n_samples=40, seed=21)
        rng = np.random.default_rng(21)
        mask = np.zeros_like(x, dtype=bool)
        for i in range(x.shape[0]):
            cols = rng.choice(x.shape[1], size=2, replace=False)
            mask[i, cols] = True
        x_obs = np.where(mask, x, np.nan)

        model = VBPCA(n_components=2, maxiters=100, bias=True)
        try:
            model.fit(x_obs, mask=mask)
            recon = model.inverse_transform()
            assert np.all(np.isfinite(recon[mask]))
        except (ValueError, np.linalg.LinAlgError):
            pass

    def test_block_missing_pattern(self) -> None:
        """Block-missing: a contiguous rectangular region is unobserved."""
        x = _make_low_rank(n_features=30, n_samples=60, seed=30)
        mask = np.ones_like(x, dtype=bool)
        # Remove a 10×20 block
        mask[5:15, 10:30] = False
        x_obs = np.where(mask, x, np.nan)

        model = VBPCA(n_components=3, maxiters=200, bias=True)
        model.fit(x_obs, mask=mask)
        recon = model.inverse_transform()

        assert np.all(np.isfinite(recon))
        # The block should still be imputed
        block_rmse = float(np.sqrt(np.mean((recon[5:15, 10:30] - x[5:15, 10:30]) ** 2)))
        # This is a rough check — block imputation won't be perfect
        assert block_rmse < 10 * x.std(), f"Block RMSE {block_rmse:.2f} too large"


class TestMinimalDimensions:
    """Tests with very small matrices."""

    def test_two_by_two(self) -> None:
        """2x2 matrix with 1 missing entry."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, True], [True, False]])
        x_obs = np.where(mask, x, np.nan)

        model = VBPCA(n_components=1, maxiters=100, bias=True)
        model.fit(x_obs, mask=mask)
        recon = model.inverse_transform()
        assert np.all(np.isfinite(recon))

    def test_tall_skinny(self) -> None:
        """100 features, 5 samples, 20% missing."""
        rng = np.random.default_rng(40)
        x = rng.standard_normal((100, 5))
        mask = rng.random(x.shape) > 0.2
        x_obs = np.where(mask, x, np.nan)

        model = VBPCA(n_components=2, maxiters=100, bias=True)
        model.fit(x_obs, mask=mask)
        recon = model.inverse_transform()
        assert np.all(np.isfinite(recon))

    def test_wide_short(self) -> None:
        """5 features, 100 samples, 20% missing."""
        rng = np.random.default_rng(41)
        x = rng.standard_normal((5, 100))
        mask = rng.random(x.shape) > 0.2
        x_obs = np.where(mask, x, np.nan)

        model = VBPCA(n_components=2, maxiters=100, bias=True)
        model.fit(x_obs, mask=mask)
        recon = model.inverse_transform()
        assert np.all(np.isfinite(recon))
