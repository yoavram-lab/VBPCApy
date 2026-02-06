"""Tests for the VBPCA estimator wrapper."""

import numpy as np

from vbpca_py.estimators import VBPCA


def test_vbpca_fit_transform_shapes() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 4))
    model = VBPCA(n_components=2, maxiters=5, tol=1e-3)
    scores = model.fit_transform(x)
    assert scores.shape == (2, x.shape[1])
    assert model.components_.shape == (x.shape[0], 2)
    recon = model.inverse_transform()
    assert recon.shape == x.shape


def test_vbpca_with_mask() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 6))
    mask = np.ones_like(x, dtype=float)
    mask[0, 0] = 0.0
    x_missing = x.copy()
    x_missing[0, 0] = np.nan
    model = VBPCA(n_components=2, maxiters=5, tol=1e-3)
    model.fit(x_missing, mask=mask)
    assert model.components_.shape[0] == x.shape[0]
    assert model.scores_.shape[1] == x.shape[1]
    recon = model.inverse_transform()
    assert recon.shape == x.shape
