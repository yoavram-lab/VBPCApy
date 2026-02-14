"""Tests for the VBPCA estimator wrapper."""

import pickle

import numpy as np
import pytest

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


def test_transform_before_fit_raises() -> None:
    model = VBPCA(n_components=2)
    with pytest.raises(RuntimeError, match="Model not fitted"):
        model.transform()


def test_inverse_transform_before_fit_raises() -> None:
    model = VBPCA(n_components=2)
    with pytest.raises(RuntimeError, match="Model not fitted"):
        model.inverse_transform()


def test_transform_new_data_not_supported_raises() -> None:
    rng = np.random.default_rng(11)
    x = rng.standard_normal((5, 6))
    model = VBPCA(n_components=2, maxiters=5)
    model.fit(x)
    with pytest.raises(NotImplementedError, match="not supported yet"):
        model.transform(x)


def test_fit_transform_deterministic_with_explicit_init() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((5, 8))
    init = {
        "A": np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, -0.25],
                [0.2, 0.7],
            ],
            dtype=float,
        ),
        "S": rng.standard_normal((2, 8)),
        "Mu": np.zeros((5, 1), dtype=float),
        "V": 1.0,
    }

    init2 = {
        "A": init["A"].copy(),
        "S": init["S"].copy(),
        "Mu": init["Mu"].copy(),
        "V": float(init["V"]),
    }

    model1 = VBPCA(n_components=2, maxiters=5, init=init, rotate2pca=0, verbose=0)
    model2 = VBPCA(
        n_components=2,
        maxiters=5,
        init=init2,
        rotate2pca=0,
        verbose=0,
    )

    z1 = model1.fit_transform(x)
    z2 = model2.fit_transform(x)

    np.testing.assert_allclose(z1, z2, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        model1.inverse_transform(), model2.inverse_transform(), rtol=1e-10, atol=1e-10
    )


def test_vbpca_pickle_roundtrip_preserves_predictions() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((6, 9))

    model = VBPCA(n_components=3, maxiters=6, verbose=0)
    model.fit(x)
    recon_before = model.inverse_transform()

    blob = pickle.dumps(model)
    model_loaded = pickle.loads(blob)
    recon_after = model_loaded.inverse_transform()

    np.testing.assert_allclose(recon_before, recon_after, rtol=1e-12, atol=1e-12)


def test_get_options_includes_default_compat_mode() -> None:
    """Resolved options should expose strict_legacy as default compat mode."""
    model = VBPCA(n_components=2)
    opts = model.get_options()
    assert opts["compat_mode"] == "strict_legacy"


def test_get_options_normalizes_compat_mode_case() -> None:
    """compat_mode should be normalized case-insensitively."""
    model = VBPCA(n_components=2, compat_mode=" MODERN ")
    opts = model.get_options()
    assert opts["compat_mode"] == "modern"


def test_get_options_rejects_invalid_compat_mode() -> None:
    """Invalid compat_mode values should raise at options resolution time."""
    model = VBPCA(n_components=2, compat_mode="legacy")
    with pytest.raises(ValueError, match="compat_mode"):
        _ = model.get_options()
