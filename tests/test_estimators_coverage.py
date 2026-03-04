"""Coverage-focused tests for estimators module."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py import estimators as est_mod
from vbpca_py.estimators import VBPCA


def test_fit_dense_with_sparse_mask_raises() -> None:
    model = VBPCA(n_components=1, maxiters=1)
    x = np.array([[1.0, 2.0]])
    mask = sp.csr_matrix(np.ones_like(x))
    with pytest.raises(ValueError):
        model.fit(x, mask=mask)


def test_fit_sparse_dense_mask_budget_error() -> None:
    model = VBPCA(n_components=1, maxiters=1, max_dense_bytes=0)
    x = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        model.fit(x, mask=mask)


def test_select_n_components_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_select(x, *, mask=None, components=None, config=None, **opts):
        captured["x_shape"] = getattr(x, "shape", None)
        captured["mask"] = mask
        captured["components"] = components
        captured["config"] = config
        captured["opts"] = opts
        return 2, {"best": True}, [{"k": 1}], "model"

    monkeypatch.setattr(est_mod, "select_n_components", fake_select)

    model = VBPCA(
        n_components=1,
        bias=False,
        maxiters=1,
        tol=1e-4,
        verbose=1,
        runtime_tuning="off",
    )
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    best_k, best_metrics, trace, best_model = model.select_n_components(
        x, components=[1, 2]
    )

    assert best_k == 2
    assert best_metrics["best"] is True
    assert trace and trace[0]["k"] == 1
    assert best_model == "model"
    assert captured["components"] == [1, 2]
    assert captured["opts"].get("bias") is False
    assert captured["opts"].get("runtime_tuning") == "off"
