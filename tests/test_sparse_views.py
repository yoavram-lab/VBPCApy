"""Sparse view lifetimes, conversion counts, and fixed-iteration trajectories."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

import vbpca_py._pca_full as fitting

OFF = dict.fromkeys(
    ("angle", "earlystop", "rms_plateau", "cost", "composite", "slowing_down"), False
)


def _fit(*, bias, rotate, order="legacy", uniquesv=False):
    rng = np.random.default_rng(294)
    x = rng.normal(size=(17, 23))
    mask = rng.random(x.shape) > 0.35
    mask[0, :] = True
    mask[:, 0] = True
    x[0, 0] = 0.0  # Membership is independent of a stored observation's value.
    rows, columns = np.nonzero(mask)
    observed = sp.csr_matrix((x[rows, columns], (rows, columns)), shape=x.shape)
    return fitting.pca_full(
        observed,
        4,
        mask=sp.csr_matrix(mask),
        maxiters=4,
        niter_broadprior=0,
        convergence_criteria=OFF,
        runtime_tuning="off",
        num_cpu=1,
        random_state=294,
        verbose=0,
        compat_mode="modern",
        record_cost=True,
        return_diagnostics=False,
        bias=bias,
        rotate2pca=rotate,
        bias_update_order=order,
        uniquesv=uniquesv,
    )


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rotate", [False, True])
def test_only_required_csc_views_are_built(monkeypatch, bias, rotate):
    real = fitting._coerce_sparse_views
    calls = []

    def record(x):
        calls.append(x)
        return real(x)

    monkeypatch.setattr(fitting, "_coerce_sparse_views", record)
    _fit(bias=bias, rotate=rotate)
    assert len(calls) == (4 if bias else 1)


@pytest.mark.parametrize("order", ["legacy", "post_factor"])
@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("uniquesv", [False, True])
def test_cached_views_preserve_uncached_trajectory(
    monkeypatch, order, rotate, uniquesv
):
    candidate = _fit(bias=True, rotate=rotate, order=order, uniquesv=uniquesv)
    monkeypatch.setattr(
        fitting,
        "_score_sparse_views",
        lambda training, x: fitting._coerce_sparse_views(x),
    )
    reference = _fit(bias=True, rotate=rotate, order=order, uniquesv=uniquesv)
    for key in ("A", "S", "Mu", "V", "Av", "Sv", "Va", "Muv"):
        assert_allclose(candidate[key], reference[key], rtol=0, atol=0)
    for key in ("rms", "cost"):
        assert_allclose(candidate["lc"][key], reference["lc"][key], rtol=0, atol=0)


def test_in_place_centering_invalidates_value_snapshots(monkeypatch):
    real_bias = fitting._update_bias

    def in_place(**kwargs):
        original = kwargs["centering"].x_data
        bias, centered = real_bias(**kwargs)
        original.data[:] = centered.x_data.data
        centered.x_data = original
        return bias, centered

    monkeypatch.setattr(fitting, "_update_bias", in_place)
    candidate = _fit(bias=True, rotate=False)
    monkeypatch.setattr(
        fitting,
        "_score_sparse_views",
        lambda training, x: fitting._coerce_sparse_views(x),
    )
    reference = _fit(bias=True, rotate=False)
    for key in ("A", "S", "Mu", "V", "Av", "Sv"):
        assert_allclose(candidate[key], reference[key], rtol=0, atol=0)
