"""Independent equations and bounded NumPy allocations for uncertainty output."""

from __future__ import annotations

import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from vbpca_py._pca_full import _marginal_variance
from vbpca_py._posterior_variance import marginal_variance


def _state(features=17, samples=23, rank=4, *, layout="C", patterns=False):
    rng = np.random.default_rng(290)
    a = rng.normal(size=(features, rank))
    s = rng.normal(size=(rank, samples))
    if layout == "F":
        a, s = np.asfortranarray(a), np.asfortranarray(s)
    elif layout == "strided":
        a = np.repeat(a, 2, axis=0)[::2]
        s = np.repeat(s, 2, axis=1)[:, ::2]
    ra = rng.normal(size=(features, rank, rank))
    rs = rng.normal(size=(2 if patterns else samples, rank, rank))
    return SimpleNamespace(
        a=a,
        s=s,
        av=ra @ ra.transpose(0, 2, 1),
        sv=rs @ rs.transpose(0, 2, 1),
        muv=rng.uniform(0.01, 0.2, size=(features, 1)),
        pattern_index=np.arange(samples) % 2 if patterns else None,
    )


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("patterns", [False, True])
@pytest.mark.parametrize("absent", ["none", "av", "sv", "both"])
def test_blocked_variance_matches_independent_cells(layout, patterns, absent):
    state = _state(layout=layout, patterns=patterns)
    if absent in {"av", "both"}:
        state.av = []
        state.muv = np.zeros(0)
    if absent in {"sv", "both"}:
        state.sv = []
    expected = np.zeros((17, 23))
    for i in range(17):
        for j in range(23):
            av = state.av[i] if len(state.av) else np.zeros((4, 4))
            index = j % 2 if patterns else j
            sv = state.sv[index] if len(state.sv) else np.zeros((4, 4))
            expected[i, j] = (
                state.a[i] @ sv @ state.a[i]
                + state.s[:, j] @ av @ state.s[:, j]
                + np.trace(av @ sv)
                + (state.muv[i, 0] if state.muv.size else 0)
            )
    actual = marginal_variance(state, block_size=5)
    assert_allclose(actual, expected, rtol=2e-14, atol=2e-13)
    assert_allclose(_marginal_variance(state), expected, rtol=2e-14, atol=2e-13)


@pytest.mark.parametrize("rank", [0, 1, 8])
def test_list_covariances_and_vector_mean(rank):
    state = _state(rank=rank)
    state.av = list(state.av)
    state.sv = list(state.sv)
    expected = _marginal_variance(state)
    state.muv = state.muv.ravel()
    assert_allclose(_marginal_variance(state), expected)


def test_variance_does_not_allocate_full_data_axis_temporaries():
    state = _state(features=640, samples=768, rank=4, patterns=True)
    tracemalloc.start()
    try:
        result = _marginal_variance(state)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The output is unavoidable. Bounded packing and a single block product
    # fit well below another full feature-by-sample temporary.
    assert peak < result.nbytes + 700_000
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("shape", [(0, 5), (5, 0)])
def test_empty_output_axes(shape):
    state = _state(features=shape[0], samples=shape[1])
    assert _marginal_variance(state).shape == shape


def test_invalid_block_size():
    state = _state()
    with pytest.raises(ValueError, match="block_size must be positive"):
        marginal_variance(state, block_size=0)
