"""Full covariance, exact observation partitions and cancellation fallbacks."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vbpca_py import _full_update as fu
from vbpca_py import _noise_statistics as ns


def _layout(array: np.ndarray, order: str) -> np.ndarray:
    if order == "strided":
        buffer = np.empty(tuple(2 * length for length in array.shape))
        view = buffer[tuple(slice(None, None, 2) for _ in array.shape)]
        view[...] = array
        return view
    return np.array(array, order=order)


def _state(q: int, order: str, *, loading_covariance: bool = True) -> tuple:
    rng = np.random.default_rng(289)
    p, n = 12, 15
    a, s = rng.normal(size=(p, q)), rng.normal(size=(q, n))
    av_root, sv_root = rng.normal(size=(p, q, q)), rng.normal(size=(n, q, q))
    av = (av_root @ av_root.transpose(0, 2, 1) + 0.2 * np.eye(q)) / q
    sv = (sv_root @ sv_root.transpose(0, 2, 1) + 0.3 * np.eye(q)) / q
    mask = np.ones((p, n), dtype=bool)
    mask.ravel()[rng.choice(p * n, 7, replace=False)] = False
    ix, jx = np.nonzero(mask)
    plan = ns.prepare_noise_complement(mask.shape, ix, jx)
    assert plan is not None
    state = fu.NoiseState(
        loadings=_layout(a, order),
        scores=_layout(s, order),
        loading_covariances=_layout(av, order) if loading_covariance else [],
        score_covariances=_layout(sv, order),
        mu_variances=np.full((p, 1), 0.03),
        pattern_index=None,
        n_data=float(ix.size),
        noise_var=0.6,
        complement=plan,
    )
    return state, ix, jx


def _manual_sum(
    state: fu.NoiseState, ix: np.ndarray, jx: np.ndarray, rms: float
) -> float:
    total = 0.0
    for i, j in zip(ix, jx, strict=True):
        sv = state.score_covariances[
            state.pattern_index[j] if state.pattern_index is not None else j
        ]
        total += float(state.loadings[i] @ sv @ state.loadings[i])
        if len(state.loading_covariances):
            av = state.loading_covariances[i]
            total += float(state.scores[:, j] @ av @ state.scores[:, j])
            total += float(np.sum(av * sv))
        if state.mu_variances.size:
            total += float(state.mu_variances[i, 0])
    return total + rms**2 * state.n_data


@pytest.mark.parametrize("q", [1, 3, 8])
@pytest.mark.parametrize("order", ["C", "F", "strided"])
@pytest.mark.parametrize("loading_covariance", [False, True])
@pytest.mark.parametrize("threads", [1, 2])
def test_complete_minus_missing_matches_independent_full_covariance_equation(
    q: int, order: str, loading_covariance: bool, threads: int
) -> None:
    state, ix, jx = _state(q, order, loading_covariance=loading_covariance)
    state.sparse_num_cpu = threads
    expected = _manual_sum(state, ix, jx, rms=0.7)
    observed_state, observed = fu._update_noise_variance(
        replace(state, complement=None), 0.7, ix, jx, hp_v=0.04
    )
    updated, actual = fu._update_noise_variance(state, 0.7, ix, jx, hp_v=0.04)
    assert updated.statistics_mode == "complement"
    assert updated.complement is state.complement
    assert actual == pytest.approx(expected, rel=2e-13, abs=2e-12)
    assert actual == pytest.approx(observed, rel=2e-13, abs=2e-12)
    assert updated.noise_var == pytest.approx(
        (expected + 0.08) / (state.n_data + 0.08), rel=2e-13
    )
    assert updated.noise_var == pytest.approx(observed_state.noise_var, rel=2e-13)


def test_pattern_indexed_covariances_are_expanded_by_sample() -> None:
    state, ix, jx = _state(3, "C")
    state.score_covariances = state.score_covariances[:2]
    state.pattern_index = np.arange(state.scores.shape[1]) % 2
    updated, actual = fu._update_noise_variance(state, 0.5, ix, jx)
    assert updated.statistics_mode == "complement"
    assert actual == pytest.approx(_manual_sum(state, ix, jx, 0.5), rel=2e-13)


def test_coordinate_partition_is_read_only_and_independent_of_data_values() -> None:
    mask = np.ones((10, 12), dtype=bool)
    mask[3, 7] = False
    ix, jx = np.nonzero(mask)
    plan = ns.prepare_noise_complement(mask.shape, ix, jx)
    assert plan is not None
    np.testing.assert_array_equal(plan.rows, [3])
    np.testing.assert_array_equal(plan.columns, [7])
    assert not plan.rows.flags.writeable
    assert not plan.columns.flags.writeable
    assert plan.rows.dtype == np.int32
    # A duplicate cannot replace another observation in a true partition.
    ix[0], jx[0] = ix[1], jx[1]
    assert ns.prepare_noise_complement(mask.shape, ix, jx) is None


@pytest.mark.parametrize(
    ("shape", "ix", "jx"),
    [
        ((0, 10), np.array([], dtype=int), np.array([], dtype=int)),
        ((2**32, 10), np.arange(3), np.arange(3)),
        ((10, 10), np.arange(3), np.arange(2)),
        ((10, 10), np.arange(95, dtype=float), np.zeros(95)),
        ((10, 10), np.arange(95)[:, None], np.zeros(95, dtype=int)),
        ((10, 10), np.zeros(95, dtype=int), np.full(95, -1)),
        ((10, 10), np.zeros(95, dtype=int), np.full(95, 10)),
        ((10, 10), np.arange(10), np.arange(10)),
    ],
)
def test_unsupported_or_low_density_partitions_do_not_allocate_a_complement(
    shape: tuple[int, int], ix: np.ndarray, jx: np.ndarray
) -> None:
    assert ns.prepare_noise_complement(shape, ix, jx) is None


@pytest.mark.parametrize("observed_scale", [0.0, 1e-9])
def test_near_cancellation_uses_the_original_observed_sum(
    observed_scale: float,
) -> None:
    p, n = 10, 10
    mask = np.ones((p, n), dtype=bool)
    mask[0, 0] = False
    ix, jx = np.nonzero(mask)
    a = np.full((p, 1), observed_scale)
    a[0, 0] = 1e8
    sv = np.zeros((n, 1, 1))
    sv[0, 0, 0] = 1.0
    state = fu.NoiseState(
        loadings=a,
        scores=np.zeros((1, n)),
        loading_covariances=[],
        score_covariances=sv,
        mu_variances=np.zeros((p, 1)),
        pattern_index=None,
        n_data=float(ix.size),
        noise_var=1.0,
        complement=ns.prepare_noise_complement(mask.shape, ix, jx),
    )
    expected = fu._update_noise_variance(replace(state, complement=None), 0.0, ix, jx)
    updated, actual = fu._update_noise_variance(state, 0.0, ix, jx)
    assert updated.statistics_mode == "observed"
    assert actual == expected[1]
    assert updated.noise_var == expected[0].noise_var
    assert updated.noise_var >= 1e-15


def test_gram_overflow_falls_back_when_cellwise_equation_is_finite() -> None:
    state, ix, jx = _state(1, "C", loading_covariance=False)
    state.loadings[:] = 1e160
    state.score_covariances[:] = 1e-300
    expected_state, expected = fu._update_noise_variance(
        replace(state, complement=None), 0.0, ix, jx
    )
    updated, actual = fu._update_noise_variance(state, 0.0, ix, jx)
    assert np.isfinite(actual)
    assert updated.statistics_mode == "observed"
    assert actual == expected
    assert updated.noise_var == expected_state.noise_var


def test_normalize_execution_mode_and_reject_unknown_option() -> None:
    assert ns.normalize_noise_statistics(" Complement ") == "complement"
    with pytest.raises(ValueError, match="noise_sufficient_statistics"):
        ns.normalize_noise_statistics("diagonal")
