"""Automatic sparse probes must not erase unselected zero observations."""

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py import VBPCA, legacy, make_xprobe_mask
from vbpca_py._validation import make_native_probe
from vbpca_py.options import CompatMode

OPTIONS = {
    "maxiters": 4,
    "niter_broadprior": 0,
    "num_cpu": 1,
    "runtime_tuning": "off",
    "verbose": 0,
    "record_cost": True,
    "convergence_criteria": dict.fromkeys(
        ["angle", "earlystop", "rms_plateau", "cost", "composite", "slowing_down"],
        False,
    ),
}


def _stored(values):
    rows, cols = np.indices(values.shape)
    return sp.csr_matrix((values.ravel(), (rows.ravel(), cols.ravel())))


def _entries(matrix):
    coordinates = matrix.tocoo()
    return dict(
        zip(
            zip(coordinates.row, coordinates.col, strict=True),
            coordinates.data,
            strict=True,
        )
    )


def _mask(kind, shape):
    if kind == "none":
        return None
    mask = np.ones(shape, dtype=bool)
    mask[0, -1] = False
    return sp.csr_matrix(mask) if kind == "sparse" else mask


@pytest.mark.parametrize("seed", [4, 19, 23])
@pytest.mark.parametrize("mask_kind", ["none", "dense", "sparse"])
@pytest.mark.parametrize(
    "kind", [sp.csr_matrix, sp.csc_matrix, sp.csr_array, sp.csc_array]
)
def test_sparse_helper_partitions_only_selected_coordinates(seed, mask_kind, kind):  # noqa: C901 - checks buffers and coordinate partitions together
    values = np.arange(12.0).reshape(3, 4)
    values[1, 1], values[2, 2] = 0, np.nan
    matrix = kind(_stored(values))
    mask = _mask(mask_kind, values.shape)
    before = [array.copy() for array in (matrix.data, matrix.indices, matrix.indptr)]
    for array in (matrix.data, matrix.indices, matrix.indptr):
        array.flags.writeable = False
    if mask is not None:
        buffers = (
            (mask.data, mask.indices, mask.indptr) if sp.issparse(mask) else (mask,)
        )
        original_mask = [array.copy() for array in buffers]
        for array in buffers:
            array.flags.writeable = False
    eligible = ~np.isnan(values)
    if mask is not None:
        eligible[0, -1] = False
    rows, cols = np.nonzero(eligible)
    chosen = np.random.default_rng(seed).choice(
        len(rows), size=round(len(rows) * 0.25), replace=False
    )
    held_out = set(zip(rows[chosen], cols[chosen], strict=True))
    training, probe = make_xprobe_mask(
        matrix, 0.25, np.random.default_rng(seed), mask=mask
    )
    assert sp.isspmatrix_csr(training)
    assert sp.isspmatrix_csr(probe)
    assert set(_entries(probe)) == held_out
    assert set(_entries(training)) == set(_entries(matrix)) - held_out
    assert training.nnz + probe.nnz == matrix.nnz
    assert not set(_entries(training)) & set(_entries(probe))
    for output in (training, probe):
        for position, value in _entries(output).items():
            np.testing.assert_equal(value, values[position])
        for actual in (output.data, output.indices, output.indptr):
            assert not any(
                np.shares_memory(actual, original)
                for original in (matrix.data, matrix.indices, matrix.indptr)
            )
    for actual, original in zip(
        (matrix.data, matrix.indices, matrix.indptr), before, strict=True
    ):
        np.testing.assert_array_equal(actual, original)
    if mask is not None:
        for actual, original in zip(buffers, original_mask, strict=True):
            np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize(("seed", "selected"), [(19, (1, 2)), (23, (0, 0))])
def test_reported_zero_loss_and_selected_zero_are_both_fixed(seed, selected):
    x = _stored(np.arange(9.0).reshape(3, 3))
    train, probe = make_xprobe_mask(x, 0.1, np.random.default_rng(seed))
    assert set(_entries(probe)) == {selected}
    assert (x.nnz, train.nnz, probe.nnz) == (9, 8, 1)
    assert (0, 0) in _entries(probe if selected == (0, 0) else train)


@pytest.mark.parametrize("mask_kind", ["none", "dense", "sparse"])
def test_sparse_nan_only_input_has_no_eligible_probes(mask_kind):
    x = _stored(np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="no observed entries"):
        make_xprobe_mask(x, mask=_mask(mask_kind, x.shape))


@pytest.mark.parametrize("mask_kind", ["dense", "sparse"])
def test_mask_requested_absent_zero_can_be_a_probe(mask_kind):
    x = sp.csr_matrix(([5.0], ([0], [1])), shape=(2, 2))
    mask = _mask(mask_kind, x.shape)
    train, probe = make_xprobe_mask(x, 0.99, np.random.default_rng(2), mask=mask)
    assert set(_entries(probe)) == {(0, 0), (1, 0), (1, 1)}
    assert list(_entries(probe).values()) == [0, 0, 0]
    # The mask restricts sampling; the unselected value outside it is retained.
    assert _entries(train) == {(0, 1): 5}


def test_sparse_partition_does_not_allocate_dense_matrix(monkeypatch):
    shape = (100_000, 100_000)
    x = sp.csr_matrix(([0.0, 3.0, 0.0], ([0, 20, 99_999], [0, 2, 99_999])), shape=shape)
    mask = x.copy()
    mask.data = np.ones(3, dtype=bool)

    def no_dense(*args, **kwargs):
        pytest.fail("sparse probe construction densified")

    for cls in (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix):
        monkeypatch.setattr(cls, "toarray", no_dense)
        monkeypatch.setattr(cls, "todense", no_dense)
    train, probe = make_xprobe_mask(x, 0.3, np.random.default_rng(19), mask=mask)
    assert train.shape == probe.shape == shape
    assert train.nnz + probe.nnz == 3


@pytest.mark.parametrize("seed", [4, 19])
@pytest.mark.parametrize("mask_kind", ["none", "dense", "sparse"])
def test_explicit_strict_legacy_split_retains_parent_behavior(seed, mask_kind):
    x = _stored(np.array([[0, 1, np.nan], [2, 0, 3], [4, 5, 6]], dtype=float))
    mask = _mask(mask_kind, x.shape)
    observed = (
        _stored(np.ones(x.shape, dtype=bool)) if mask is None else sp.csr_matrix(mask)
    )
    rows, cols = observed.nonzero()
    chosen = np.random.default_rng(seed).choice(len(rows), size=2, replace=False)
    probe = sp.csr_matrix(
        (
            np.asarray(x[rows[chosen], cols[chosen]]).ravel(),
            (rows[chosen], cols[chosen]),
        ),
        shape=x.shape,
    )
    indicator = probe.copy()
    indicator.data = np.ones(probe.nnz, dtype=float)
    expected = x - x.multiply(indicator)
    expected.eliminate_zeros()
    expected.sort_indices()
    actual, actual_probe = make_xprobe_mask(
        x, 0.25, np.random.default_rng(seed), mask=mask, compat_mode=" STRICT_LEGACY "
    )
    for left, right in [(actual, expected), (actual_probe, probe)]:
        for name in ["data", "indices", "indptr"]:
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


@pytest.mark.parametrize("mode", ["unknown", 123])
def test_probe_compatibility_mode_is_validated(mode):
    with pytest.raises((ValueError, TypeError), match="CompatMode"):
        make_xprobe_mask(np.ones((2, 2)), compat_mode=mode)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("algorithm", ["vb", "map", "ppca"])
@pytest.mark.parametrize("mask_kind", ["none", "dense", "sparse"])
@pytest.mark.parametrize("seed", [19, 23])
def test_modern_legacy_automatic_probe_matches_explicit_dense_split(
    rank, algorithm, mask_kind, seed
):
    values = np.array(
        [
            [0, 1e-10, -2, 4, 3, 2, 5, 1],
            [2, 0, 1, 3, 4, 5, -1, 2],
            [1, 2, 3, 0, 5, 6, 1, 3],
        ],
        dtype=float,
    )
    if mask_kind == "none":
        values[2, 6] = np.nan
    x = _stored(values)
    mask = _mask(mask_kind, x.shape)
    if mask is not None:
        # Request an absent zero without replacing it by epsilon.
        x[0, 0] = 0
        x.eliminate_zeros()
    eligible = ~np.isnan(values)
    if mask is not None:
        eligible[0, -1] = False
    rows, cols = np.nonzero(eligible)
    selected = np.random.default_rng(seed).choice(
        len(rows), size=round(len(rows) * 0.1), replace=False
    )
    probe = np.full_like(values, np.nan)
    probe[rows[selected], cols[selected]] = values[rows[selected], cols[selected]]
    expected_values = np.where(eligible, values, np.nan)
    expected_values[rows[selected], cols[selected]] = np.nan
    options = {
        **OPTIONS,
        "algorithm": algorithm,
        "random_state": seed,
        "compat_mode": CompatMode.MODERN,
    }
    actual = legacy.VBPCA(rank, xprobe_fraction=0.1, **options).fit(x, mask=mask)
    expected = legacy.VBPCA(rank, **options).fit(expected_values, xprobe=probe)
    for name in [
        "components_",
        "scores_",
        "mean_",
        "reconstruction_",
        "variance_",
        "predictive_variance_",
        "rms_",
        "prms_",
        "cost_",
    ]:
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=2e-10,
            atol=2e-12,
            equal_nan=True,
            err_msg=name,
        )


@pytest.mark.parametrize("mode", ["modern", "strict_legacy"])
def test_legacy_automatic_fit_passes_its_mode_to_split(monkeypatch, mode):
    original = legacy.make_xprobe_mask
    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs["compat_mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(legacy, "make_xprobe_mask", capture)
    legacy.VBPCA(
        1, compat_mode=mode, xprobe_fraction=0.1, random_state=19, **OPTIONS
    ).fit(_stored(np.arange(24.0).reshape(3, 8)))
    assert seen == [mode]


def test_native_sparse_split_policy_is_unchanged():
    x = _stored(np.arange(24.0).reshape(8, 3))
    expected_indices = np.random.default_rng(19).choice(x.nnz, size=2, replace=False)
    train, probe = make_native_probe(x, 0.1, np.random.default_rng(19))
    all_positions = list(np.ndindex(x.shape))
    assert set(_entries(probe)) == {all_positions[i] for i in expected_indices}
    np.testing.assert_array_equal(probe.data, x.data[np.sort(expected_indices)])
    assert train.nnz + probe.nnz == x.nnz
    actual = VBPCA(1, xprobe_fraction=0.1, random_state=19, **OPTIONS).fit(x)
    expected = VBPCA(1, random_state=19, **OPTIONS).fit(train, xprobe=probe)
    np.testing.assert_allclose(
        actual.components_, expected.components_, rtol=2e-12, atol=0
    )


def test_duplicate_coordinates_are_sampled_once_after_combining_values():
    x = sp.csr_matrix(([1.0, -1.0, 5.0, 0.0], [0, 0, 1, 0], [0, 3, 4]), shape=(2, 2))
    mask = sp.csr_matrix(
        ([True, True, True, True, False], [0, 0, 1, 0, 1], [0, 3, 5]), shape=x.shape
    )
    before = [
        a.copy()
        for matrix in (x, mask)
        for a in (matrix.data, matrix.indices, matrix.indptr)
    ]
    train, probe = make_xprobe_mask(x, 0.99, np.random.default_rng(2), mask=mask)
    assert train.nnz == 0
    assert _entries(probe) == {(0, 0): 0, (0, 1): 5, (1, 0): 0}
    for actual, original in zip(
        [
            a
            for matrix in (x, mask)
            for a in (matrix.data, matrix.indices, matrix.indptr)
        ],
        before,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize(
    "mask", [np.ones((2, 3)), sp.csr_matrix((2, 3)), sp.csr_matrix((2, 2))]
)
def test_sparse_probe_mask_shape_and_empty_observation_validation(mask):
    with pytest.raises(ValueError, match=r"same shape|no observed entries"):
        make_xprobe_mask(_stored(np.ones((2, 2))), mask=mask)
