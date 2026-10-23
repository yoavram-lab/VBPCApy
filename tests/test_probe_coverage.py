"""A supplied probe cannot silently disappear during training-axis removal."""

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py._pca_full import pca_full
from vbpca_py.legacy import VBPCA


def _settings(compat_mode):
    return {
        "compat_mode": compat_mode,
        "maxiters": 4,
        "niter_broadprior": 0,
        "rmsstop": None,
        "minangle": 0.0,
        "random_state": 3,
        "num_cpu": 1,
        "runtime_tuning": "off",
        "verbose": 0,
    }


def _stored(x):
    rows, columns = np.nonzero(~np.isnan(x))
    return sp.csr_matrix((x[rows, columns], (rows, columns)), shape=x.shape)


def _bad_probe():
    x = np.array([[10.0, np.nan], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    probe = np.full(x.shape, np.nan)
    probe[0, 0], probe[3, 1] = 10.0, 5.0
    return x.T, probe.T


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("compat_mode", ["modern", "strict_legacy"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("probe_only_feature", [False, True])
def test_refuses_finite_wrong_aggregate_and_probe_only_features(
    rank, compat_mode, sparse, probe_only_feature
):
    # At the parent revision, rank-zero dense modern mode reports PRMS=2.247087
    # although both supplied cells give RMS=5.602611 against the returned model.
    x, probe = _bad_probe()
    if probe_only_feature:
        x, probe = x.T, probe.T
    if sparse:
        x, probe = _stored(x), _stored(probe)
    with pytest.raises(ValueError, match="Probe entries would be discarded"):
        pca_full(x, rank, xprobe=probe, **_settings(compat_mode))


@pytest.mark.parametrize("compat_mode", ["modern", "strict_legacy"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("explicit_mask", [False, True])
def test_zero_probe_on_removed_sample_is_still_observed(
    compat_mode, sparse, explicit_mask
):
    x, probe = _bad_probe()
    x[0, 0] = probe[0, 0] = 0.0
    mask = ~np.isnan(x) if explicit_mask else None
    if sparse:
        x, probe = _stored(x), _stored(probe)
        if explicit_mask:
            mask = sp.csr_matrix(mask)
    with pytest.raises(ValueError, match="Probe entries would be discarded"):
        pca_full(x, 0, xprobe=probe, mask=mask, **_settings(compat_mode))


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("compat_mode", ["modern", "strict_legacy"])
@pytest.mark.parametrize("sparse", [False, True])
def test_legacy_automatic_probe_cannot_silently_drop_its_only_sample_observation(
    rank, compat_mode, sparse
):
    x, _ = _bad_probe()
    if sparse:
        x = _stored(x)
    model = VBPCA(rank, xprobe_fraction=2 / 9, **_settings(compat_mode))
    with pytest.raises(ValueError, match="Probe entries would be discarded"):
        model.fit(x)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("compat_mode", ["modern", "strict_legacy"])
@pytest.mark.parametrize("sparse", [False, True])
def test_valid_probes_and_unobserved_axes_without_probes_remain_supported(
    rank, compat_mode, sparse
):
    x = np.array([
        [1.0, 2.0, 3.0, 4.0, np.nan],
        [2.0, 3.0, 4.0, 5.0, np.nan],
        [np.nan] * 5,
    ])
    probe = np.full(x.shape, np.nan)
    probe[0, 0], probe[1, 1] = x[0, 0], x[1, 1]
    observed = ~np.isnan(probe)
    data, held_out = (
        (_stored(x), _stored(probe)) if sparse else (x.copy(), probe.copy())
    )
    if sparse:
        snapshots = [
            a.copy()
            for table in (data, held_out)
            for a in (table.data, table.indices, table.indptr)
        ]
        for table in (data, held_out):
            for array in (table.data, table.indices, table.indptr):
                array.setflags(write=False)
    else:
        snapshots = [data.copy(), held_out.copy()]
        data.setflags(write=False)
        held_out.setflags(write=False)
    result = pca_full(data, rank, xprobe=held_out, **_settings(compat_mode))
    expected = np.sqrt(np.mean((probe[observed] - result["Xrec"][observed]) ** 2))
    assert result["PRMS"] == pytest.approx(expected, rel=2e-12, abs=2e-12)
    actual = (
        [
            a
            for table in (data, held_out)
            for a in (table.data, table.indices, table.indptr)
        ]
        if sparse
        else [data, held_out]
    )
    for array, before in zip(actual, snapshots, strict=True):
        np.testing.assert_array_equal(array, before)


@pytest.mark.parametrize("compat_mode", ["modern", "strict_legacy"])
def test_sparse_nan_probe_on_removed_axes_is_missing(compat_mode):
    x = _stored(np.array([[1.0, 2.0, np.nan], [3.0, 4.0, np.nan], [np.nan] * 3]))
    probe = sp.csr_matrix(([1.0, np.nan], ([0, 2], [0, 2])), shape=x.shape)
    result = pca_full(x, 0, xprobe=probe, **_settings(compat_mode))
    assert result["PRMS"] == pytest.approx(abs(1.0 - result["Xrec"][0, 0]))
    np.testing.assert_array_equal(probe.data, [1.0, np.nan])


@pytest.mark.parametrize("transpose", [False, True])
def test_sparse_signed_cancellation_must_not_silently_remove_probe_axes(transpose):
    x = np.array([[3.0, 1.0, -1.0, np.nan], [7.0, 1.0, 2.0, 3.0]])
    probe = np.full(x.shape, np.nan)
    probe[0, 0] = x[0, 0]
    if transpose:
        x, probe = x.T, probe.T
    data, held_out = _stored(x), _stored(probe)
    with pytest.raises(ValueError, match="Probe entries would be discarded"):
        pca_full(data, 0, xprobe=held_out, **_settings("strict_legacy"))
    result = pca_full(data, 0, xprobe=held_out, **_settings("modern"))
    assert result["PRMS"] == pytest.approx(abs(3.0 - result["Xrec"][0, 0]))
