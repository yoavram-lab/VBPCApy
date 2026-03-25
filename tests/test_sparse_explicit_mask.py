import numpy as np
import scipy.sparse as sp

from vbpca_py import VBPCA
from vbpca_py._full_update import _build_masks_and_counts, _prepare_data


def test_sparse_explicit_mask_counts_and_zero_preserved():
    x = sp.csr_matrix([[0.0, 1.0], [2.0, 0.0]])
    mask = np.array([[True, True], [True, False]], dtype=bool)
    opts = {"compat_mode": "modern", "init": None}

    x_data, x_probe, mask_prepared, *_ = _prepare_data(
        x,
        opts,
        mask_override=mask,
    )
    x_data, _, mask_out, _, n_obs_row, n_data, n_probe = _build_masks_and_counts(
        x_data,
        x_probe,
        opts,
        mask_override=mask_prepared,
    )

    assert sp.isspmatrix(mask_out)
    assert mask_out.count_nonzero() == int(mask.sum())
    np.testing.assert_array_equal(n_obs_row, np.array([2.0, 1.0]))
    assert n_data == float(mask.sum())
    assert n_probe == 0
    assert x_data.shape == (2, 2)


def test_sparse_implicit_zero_legacy_counts_nonzero_only():
    x = sp.csr_matrix([[0.0, 1.0], [2.0, 0.0]])
    opts = {"compat_mode": "modern", "init": None}

    x_data, x_probe, mask_prepared, *_ = _prepare_data(
        x,
        opts,
        mask_override=None,
    )
    x_data, _, mask_out, _, n_obs_row, n_data, n_probe = _build_masks_and_counts(
        x_data,
        x_probe,
        opts,
        mask_override=mask_prepared,
    )

    assert sp.isspmatrix(mask_out)
    # Only the stored nonzeros should be counted here (legacy behavior)
    assert mask_out.count_nonzero() == 2
    np.testing.assert_array_equal(n_obs_row, np.array([1.0, 1.0]))
    assert n_data == 2.0
    assert n_probe == 0
    assert x_data.shape == (2, 2)


def test_vbpca_runs_with_sparse_mask_and_retains_observed_zero():
    x = sp.csr_matrix([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    mask = np.array(
        [
            [True, True, True],
            [True, False, True],
        ],
        dtype=bool,
    )

    model = VBPCA(
        n_components=1,
        maxiters=10,
        compat_mode="modern",
        verbose=0,
    )
    model.fit(x, mask=mask)
    recon = np.asarray(model.inverse_transform(), dtype=float)

    assert recon.shape == (2, 3)
    # Observed zero entry should remain near zero after reconstruction
    assert abs(recon[0, 0]) < 5e-2
    assert model.rms_ is not None
