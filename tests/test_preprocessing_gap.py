"""Add coverage for preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py.preprocessing import (
    AutoEncoder,
    MissingAwareMinMaxScaler,
    MissingAwareOneHotEncoder,
    MissingAwareSparseOneHotEncoder,
    MissingAwareStandardScaler,
    _ensure_mask,
    _sparse_safe_max,
    _sparse_safe_min,
)


def test_one_hot_encoder_binary_and_multicat_paths() -> None:
    enc = MissingAwareOneHotEncoder(handle_unknown="raise", mean_center=True)
    x = np.array([["a", "x"], ["b", "y"], ["a", "y"]], dtype=object)
    mask = np.array([[1, 1], [1, 1], [0, 1]], dtype=bool)
    enc.fit(x, mask=mask)

    with pytest.raises(ValueError):
        enc.transform(np.array([["c", "x"]], dtype=object), mask=np.array([[1, 1]]))

    z = enc.transform(x, mask=mask)
    assert z.shape[1] == len(enc.feature_names_out_)

    x_decoded = enc.inverse_transform(z, mask=mask)
    assert x_decoded.shape == x.shape


def test_one_hot_encoder_empty_categories_branch() -> None:
    enc = MissingAwareOneHotEncoder(mean_center=False)
    x = np.array([[np.nan], [np.nan]])
    mask = np.array([[0], [0]], dtype=bool)
    enc.fit(x, mask=mask)
    z = enc.transform(x, mask=mask)
    assert z.shape[1] == 0


def test_sparse_one_hot_encoder_error_paths_and_roundtrip() -> None:
    enc = MissingAwareSparseOneHotEncoder()
    data = sp.csr_matrix([[1.0], [2.0], [0.0]])

    with pytest.raises(ValueError):
        enc.fit(data, mask=np.ones((3, 1)))

    bad_shape = sp.csr_matrix(np.ones((3, 2)))
    with pytest.raises(ValueError):
        enc.fit(bad_shape)

    enc_unknown = MissingAwareSparseOneHotEncoder(handle_unknown="raise")
    enc_unknown.fit(data)
    with pytest.raises(ValueError):
        enc_unknown.transform(sp.csr_matrix([[5.0], [1.0], [2.0]]))

    enc_mc = MissingAwareSparseOneHotEncoder(mean_center=True)
    enc_mc.fit(data)
    z = enc_mc.transform(data)
    inv = enc_mc.inverse_transform(z)
    assert inv.shape == data.shape

    # Empty data triggers zero-width branch.
    empty_enc = MissingAwareSparseOneHotEncoder()
    empty_enc.fit(sp.csr_matrix((3, 1)))
    empty_z = empty_enc.transform(sp.csr_matrix((3, 1)))
    assert empty_z.shape[1] == 0

    # Direct binary inverse helper coverage.
    enc_mc.categories_ = [0.0, 1.0]
    enc_mc.column_means_ = np.zeros(2)
    enc_mc.n_features_in_ = 1
    z_bin = sp.csr_matrix([[1.0, 0.0]])
    restored = enc_mc._inverse_binary_sparse(z_bin, n_rows=1)
    assert restored.shape == (1, 1)


def test_standard_scaler_dense_and_sparse() -> None:
    dense = np.array([[1.0, 0.0], [3.0, np.nan]])
    mask = ~np.isnan(dense)
    scaler = MissingAwareStandardScaler()
    scaler.fit(dense, mask=mask)
    z = scaler.transform(dense, mask=mask)
    back = scaler.inverse_transform(z, mask=mask)
    assert np.allclose(back[mask], dense[mask])

    sparse = sp.csr_matrix([[0.0, 1.0], [1.0, 2.0]])
    scaler_sparse = MissingAwareStandardScaler()
    scaler_sparse.fit(sparse)
    z_sparse = scaler_sparse.transform(sparse)
    back_sparse = scaler_sparse.inverse_transform(z_sparse)
    assert back_sparse.shape == sparse.shape

    # No observed entries -> nan means/vars path.
    dense_empty = np.array([[np.nan, np.nan]])
    mask_empty = np.array([[0, 0]], dtype=bool)
    scaler_empty = MissingAwareStandardScaler()
    scaler_empty.fit(dense_empty, mask=mask_empty)
    z_empty = scaler_empty.transform(dense_empty, mask=mask_empty)
    back_empty = scaler_empty.inverse_transform(z_empty, mask=mask_empty)
    assert np.isnan(back_empty).all()


def test_minmax_scaler_dense_and_sparse() -> None:
    dense = np.array([[2.0, 2.0], [2.0, 4.0]])
    mask = np.array([[1, 1], [0, 1]], dtype=bool)
    scaler = MissingAwareMinMaxScaler()
    scaler.fit(dense, mask=mask)
    z = scaler.transform(dense, mask=mask)
    back = scaler.inverse_transform(z, mask=mask)
    assert np.allclose(back[mask], dense[mask])

    sparse = sp.csr_matrix([[1.0, 0.0], [2.0, 3.0]])
    scaler_sparse = MissingAwareMinMaxScaler()
    scaler_sparse.fit(sparse)
    z_sparse = scaler_sparse.transform(sparse)
    back_sparse = scaler_sparse.inverse_transform(z_sparse)
    assert back_sparse.shape == sparse.shape

    dense_missing = np.array([[np.nan, np.nan]])
    mask_missing = np.array([[0, 0]], dtype=bool)
    scaler_missing = MissingAwareMinMaxScaler()
    scaler_missing.fit(dense_missing, mask=mask_missing)
    z_missing = scaler_missing.transform(dense_missing, mask=mask_missing)
    back_missing = scaler_missing.inverse_transform(z_missing, mask=mask_missing)
    assert np.isnan(back_missing).all()


def test_autoencoder_dense_and_sparse_paths() -> None:
    x_dense = np.array([[1.0, 0.0], [np.nan, 1.0]])
    mask = ~np.isnan(x_dense)
    ae = AutoEncoder(column_types=["continuous", "continuous"], mean_center_ohe=True)
    ae.fit(x_dense, mask=mask)
    z = ae.transform(x_dense, mask=mask)
    x_back = ae.inverse_transform(z, mask=mask)
    assert x_back.shape == x_dense.shape

    x_sparse = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    ae_sparse = AutoEncoder(column_types=["categorical", "continuous"])
    ae_sparse.fit(x_sparse)
    z_sparse = ae_sparse.transform(x_sparse)
    assert z_sparse.shape[0] == x_sparse.shape[0]

    z_sparse_back = ae_sparse.inverse_transform(z_sparse)
    assert z_sparse_back.shape[0] == x_sparse.shape[0]

    with pytest.raises(ValueError):
        ae_sparse.transform(x_sparse, mask=np.ones_like(x_sparse.toarray()))


def test_helper_branches_for_sparse_stats_and_masks() -> None:
    # _ensure_mask for non-numeric dtype uses equality path.
    text = np.array(["a", "b"], dtype=object)[:, None]
    mask_out = _ensure_mask(text, mask=None)
    assert mask_out.shape == text.shape

    # _sparse_safe_min/_max continue when counts are zero.
    csc = sp.csc_matrix((2, 2))
    counts = np.array([0, 0])
    assert np.all(np.isnan(_sparse_safe_min(csc, counts)))
    assert np.all(np.isnan(_sparse_safe_max(csc, counts)))
