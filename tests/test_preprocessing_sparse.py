import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py.preprocessing import (
    AutoEncoder,
    MissingAwareMinMaxScaler,
    MissingAwareSparseOneHotEncoder,
    MissingAwareStandardScaler,
)


def test_standard_scaler_sparse_matches_dense_when_full_observed():
    x_dense = np.array([[1.0, 2.0], [3.0, 6.0]])
    x_sparse = sp.csr_matrix(x_dense)

    dense_scaler = MissingAwareStandardScaler().fit(x_dense)
    sparse_scaler = MissingAwareStandardScaler().fit(x_sparse)

    z_dense = dense_scaler.transform(x_dense)
    z_sparse = sparse_scaler.transform(x_sparse)

    assert sp.issparse(z_sparse)
    np.testing.assert_allclose(z_sparse.toarray(), z_dense, rtol=1e-6, atol=1e-6)

    x_roundtrip = sparse_scaler.inverse_transform(z_sparse)
    np.testing.assert_allclose(x_roundtrip.toarray(), x_dense, rtol=1e-6, atol=1e-6)


def test_minmax_scaler_sparse_matches_dense_when_full_observed():
    x_dense = np.array([[1.0, 2.0], [3.0, 6.0]])
    x_sparse = sp.csr_matrix(x_dense)

    dense_scaler = MissingAwareMinMaxScaler().fit(x_dense)
    sparse_scaler = MissingAwareMinMaxScaler().fit(x_sparse)

    z_dense = dense_scaler.transform(x_dense)
    z_sparse = sparse_scaler.transform(x_sparse)

    assert sp.issparse(z_sparse)
    np.testing.assert_allclose(z_sparse.toarray(), z_dense, rtol=1e-6, atol=1e-6)

    x_roundtrip = sparse_scaler.inverse_transform(z_sparse)
    np.testing.assert_allclose(x_roundtrip.toarray(), x_dense, rtol=1e-6, atol=1e-6)


def test_autoencoder_sparse_continuous_supported():
    x_dense = np.array([[1.0, 2.0], [3.0, 6.0]])
    x_sparse = sp.csr_matrix(x_dense)

    dense_enc = AutoEncoder().fit(x_dense)
    sparse_enc = AutoEncoder().fit(x_sparse)

    dense_enc.transform(x_dense)
    z_sparse = sparse_enc.transform(x_sparse)

    assert sp.issparse(z_sparse)


def test_autoencoder_sparse_categorical_supported():
    x_dense = np.array([[1.0, 2.0], [3.0, 6.0]])
    x_sparse = sp.csr_matrix(x_dense)

    dense_enc = AutoEncoder(column_types=["categorical", "categorical"]).fit(x_dense)
    sparse_enc = AutoEncoder(column_types=["categorical", "categorical"]).fit(x_sparse)

    dense_enc.transform(x_dense)
    z_sparse = sparse_enc.transform(x_sparse)

    assert sp.issparse(z_sparse)

    x_roundtrip_sparse = sparse_enc.inverse_transform(z_sparse)
    assert sp.issparse(x_roundtrip_sparse)
    np.testing.assert_allclose(
        x_roundtrip_sparse.toarray(), x_dense, rtol=1e-6, atol=1e-6
    )


def test_sparse_one_hot_roundtrip_preserves_structure():
    rows = np.array([0, 2])
    cols = np.array([0, 0])
    data = np.array([1.0, 2.0])
    x_sparse = sp.csr_matrix((data, (rows, cols)), shape=(3, 1))

    enc = MissingAwareSparseOneHotEncoder().fit(x_sparse)

    z = enc.transform(x_sparse)
    assert sp.issparse(z)
    expected_z = sp.csr_matrix([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=float)
    np.testing.assert_allclose(z.toarray(), expected_z.toarray(), rtol=1e-6, atol=1e-6)

    x_roundtrip = enc.inverse_transform(z)
    assert sp.issparse(x_roundtrip)
    np.testing.assert_allclose(
        x_roundtrip.toarray(), x_sparse.toarray(), rtol=1e-6, atol=1e-6
    )


def test_sparse_one_hot_unknown_category_raises_when_configured():
    x_sparse = sp.csr_matrix(np.array([[1.0], [2.0]], dtype=float))
    enc = MissingAwareSparseOneHotEncoder(handle_unknown="raise").fit(x_sparse)

    x_with_unknown = sp.csr_matrix(np.array([[1.0], [3.0]], dtype=float))
    with pytest.raises(ValueError):
        enc.transform(x_with_unknown)
