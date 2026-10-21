"""Construct observed zeros explicitly instead of dropping them in conversion."""

import scipy.sparse as sp

x_sparse = sp.csr_matrix(
    ([1.0, 0.0, 0.0, 2.0], ([0, 0, 1, 1], [0, 1, 0, 1])),
    shape=(2, 2),
)
print(x_sparse.nnz)  # 4 stored observations, including both zeros
