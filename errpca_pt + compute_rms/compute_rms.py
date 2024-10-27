import numpy as np
import scipy.sparse as sp
from errpca_pt import errpca_pt
def compute_rms(X, A, S, M, ndata, numCPU=1):
    """
    Computes the RMS error and the reconstruction error matrix.
    """
    if X.size == 0:
        return np.nan, []  # Equivalent to MATLAB's empty array

    if sp.issparse(X):
        # Handle sparse matrices
        # Extract components of the sparse matrix X
        X_data = X.data.astype(np.float64)
        X_indices = X.indices.astype(np.int32)
        X_indptr = X.indptr.astype(np.int32)

        # Ensure A and S are float64 arrays
        A = np.array(A, dtype=np.float64)
        S = np.array(S, dtype=np.float64)

        # Call errpca_pt with the correct arguments
        errMx_dict = errpca_pt(X_data, X_indices, X_indptr, A, S, numCPU)

        # Reconstruct errMx from the returned dictionary
        errMx_data = errMx_dict['data']
        errMx_indices = errMx_dict['indices']
        errMx_indptr = errMx_dict['indptr']
        errMx_shape = errMx_dict['shape']
        errMx = sp.csr_matrix((errMx_data, errMx_indices, errMx_indptr), shape=errMx_shape)

    else:
        # Handle dense matrices
        residual = X - A @ S
        if sp.issparse(M):
            M = M.toarray()
        errMx = np.multiply(residual, M)  # Element-wise multiplication

    # Compute RMS
    if sp.issparse(errMx):
        rms = np.sqrt((errMx.data ** 2).sum() / ndata)
    else:
        rms = np.sqrt(np.sum(errMx ** 2) / ndata)

    return rms, errMx
