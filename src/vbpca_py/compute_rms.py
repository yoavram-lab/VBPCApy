#Computes the root mean square (RMS) error and the error matrix for a probabilistic PCA model, handling both dense and sparse input matrices.
#Gets observed data matrix (X), factor matrices (A, S), mask for missing data (M), number of data points (ndata), and optional CPU count (numCPU).
#Returns the RMS error (rms) and the error matrix (errMx), which is sparse if the input is sparse.

import numpy as np
import scipy.sparse as sp
from errpca_pt import errpca_pt

def compute_rms(X, A, S, M, ndata, numCPU=1):
    print("in compute rms", flush=True)
    if X.size == 0:
        return np.nan, []  # Equivalent to MATLAB's empty array
    
    if sp.issparse(X):
        # Prepare arguments for errpca_pt
        X_data = X.data.astype(np.float64)
        X_indices = X.indices.astype(np.int32)
        X_indptr = X.indptr.astype(np.int32)
        A = np.array(A, dtype=np.float64)
        S = np.array(S, dtype=np.float64)

        errMx_dict = errpca_pt(X_data, X_indices, X_indptr, A, S, numCPU)

        # Reconstruct errMx from the dictionary
        errMx = sp.csr_matrix(
            (errMx_dict['data'], errMx_dict['indices'], errMx_dict['indptr']),
            shape=errMx_dict['shape']
        )
    else:
        residual = X - A @ S
        if sp.issparse(M):
            M = M.toarray()
        errMx = np.multiply(residual, M)

    # Compute RMS
    rms = np.sqrt((errMx.data ** 2).sum() / ndata) if sp.issparse(errMx) else np.sqrt(np.sum(errMx ** 2) / ndata)
    return rms, errMx
