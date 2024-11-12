import numpy as np
import scipy.sparse as sp
from errpca_pt import errpca_pt

def compute_rms(X, A, S, M, ndata, numCPU=1):
    if X.size == 0:
        return np.nan, []  # Equivalent to MATLAB's empty array
    
    print("befor is sparse")
    if sp.issparse(X):
        # Prepare arguments for errpca_pt
        X_data = X.data.astype(np.float64)
        X_indices = X.indices.astype(np.int32)
        X_indptr = X.indptr.astype(np.int32)
        A = np.array(A, dtype=np.float64)
        S = np.array(S, dtype=np.float64)
        print("inside is sparse")
        # Print arguments before calling errpca_pt
        print("Arguments sent to errpca_pt:")
        print("X_data:", X_data)
        print("X_indices:", X_indices)
        print("X_indptr:", X_indptr)
        print("A:", A)
        print("S:", S)
        print("numCPU:", numCPU)

        # Call errpca_pt and get result
        errMx_dict = errpca_pt(X_data, X_indices, X_indptr, A, S, numCPU)

        # Print returned data from errpca_pt
        print("Returned from errpca_pt:")
        print("errMx data:", errMx_dict['data'])
        print("errMx indices:", errMx_dict['indices'])
        print("errMx indptr:", errMx_dict['indptr'])
        print("errMx shape:", errMx_dict['shape'])

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
