import numpy as np
import scipy.sparse as sp

def compute_rms(X, A, S, M, ndata, numCPU=1):
    """
    Computes the RMS error and the reconstruction error matrix.

    Parameters:
    - X (numpy.ndarray or scipy.sparse matrix): Original data matrix.
    - A (numpy.ndarray): Basis matrix.
    - S (numpy.ndarray): Coefficient matrix.
    - M (numpy.ndarray or scipy.sparse matrix): Mask matrix.
    - ndata (int): Number of data points.
    - numCPU (int, optional): Number of CPUs for parallel computation (default is 1).

    Returns:
    - rms (float): Root Mean Square error.
    - errMx (numpy.ndarray or scipy.sparse matrix): Reconstruction error matrix.
    """

    EPS = 1e-15  # Small epsilon to replace exact zeros in sparse matrices

    if X.size == 0:
        return np.nan, []  # Equivalent to MATLAB's empty array

    if sp.issparse(X):
        # Handle sparse matrices
        errMx = errpca_pt(X, A, S, numCPU)  # Placeholder for the C++ function
    else:
        # Handle dense matrices
        errMx = (X - A @ S) * M  # Element-wise multiplication

    # Compute RMS
    if sp.issparse(errMx):
        # For sparse matrices, compute RMS only on non-zero elements
        rms = np.sqrt((errMx.data ** 2).sum() / ndata)
    else:
        # For dense matrices, compute RMS over all elements
        rms = np.sqrt(np.sum(errMx ** 2) / ndata)

    return rms, errMx
