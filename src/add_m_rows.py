#Expands or updates the matrix A and its associated covariance Av, handling observed and unobserved rows with variance specifications (Va).
#Input matrix (A), covariance (Av), observed row indices (Ir), total rows (n1x), and optionally variance values (Va).
#Returns updated matrix (A_new) and associated covariance (Av_new).

import numpy as np

def add_m_rows(A, Av, Ir, n1x, Va=None):
    # Determine the number of components (ncomp)
    if A.ndim >= 2:
        ncomp = A.shape[1]
    else:
        ncomp = 0

    # Set default Va if not provided
    if Va is None:
        Va = np.full(ncomp, np.inf)
    else:
        Va = np.asarray(Va)
        if Va.size == 1:
            Va = np.full(ncomp, Va)
        elif Va.size != ncomp:
            raise ValueError("Length of Va must be 1 or equal to the number of components (ncomp).")

    # Compute Ir2 as the set difference between all rows and Ir
    all_rows = set(range(1, n1x + 1))
    Ir_set = set(Ir)
    Ir2 = sorted(list(all_rows - Ir_set))

    # Initialize A_new
    if np.isinf(Va).any():
        A_new = np.full((n1x, ncomp), np.nan)
    else:
        A_new = np.zeros((n1x, ncomp))

    # Convert Ir to 0-based indices for Python
    Ir_zero_based = [i - 1 for i in Ir]

    # Assign observed rows
    if Ir_zero_based and ncomp > 0:
        A_new[np.ix_(Ir_zero_based, range(ncomp))] = A

    # Handle Av_new
    if Av and ncomp > 0:
        if isinstance(Av, list):
            Av_new = [None] * n1x
            # Assign existing Av elements to observed rows
            for j, ir in enumerate(Ir_zero_based):
                Av_new[ir] = Av[j]
            # Assign diagonal matrices with variances Va to unobserved rows
            for ir in Ir2:
                Av_new[ir - 1] = np.diag(Va)
        else:
            if Av.ndim != 2 or Av.shape[1] != ncomp:
                raise ValueError("Av must be a 2D array with shape (n_observed_rows, ncomp).")
            Av = np.asarray(Va)
            Av_new = np.tile(Va, (n1x, 1))
            Av_new[np.ix_(Ir_zero_based, range(ncomp))] = Av
    else:
        Av_new = []

    return A_new, Av_new

