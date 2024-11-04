import numpy as np

def add_m_rows(A, Av, Ir, n1x, Va=None):
    """
    Add unobserved rows to PCA solution.

    Parameters:
    ----------
    A : numpy.ndarray
        The parameter matrix of shape (n_observed_rows, ncomp).
    Av : list or numpy.ndarray
        Covariance information. Can be a list (analogous to a cell array in MATLAB)
        or a 2D NumPy array where each row corresponds to covariance data for a row in A.
    Ir : list or array-like
        List of 1-based indices indicating observed rows.
    n1x : int
        Total number of rows after addition.
    Va : array-like or float, optional
        Variance of the prior for A. If not provided, defaults to infinity for all components.

    Returns:
    -------
    A_new : numpy.ndarray
        The updated parameter matrix with shape (n1x, ncomp).
    Av_new : list or numpy.ndarray
        The updated covariance information corresponding to A_new.
    """
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
    all_rows = set(range(1, n1x + 1))  # 1-based indices
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
            # Av is a list (cell array in MATLAB)
            Av_new = [None] * n1x
            # Assign existing Av elements to observed rows
            for j, ir in enumerate(Ir_zero_based):
                Av_new[ir] = Av[j]
            # Assign diagonal matrices with variances Va to unobserved rows
            for ir in Ir2:
                Av_new[ir - 1] = np.diag(Va)
        else:
            # Av is a NumPy array
            if Av.ndim != 2 or Av.shape[1] != ncomp:
                raise ValueError("Av must be a 2D array with shape (n_observed_rows, ncomp).")
            # Va must be a 1D array with length ncomp
            Av = np.asarray(Va)
            Av_new = np.tile(Va, (n1x, 1))
            Av_new[np.ix_(Ir_zero_based, range(ncomp))] = Av
    else:
        Av_new = []

    return A_new, Av_new

