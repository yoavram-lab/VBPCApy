import numpy as np

def add_m_cols(S, Sv, Ic, n2x, Isv=None):
    """
    Add unobserved columns to PCA solution.

    Parameters:
    ----------
    S : numpy.ndarray
        The parameter matrix of shape (ncomp, len(Ic)).
    Sv : list or numpy.ndarray
        Either a list (analogous to a cell array in MATLAB) or a 1D array
        representing the covariance information.
    Ic : list or array-like
        List of 1-based column indices where S should be inserted.
    n2x : int
        Total number of columns after addition.
    Isv : list or array-like, optional
        Additional covariance index information.

    Returns:
    -------
    S_new : numpy.ndarray
        The new parameter matrix with shape (ncomp, n2x).
    Sv_new : list or numpy.ndarray
        The updated covariance information.
    Isv_new : list or numpy.ndarray
        The updated covariance index information.
    """
    ncomp = S.shape[0]

    # Compute Ic2 as the set difference between all columns and Ic
    all_columns = set(range(1, n2x + 1))
    Ic_set = set(Ic)
    Ic2 = sorted(list(all_columns - Ic_set))

    # Initialize S_new with zeros and assign S to the specified columns
    S_new = np.zeros((ncomp, n2x))
    # Convert 1-based indices to 0-based for Python
    Ic_zero_based = [i - 1 for i in Ic]
    S_new[:, Ic_zero_based] = S

    # Handle Sv based on its type
    if isinstance(Sv, list):
        if Isv is None or len(Isv) == 0:
            # Initialize Sv_new as a list with n2x elements
            Sv_new = [None] * n2x
            # Assign existing Sv elements to their respective columns
            for j, ic in enumerate(Ic_zero_based):
                Sv_new[ic] = Sv[j]
            # Assign identity matrices to the remaining columns
            for j in Ic2:
                Sv_new[j - 1] = np.eye(ncomp)
            # Isv_new is an empty list
            Isv_new = []
        else:
            # Copy existing Sv and append a new identity matrix
            Sv_new = Sv.copy()
            Sv_new.append(np.eye(ncomp))
            # Initialize Isv_new with the index of the new covariance
            Isv_new = [len(Sv_new)] * n2x
            # Assign existing Isv indices to their respective columns
            for j, ic in enumerate(Ic_zero_based):
                Isv_new[ic] = Isv[j]
    else:
        # Sv is assumed to be a 1D numpy array representing diagonal covariances
        Sv_new = np.ones((ncomp, n2x))
        # Assign the provided Sv to the specified columns
        Sv_new[:, Ic_zero_based] = Sv[:, np.newaxis]

        # If Sv is not a list, Isv_new is not defined in MATLAB either
        Isv_new = None

    return S_new, Sv_new, Isv_new
