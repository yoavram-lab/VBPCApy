#expands or updates matrices S and Sv and optionally updates covariance indices (Isv) for a new dimensionality (n2x).
#Input matrices (S, Sv), selected column indices (Ic), total columns (n2x), and optionally covariance indices (Isv).
#returns: Updated matrices (S_new, Sv_new) and optionally updated covariance indices (Isv_new).

import numpy as np

def add_m_cols(S, Sv, Ic, n2x, Isv=None):
    print("in add m cols", flush=True)
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
            Isv_new = []
        else:
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

        Isv_new = None

    return S_new, Sv_new, Isv_new
