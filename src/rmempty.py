#Removes empty rows and columns (entirely zeros or NaNs) from matrices and updates related initialization structures.
#Gets input matrix (X), optional matrix (Xprobe), initialization dictionary (init), and verbosity level (verbose).
#Returns the cleaned matrices (X, Xprobe), indices of non-empty rows (Ir) and columns (Ic), and the updated initialization dictionary (init).

import numpy as np
from scipy import sparse

def rmempty(flag, X, Xprobe, init, verbose):

    if verbose == 2:
        print("Checking for empty rows or columns...")

    n1x, n2x = X.shape
    if sparse.issparse(X):
        Ic = np.where(np.array(X.sum(axis=0)).flatten() != 0)[0]
        Ir = np.where(np.array(X.sum(axis=1)).flatten() != 0)[0]

    else:
        Ic = np.where(np.sum(~np.isnan(X), axis=0) > 0)[0]
        Ir = np.where(np.sum(~np.isnan(X), axis=1) > 0)[0]

    n1 = len(Ir)
    n2 = len(Ic)

    if n1 == n1x and n2 == n2x:
        if verbose:
            print("No empty rows or columns")
        return X, Xprobe, Ir, Ic, init

    if n1 < n1x and n2 < n2x:
        X = X[np.ix_(Ir, Ic)]
        if Xprobe is not None:
            Xprobe = Xprobe[np.ix_(Ir, Ic)]

    elif n1 < n1x:
        X = X[Ir, :]
        if Xprobe is not None:
            Xprobe = Xprobe[Ir, :]

    elif n2 < n2x:
        X = X[:, Ic]
        if Xprobe is not None:
            Xprobe = Xprobe[:, Ic]

    else:
        Ir = []
        Ic = []

    if verbose:
        print(f"{n1x - n1} empty rows and {n2x - n2} empty columns removed")

    if not isinstance(init, dict):
        return X, Xprobe, Ir, Ic, init

    # Update init['A'] along rows only
    if n1 < n1x:
        if "A" in init:
            init["A"] = init["A"][Ir, :]
        if "Av" in init and len(init["Av"]) > 0:
            if isinstance(init["Av"], list):
                init["Av"] = [init["Av"][i] for i in Ir]
            else:
                init["Av"] = init["Av"][Ir, :]

    # Update init['S'] along columns only
    if n2 < n2x:
        if "S" in init:
            init["S"] = init["S"][:, Ic]
        if "Sv" in init and len(init["Sv"]) > 0:
            if isinstance(init["Sv"], list):
                init["Sv"] = [init["Sv"][j] for j in Ic]
            else:
                init["Sv"] = init["Sv"][:, Ic]

    return X, Xprobe, Ir, Ic, init
