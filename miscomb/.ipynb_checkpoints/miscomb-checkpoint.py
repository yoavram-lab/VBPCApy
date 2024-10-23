# %  MISCOMB - Find combinations of missing values in columns
# %
# %  [ nobscomb, obscombj, Isv ] = MISCOMB(M,verbose) computes the
# %  combinations of missing values in each column of a data matrix.
# %  This is needed for faster implementation.
# %
# %  See also PCA_FULL, PCA_PT, LSPCA

# %  This software is provided "as is", without warranty of any kind.
# %  Alexander Ilin, Tapani Raiko

import numpy as np

def miscomb(M, verbose):

    n2 = M.shape[1]

    if verbose:
        print("calculating Isv..")

    tmp, Isv = np.unique(M.T, axis=0, return_inverse=True)
    nobscomb = tmp.shape[0]

    if nobscomb < n2:
        obscombj = [[] for _ in range(nobscomb)]  # Initialize empty lists for each combination
        for i in range(n2):
            obscombj[Isv[i]].append(i)
    else:
        obscombj = []
        Isv = []

    if verbose:
        print('done')
        print(f'Missing values combinations: found {nobscomb} in {n2} columns')
    
            
    
    return nobscomb, obscombj, Isv