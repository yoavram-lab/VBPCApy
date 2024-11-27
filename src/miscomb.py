#Identifies unique patterns of missing values in the columns of a matrix and groups columns with the same pattern.
#Gets a binary mask matrix (M) indicating observed/missing values and a verbosity flag (verbose).
#Returns the number of unique missing value patterns (nobscomb), a list of column indices for each pattern (obscombj), and an array mapping columns to patterns (Isv).

import numpy as np

def miscomb(M, verbose):

    n2 = M.shape[1]

    if verbose:
        print("calculating Isv..")

    tmp, Isv = np.unique(M.T, axis=0, return_inverse=True)
    nobscomb = tmp.shape[0]

    if nobscomb < n2:
        obscombj = [[] for _ in range(nobscomb)]
        for i in range(n2):
            obscombj[Isv[i]].append(i)
    else:
        obscombj = []
        Isv = []

    if verbose:
        print('done')
        print(f'Missing values combinations: found {nobscomb} in {n2} columns')
    
            
    
    return nobscomb, obscombj, Isv