import sys
from pathlib import Path

# Define the path to the directory containing compute_rms.py
path_to_compute_rms = Path("../errpca_pt_compute_rms")

# Add this path to sys.path
sys.path.append(str(path_to_compute_rms))

import numpy as np
from numpy.linalg import slogdet
from scipy.sparse import issparse, csr_matrix
from compute_rms import compute_rms  # Ensure this module is available

def cf_full(X, A, S, Mu, V, Av=None, Sv=None, Isv=None, Muv=None, Va=None, Vmu=None, M=None, sXv=None, ndata=None):
    n1, n2 = X.shape
    ncomp = A.shape[1]

    # Handle M, sXv, ndata if not provided
    if M is None:
        M = ~np.isnan(X)
        X = np.where(M, X, 0)

    if sXv is None:
        IX, JX = np.where(M)
        ndata = len(IX)

        # Call compute_rms and capture results
        rms, errMx = compute_rms(X, A, S, M, ndata)

        # Initialize sXv based on rms
        sXv = (rms ** 2) * ndata
        if Sv is None:
            for r in range(ndata):
                a_IXr = A[IX[r], :]
                sv_JXr = np.eye(ncomp)
                sXv += a_IXr @ sv_JXr @ a_IXr.T
                if Av is not None:
                    sXv += S[:, JX[r]].T @ Av[IX[r]] @ S[:, JX[r]] + np.sum(Sv[JX[r]] * Av[IX[r]])
        else:
            for r in range(ndata):
                a_IXr = A[IX[r], :]
                sv_IJXr = Sv[Isv[JX[r]]] if Isv else Sv[JX[r]]
                sXv += a_IXr @ sv_IJXr @ a_IXr.T
                if Av is not None:
                    sXv += S[:, JX[r]].T @ Av[IX[r]] @ S[:, JX[r]] + np.sum(Sv[JX[r]] * Av[IX[r]])

        if Muv:
            sXv += np.sum(Muv[IX])

    # Determine whether to use priors based on Va
    use_prior = Va is not None and not np.any(np.isinf(Va))

    cost_x = 0.5 / V * sXv + 0.5 * ndata * np.log(2 * np.pi * V)
    cost_mu = 0.0
    cost_a = 0.0

    if use_prior:
        if Muv:
            cost_mu = 0.5 / Vmu * np.sum(Mu ** 2 + Muv) - 0.5 * np.sum(np.log(Muv)) + (n1 / 2) * np.log(Vmu) - (n1 / 2)
        elif Vmu != 0:
            cost_mu = 0.5 / Vmu * np.sum(Mu ** 2) + (n1 / 2) * np.log(2 * np.pi * Vmu)

        if Av:
            cost_a = 0.5*np.sum(np.sum(A @ A, axis = 0) / Va) + (n1 / 2) * np.sum(np.log(Va), axis = 0) - (n1 * ncomp / 2)
            print("cost_a after initial assignment:", cost_a)
            for i in range(n1):
                trace_term = 0.5 * np.sum(np.diag(Av[i]) / Va)
                sign, logdet = slogdet(Av[i])
                cost_a += trace_term - 0.5 * logdet if sign > 0 else trace_term - 0.5 * (-np.inf)
                print(f"cost_a after updating with trace and determinant of Av[{i}]:", cost_a)
        else:
            cost_a = 0.5 * np.sum(A ** 2 / Va) + (n1 / 2) * np.sum(np.log(2 * np.pi * Va))
            print("cost_a after assignment with Va:", cost_a)
    else:
        if Muv:
            cost_mu = -0.5 * np.sum(np.log(2 * np.pi * Muv)) - (n1 / 2)
        if Av:
            cost_a = - (n1 * ncomp / 2) * (1 + np.log(2 * np.pi))
            print("cost_a after no-prior initial assignment:", cost_a)
            for i in range(n1):
                sign, logdet = slogdet(Av[i])
                cost_a -= 0.5 * logdet if sign > 0 else 0.5 * (-np.inf)
                print(f"cost_a after updating with determinant of Av[{i}]:", cost_a)

    cost_s = 0.5 * np.sum(S ** 2)
    if Sv:
        if Isv:
            for j in range(n2):
                sv_idx = Isv[j]
                trace_svj = 0.5 * np.trace(Sv[sv_idx])
                sign, logdet_svj = slogdet(Sv[sv_idx])
                cost_s += trace_svj - 0.5 * logdet_svj if sign > 0 else trace_svj - 0.5 * (-np.inf)
        else:
            for j in range(n2):
                trace_svj = 0.5 * np.trace(Sv[j])
                sign, logdet_svj = slogdet(Sv[j])
                cost_s += trace_svj - 0.5 * logdet_svj if sign > 0 else trace_svj - 0.5 * (-np.inf)
    cost_s -= (ncomp * n2) / 2
    cost = cost_mu + cost_a + cost_x + cost_s


    return cost, cost_x, cost_a, cost_mu, cost_s
