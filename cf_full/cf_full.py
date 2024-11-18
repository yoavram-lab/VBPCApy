import sys
from pathlib import Path

# Define the path to the directory containing compute_rms.py
path_to_compute_rms = Path("../errpca_pt_compute_rms")

# Add this path to sys.path
sys.path.append(str(path_to_compute_rms))

import numpy as np
from numpy.linalg import slogdet
from scipy.sparse import issparse, csr_matrix
# from compute_rms import compute_rms  # Ensure this module is available or replace with appropriate code

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

        # Since compute_rms is not provided, we'll compute rms as an example
        X_hat = A @ S + Mu[:, np.newaxis]
        diff = (X - X_hat) * M
        rms = np.sqrt(np.sum(diff ** 2) / ndata)

        # Initialize sXv based on rms
        sXv = (rms ** 2) * ndata
        if Sv is None:
            for r in range(ndata):
                i = IX[r]
                j = JX[r]
                a_i = A[i, :]
                sv_j = np.eye(ncomp)
                sXv += a_i @ sv_j @ a_i.T
                if Av is not None and len(Av) > i:
                    s_j = S[:, j]
                    av_i = Av[i]
                    sXv += s_j.T @ av_i @ s_j + np.sum(sv_j * av_i)
        else:
            for r in range(ndata):
                i = IX[r]
                j = JX[r]
                a_i = A[i, :]
                sv_j = Sv[Isv[j]] if Isv is not None and len(Isv) > j else Sv[j]
                sXv += a_i @ sv_j @ a_i.T
                if Av is not None and len(Av) > i:
                    s_j = S[:, j]
                    av_i = Av[i]
                    sXv += s_j.T @ av_i @ s_j + np.sum(sv_j * av_i)

        if Muv is not None and Muv.size > 0:
            sXv += np.sum(Muv[IX])

    # Determine whether to use priors based on Va
    use_prior = Va is not None and not np.any(np.isinf(Va))

    cost_x = 0.5 / V * sXv + 0.5 * ndata * np.log(2 * np.pi * V)
    cost_mu = 0.0
    cost_a = 0.0

    if use_prior:
        if Muv is not None and Muv.size > 0:
            cost_mu = 0.5 / Vmu * np.sum(Mu ** 2 + Muv) - 0.5 * np.sum(np.log(Muv)) + (n1 / 2) * np.log(Vmu) - (n1 / 2)
        elif Vmu != 0:
            cost_mu = 0.5 / Vmu * np.sum(Mu ** 2) + (n1 / 2) * np.log(2 * np.pi * Vmu)

        if Av is not None and len(Av) > 0:
            var1 = np.sum(np.sum(A ** 2, axis=0) / Va)
            var2 = (n1 / 2) * np.sum(np.log(Va), axis=0)
            var3 = (n1 * ncomp / 2)
    

            cost_a = 0.5 * np.sum(np.sum(A ** 2, axis=0) / Va) + (n1 / 2) * np.sum(np.log(Va), axis=0) - (n1 * ncomp / 2)
            for i in range(n1):
                trace_term = 0.5 * np.sum(np.diag(Av[i]) / Va)
                sign, logdet = slogdet(Av[i])
                if sign > 0:
                    cost_a += trace_term - 0.5 * logdet
                else:
                    cost_a += trace_term - 0.5 * (-np.inf)
        else:
            cost_a = 0.5 * np.sum(A ** 2 / Va) + (n1 / 2) * np.sum(np.log(2 * np.pi * Va))
    else:
        if Muv is not None and Muv.size > 0:
            cost_mu = -0.5 * np.sum(np.log(2 * np.pi * Muv)) - (n1 / 2)
        if Av is not None and len(Av) > 0:
            cost_a = - (n1 * ncomp / 2) * (1 + np.log(2 * np.pi))
            for i in range(n1):
                sign, logdet = slogdet(Av[i])
                if sign > 0:
                    cost_a -= 0.5 * logdet
                else:
                    cost_a -= 0.5 * (-np.inf)

    cost_s = 0.5 * np.sum(S ** 2)
    if Sv is not None and len(Sv) > 0:
        if Isv is not None and len(Isv) > 0:
            for j in range(n2):
                sv_idx = Isv[j]
                if Sv[sv_idx] is not None:
                    trace_svj = 0.5 * np.trace(Sv[sv_idx])
                    sign, logdet_svj = slogdet(Sv[sv_idx])
                    if sign > 0:
                        cost_s += trace_svj - 0.5 * logdet_svj
                    else:
                        cost_s += trace_svj - 0.5 * (-np.inf)
        else:
            for j in range(n2):
                if Sv[j] is not None:
                    trace_svj = 0.5 * np.trace(Sv[j])
                    sign, logdet_svj = slogdet(Sv[j])
                    if sign > 0:
                        cost_s += trace_svj - 0.5 * logdet_svj
                    else:
                        cost_s += trace_svj - 0.5 * (-np.inf)
    cost_s -= (ncomp * n2) / 2
    cost = cost_mu + cost_a + cost_x + cost_s

    return cost, cost_x, cost_a, cost_mu, cost_s

