#Performs probabilistic PCA (principal component analysis) on input data with missing values, supporting various algorithms and priors.
#Gets input data matrix (X), number of components (ncomp), and optional keyword arguments for configurations and options.
#Returns a dictionary containing trained parameters (`A`, `S`, `Mu`, `V`, etc.), convergence logs (`lc`), and other auxiliary results.

import pdb
import pandas as pd
import numpy as np
import scipy.sparse as sp 
from scipy.io import savemat
import scipy.linalg
import scipy.io
from scipy.sparse import issparse, csr_matrix
import numpy as np
from scipy.io import loadmat
from scipy.linalg import orth  
import matplotlib.pyplot as plt
import time
from scipy.linalg import subspace_angles
import sys

from add_m_cols import add_m_cols 
from add_m_rows import add_m_rows 
from argschk import argschk
from cf_full import cf_full
from converg_check import converg_check
from compute_rms import compute_rms 
from miscomb import miscomb
from rmempty import rmempty
from subtract_mu_from_sparse import subtract_mu_from_sparse

def pca_full(X, ncomp, **kwargs):
    print("in pca full", flush=True)
    opts = { 'init':'random',
    'maxiters':1000,
    'bias':1,
    'uniquesv':0,
    'autosave':600,
    'filename':'pca_f_autosave',
    'minangle':1e-8,
    'algorithm':'vb',
    'niter_broadprior':100,
    'earlystop':0,
    'rmsstop':np.array([100, 1e-4, 1e-3]),
    'cfstop':np.array([]), # 
    'verbose':1,
    'xprobe':None,
    'rotate2pca':1,
    'display':0 }

    opts, wrnmsg = argschk(opts, **kwargs)
    
    if wrnmsg:
        print(f"warning: {wrnmsg}")

    algorithmVal = opts["algorithm"]
    if (algorithmVal == "ppca"):
        use_prior = False
        use_postvar = False
    elif (algorithmVal == "map"):
        use_prior = True
        use_postvar = False
    elif (algorithmVal == "vb"):
        use_prior = True
        use_postvar = True
    else:
        raise ValueError(f"Wrong value if the argument 'algorithm': {algorithmVal}")

    Xprobe = opts['xprobe']
    
    n1x, n2x = X.shape
    X, Xprobe, Ir, Ic, opts['init'] = rmempty(X, Xprobe, opts['init'], opts['verbose'])

    n1, n2 = X.shape
    [n1x,n2x] = X.shape

    # Handle the case where X is sparse
    if issparse(X):
        M = (X != 0).astype(float)
        if Xprobe is not None:
            Mprobe = (Xprobe != 0).astype(float)
        else:
            Mprobe = None
            
    else:
        M = ~np.isnan(X)
        if Xprobe is not None:
            Mprobe = ~np.isnan(Xprobe)
        else:
            Mprobe = None

        X[X == 0] = np.finfo(float).eps  # Replace zeros with a small number
        
        if Xprobe is not None:
            Xprobe[Xprobe == 0] = np.finfo(float).eps
            Xprobe[np.isnan(Xprobe)] = 0
            
        X[np.isnan(X)] = 0  # Replace NaNs with 0
        
    
    # Compute the number of observed (non-missing) values in each row of X
    Nobs_i = np.sum(M, axis=1)
    ndata = np.sum(Nobs_i)
    
    # Compute the number of observed values in the probe data (Xprobe)
    if issparse(Mprobe):
        nprobe = Mprobe.count_nonzero() 
    else:
        nprobe = np.count_nonzero(Mprobe)
    # If no observed values in Xprobe, set it to empty and disable early stopping
    if nprobe == 0:
        Xprobe = np.array([])
        opts['earlystop'] = 0

    IX, JX = np.nonzero(X)

    # Compute indices Isv: Sv[Isv[j]] gives Sv for j, j=1...n2
    if opts['uniquesv']:
        nobscomb, obscombj, Isv = miscomb(M, opts['verbose'])
    else:
        nobscomb = n2
        Isv = []
        obscombj = {}

    A, S, Mu, V, Av, Sv, Muv = init_parms(opts["init"], n1, n2, ncomp, nobscomb, Isv)

    if use_prior:
        Va = 1000 * np.ones((1, ncomp))
        Vmu = 1000
    else:
        Va = np.full(ncomp, np.inf)
        Vmu = np.inf

    if not use_postvar:
        Muv = np.array([])
        Av = []

    if not opts['bias']:
        Muv = np.array([])
        Vmu = 0
    
    if np.size(Mu) == 0:
        if opts['bias']:
            Mu = (np.sum(X, axis=1) / Nobs_i).reshape(-1, 1)  # Shape: (n1, 1)
        else:
            Mu = np.zeros((n1, 1))


    X, Xprobe = subtract_mu(Mu, X, M, Xprobe, Mprobe, opts['bias'])

    rms, errMx = compute_rms(X, A, S, M, ndata)
    prms, _ = compute_rms(Xprobe, A, S, Mprobe, nprobe)

    
    lc = {
        'rms': [rms],
        'prms': [prms],
        'time': [0],
        'cost': [np.nan]
    }

    dsph = display_init(opts['display'], lc)
    print_first_step(opts['verbose'], rms, prms)
    Aold = A.copy()

    # Parameters of the prior for variance parameters
    hpVa = 0.001
    hpVb = 0.001
    hpV = 0.001

    time_start = time.time()
    time_autosave = time_start
    tic = time.time()

    total = max(range(1, opts['maxiters'] + 1));
    for iter in range(1, opts['maxiters'] + 1):
        print(f"iteration {iter} out of {total}")
        if use_prior and iter > opts['niter_broadprior']:
            # Update Va and Vmu
            if opts['bias']:
                Vmu = np.sum(Mu ** 2)
                if Muv.size > 0:
                    Vmu = Vmu + np.sum(Muv)
                Vmu = (Vmu + 2 * hpVa) / (n1 + 2 * hpVb)
            Va = np.sum(A ** 2, axis=0)
            if Av:
                for i in range(n1):
                    Va = Va + np.diag(Av[i])
            Va = (Va + 2 * hpVa) / (n1 + 2 * hpVb)

        if opts['bias']:
            dMu = np.sum(errMx, axis=1) / Nobs_i  # Shape: (n1,)
            if Muv.size > 0:
                Muv = V / (Nobs_i + V / Vmu)  # Shape: (n1,)
            th = 1 / (1 + V / (Nobs_i * Vmu))  # Shape: (n1,)
            th = th.reshape(-1, 1)  # Shape: (n1, 1)
            
            # Reshape dMu to (n1, 1)
            dMu = dMu.reshape(-1, 1)  # Shape: (n1, 1)
            
            Mu_old = Mu.copy()
            Mu = th * (Mu + dMu)
            dMu = Mu - Mu_old  # Shape: (n1,1)
            X, Xprobe = subtract_mu(dMu, X, M, Xprobe, Mprobe, update_bias=True)

        # Update S
        print("update S", flush=True)
        if not Isv:
            for j in range(n2):
                # print(j, "out of ", range(n2), flush=True)
                A_j = (M[:, j][:, np.newaxis]) * A
                Psi = A_j.T @ A_j + np.diag(np.full(ncomp, V))
                if Av:
                    for i in np.where(M[:, j])[0]:
                        Psi = Psi + Av[i]
                invPsi = np.linalg.inv(Psi)
                S[:, j] = invPsi @ A_j.T @ X[:, j]
                Sv[j] = V * invPsi
                print_progress(opts['verbose'], j + 1, n2, 'Updating S:')
        else:
            for k in range(nobscomb):
                j = obscombj[k][0]
                A_j = (M[:, j][:, np.newaxis]) * A
                Psi = A_j.T @ A_j + np.diag(np.full(ncomp, V))
                if Av:
                    for i in np.where(M[:, j])[0]:
                        Psi = Psi + Av[i]
                invPsi = np.linalg.inv(Psi)
                Sv[k] = V * invPsi
                tmp = invPsi @ A_j.T
                for j_idx in obscombj[k]:
                    S[:, j_idx] = tmp @ X[:, j_idx]
                    
                print_progress(opts['verbose'], k + 1, nobscomb, 'Updating S:')
        
        if opts['verbose'] == 2:
            print('\r', end='')
        
        if opts['rotate2pca']:
            dMu, A, Av, S, Sv = rotate_to_pca(A, Av, S, Sv, Isv, obscombj, opts['bias'])
            if opts['bias']:
                X, Xprobe = subtract_mu(dMu, X, M, Xprobe, Mprobe, update_bias=True)
                Mu = Mu + dMu

        # Update A
        print("Update A", flush=True)
        if opts['verbose'] == 2:
            print('\r', end='')
        for i in range(n1):
            S_i = (M[i, :][np.newaxis, :]) * S
            Phi = S_i @ S_i.T + np.diag(V / Va)
            for j_idx in np.where(M[i, :])[0]:
                if Isv:
                    Phi = Phi + Sv[Isv[j_idx]]
                else:
                    Phi = Phi + Sv[j_idx]
            invPhi = np.linalg.inv(Phi)
            A[i, :] = X[i, :] @ S_i.T @ invPhi

            if Av:
                Av[i] = V * invPhi

            print_progress(opts['verbose'], i + 1, n1, 'Updating A:')
        if opts['verbose'] == 2:
            print('\r', end='')
    
        rms, errMx = compute_rms(X, A, S, M, ndata)
        prms = compute_rms(Xprobe, A, S, Mprobe, nprobe) if nprobe > 0 else np.nan


        # Update V
        print("Update V", flush=True)
        sXv = 0
        if not Isv:
            # print("1", flush=True)

            for r in range(ndata):
                i = IX[r]
                j = JX[r]
                a_i = A[i, :].reshape(1, -1)  # Shape (1, ncomp)
                sXv += (a_i @ Sv[j] @ a_i.T).item()
                if Av:
                    s_j = S[:, j].reshape(-1, 1)  # Shape (ncomp, 1)
                    sXv += (s_j.T @ Av[i] @ s_j).item() + np.sum(Sv[j] * Av[i])
        else:
            for r in range(ndata):
                i = IX[r]
                j = JX[r]
                a_i = A[i, :].reshape(1, -1)  # Shape (1, ncomp)
                sXv += (a_i @ Sv[Isv[j]] @ a_i.T).item()
                if Av:
                    s_j = S[:, j].reshape(-1, 1)  # Shape (ncomp, 1)
                    sXv += (s_j.T @ Av[i] @ s_j).item() + np.sum(Sv[Isv[j]] * Av[i])
        
        if Muv.size > 0:
            sXv += np.sum(Muv[IX])

        sXv = sXv + (rms ** 2) * ndata

        V = (sXv + 2 * hpV) / (ndata + 2 * hpV)

        t = time.time() - tic
        lc['rms'].append(rms)
        lc['prms'].append(prms)
        lc['time'].append(t)

        if np.size(opts['cfstop']) > 0:
            cost, cost_x, cost_a, cost_mu, cost_s = cf_full(X, A, S, Mu, V, Av, Sv, Isv, Muv, Va, Vmu, M, sXv, ndata)
            lc['cost'].append(cost)

        display_progress(dsph, lc)
        angles = subspace_angles(A, Aold)
        angleA = np.max(angles)
        print_step(opts['verbose'], lc, angleA)

        convmsg = converg_check(opts, lc, angleA)
        if convmsg:
            if use_prior and iter <= opts['niter_broadprior']:
                pass
            elif opts['verbose']:
                print(f'{convmsg}')
            break
        Aold = A.copy()

        current_time = time.time()
        if (current_time - time_autosave) > opts['autosave']:
            time_autosave = current_time
            if opts['verbose'] == 2:
                print('Saving ... ', end='')
            try:
                    savemat(opts['filename'], {
                        'A': A,
                        'S': S,
                        'Mu': Mu,
                        'V': V,
                        'Av': Av,
                        'Muv': Muv,
                        'Sv': Sv,
                        'Isv': Isv,
                        'Va': Va,
                        'Vmu': Vmu,
                        'lc': lc,
                        'Ir': Ir,
                        'Ic': Ic,
                        'n1x': n1x,
                        'n2x': n2x,
                        'n1': n1,
                        'n2': n2
                    })
                    if opts['verbose'] == 2:
                        print('done')
            except Exception as e:
                if opts['verbose']:
                    print(f"Error saving to {opts['filename']}: {e}")


    # Finally rotate to the PCA solution
    if not opts['rotate2pca']:
        dMu, A, Av, S, Sv = rotate_to_pca(A, Av, S, Sv, Isv, obscombj, opts['bias'])
        if opts['bias']:
            Mu = Mu + dMu

    if n1 < n1x:
        A, Av = add_m_rows(A, Av, Ir, n1x, Va)
        Mu, Muv = add_m_rows(Mu, Muv, Ir, n1x, Vmu)
    if n2 < n2x:
        S, Sv, Isv = add_m_cols(S, Sv, Ic, n2x, Isv)

    result = {
        'A': A,
        'S': S,
        'Mu': Mu,
        'V': V,
        'Av': Av,
        'Sv': Sv,
        'Isv': Isv,
        'Muv': Muv,
        'Va': Va,
        'Vmu': Vmu,
        'lc': lc,
        'cv': {
            'A': Av,
            'S': Sv,
            'Isv': Isv,
            'Mu': Muv
        },
        'hp': {
            'Va': Va,
            'Vmu': Vmu
        }
    }


    return result

##############################################################################################
#Subtracts the row-wise mean (`Mu`) from a data matrix (`X`) and optionally from a probe matrix (`Xprobe`), while handling sparse matrices efficiently.
#Gets a mean vector (`Mu`), data matrix (`X`), missing data mask (`M`), optional probe matrix (`Xprobe`), probe mask (`Mprobe`), and a bias update flag (`update_bias`).
#Returns the modified data matrix (`X`) and probe matrix (`Xprobe`) with the mean subtracted.

def subtract_mu(Mu, X, M, Xprobe=None, Mprobe=None, update_bias=True):
    print("in subtract mu", flush=True)
    n2 = X.shape[1]

    if not update_bias:
        return X, Xprobe

    if sp.isspmatrix(X):
        # Handle sparse case
        data = X.data
        indices = X.indices
        indptr = X.indptr
        shape = X.shape

        X_data = subtract_mu_from_sparse(data, indices, indptr, shape, Mu)

        X = sp.csr_matrix((X_data, indices, indptr), shape=shape)

        if Xprobe is not None and Xprobe.size > 0:
            data_probe = Xprobe.data
            indices_probe = Xprobe.indices
            indptr_probe = Xprobe.indptr
            shape_probe = Xprobe.shape

            Xprobe_data = subtract_mu_from_sparse(data_probe, indices_probe, indptr_probe, shape_probe, Mu)
            Xprobe = sp.csr_matrix((Xprobe_data, indices_probe, indptr_probe), shape=shape_probe)
    else:
        X = X - (Mu * M.astype(int)) 
        if Xprobe is not None and Xprobe.size > 0 and Mprobe is not None:
            Xprobe = Xprobe - (Mu * Mprobe)  # Similarly handled

    return X, Xprobe

##############################################################################################
#Aligns factor matrices (`A` and `S`) to the PCA solution by performing eigen decomposition on covariance matrices and applying rotations, optionally adjusting bias.
#Gets loading matrix (`A`), loading covariance (`Av`), component matrix (`S`), component covariance (`Sv`), indices of covariance matrices (`Isv`), observation combinations (`obscombj`), and a bias update flag (`update_bias`).
#Returns the adjusted bias (`dMu`), rotated loading matrix (`A`), rotated loading covariance (`Av`), rotated component matrix (`S`), and rotated component covariance (`Sv`).

def rotate_to_pca(A, Av, S, Sv, Isv, obscombj, update_bias):
    print("in rotate to pca", flush=True)
    n1 = A.shape[0]
    n2 = S.shape[1]

    if update_bias:
        mS = np.mean(S, axis=1, keepdims=True)
        dMu = A @ mS
        S = S - mS
    else:
        dMu = 0

    covS = S @ S.T
    
    if len(Isv) == 0:
        for j in range(n2):
            covS += Sv[j]
    else:
        nobscomb = len(obscombj)
        for j in range(nobscomb):
            covS += len(obscombj[j]) * Sv[j]

    covS /= n2

    eigvals, VS = np.linalg.eigh(covS)

    # Construct D and RA as in MATLAB
    D = np.diag(eigvals)
    sqrt_D = np.sqrt(D)  # Square root of diagonal elements
    
    RA = VS @ sqrt_D
    A = A @ RA

    covA = A.T @ A
    
    if Av:
        for i in range(n1):
            Av[i] = RA.T @ Av[i] @ RA
            covA += Av[i]

    covA /= n1

    eigvals, VA = np.linalg.eigh(covA)

    I = np.argsort(-eigvals)
    eigvals = eigvals[I]
    VA = VA[:, I]

    A = A @ VA

    if Av:
        for i in range(n1):
            Av[i] = VA.T @ Av[i] @ VA

    # Adjust R calculation using D_inv
    epsilon = 1e-10
    D_diag = np.sqrt(np.diag(D))
    D_inv = np.diag(np.where(D_diag > epsilon, 1 / D_diag, 0)) 
    R = VA.T @ D_inv @ VS.T

    S = R @ S
    
    for j in range(len(Sv)):
        Sv[j] = R @ Sv[j] @ R.T

    return dMu, A, Av, S, Sv

##############################################################################################
#Initializes the parameters (`A`, `S`, `Mu`, etc.) for probabilistic PCA based on a provided initialization method or file, ensuring defaults for missing values.
#Gets the initialization method or file (`init`), dimensions of the data (`n1`, `n2`), number of components (`ncomp`), number of observation combinations (`nobscomb`), and indices of covariance matrices (`Isv`).
#Returns initialized parameters: loading matrix (`A`), component matrix (`S`), mean vector (`Mu`), variance scalar (`V`), loading covariance (`Av`), component covariance (`Sv`), and mean covariance (`Muv`).

def init_parms(init, n1, n2, ncomp, nobscomb, Isv):
    print("in init_parms", flush=True)
    if isinstance(init, str):
        if init.lower() == 'random':
            init = {}
        else:
            # Load parameters from a .mat file
            mat_data = loadmat(init)
            init = mat_data.get('init', {})
    
    # If 'init' is a dictionary
    if isinstance(init, dict):
        if 'A' in init:
            A = init['A']
        else:
            A = orth(np.random.randn(n1, ncomp))
        
        if 'Av' in init and init['Av'] is not None and init['Av'].size != 0:
            if isinstance(init['Av'], list):
                Av = init['Av']
            else:
                Av = []
                for i in range(n1):
                    Av.append(np.diag(init['Av'][i, :]))
        else:
            # Default Av as list of identity matrices
            Av = [np.eye(ncomp) for _ in range(n1)]

        Mu = init.get('Mu', np.array([])) 
        Muv = init.get('Muv', np.ones((n1, 1)))
        V = init.get('V', 1)
        S = init.get('S', np.random.randn(ncomp, n2))
        
        if 'Sv' in init and init['Sv'] is not None and np.size(init['Sv']) > 0:
            if nobscomb < n2:
                # Get unique elements and their first indices
                B, I = np.unique(Isv, return_index=True)
                if not isinstance(init['Sv'], list):
                    Sv = []
                    for j in range(nobscomb):
                        Sv.append(np.diag(init['Sv'][:, Isv[I[j]]]))
                elif 'Isv' in init and init['Isv']:
                    Sv = [init['Sv'][i] for i in init['Isv'][I]]
                else:
                    Sv = [init['Sv'][Isv[I[j]]] for j in range(nobscomb)]
            else:
                if not isinstance(init['Sv'], list):
                    Sv = []
                    for j in range(n2):
                        Sv.append(np.diag(init['Sv'][:, j]))
                elif 'Isv' in init and init['Isv']:
                    Sv = [init['Sv'][i] for i in init['Isv']]
                elif len(init['Sv']) == n2:
                    Sv = init['Sv']
                else:
                    Sv = []
        else:
            Sv = [np.eye(ncomp) for _ in range(nobscomb)]
    else:
        # If 'init' is neither str nor dict, raise an error
        raise ValueError("init must be either a string or a dictionary.")
    
    return A, S, Mu, V, Av, Sv, Muv

##############################################################################################
#Print_first_step
#Prints the initial step's root mean square (RMS) error and probe RMS error if verbosity is enabled.
#Gets a verbosity flag (`verbose`), initial RMS error (`rms`), and initial probe RMS error (`prms`).
#Returns nothing; outputs formatted step details to the console if `verbose` is True.

def print_first_step(verbose, rms, prms):
    if not verbose:
        return

    print(f"Step 0: rms = {rms:.6f}")
    if not np.isnan(prms):
        print(f" ({prms:.6f})")

import numpy as np

##############################################################################################
#Print_step
#Logs the current iteration's metrics, including cost, RMS error, probe RMS error, and subspace angle, if verbosity is enabled.
#Gets a verbosity flag (`verbose`), log container (`lc`) with metrics, and subspace angle (`a_angle`).
#Returns nothing; outputs formatted details of the current step to the console if `verbose` is True.
def print_step(verbose, lc, a_angle):
    if not verbose:
        return

    iter = len(lc['rms']) - 1
    steptime = lc['time'][-1] - lc['time'][-2]

    print(f"Step {iter}: ", end='')


    cost_last = lc['cost'][-1]
    if isinstance(cost_last, np.ndarray):
        if cost_last.size == 1:
            cost_val = cost_last.item()
        else:
            cost_val = cost_last[0]
    else:
        cost_val = cost_last

    if not np.isnan(cost_val):
        print(f"cost = {cost_val:.6f}, ", end='')


    rms_last = lc['rms'][-1]
    if isinstance(rms_last, np.ndarray):
        if rms_last.size == 1:
            rms_val = rms_last.item()
        else:
            rms_val = rms_last[0]
    else:
        rms_val = rms_last

    print(f"rms = {rms_val:.6f}", end='')


    if 'prms' in lc and len(lc['prms']) > 0:
        prms_last = lc['prms'][-1]
        if isinstance(prms_last, np.ndarray):
            if prms_last.size == 1:
                prms_val = prms_last.item()
            else:
                prms_val = prms_last[0]
        else:
            prms_val = prms_last

        if not np.isnan(prms_val):
            print(f", prms = {prms_val:.6f}")
        else:
            print()
    else:
        print()

##############################################################################################
#Print_progress_bar
#Displays a progress bar message if verbosity level is set to 2.
#Gets a verbosity level (`verbose`) and a descriptive message (`string`).
#Returns nothing; outputs the provided message to the console if `verbose` equals 2.

def print_progress_bar(verbose, string):
    if verbose == 2:
        print(f"print_progress_bar: {string}", flush=True)
        # print("\n|                                                  |\r|")

##############################################################################################
#Print_progress
#Displays the progress of an iterative process if verbosity level is set to 2.
#Gets a verbosity level (`verbose`), current iteration index (`i`), total iterations (`n`), and a descriptive message (`string`).
#Returns nothing; outputs the progress message to the console if `verbose` equals 2.

def print_progress(verbose, i, n, string):
    if verbose == 2:
        print("print_progress", flush=True)
        print(f"\r{string} {i}/{n}", end='')

##############################################################################################
#Display_init
#Initializes and displays plots for tracking RMS training and test errors during an iterative process.
#Gets a display flag (`display`) and a log container (`lc`) with RMS training and test errors.
#Returns a dictionary (`dsph`) containing display flags and plot handles for interactive updates.

def display_init(display, lc):
    dsph = {'display': display}

    if not dsph['display']:
        return dsph

    # Create a new figure with 2 subplots (2 rows, 1 column)
    dsph['fig'], axes = plt.subplots(2, 1, figsize=(8, 6))

    # First Subplot: RMS Training Error
    ax1 = axes[0]
    steps_rms = np.arange(len(lc['rms']))  # Generate step indices [0, 1, 2, ...]
    line_rms, = ax1.plot(steps_rms, lc['rms'], label='RMS Training Error')
    ax1.set_title('RMS Training Error')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('RMS Error')
    ax1.legend()
    ax1.grid(True)

    # Second Subplot: RMS Test Error
    ax2 = axes[1]
    steps_prms = np.arange(len(lc['prms']))  # Generate step indices [0, 1, 2, ...]
    line_prms, = ax2.plot(steps_prms, lc['prms'], label='RMS Test Error', color='orange')
    ax2.set_title('RMS Test Error')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('RMS Error')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Force the plots to render
    plt.draw()

    # Store plot handles in the dsph dictionary
    dsph['rms'] = line_rms
    dsph['prms'] = line_prms

    return dsph

    
##############################################################################################
#Display_progress
#Updates the plots for RMS training and test errors during an iterative process if display is enabled.
#Gets a display dictionary (`dsph`) with plot handles and a log container (`lc`) with updated RMS data.
#Returns nothing; updates the plots in real-time if `dsph['display']` is True.

def display_progress(dsph, lc):
    if dsph['display']:
        dsph['rms'].set_xdata(np.arange(len(lc['rms'])))
        dsph['rms'].set_ydata(lc['rms'])
        dsph['prms'].set_xdata(np.arange(len(lc['prms'])))
        dsph['prms'].set_ydata(lc['prms'])
        import matplotlib.pyplot as plt
        plt.draw()