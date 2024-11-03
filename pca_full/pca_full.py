# return dictionary with A, S, Mu, V, cv, hp, lc
# A: Likely represents a matrix of principal components (loadings).# A is the liner transition matrix - Josh
# S: Another output matrix, possibly the transformed data in the new PCA space (scores). # this is the matrix of prinicipal components
# Mu: The mean values of the data. # this is the bias, like the y intecept in a linear regression, but for a matrix
# V: Variance, possibly of the noise.
# cv: Posterior covariances (likely a structure with details for A and S). # as well as Mu
# hp: Hyperparameters (related to prior distributions).
# lc: Learning curves (metrics for performance over iterations).

# X: The input data matrix.
# ncomp: The number of principal components to compute.
# kwargs: Stands for "keyword arguments". It allows you to pass optional parameters as a dictionary.
import numpy as np
from scipy.io import savemat
import scipy.linalg
import scipy.io
from scipy.sparse import issparse, csr_matrix
import numpy as np
from scipy.io import loadmat
from scipy.linalg import orth  # Use orth from scipy.linalg
import matplotlib.pyplot as plt
import time

# from argschk import argschk
# from rmempty import rmempty



def pca_full(X, ncomp, **kwargs):

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
    'rmsstop':np.array([100, 1e-4, 1e-3]), # () means no rms stop criteria
    'cfstop':np.array([]), # () means no cost stop criteria
    'verbose':1,
    'xprobe':np.array([]),
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

    n1x, n2x = X.shape
    X, Xprobe, Ir, Ic, opts.init = rmempty(X, Xprobe, opts.init, opts.verbose)
    n1, n2 = X.shape

    [n1x,n2x] = X.shape()

    # Handle the case where X is sparse
    if issparse(X):
        M = (X != 0).astype(float)  # Equivalent to spones(X)
        Mprobe = (Xprobe != 0).astype(float)
    else:
        M = ~np.isnan(X)
        Mprobe = ~np.isnan(Xprobe)
    
        X[X == 0] = np.finfo(float).eps  # Replace zeros with a small number
        Xprobe[Xprobe == 0] = np.finfo(float).eps
    
        X[np.isnan(X)] = 0  # Replace NaNs with 0
        Xprobe[np.isnan(Xprobe)] = 0
    
    # Compute the number of observed (non-missing) values in each row of X
    Nobs_i = np.sum(M, axis=1)
    
    # Total number of observed data points
    ndata = np.sum(Nobs_i)
    
    # Compute the number of observed values in the probe data (Xprobe)
    if issparse(Mprobe):
        nprobe = Mprobe.count_nonzero()  # Use count_nonzero for sparse matrices
    else:
        nprobe = np.count_nonzero(Mprobe)
    # If no observed values in Xprobe, set it to empty and disable early stopping
    if nprobe == 0:
        Xprobe = np.array([])  # Set Xprobe to an empty array
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
        Va = 1000 * no.ones(1, ncomp)
        Vmu = 1000
    else:
        Va = np.full(ncomp, np.inf)
        Vmu = np.inf

    if not use_postvar:
        Muv = np.array([])
        Av = []

    if not opts.bias:
        Muv = np.array([])
        Vmu = 0

    if np.size(Mu) > 0:
        if opts['bias']:
            Mu = np.sum(X, axis=1) / Nobs_i  # Sum over rows and divide by Nobs_i
        else:
            Mu = np.zeros((n1, 1))

    X, Xprobe = subtract_mu(Mu, X, M, Xprobr, Mprobe, opts['bias'])

    rms, errMx = compute_rms(X, A, S, M, ndata) #still need to find the cpp file computr_rms refers too and test it
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

    for iter in range(1, opts['maxiters'] + 1):

        # The prior is not updated at the beginning of learning to avoid killing sources
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
            dMu = np.sum(errMx, axis=1) / Nobs_i
            if Muv.size > 0:
                Muv = V / (Nobs_i + V / Vmu)
            th = 1 / (1 + V / (Nobs_i * Vmu))
            Mu_old = Mu.copy()
            Mu = th * (Mu + dMu)
            dMu = Mu - Mu_old
            X, Xprobe = subtract_mu(dMu, X, M, Xprobe, Mprobe, update_bias=True)

        # Update S
        if not Isv:
            for j in range(n2):
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
        if opts['verbose'] == 2:
            print('                                              \r', end='')
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
        sXv = 0
        if not Isv:
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

        if opts['cfstop'].size > 0:
            cost = cf_full(X, A, S, Mu, V, Av, Sv, Isv, Muv, Va, Vmu, M, sXv, ndata)
            lc['cost'].append(cost)

        display_progress(dsph, lc)
        angles = subspace_angle(A, Aold)
        angleA = angles[0]
        print_step(opts['verbose'], lc, angleA)

        convmsg = convergence_check(opts, lc, angleA)
        if convmsg:
            if use_prior and iter <= opts['niter_broadprior']:
                # if the prior has never been updated: do nothing
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
            # Save parameters (assuming save_parameters is a helper function)
            # save_parameters(opts['filename'], A, S, Mu, V, Av, Muv, Sv, Isv, Va, Vmu, lc, Ir, Ic, n1x, n2x, n1, n2
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
def subtract_mu(Mu, X, M, Xprobe=None, Mprobe=None, update_bias=True):
    """
    Subtracts the bias vector Mu from the data matrix X. If X is sparse,
    uses the SUBTRACT_MU function from the compiled module. Otherwise,
    performs element-wise subtraction.

    Parameters:
    - Mu: (n1,) bias vector.
    - X: (n1, n2) data matrix (sparse or dense).
    - M: (n1, n2) mask matrix.
    - Xprobe: (n1, n2_probe) probe matrix (optional).
    - Mprobe: (n1, n2_probe) probe mask matrix (optional).
    - update_bias: bool flag to determine whether to update X and Xprobe.

    Returns:
    - X_new: Updated data matrix after subtraction.
    - Xprobe_new: Updated probe matrix after subtraction (if provided).
    """
    n2 = X.shape[1]

    if not update_bias:
        return X, Xprobe

    if issparse(X):
        # Ensure that Xprobe is also sparse if provided
        X = subtract_mu_from_sparse(X, Mu)
        if Xprobe is not None and Xprobe.size > 0:
            Xprobe = subtract_mu_from_sparse(Xprobe, Mu)
    else:
        X = X - np.multiply(np.tile(Mu, (1, n2)), M)
        if Xprobe is not None and Xprobe.size > 0 and Mprobe is not None:
            Xprobe = Xprobe - np.multiply(np.tile(Mu, (1, Xprobe.shape[1])), Mprobe)

    return X, Xprobe


##############################################################################################
def rotate_to_pca(A, Av, S, Sv, Isv, obscombj, update_bias):
    n1 = A.shape[0]
    n2 = S.shape[1]

    # Initialize dMu based on update_bias flag
    if update_bias:
        mS = np.mean(S, axis=1, keepdims=True)
        dMu = A @ mS
        S = S - mS
    else:
        dMu = 0

    # Calculate covariance of S
    covS = S @ S.T
    if len(Isv) == 0:
        for j in range(n2):
            covS += Sv[j]
    else:
        nobscomb = len(obscombj)
        for j in range(nobscomb):
            covS += len(obscombj[j]) * Sv[j]

    covS /= n2

    # Perform PCA on covS
    eigvals, VS = np.linalg.eigh(covS)

    D = np.diag(np.sqrt(eigvals))
    RA = VS @ D
    A = A @ RA

    # Calculate covariance of A
    covA = A.T @ A
    if Av:
        for i in range(n1):
            Av[i] = RA.T @ Av[i] @ RA
            covA += Av[i]

    covA /= n1

    # Perform PCA on covA
    eigvals, VA = np.linalg.eigh(covA)
    DA = np.sort(-eigvals)[::-1]
    I = np.argsort(-eigvals)
    VA = VA[:, I]
    A = A @ VA

    if Av:
        for i in range(n1):
            Av[i] = VA.T @ Av[i] @ VA

    # R = VA.T @ np.diag(1 / np.sqrt(np.diag(D))) @ VS.T
    # Calculate the square root of the diagonal of D
    D_diag = np.sqrt(np.diag(D))
    
    # Initialize an array for the inverse of D's diagonal, handling zeros
    D_inv = np.zeros_like(D_diag)
    non_zero_indices = D_diag > 0  # Identify non-zero elements in D
    D_inv[non_zero_indices] = 1 / D_diag[non_zero_indices]  # Invert only non-zero elements
    
    # Calculate R using the adjusted inverse diagonal matrix
    R = VA.T @ np.diag(D_inv) @ VS.T


    # Transform S and Sv
    S = R @ S
    for j in range(len(Sv)):
        Sv[j] = R @ Sv[j] @ R.T
        
    return dMu, A, Av, S, Sv

##############################################################################################

def init_parms(init, n1, n2, ncomp, nobscomb, Isv):
    """
    Initialize parameters based on the input.

    Parameters:
    - init: str or dict
        If str, specifies 'random' or a filename to load parameters from.
        If dict, contains initialization parameters.
    - n1, n2, ncomp, nobscomb: int
        Dimensions for initializing matrices.
    - Isv: array-like
        Index array used for Sv initialization.

    Returns:
    - A, S, Mu, V, Av, Sv, Muv: Initialized parameters.
    """
    
    # Handle the 'init' input
    if isinstance(init, str):
        if init.lower() == 'random':
            init = {}
        else:
            # Load parameters from a .mat file
            mat_data = loadmat(init)
            # Convert MATLAB struct to Python dict (assuming 'init' is the variable name)
            # Adjust 'init' to the actual variable name in the .mat file if different
            init = mat_data.get('init', {})
    
    # If 'init' is a dictionary (similar to MATLAB struct)
    if isinstance(init, dict):
        # Initialize A
        if 'A' in init:
            A = init['A']
        else:
            # Generate a random matrix and orthogonalize it
            A = orth(np.random.randn(n1, ncomp))
        
        # Initialize Av
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
        
        # Initialize Mu
        Mu = init.get('Mu', np.array([])) 
        
        # Initialize Muv
        Muv = init.get('Muv', np.ones((n1, 1)))
        
        # Initialize V
        V = init.get('V', 1)
        
        # Initialize S
        S = init.get('S', np.random.randn(ncomp, n2))
        
        # Initialize Sv
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
            # Default Sv as list of identity matrices
            Sv = [np.eye(ncomp) for _ in range(nobscomb)]
    else:
        # If 'init' is neither str nor dict, raise an error
        raise ValueError("init must be either a string or a dictionary.")
    
    return A, S, Mu, V, Av, Sv, Muv

##############################################################################################

def print_first_step(verbose, rms, prms):
    if not verbose:
        return

    print(f"Step 0: rms = {rms:.6f}")
    if not np.isnan(prms):
        print(f" ({prms:.6f})")

import numpy as np

##############################################################################################

def print_step(verbose, lc, a_angle):
    if not verbose:
        return

    iter = len(lc['rms']) - 1
    steptime = lc['time'][-1] - lc['time'][-2]

    print(f"Step {iter}: ", end='')
    if not np.isnan(lc['cost'][-1]):
        print(f"cost = {lc['cost'][-1]:.6f}, ", end='')
    print(f"rms = {lc['rms'][-1]:.6f}", end='')
    if not np.isnan(lc['prms'][-1]):
        print(f" ({lc['prms'][-1]:.6f})", end='')
    print(f", angle = {a_angle:.2e}", end='')
    if steptime > 1:
        print(f" ({round(steptime)} sec)")
    else:
        print(f" ({steptime:.0e} sec)")

##############################################################################################

def print_progress_bar(verbose, string):
    if verbose == 2:
        print(string)
        # print("\n|                                                  |\r|")

##############################################################################################

def print_progress(verbose, i, n, string):
    if verbose == 2:
        print(f"\r{string} {i}/{n}", end='')

##############################################################################################

def display_init(display, lc):
    """
    Initializes the display for plotting RMS errors.

    Parameters:
    - display (bool): Flag to determine whether to initialize and display plots.
    - lc (dict): Logging structure containing 'rms' and 'prms' lists or arrays.

    Returns:
    - dsph (dict): Display structure containing display flags and plot handles.
    """
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

def display_progress(dsph, lc):
    if dsph['display']:
        dsph['rms'].set_xdata(np.arange(len(lc['rms'])))
        dsph['rms'].set_ydata(lc['rms'])
        dsph['prms'].set_xdata(np.arange(len(lc['prms'])))
        dsph['prms'].set_ydata(lc['prms'])
        import matplotlib.pyplot as plt
        plt.draw()