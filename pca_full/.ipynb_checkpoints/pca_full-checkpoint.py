# return dictionary with A, S, Mu, V, cv, hp, lc
# A: Likely represents a matrix of principal components (loadings).
# S: Another output matrix, possibly the transformed data in the new PCA space (scores).
# Mu: The mean values of the data.
# V: Variance, possibly of the noise.
# cv: Posterior covariances (likely a structure with details for A and S).
# hp: Hyperparameters (related to prior distributions).
# lc: Learning curves (metrics for performance over iterations).

# X: The input data matrix.
# ncomp: The number of principal components to compute.
# kwargs: Stands for "keyword arguments". It allows you to pass optional parameters as a dictionary.
import numpy as np
import scipy.linalg
import scipy.io
from scipy.sparse import issparse, csr_matrix
import numpy as np
from scipy.io import loadmat
from scipy.linalg import orth  # Use orth from scipy.linalg
import matplotlib.pyplot as plt

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
        'rms': rms,
        'prms': prms,
        'time': 0,
        'cost': float('nan')
    }
    
    dsph = display_init(opts.display, lc)
    print_first_step(opts.verbose, rms, prms)
    Aold = A


    return A, S, Mu, V, cv, hp, lc

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

def print_first_step(verbose, rms, prms):
    if not verbose:
        return

    print(f"Step 0: rms = {rms:.6f}")
    if not np.isnan(prms):
        print(f" ({prms:.6f})")

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

    
