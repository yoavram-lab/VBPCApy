import numpy as np
from scipy.io import loadmat

from vbpca_py.pca_full import pca_full

mat = loadmat("tests/data/legacy_pca_full_dense.mat", squeeze_me=True, struct_as_record=False)
x = np.asarray(mat["x"], float)
mat_res = mat["result"]
A_mat = np.asarray(mat_res.A, float)
S_mat = np.asarray(mat_res.S, float)
Mu_mat = np.asarray(mat_res.Mu, float)
if Mu_mat.ndim == 1:
    Mu_mat = Mu_mat[:, None]


def run(iters: int, rotate: int, niter_broadprior: int | None = None):
    kwargs = dict(
        x=x,
        n_components=int(mat["k"]),
        algorithm="vb",
        maxiters=iters,
        bias=int(getattr(mat_res, "bias", 1)),
        uniquesv=int(getattr(mat_res, "uniquesv", 0)),
        init=mat_res,
        autosave=0,
        display=0,
        verbose=0,
        rotate2pca=rotate,
    )
    if niter_broadprior is not None:
        kwargs["niter_broadprior"] = niter_broadprior
    return pca_full(**kwargs)

configs = [
    (200, 1, None),      # baseline
    (200, 0, None),      # rotation off
    (200, 1, 10000),     # hyperprior frozen
]

for iters, rotate, nbp in configs:
    res = run(iters, rotate, nbp)
    A, S, Mu, V = res["A"], res["S"], res["Mu"], float(res["V"])
    x_rec = A @ S + Mu
    x_rec_mat = A_mat @ S_mat + Mu_mat
    maxdiff = float(np.max(np.abs(x_rec - x_rec_mat)))
    mudiff = float(np.max(np.abs(Mu - Mu_mat)))
    print({"iters": iters, "rotate2pca": rotate, "niter_broadprior": nbp, "max_recon_abs_diff": maxdiff, "max_mu_abs_diff": mudiff, "V": V})
