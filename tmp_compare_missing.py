import numpy as np
from scipy.io import loadmat

from vbpca_py.pca_full import pca_full

mat = loadmat("tests/data/legacy_pca_full_missing.mat", squeeze_me=True, struct_as_record=False)
x = np.asarray(mat["x"], float)
mat_res = mat["result"]
A_mat = np.asarray(mat_res.A, float)
S_mat = np.asarray(mat_res.S, float)
Mu_mat = np.asarray(mat_res.Mu, float)
if Mu_mat.ndim == 1:
    Mu_mat = Mu_mat[:, None]
V_mat = float(mat_res.V)

res = pca_full(
    x,
    n_components=int(mat["k"]),
    algorithm="vb",
    maxiters=int(getattr(mat_res, "maxiters", 200)),
    bias=int(getattr(mat_res, "bias", 1)),
    uniquesv=int(getattr(mat_res, "uniquesv", 0)),
    init=mat_res,
    autosave=0,
    display=0,
    verbose=0,
    rotate2pca=1,
)
A_py, S_py, Mu_py, V_py = res["A"], res["S"], res["Mu"], float(res["V"])
x_rec_py = A_py @ S_py + Mu_py
x_rec_mat = A_mat @ S_mat + Mu_mat
mask = ~np.isnan(x)
diff = (x_rec_py - x_rec_mat) * mask
mse = float(np.sum(diff**2) / float(np.count_nonzero(mask)))
rms = float(np.sqrt(mse))
print({"rms": rms, "V_py": V_py, "V_mat": V_mat, "lc_len": len(res["lc"]["rms"])})
