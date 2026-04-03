"""Compare internal RMS (on re-centered working copy) vs RMS on original data."""

import numpy as np

import vbpca_py._pca_full as pf
from vbpca_py._pca_full import pca_full

rng = np.random.default_rng(42)
n, p, k = 50, 100, 5
U = rng.standard_normal((p, k))
S = rng.standard_normal((k, n))
X = U @ S + 0.3 * rng.standard_normal((p, n))

X_original = X.copy()
mask_full = np.ones_like(X)
n_data = float(mask_full.sum())

orig_step = pf._iteration_step
rms_on_original = []
noise_vars = []


def capture_step(ctx):
    orig_step(ctx)
    a = ctx.training.model.a
    s = ctx.training.model.s
    mu = ctx.training.model.mu
    # RMS on original data: || X_orig - mu - A@S ||
    resid = X_original - mu - a @ s
    rms = float(np.sqrt(np.sum(resid**2) / n_data))
    rms_on_original.append(rms)
    noise_vars.append(float(ctx.training.model.noise_var))


pf._iteration_step = capture_step
try:
    result = pca_full(X, k, maxiters=15, verbose=0, niter_broadprior=0, minangle=1e-15)
finally:
    pf._iteration_step = orig_step

rms_internal = np.array(result["lc"]["rms"])

print("iter | internal_rms | original_rms | noise_var")
print("-----|-------------|-------------|----------")
for i in range(len(rms_on_original)):
    print(
        f"{i + 1:4d} | {rms_internal[i + 1]:.8f} | "
        f"{rms_on_original[i]:.8f} | {noise_vars[i]:.8f}"
    )

print("\nInternal deltas:", [f"{d:+.3e}" for d in np.diff(rms_internal[1:])])
print("Original deltas:", [f"{d:+.3e}" for d in np.diff(rms_on_original)])
print("Noise V  deltas:", [f"{d:+.3e}" for d in np.diff(noise_vars)])
