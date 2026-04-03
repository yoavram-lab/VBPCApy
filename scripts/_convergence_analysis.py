"""Convergence analysis: compare stop criteria across scenarios."""

import numpy as np

from vbpca_py._pca_full import pca_full


def analyze(label, n, p, k, miss_frac=0.0, seed=42, maxiters=200):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((p, k))
    S = rng.standard_normal((k, n))
    X = U @ S + 0.3 * rng.standard_normal((p, n))
    mask = None
    if miss_frac > 0:
        mask = rng.random((p, n)) > miss_frac
        X[~mask] = np.nan

    # --- Monkey-patch to capture per-iteration angles ---
    import vbpca_py._converge as conv

    orig_angle = conv._angle_stop_message
    angles = []

    def capture(opts, angle_a):
        angles.append(angle_a)
        return orig_angle(opts, angle_a)

    conv._angle_stop_message = capture
    try:
        result = pca_full(
            X, k, mask=mask, maxiters=maxiters, verbose=0, niter_broadprior=0
        )
    finally:
        conv._angle_stop_message = orig_angle

    lc = result["lc"]
    rms = np.array(lc["rms"])
    stop = np.array(lc.get("_stop", []))
    n_iter = len(rms)
    early = any(s > 0 for s in stop) if len(stop) else False
    angles_arr = np.array(angles)

    # --- Smoothed RMS (2-point moving average removes 2-cycle) ---
    if len(rms) >= 3:
        rms_smooth = (rms[:-1] + rms[1:]) / 2
        smooth_deltas = np.abs(np.diff(rms_smooth))
        smooth_rel = smooth_deltas / (np.abs(rms_smooth[1:]) + 1e-15)
    else:
        smooth_rel = np.array([])

    # --- Raw RMS relative deltas ---
    raw_deltas = np.abs(np.diff(rms))
    raw_rel = raw_deltas / (np.abs(rms[1:]) + 1e-15)

    def first_below(arr, thr):
        idx = np.where(arr < thr)[0]
        return int(idx[0]) + 2 if len(idx) else None

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  n={n}, p={p}, k={k}, miss={miss_frac:.0%}, maxiters={maxiters}")
    print(f"  Iterations run: {n_iter}, Early stopped: {early}")
    print(f"{'=' * 70}")

    # Angle convergence
    print("\n  Subspace angle convergence:")
    for thr in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
        idx = np.where(angles_arr < thr)[0]
        first = int(idx[0]) + 1 if len(idx) else None
        print(f"    angle < {thr:.0e}:  iter {first}")

    # Raw RMS (will show None for oscillating dense cases)
    print("\n  Raw RMS rel_delta convergence:")
    for thr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        print(f"    rel_delta < {thr:.0e}:  iter {first_below(raw_rel, thr)}")

    # Smoothed RMS
    print("\n  Smoothed RMS (2-pt avg) rel_delta convergence:")
    for thr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        print(f"    rel_delta < {thr:.0e}:  iter {first_below(smooth_rel, thr)}")

    # Derivative of angle (rate of angle change)
    if len(angles_arr) >= 3:
        angle_ratios = angles_arr[1:] / (angles_arr[:-1] + 1e-30)
        print("\n  Angle contraction ratio (angle[i+1]/angle[i]):")
        for i in [1, 2, 3, 5, 10, 15, 20]:
            if i < len(angle_ratios):
                print(f"    iter {i + 1}: {angle_ratios[i]:.6f}")

    # Composite: angle * smoothed_rms_delta
    # Idea: both must be small for true convergence
    print()


scenarios = [
    ("Dense 20x50 k=3", 20, 50, 3, 0.0),
    ("Dense 50x100 k=5", 50, 100, 5, 0.0),
    ("Dense 100x200 k=10", 100, 200, 10, 0.0),
    ("Dense 200x200 k=5", 200, 200, 5, 0.0),
    ("MCAR10% 50x100 k=5", 50, 100, 5, 0.1),
    ("MCAR30% 50x100 k=5", 50, 100, 5, 0.3),
    ("MCAR50% 100x200 k=10", 100, 200, 10, 0.5),
]

for label, n, p, k, mf in scenarios:
    analyze(label, n, p, k, miss_frac=mf)

# --- Now test: what if we just used angle stop with niter_broadprior=0? ---
print("\n" + "=" * 70)
print("  SUMMARY: Would angle stop (minangle=1e-8, niter_broadprior=0) work?")
print("=" * 70)
rng = np.random.default_rng(42)
for label, n, p, k, mf in scenarios:
    U = rng.standard_normal((p, k))
    S = rng.standard_normal((k, n))
    X = U @ S + 0.3 * rng.standard_normal((p, n))
    mask = None
    if mf > 0:
        mask = rng.random((p, n)) > mf
        X[~mask] = np.nan

    import vbpca_py._converge as conv

    orig_angle = conv._angle_stop_message
    angles = []

    def capture(opts, angle_a):
        angles.append(angle_a)
        return orig_angle(opts, angle_a)

    conv._angle_stop_message = capture
    try:
        result = pca_full(X, k, mask=mask, maxiters=200, verbose=0, niter_broadprior=0)
    finally:
        conv._angle_stop_message = orig_angle

    stop = np.array(result["lc"].get("_stop", []))
    n_iter = len(result["lc"]["rms"])
    stopped_at = np.where(np.array(stop) > 0)[0]
    first_stop = int(stopped_at[0]) + 1 if len(stopped_at) else None
    print(f"  {label:30s} | stopped at iter {first_stop} (ran {n_iter} iters)")
