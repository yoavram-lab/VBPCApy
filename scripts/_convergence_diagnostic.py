"""Diagnostic: measure how quickly VBPCA converges in practice."""

import numpy as np

from vbpca_py._pca_full import pca_full


def run_diagnostic(
    n: int,
    p: int,
    k: int,
    maxiters: int = 200,
    seed: int = 42,
    miss_frac: float = 0.0,
    label: str = "",
):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((p, k))
    S = rng.standard_normal((k, n))
    X = U @ S + 0.3 * rng.standard_normal((p, n))

    mask = None
    if miss_frac > 0:
        mask = rng.random((p, n)) > miss_frac
        X[~mask] = np.nan

    result = pca_full(X, k, mask=mask, maxiters=maxiters, verbose=0)

    lc = result["lc"]
    rms = np.array(lc["rms"])
    cost = np.array(lc["cost"])
    stop = np.array(lc.get("_stop", []))

    tag = f"{label} " if label else ""
    print(
        f"\n=== {tag}n={n}, p={p}, k={k}, maxiters={maxiters}, miss={miss_frac:.0%} ==="
    )
    print(f"Iterations run: {len(rms)}")
    early_stopped = any(s > 0 for s in stop) if len(stop) else False
    print(f"Early stopped: {early_stopped}")
    print(f"Cost computed: {not np.all(np.isnan(cost))}")
    print(f"RMS: first={rms[0]:.6f}, last={rms[-1]:.6f}")

    # Show per-iteration RMS delta
    print("\n  iter |     rms      |  delta_abs  |  delta_rel  | cumul_rel")
    print("  -----|-------------|-------------|-------------|----------")
    rms_start = rms[0]
    for i in [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 199]:
        if i < len(rms):
            delta = abs(rms[i] - rms[i - 1])
            rel = delta / (abs(rms[i]) + 1e-15)
            cumul = abs(rms[i] - rms_start) / (abs(rms_start) + 1e-15)
            print(f"  {i:4d} | {rms[i]:.6f}  | {delta:.3e} | {rel:.3e} | {cumul:.3e}")

    # Check: after how many iters does delta drop below thresholds?
    deltas = np.abs(np.diff(rms))
    rel_deltas = deltas / (np.abs(rms[1:]) + 1e-15)
    for thr in [1e-3, 1e-4, 1e-5, 1e-6]:
        idx = np.where(rel_deltas < thr)[0]
        first = idx[0] + 1 if len(idx) > 0 else None
        print(f"  First iter with rel_delta < {thr:.0e}: {first}")


# Dense, complete data
for n, p, k in [(20, 50, 3), (50, 100, 5), (100, 200, 10), (200, 200, 5)]:
    run_diagnostic(n, p, k)

# With missing data
print("\n\n===== WITH MISSING DATA =====")
for n, p, k in [(50, 100, 5), (100, 200, 10)]:
    for miss in [0.1, 0.3, 0.5]:
        run_diagnostic(n, p, k, miss_frac=miss, label=f"MCAR{int(miss * 100)}%")
