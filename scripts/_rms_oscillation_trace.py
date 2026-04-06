"""Instrument one iteration to find root cause of RMS 2-cycle oscillation."""

import numpy as np

from vbpca_py._pca_full import pca_full


def trace_rms_sources():
    """Compute RMS at multiple points within each iteration to find oscillation source."""
    rng = np.random.default_rng(42)
    n, p, k = 50, 100, 5
    u = rng.standard_normal((p, k))
    s = rng.standard_normal((k, n))
    x_true = u @ s + 0.3 * rng.standard_normal((p, n))

    # Run 20 iters with niter_broadprior=0, capture intermediate states
    # We'll monkey-patch _iteration_step to measure RMS before/after each phase
    import vbpca_py._pca_full as pf

    orig_step = pf._iteration_step

    # Track RMS computation at different points
    traces = []

    def instrumented_step(ctx):
        """Wrap _iteration_step to measure intermediate RMS values."""
        iteration = ctx.iteration
        m = ctx.training.model
        prepared = ctx.prepared
        mask = prepared.mask

        def quick_rms(a, s):
            resid = prepared.x_data - a @ s
            err = resid * np.asarray(mask, dtype=float)
            return float(np.sqrt(np.sum(err**2) / prepared.n_data))

        # Snapshot BEFORE any updates
        rms_before = quick_rms(m.a, m.s)
        a_before = m.a.copy()
        s_before = m.s.copy()

        # Run actual step
        orig_step(ctx)

        # Snapshot AFTER all updates
        rms_after = quick_rms(m.a, m.s)

        # Also compute: RMS with old-A + new-S, and new-A + old-S
        rms_old_a_new_s = quick_rms(a_before, m.s)
        rms_new_a_old_s = quick_rms(m.a, s_before)

        # Check if A changed much
        a_delta = np.linalg.norm(m.a - a_before) / (np.linalg.norm(a_before) + 1e-15)
        s_delta = np.linalg.norm(m.s - s_before) / (np.linalg.norm(s_before) + 1e-15)

        # Noise variance
        v = float(m.noise_var)

        traces.append({
            "iter": iteration,
            "rms_before": rms_before,
            "rms_after": rms_after,
            "rms_oldA_newS": rms_old_a_new_s,
            "rms_newA_oldS": rms_new_a_old_s,
            "a_rel_change": a_delta,
            "s_rel_change": s_delta,
            "noise_var": v,
        })

    pf._iteration_step = instrumented_step
    try:
        pca_full(
            x_true,
            k,
            maxiters=20,
            verbose=0,
            niter_broadprior=0,
            minangle=1e-15,  # prevent early stop so we can observe oscillation
        )
    finally:
        pf._iteration_step = orig_step

    print(
        "iter | rms_before | rms_after  | oldA+newS  | newA+oldS  | A_chg     | S_chg     | V"
    )
    print(
        "-----|------------|------------|------------|------------|-----------|-----------|-------"
    )
    for t in traces:
        print(
            f"{t['iter']:4d} | {t['rms_before']:.6f}  | {t['rms_after']:.6f}  | "
            f"{t['rms_oldA_newS']:.6f}  | {t['rms_newA_oldS']:.6f}  | "
            f"{t['a_rel_change']:.3e} | {t['s_rel_change']:.3e} | {t['noise_var']:.6f}"
        )

    # Now check: is the oscillation in A, S, or V?
    print("\n--- RMS delta pattern ---")
    for i in range(1, len(traces)):
        d = traces[i]["rms_after"] - traces[i - 1]["rms_after"]
        print(f"  iter {traces[i]['iter']}: delta = {d:+.6e}")


def check_rotation_effect():
    """Check if RotateToPCA causes the oscillation."""
    rng = np.random.default_rng(42)
    n, p, k = 50, 100, 5
    u = rng.standard_normal((p, k))
    s = rng.standard_normal((k, n))
    x_true = u @ s + 0.3 * rng.standard_normal((p, n))

    print("\n\n=== WITH rotate2pca=1 (default) ===")
    r1 = pca_full(x_true, k, maxiters=15, verbose=0, niter_broadprior=0, minangle=1e-15)
    rms1 = np.array(r1["lc"]["rms"])
    print("RMS:", [f"{v:.6f}" for v in rms1])
    print("Deltas:", [f"{d:+.6e}" for d in np.diff(rms1)])

    print("\n=== WITH rotate2pca=0 ===")
    r2 = pca_full(
        x_true,
        k,
        maxiters=15,
        verbose=0,
        niter_broadprior=0,
        minangle=1e-15,
        rotate2pca=0,
    )
    rms2 = np.array(r2["lc"]["rms"])
    print("RMS:", [f"{v:.6f}" for v in rms2])
    print("Deltas:", [f"{d:+.6e}" for d in np.diff(rms2)])


def check_bias_effect():
    """Check if bias (mu) update causes the oscillation."""
    rng = np.random.default_rng(42)
    n, p, k = 50, 100, 5
    u = rng.standard_normal((p, k))
    s = rng.standard_normal((k, n))
    x_true = u @ s + 0.3 * rng.standard_normal((p, n))

    print("\n\n=== WITH bias=1 (default) ===")
    r1 = pca_full(x_true, k, maxiters=15, verbose=0, niter_broadprior=0, minangle=1e-15)
    rms1 = np.array(r1["lc"]["rms"])
    print("Deltas:", [f"{d:+.6e}" for d in np.diff(rms1)])

    print("\n=== WITH bias=0 ===")
    r2 = pca_full(
        x_true, k, maxiters=15, verbose=0, niter_broadprior=0, minangle=1e-15, bias=0
    )
    rms2 = np.array(r2["lc"]["rms"])
    print("RMS:", [f"{v:.6f}" for v in rms2])
    print("Deltas:", [f"{d:+.6e}" for d in np.diff(rms2)])


def check_precenter_data():
    """Pre-center per row (feature-wise), run with bias=0."""
    rng = np.random.default_rng(42)
    n, p, k = 50, 100, 5
    u = rng.standard_normal((p, k))
    s = rng.standard_normal((k, n))
    x = u @ s + 0.3 * rng.standard_normal((p, n))

    # Pre-center: subtract row means (feature means)
    row_mean = np.nanmean(x, axis=1, keepdims=True)
    x_centered = x - row_mean

    print("\n\n=== PRE-CENTERED (row mean) + bias=0 ===")
    r = pca_full(
        x_centered,
        k,
        maxiters=15,
        verbose=0,
        niter_broadprior=0,
        minangle=1e-15,
        bias=0,
    )
    rms = np.array(r["lc"]["rms"])
    print("RMS:", [f"{v:.6f}" for v in rms])
    print("Deltas:", [f"{d:+.6e}" for d in np.diff(rms)])

    print("\n=== PRE-CENTERED (row mean) + bias=1 (default) ===")
    r2 = pca_full(
        x_centered,
        k,
        maxiters=15,
        verbose=0,
        niter_broadprior=0,
        minangle=1e-15,
        bias=1,
    )
    rms2 = np.array(r2["lc"]["rms"])
    print("RMS:", [f"{v:.6f}" for v in rms2])
    print("Deltas:", [f"{d:+.6e}" for d in np.diff(rms2)])

    print("\n=== RAW (no centering) + bias=1 (default) ===")
    r3 = pca_full(
        x, k, maxiters=15, verbose=0, niter_broadprior=0, minangle=1e-15, bias=1
    )
    rms3 = np.array(r3["lc"]["rms"])
    print("RMS:", [f"{v:.6f}" for v in rms3])
    print("Deltas:", [f"{d:+.6e}" for d in np.diff(rms3)])


def trace_rms_within_iteration():
    """Compute RMS at every sub-step within iterations to find where oscillation enters."""

    import vbpca_py._pca_full as pf

    rng = np.random.default_rng(42)
    n, p, k = 50, 100, 5
    u = rng.standard_normal((p, k))
    s = rng.standard_normal((k, n))
    x = u @ s + 0.3 * rng.standard_normal((p, n))

    step_traces = []

    def quick_rms(ctx):
        """RMS from current x_data, A, S.

        Returns:
            float: The root mean square error on observed entries.
        """
        x_data = ctx.centering_state.x_data
        a = ctx.training.model.a
        s_mat = ctx.training.model.s
        mask = ctx.prepared.mask
        resid = np.asarray(x_data) - a @ s_mat
        err = resid * np.asarray(mask, dtype=float)
        return float(np.sqrt(np.sum(err**2) / ctx.prepared.n_data))

    # We need to replace _iteration_step entirely to capture sub-step RMS
    orig_step = pf._iteration_step

    def substep_trace(ctx):
        it = ctx.iteration
        rec = {"iter": it}

        # 1) Before anything
        rec["rms_0_start"] = quick_rms(ctx)

        # Run hyperprior update
        pf._update_hyperpriors_phase(ctx)
        rec["rms_1_after_hyper"] = quick_rms(ctx)

        # Run bias/center
        x_data, x_probe = pf._bias_and_center(ctx)
        rec["rms_2_after_bias"] = quick_rms(ctx)

        # Run scores + rotate
        x_data, x_probe, _ = pf._score_and_rotate(ctx, x_data, x_probe)
        rec["rms_3_after_scores"] = quick_rms(ctx)

        # Run loadings
        pf._update_loadings_phase(ctx, x_data)
        rec["rms_4_after_loadings"] = quick_rms(ctx)

        # This is where the official RMS is computed
        rms_val, prms_val, _ = pf._rms_phase(ctx, x_data, x_probe)
        rec["rms_5_official"] = rms_val

        # Run noise
        s_xv, _ = pf._noise_phase(ctx, rms_val)

        # Run convergence
        _, _ = pf._convergence_phase(ctx, x_data, rms_val, prms_val, s_xv)
        pf._append_phase_timings(
            ctx.training.lc,
            {
                "phase_scores_sec": 0,
                "phase_loadings_sec": 0,
                "phase_rms_sec": 0,
                "phase_noise_sec": 0,
                "phase_converge_sec": 0,
                "phase_total_sec": 0,
            },
        )
        ctx.training.lc.setdefault("_stop", []).append(0.0)

        rec["noise_var"] = float(ctx.training.model.noise_var)
        step_traces.append(rec)

    pf._iteration_step = substep_trace
    try:
        pca_full(x, k, maxiters=15, verbose=0, niter_broadprior=0, minangle=1e-15)
    finally:
        pf._iteration_step = orig_step

    print("\n\n=== RMS AT EACH SUB-STEP ===")
    print(
        "iter | start      | +hyper     | +bias      | +scores    | +loadings  | official   | V"
    )
    print(
        "-----|------------|------------|------------|------------|------------|------------|-------"
    )
    for t in step_traces:
        print(
            f"{t['iter']:4d} | {t['rms_0_start']:.6f}  | {t['rms_1_after_hyper']:.6f}  | "
            f"{t['rms_2_after_bias']:.6f}  | {t['rms_3_after_scores']:.6f}  | "
            f"{t['rms_4_after_loadings']:.6f}  | {t['rms_5_official']:.6f}  | "
            f"{t['noise_var']:.6f}"
        )

    # Show which sub-step introduces the alternation
    print("\n--- Per-step deltas (iter N vs iter N-1) ---")
    print("iter | d_start    | d_+bias    | d_+scores  | d_+loadings | d_official")
    for i in range(1, len(step_traces)):
        p, c = step_traces[i - 1], step_traces[i]
        print(
            f"{c['iter']:4d} | "
            f"{c['rms_0_start'] - p['rms_0_start']:+.3e} | "
            f"{c['rms_2_after_bias'] - p['rms_2_after_bias']:+.3e} | "
            f"{c['rms_3_after_scores'] - p['rms_3_after_scores']:+.3e} | "
            f"{c['rms_4_after_loadings'] - p['rms_4_after_loadings']:+.3e} | "
            f"{c['rms_5_official'] - p['rms_5_official']:+.3e}"
        )


if __name__ == "__main__":
    trace_rms_sources()
    check_rotation_effect()
    check_bias_effect()
    check_precenter_data()
    trace_rms_within_iteration()
