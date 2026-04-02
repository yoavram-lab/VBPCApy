#!/usr/bin/env python3
"""Compute all key paper numbers with 95% CIs from the JSON data."""

import json
from collections import defaultdict

import numpy as np

with open("paper/stability_results.json") as f:
    stab = json.load(f)
with open("paper/coverage_results.json") as f:
    cov = json.load(f)


def ci95(values, directional=None):
    a = np.array(values, dtype=float)
    if directional == "upper":
        return np.mean(a), np.percentile(a, 5), np.percentile(a, 100)
    if directional == "lower":
        return np.mean(a), np.percentile(a, 0), np.percentile(a, 95)
    return np.mean(a), np.percentile(a, 2.5), np.percentile(a, 97.5)


def boot_ci(arr, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.array(arr, dtype=float)
    boots = [a[rng.choice(len(a), len(a), replace=True)].mean() for _ in range(n)]
    boots = np.array(boots)
    return a.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


cov95 = [r for r in cov if abs(r["nominal"] - 0.95) < 0.01]

# ── 1. Accuracy range ──
print("=" * 70)
print("1. ACCURACY RANGE (exact rank recovery, cost metric)")
acc_cells = defaultdict(list)
for r in stab:
    if r["metric"] == "cost":
        acc_cells[(r["n"], r["p"], r["missingness"])].append(
            1 if r["selected_k"] == r["true_rank"] else 0
        )
cell_accs = [np.mean(v) for v in acc_cells.values()]
print(f"  Cell-level range: {min(cell_accs) * 100:.0f}%–{max(cell_accs) * 100:.0f}%")
m, lo, hi = boot_ci(
    [
        1 if r["selected_k"] == r["true_rank"] else 0
        for r in stab
        if r["metric"] == "cost"
    ]
)
print(f"  Overall mean: {m * 100:.1f}% (95% CI {lo * 100:.1f}–{hi * 100:.1f}%)")

# ── 2. Power ──
print("\n" + "=" * 70)
print("2. DETECTION POWER (cost metric)")
for kt in [2, 5, 10]:
    vals = [
        1 if r["selected_k"] >= r["true_rank"] else 0
        for r in stab
        if r["metric"] == "cost" and r["true_rank"] == kt
    ]
    m, lo, hi = boot_ci(vals)
    print(f"  k={kt}: {m * 100:.1f}% (95% CI {lo * 100:.1f}–{hi * 100:.1f}%)")

# ── 3. RMSE improvement by pattern ──
print("\n" + "=" * 70)
print("3. RMSE IMPROVEMENT BY PATTERN")
per_pat = defaultdict(list)
for r in cov95:
    bl, vb = r["baseline_rmse"], r["holdout_rmse"]
    if bl > 0:
        per_pat[r["missingness"]].append((bl - vb) / bl * 100)
for pat in ["complete", "mcar", "mnar", "block"]:
    m, lo, hi = boot_ci(per_pat[pat])
    print(f"  {pat}: {m:.1f}% (95% CI {lo:.1f}–{hi:.1f}%)")
pat_means = {p: np.mean(v) for p, v in per_pat.items()}
print(
    f"  Range of pattern means: {min(pat_means.values()):.1f}–{max(pat_means.values()):.1f}%"
)

# ── 4. RMSE heatmap by (n,p) ──
print("\n" + "=" * 70)
print("4. RMSE HEATMAP by (n,p)")
cell_impr = defaultdict(list)
for r in cov95:
    bl, vb = r["baseline_rmse"], r["holdout_rmse"]
    if bl > 0:
        cell_impr[(r["n"], r["p"])].append((bl - vb) / bl * 100)
np_impr = {k: np.mean(v) for k, v in cell_impr.items()}
min_c = min(np_impr, key=np_impr.get)
max_c = max(np_impr, key=np_impr.get)
for label, key in [("Min", min_c), ("Max", max_c)]:
    m, lo, hi = boot_ci(cell_impr[key])
    print(f"  {label}: (n={key[0]}, p={key[1]}) = {m:.1f}% (95% CI {lo:.1f}–{hi:.1f}%)")

# ── 5. High-p (p>=50) improvement ──
print("\n" + "=" * 70)
print("5. HIGH-P (p>=50) IMPROVEMENT")
hp = [v for (n, p), vals in cell_impr.items() if p >= 50 for v in vals]
m, lo, hi = boot_ci(hp)
print(f"  Mean: {m:.1f}% (95% CI {lo:.1f}–{hi:.1f}%)")
hp_cell = [np_impr[k] for k in np_impr if k[1] >= 50]
print(f"  Cell-mean range: {min(hp_cell):.1f}–{max(hp_cell):.1f}%")

# ── 6. Coverage at 95% nominal ──
print("\n" + "=" * 70)
print("6. COVERAGE AT 95% NOMINAL")
c_vals = [r["coverage"] for r in cov95]
m, lo, hi = boot_ci(c_vals)
print(f"  Overall: {m * 100:.1f}% (95% CI {lo * 100:.1f}–{hi * 100:.1f}%)")
for pat in ["complete", "mcar", "mnar", "block"]:
    vals = [r["coverage"] for r in cov95 if r["missingness"] == pat]
    m, lo, hi = boot_ci(vals)
    print(f"  {pat}: {m * 100:.1f}% (95% CI {lo * 100:.1f}–{hi * 100:.1f}%)")

# ── 7. Error decomposition ──
print("\n" + "=" * 70)
print("7. ERROR DECOMPOSITION (MAE, over/under rates)")
for metric in ["cost", "prms", "evr95"]:
    t = [r for r in stab if r["metric"] == metric]
    if not t:
        continue
    maes = [abs(r["selected_k"] - r["true_rank"]) for r in t]
    m, lo, hi = boot_ci(maes)
    print(f"  {metric} MAE: {m:.2f} (95% CI {lo:.2f}–{hi:.2f})")
    ov = [1 if r["selected_k"] > r["true_rank"] else 0 for r in t]
    un = [1 if r["selected_k"] < r["true_rank"] else 0 for r in t]
    mo, lo2, hi2 = boot_ci(ov)
    mu, lo3, hi3 = boot_ci(un)
    print(f"    Over: {mo * 100:.1f}% (95% CI {lo2 * 100:.1f}–{hi2 * 100:.1f}%)")
    print(f"    Under: {mu * 100:.1f}% (95% CI {lo3 * 100:.1f}–{hi3 * 100:.1f}%)")

# ── 8. Asymmetry paragraph numbers ──
print("\n" + "=" * 70)
print("8. ASYMMETRY / BIAS")
for metric in ["cost", "evr95"]:
    t = [r for r in stab if r["metric"] == metric]
    biases = [r["selected_k"] - r["true_rank"] for r in t]
    m, lo, hi = boot_ci(biases)
    print(f"  {metric} bias: {m:.2f} (95% CI {lo:.2f}–{hi:.2f})")

evr_mnar = [r for r in stab if r["metric"] == "evr95" and r["missingness"] == "mnar"]
vals = [1 if r["selected_k"] > r["true_rank"] else 0 for r in evr_mnar]
m, lo, hi = boot_ci(vals)
print(
    f"  EVR95 MNAR over-select: {m * 100:.1f}% (95% CI {lo * 100:.1f}–{hi * 100:.1f}%)"
)

evr_all = [r for r in stab if r["metric"] == "evr95"]
vals_u = [1 if r["selected_k"] < r["true_rank"] else 0 for r in evr_all]
m_u, lo_u, hi_u = boot_ci(vals_u)
print(
    f"  EVR95 under-select: {m_u * 100:.1f}% (95% CI {lo_u * 100:.1f}–{hi_u * 100:.1f}%)"
)
vals_o = [1 if r["selected_k"] > r["true_rank"] else 0 for r in evr_all]
m_o, lo_o, hi_o = boot_ci(vals_o)
print(
    f"  EVR95 over-select: {m_o * 100:.1f}% (95% CI {lo_o * 100:.1f}–{hi_o * 100:.1f}%)"
)

# ── 9. EVR95 MAE by pattern ──
print("\n" + "=" * 70)
print("9. EVR95 MAE by pattern")
for pat in ["complete", "mcar", "mnar", "block"]:
    t = [r for r in stab if r["metric"] == "evr95" and r["missingness"] == pat]
    maes = [abs(r["selected_k"] - r["true_rank"]) for r in t]
    m, lo, hi = boot_ci(maes)
    print(f"  {pat}: {m:.2f} (95% CI {lo:.2f}–{hi:.2f})")
print("  Cost MAE by pattern:")
for pat in ["complete", "mcar", "mnar", "block"]:
    t = [r for r in stab if r["metric"] == "cost" and r["missingness"] == pat]
    maes = [abs(r["selected_k"] - r["true_rank"]) for r in t]
    m, lo, hi = boot_ci(maes)
    print(f"  {pat}: {m:.2f} (95% CI {lo:.2f}–{hi:.2f})")

# ── 10. MAE ratio ──
print("\n" + "=" * 70)
print("10. MAE RATIO")
cost_m = [abs(r["selected_k"] - r["true_rank"]) for r in stab if r["metric"] == "cost"]
evr_m = [abs(r["selected_k"] - r["true_rank"]) for r in stab if r["metric"] == "evr95"]
ratio = np.mean(evr_m) / np.mean(cost_m)
print(f"  EVR95/Cost = {ratio:.2f}x")
rng = np.random.default_rng(42)
ca, ea = np.array(cost_m, dtype=float), np.array(evr_m, dtype=float)
br = []
for _ in range(10000):
    cm = ca[rng.choice(len(ca), len(ca), replace=True)].mean()
    em = ea[rng.choice(len(ea), len(ea), replace=True)].mean()
    if cm > 0:
        br.append(em / cm)
br = np.array(br)
print(
    f"  Bootstrap 95% CI: {np.percentile(br, 2.5):.2f}–{np.percentile(br, 97.5):.2f}x"
)

print("\n" + "=" * 70)
print("DONE")
