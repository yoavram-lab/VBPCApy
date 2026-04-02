#!/usr/bin/env python3
"""Check within-cell SD and t-based CI widths at 10 vs 25 reps."""

import json
from collections import defaultdict

import numpy as np
from scipy import stats

stab = json.load(open("paper/stability_results.json"))
cov = json.load(open("paper/coverage_results.json"))
cov95 = [r for r in cov if abs(r["nominal"] - 0.95) < 0.01]

t9 = stats.t.ppf(0.975, df=9)
t24 = stats.t.ppf(0.975, df=24)
print(f"t_crit: df=9 -> {t9:.3f}, df=24 -> {t24:.3f}")
print()


def summarize(name, cell_dict):
    sds = [np.std(v, ddof=1) for v in cell_dict.values() if len(v) >= 2]
    means = [np.mean(v) for v in cell_dict.values()]
    print(f"--- {name} ---")
    print(f"  Cells: {len(cell_dict)}, reps/cell: {len(list(cell_dict.values())[0])}")
    print(
        f"  Cell SD:  min={min(sds):.3f}  median={np.median(sds):.3f}  mean={np.mean(sds):.3f}  max={max(sds):.3f}"
    )
    hw10 = t9 * np.array(sds) / np.sqrt(10)
    hw25 = t24 * np.array(sds) / np.sqrt(25)
    print(
        f"  95% CI half-width (10 reps):  median=+/-{np.median(hw10):.3f}  max=+/-{max(hw10):.3f}"
    )
    print(
        f"  95% CI half-width (25 reps):  median=+/-{np.median(hw25):.3f}  max=+/-{max(hw25):.3f}"
    )
    print(
        f"  Shrink factor: {np.median(hw10) / np.median(hw25):.2f}x narrower at 25 reps"
    )
    print()


# 1. Accuracy
acc = defaultdict(list)
for r in stab:
    if r["metric"] == "cost":
        acc[(r["n"], r["p"], r["missingness"], r["true_rank"])].append(
            1 if r["selected_k"] == r["true_rank"] else 0
        )
summarize("ACCURACY (exact match, cost)", acc)

# 2. Power
pwr = defaultdict(list)
for r in stab:
    if r["metric"] == "cost":
        pwr[(r["n"], r["p"], r["missingness"], r["true_rank"])].append(
            1 if r["selected_k"] >= r["true_rank"] else 0
        )
summarize("POWER (cost)", pwr)

# 3. RMSE improvement
impr = defaultdict(list)
for r in cov95:
    bl, vb = r["baseline_rmse"], r["holdout_rmse"]
    if bl > 0:
        impr[(r["n"], r["p"], r["missingness"], r["true_rank"])].append(
            (bl - vb) / bl * 100
        )
summarize("RMSE IMPROVEMENT (%)", impr)

# 4. Coverage
covc = defaultdict(list)
for r in cov95:
    covc[(r["n"], r["p"], r["missingness"], r["true_rank"])].append(r["coverage"])
summarize("COVERAGE (at 95% nominal)", covc)

# 5. Bias
for metric in ["cost", "evr95"]:
    bias = defaultdict(list)
    for r in stab:
        if r["metric"] == metric:
            bias[(r["n"], r["p"], r["missingness"], r["true_rank"])].append(
                r["selected_k"] - r["true_rank"]
            )
    summarize(f"BIAS ({metric})", bias)

# 6. MAE
for metric in ["cost", "evr95"]:
    mae = defaultdict(list)
    for r in stab:
        if r["metric"] == metric:
            mae[(r["n"], r["p"], r["missingness"], r["true_rank"])].append(
                abs(r["selected_k"] - r["true_rank"])
            )
    summarize(f"MAE ({metric})", mae)

# Grand summary
print("=" * 70)
print("SUMMARY: CI half-width comparison (10 vs 25 reps)")
print(f"  General shrink factor: {(t9 / np.sqrt(10)) / (t24 / np.sqrt(25)):.2f}x")
print(
    f"  i.e., 25-rep CIs are ~{(t24 / np.sqrt(25)) / (t9 / np.sqrt(10)):.0%} the width of 10-rep CIs"
)
