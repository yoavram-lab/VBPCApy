"""Missingness and scale diagnostics for check_data (#249)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from ._encoding_checks import _is_indicator

if TYPE_CHECKING:
    from collections.abc import Sequence

SCALE_RATIO_WARN = 10.0
UNIT_SCALE_RANGE = (0.1, 10.0)
MCAR_MIN_GROUP = 5
MCAR_MIN_FRACTION = 0.05


def _name(names: Sequence[str] | None, j: int) -> str:
    return names[j] if names is not None else str(j)


def missingness_summary(
    observed: np.ndarray, row_threshold: float
) -> tuple[dict[str, Any], list[str]]:
    """Summarize where values are missing.

    Returns:
        The overall and per-row missing fractions, complete rows, the number
        of distinct missingness patterns and the rows above
        ``row_threshold``, with a warning when any row exceeds it.
    """
    n_samples, n_features = observed.shape
    row_missing = 1.0 - observed.mean(axis=1) if n_features else np.zeros(n_samples)
    above = int(np.sum(row_missing > row_threshold))
    patterns = np.unique(observed, axis=0).shape[0] if n_samples else 0
    summary: dict[str, Any] = {
        "fraction_missing": float(1.0 - observed.mean()) if observed.size else 0.0,
        "complete_rows": int(np.sum(observed.all(axis=1))),
        "n_patterns": int(patterns),
        "max_row_missing_fraction": float(row_missing.max()) if n_samples else 0.0,
        "rows_above_threshold": above,
    }
    messages = []
    if above:
        messages.append(
            f"{above} of {n_samples} rows have more than {row_threshold:.0%} of "
            "their values missing; they carry little information and may "
            "destabilize the fit"
        )
    return summary, messages


def _continuous_columns(
    x: np.ndarray, observed: np.ndarray, groups: np.ndarray
) -> list[int]:
    """Return columns that are not indicators and not part of a one-hot block."""
    sizes = np.bincount(groups)
    return [
        j
        for j in range(x.shape[1])
        if sizes[groups[j]] == 1
        and np.sum(observed[:, j]) >= 2
        and not _is_indicator(x[observed[:, j], j])
    ]


def scale_warnings(
    x: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    names: Sequence[str] | None,
) -> list[str]:
    """Flag continuous columns on very different or far-from-unit scales.

    VBPCA's default priors on loading and noise variances are absolute, so
    they behave as intended only on roughly unit-scale data; and a column
    with a much larger spread dominates the leading components.

    Returns:
        Warnings recommending per-feature standardization.
    """
    columns = _continuous_columns(x, observed, groups)
    sds = {j: float(np.std(x[observed[:, j], j])) for j in columns}
    sds = {j: sd for j, sd in sds.items() if sd > 0.0}
    if not sds:
        return []
    messages = []
    largest = max(sds, key=sds.__getitem__)
    smallest = min(sds, key=sds.__getitem__)
    ratio = sds[largest] / sds[smallest]
    if ratio > SCALE_RATIO_WARN:
        messages.append(
            f"continuous feature scales differ by a factor of {ratio:.0f} "
            f"({_name(names, largest)} vs {_name(names, smallest)}); standardize "
            "each feature (MissingAwareStandardScaler) so no feature dominates"
        )
    median = float(np.median(list(sds.values())))
    low, high = UNIT_SCALE_RANGE
    if not low <= median <= high:
        messages.append(
            f"the median standard deviation of continuous features is {median:.3g}; "
            "VBPCA's default priors assume roughly unit-scale data, so standardize "
            "the features before fitting"
        )
    return messages


def mcar_screen(
    x: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    names: Sequence[str] | None,
    *,
    alpha: float = 0.05,
) -> tuple[dict[str, Any], list[str]]:
    """Screen for missingness that depends on observed values.

    For every column with between 5% and 95% missing, compare each other
    continuous column between rows where the first is missing and rows
    where it is observed (Welch's t-test), with a Bonferroni correction over
    all pairs. A significant pair is evidence against MCAR; no significant
    pair does not establish MCAR.

    Returns:
        The number of pairs tested and flagged, and a warning naming the
        strongest flagged pair.
    """
    continuous = _continuous_columns(x, observed, groups)
    missing_fraction = 1.0 - observed.mean(axis=0)
    targets = [
        j
        for j in range(x.shape[1])
        if MCAR_MIN_FRACTION <= missing_fraction[j] <= 1.0 - MCAR_MIN_FRACTION
    ]
    results: list[tuple[float, int, int]] = []
    for j in targets:
        for k in continuous:
            if k == j:
                continue
            values_k = observed[:, k]
            absent = x[~observed[:, j] & values_k, k]
            present = x[observed[:, j] & values_k, k]
            if min(absent.size, present.size) < MCAR_MIN_GROUP:
                continue
            p_value = float(stats.ttest_ind(absent, present, equal_var=False).pvalue)
            if np.isfinite(p_value):
                results.append((p_value, j, k))
    tested = len(results)
    flagged = [r for r in results if r[0] < alpha / max(tested, 1)]
    summary = {"mcar_pairs_tested": tested, "mcar_pairs_flagged": len(flagged)}
    messages = []
    if flagged:
        p_value, j, k = min(flagged)
        messages.append(
            f"missingness is not completely at random: {len(flagged)} of {tested} "
            f"column pairs differ between missing and observed rows (strongest: "
            f"{_name(names, k)} by missingness of {_name(names, j)}, "
            f"p = {p_value:.2g}, Bonferroni-corrected at {alpha})"
        )
    return summary, messages
