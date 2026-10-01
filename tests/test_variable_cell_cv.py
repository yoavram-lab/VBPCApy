"""Tests for variable-cell cross-validation and the first-minimum rule (#250, #248)."""

from __future__ import annotations

import numpy as np
import pytest

from vbpca_py import model_selection
from vbpca_py.model_selection import CVConfig, cross_validate_components
from vbpca_py.preprocessing import MissingAwareOneHotEncoder

FIT = {"maxiters": 300, "verbose": 0}


def _categorical_null(
    n: int = 120, n_variables: int = 15, levels: int = 4
) -> np.ndarray:
    rng = np.random.default_rng(3)
    codes = rng.integers(0, levels, size=(n, n_variables)).astype(float)
    codes[rng.random(codes.shape) < 0.05] = np.nan
    return codes


def _one_hot(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    encoder = MissingAwareOneHotEncoder(binary="both")
    z = encoder.fit_transform(codes, mask=~np.isnan(codes))
    assert encoder.feature_groups_ is not None
    return z.T, encoder.feature_groups_


def test_group_folds_hold_out_whole_cells_and_keep_coverage() -> None:
    x, groups = _one_hot(_categorical_null(40, 6))
    rows, cols = np.nonzero(~np.isnan(x))

    folds = model_selection._make_group_folds(x, groups, 5, np.random.default_rng(0))

    held = np.concatenate([probe for probe, _ in folds])
    assert len(np.unique(held)) == len(held)
    for probe, train in folds:
        cells = {(groups[r], c) for r, c in zip(rows[probe], cols[probe], strict=True)}
        for group, column in cells:
            members = np.flatnonzero((groups[rows] == group) & (cols == column))
            assert set(members) <= set(probe)
        assert set(rows[train]) == set(rows)


def test_group_labels_must_cover_every_feature() -> None:
    x, groups = _one_hot(_categorical_null(30, 4))

    with pytest.raises(ValueError, match="label all"):
        model_selection._make_group_folds(x, groups[:-1], 3, np.random.default_rng(0))


def test_first_minimum_stops_at_the_first_flat_step() -> None:
    curve = [
        {"k": k, "mean_prms": m, "se_prms": 0.01}
        for k, m in enumerate([1.0, 0.8, 0.7, 0.72, 0.6])
    ]

    assert model_selection._first_minimum(curve, "prms") == 2


def test_invalid_rule_combinations_are_rejected() -> None:
    with pytest.raises(ValueError, match="early_stop requires"):
        CVConfig(early_stop=True).resolved_rule()
    with pytest.raises(ValueError, match="selection_rule must be"):
        CVConfig(selection_rule="global").resolved_rule()  # type: ignore[arg-type]


def test_early_stop_matches_the_full_sweep() -> None:
    x, groups = _one_hot(_categorical_null(80, 10))
    common = {"feature_groups": groups, "selection_rule": "first_minimum", "seed": 1}

    full_k, full = cross_validate_components(
        x, components=range(6), config=CVConfig(**common), **FIT
    )
    early_k, early = cross_validate_components(
        x, components=range(6), config=CVConfig(**common, early_stop=True), **FIT
    )

    assert early_k == full_k
    assert len(early) <= len(full)
    for short, long in zip(early, full, strict=False):
        assert short["mean_prms"] == pytest.approx(long["mean_prms"])


def _archetypes(n: int = 130, n_variables: int = 6, levels: int = 3) -> np.ndarray:
    """Draw codes from three archetypes (true centered rank 2), with noise.

    Returns:
        Samples-by-variables level codes with 5% missing.
    """
    rng = np.random.default_rng(3)
    profiles = rng.integers(0, levels, size=(3, n_variables))
    member = rng.integers(0, 3, n)
    noise = rng.integers(0, levels, size=(n, n_variables))
    codes = np.where(rng.random((n, n_variables)) < 0.7, profiles[member], noise)
    codes = codes.astype(float)
    codes[rng.random(codes.shape) < 0.05] = np.nan
    return codes


def test_variable_cells_avoid_the_entrywise_leak_on_structured_data() -> None:
    x, groups = _one_hot(_archetypes())

    entrywise_k, _ = cross_validate_components(
        x, components=range(10), config=CVConfig(seed=2), **FIT
    )
    cell_k, _ = cross_validate_components(
        x,
        components=range(10),
        config=CVConfig(feature_groups=groups, selection_rule="first_minimum", seed=2),
        **FIT,
    )

    assert cell_k <= 2
    assert entrywise_k > 2 + cell_k
