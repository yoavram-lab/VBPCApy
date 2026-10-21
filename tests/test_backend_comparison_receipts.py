"""The profiling comparator must reject incomplete or changed fit snapshots."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

_COMPARE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "analysis/compare_backend_revisions.py")
)["_compare_snapshots"]


@pytest.mark.parametrize(
    "change", ["none", "value", "dtype", "field", "extra", "missing", "empty"]
)
def test_comparison_requires_equal_complete_snapshots(tmp_path, change) -> None:
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    values = np.array([1.0, np.nan, np.inf])
    np.savez(reference / "fit-0.npz", values=values)
    if change == "value":
        values[0] = 2
    elif change == "dtype":
        values = values.astype(np.float32)
    if change == "field":
        np.savez(candidate / "fit-0.npz", other=values)
    elif change != "missing":
        np.savez(candidate / "fit-0.npz", values=values)
    if change == "extra":
        np.savez(candidate / "fit-1.npz", values=values)
    elif change == "empty":
        (reference / "fit-0.npz").unlink()
        (candidate / "fit-0.npz").unlink()
    if change == "none":
        receipt = _COMPARE(reference, candidate)
        assert receipt["snapshots"][0]["arrays"] == ["values"]
    else:
        with pytest.raises((AssertionError, ValueError, FileNotFoundError)):
            _COMPARE(reference, candidate)
