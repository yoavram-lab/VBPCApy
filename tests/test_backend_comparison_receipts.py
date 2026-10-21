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


def _gaussian_snapshot():
    rng = np.random.default_rng(289)
    a, s = rng.normal(size=(5, 3)), rng.normal(size=(3, 7))
    ca, cs = rng.normal(size=(5, 3, 3)), rng.normal(size=(7, 3, 3))
    av, sv = ca @ ca.transpose(0, 2, 1), cs @ cs.transpose(0, 2, 1)
    mu, muv = rng.normal(size=(5, 1)), rng.uniform(size=(5, 1))
    # Independent cell-wise uncertainty equation; no production contraction.
    vr = np.array([
        [
            a[i] @ sv[j] @ a[i]
            + s[:, j] @ av[i] @ s[:, j]
            + np.trace(av[i] @ sv[j])
            + muv[i, 0]
            for j in range(7)
        ]
        for i in range(5)
    ])
    return {
        "A": a,
        "S": s,
        "Av": av,
        "Sv": sv,
        "Mu": mu,
        "Muv": muv,
        "V": np.array(0.5),
        "Va": np.ones(3),
        "Xrec": a @ s + mu,
        "Vr": vr,
        "Vpred": vr + 0.5,
    }


@pytest.mark.parametrize(
    "change", ["none", "covariance", "prediction", "noise", "scale", "prior", "missing"]
)
def test_latent_rotation_comparison_preserves_full_posterior_gates(tmp_path, change):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    expected = _gaussian_snapshot()
    actual = {key: value.copy() for key, value in expected.items()}
    angle = 0.0002
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    actual.update(
        A=expected["A"] @ rotation,
        S=rotation.T @ expected["S"],
        Av=rotation.T @ expected["Av"] @ rotation,
        Sv=rotation.T @ expected["Sv"] @ rotation,
    )
    if change == "covariance":
        actual["Av"][0, 0, 0] += 0.1
    elif change == "prediction":
        actual["Xrec"][0, 0] += 0.1
    elif change == "noise":
        actual["V"] += 0.1
    elif change == "scale":
        actual["A"] *= 2
        actual["S"] /= 2
    elif change == "prior":
        actual["Va"][0] += 0.1
    elif change == "missing":
        expected.pop("Vpred")
        actual.pop("Vpred")
    np.savez(reference / "fit-0.npz", **expected)
    np.savez(candidate / "fit-0.npz", **actual)
    if change == "none":
        receipt = _COMPARE(reference, candidate, tolerant=True, align_latent=True)
        assert receipt["latent_alignment"] is True
        assert (
            "A"
            in receipt["snapshots"][0]["latent_alignment"][
                "raw_component_gate_failures"
            ]
        )
        # The historical allocation comparator retains exact coordinates.
        with pytest.raises(AssertionError):
            _COMPARE(reference, candidate)
    else:
        with pytest.raises((AssertionError, ValueError)):
            _COMPARE(reference, candidate, tolerant=True, align_latent=True)
