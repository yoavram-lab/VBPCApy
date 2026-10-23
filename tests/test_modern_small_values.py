"""An explicit modern observation mask must not round finite measurements."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from vbpca_py import VBPCA
from vbpca_py._full_update import _prepare_dense_with_mask_override
from vbpca_py.legacy import VBPCA as LegacyVBPCA  # noqa: N811

OPTIONS = {
    "maxiters": 4,
    "niter_broadprior": 0,
    "random_state": 350,
    "runtime_tuning": "off",
    "num_cpu": 1,
    "verbose": 0,
    "record_cost": True,
    "convergence_criteria": dict.fromkeys(
        ["angle", "earlystop", "rms_plateau", "cost", "composite", "slowing_down"],
        False,
    ),
}


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("algorithm", ["vb", "map", "ppca"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("scale", [1e-6, 1e-10])
@pytest.mark.parametrize("kind", ["all_true", "masked", "probe"])
def test_modern_public_masked_fit_matches_identical_nan_observations(
    rank, algorithm, bias, scale, kind
):
    x = np.array(
        [
            [1, 2, -3, 4, 0, -2, 5, 1],
            [2, -1, 0, 3, 1, 4, -2, 5],
            [1, 4, 2, 3, 5, 2, 6, 0],
        ],
        dtype=float,
    )
    x[:2] *= scale
    mask = np.ones(x.shape, dtype=bool)
    if kind == "masked":
        mask[0, 2], mask[1, 4], mask[2, 5] = False, False, False
        x[~mask] = 19.0
    probe = None
    if kind == "probe":
        probe = np.full_like(x, np.nan)
        probe[0, 0], probe[1, 1] = x[0, 0], x[1, 1]
        probe.flags.writeable = False
    expected = np.where(mask, x, np.nan)
    before_probe = None if probe is None else probe.copy()
    before, before_mask = x.copy(), mask.copy()
    x.flags.writeable = False
    mask.flags.writeable = False
    options = {**OPTIONS, "algorithm": algorithm, "bias": bias}
    actual = LegacyVBPCA(rank, compat_mode="modern", **options).fit(
        x, mask=mask, xprobe=probe
    )
    reference = LegacyVBPCA(rank, compat_mode="modern", **options).fit(
        expected, xprobe=probe
    )
    native = VBPCA(rank, **options).fit(
        x.T, mask=mask.T, xprobe=None if probe is None else probe.T
    )
    for name in [
        "components_",
        "scores_",
        "mean_",
        "noise_variance_",
        "reconstruction_",
        "variance_",
        "predictive_variance_",
        "rms_",
        "prms_",
        "cost_",
    ]:
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(reference, name),
            rtol=2e-12,
            atol=0,
            equal_nan=True,
            err_msg=name,
        )
    np.testing.assert_allclose(
        actual.components_.T, native.components_, rtol=2e-12, atol=0
    )
    np.testing.assert_allclose(actual.scores_.T, native.scores_, rtol=2e-12, atol=0)
    for field in ["rms", "prms", "cost"]:
        np.testing.assert_allclose(
            actual.learning_curve_[field],
            reference.learning_curve_[field],
            rtol=2e-12,
            atol=0,
            equal_nan=True,
        )
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(mask, before_mask)
    if probe is not None:
        np.testing.assert_array_equal(probe, before_probe)


def test_strict_legacy_mask_normalization_retains_historical_epsilon_policy():
    x = np.array([[0.0, 1e-10, -1e-10, 1e-6, np.nan]])
    probe = np.array([[1e-10, -1e-10, 0.0, np.nan, 2.0]])
    mask = np.ones(x.shape, dtype=bool)
    actual, actual_probe, _, probe_mask = _prepare_dense_with_mask_override(
        x.copy(), probe.copy(), mask, "strict_legacy"
    )
    epsilon = np.finfo(float).eps
    np.testing.assert_array_equal(actual, [[epsilon, epsilon, epsilon, 1e-6, 0]])
    np.testing.assert_array_equal(actual_probe, [[epsilon, epsilon, epsilon, 0, 2]])
    np.testing.assert_array_equal(probe_mask, ~np.isnan(probe))


def test_executed_compatibility_example_reports_correct_defaults_and_orders():
    script = Path(__file__).parents[1] / "scripts/example_compatibility_modes.py"
    result = runpy.run_path(str(script))["run_example"]()
    assert result["native"].compat_mode == "modern"
    assert result["historical"].compat_mode == "strict_legacy"
    for name, bias, variance in [
        ("native", "post_factor", "post_rotation"),
        ("historical", "legacy", "legacy"),
    ]:
        report = result[name].runtime_report_
        assert report["bias_update_order"] == bias
        assert report["variance_update_order"] == variance
