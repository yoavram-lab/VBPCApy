"""Output budgets cover public materialization paths without promising peak RSS."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py import VBPCA, _memory, estimators
from vbpca_py import _pca_full as pf
from vbpca_py import model_selection as ms


def _options() -> dict[str, object]:
    return {
        "maxiters": 2,
        "niter_broadprior": 0,
        "runtime_tuning": "off",
        "num_cpu": 1,
        "random_state": 8,
        "verbose": 0,
    }


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("estimator", [False, True])
def test_fit_budget_uses_original_shape_before_preparation(
    sparse: bool, estimator: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = np.full((6, 9), np.nan)
    x[:2, :3] = np.arange(6).reshape(2, 3)
    data = sp.csr_matrix(np.nan_to_num(x)) if sparse else x
    # A budget sufficient for reduced data must still reject restored outputs.
    monkeypatch.setattr(
        pf, "_prepare_problem", lambda *_a, **_k: pytest.fail("prepared")
    )
    monkeypatch.setattr(
        estimators, "make_xprobe_mask", lambda *_a, **_k: pytest.fail("probe copied")
    )
    if estimator:
        with pytest.raises(ValueError, match=r"Fit diagnostics.*1296 bytes.*3 float64"):
            VBPCA(
                1, max_dense_bytes=3 * 2 * 3 * 8, xprobe_fraction=0.1, **_options()
            ).fit(data)
    else:
        with pytest.raises(ValueError, match=r"Fit diagnostics.*1296 bytes.*3 float64"):
            pf.pca_full(data, 1, max_dense_bytes=3 * 2 * 3 * 8, **_options())


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("budget", [192, None])
def test_combined_diagnostic_boundary_and_explicit_opt_out(
    sparse: bool, budget: int | None
) -> None:
    x = np.arange(8, dtype=np.float32).reshape(2, 4) + 1
    data = sp.csr_matrix(x) if sparse else x
    model = VBPCA(1, max_dense_bytes=budget, **_options()).fit(data)
    outputs = (model.reconstruction_, model.variance_, model.predictive_variance_)
    assert all(a is not None and a.dtype == np.float64 for a in outputs)
    assert sum(a.nbytes for a in outputs if a is not None) == 192
    np.testing.assert_array_equal(data.toarray() if sparse else data, x)
    with pytest.raises(ValueError, match="Fit diagnostics"):
        VBPCA(1, max_dense_bytes=191, **_options()).fit(data)
    # Each array fits individually, but the three-output operation does not.
    with pytest.raises(ValueError, match="Fit diagnostics"):
        VBPCA(1, max_dense_bytes=64, **_options()).fit(data)


@pytest.mark.parametrize("sparse", [False, True])
def test_omitted_diagnostics_allow_factors_and_temporary_work(
    sparse: bool,
) -> None:
    x = np.arange(8, dtype=float).reshape(2, 4) + 1
    data = sp.csr_matrix(x) if sparse else x
    model = VBPCA(1, max_dense_bytes=1, return_diagnostics=False, **_options()).fit(
        data
    )
    assert model.reconstruction_ is None
    assert model.variance_ is None
    assert model.predictive_variance_ is None
    assert model.components_ is not None
    assert model.components_.nbytes > 1
    with pytest.raises(ValueError, match="Inverse reconstruction"):
        model.inverse_transform()


def test_dense_mask_budget_still_applies_when_diagnostics_are_omitted() -> None:
    x = sp.csr_matrix(np.ones((2, 4)))
    with pytest.raises(ValueError, match="Dense mask"):
        VBPCA(1, max_dense_bytes=1, return_diagnostics=False, **_options()).fit(
            x, mask=np.ones(x.shape, dtype=bool)
        )


def test_inverse_transform_checks_actual_new_sample_count() -> None:
    model = VBPCA(1, max_dense_bytes=64, return_diagnostics=False, **_options()).fit(
        np.arange(8, dtype=float).reshape(2, 4) + 1
    )
    assert model.inverse_transform().nbytes == 64
    with pytest.raises(ValueError, match=r"Inverse reconstruction.*80 bytes"):
        model.inverse_transform(np.ones((1, 5)))
    with pytest.raises(ValueError, match="scores must have shape"):
        model.inverse_transform(np.ones(4))
    model.opts["max_dense_bytes"] = None
    assert model.inverse_transform(np.ones((1, 5))).shape == (2, 5)


def test_model_selection_checks_requested_outputs_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ms, "_fit_candidate", lambda *_a, **_k: pytest.fail("fitted"))
    with pytest.raises(ValueError, match="Selected-model diagnostics"):
        ms.select_n_components(
            np.ones((2, 4)),
            components=[1],
            config=ms.SelectionConfig(metric="rms"),
            max_dense_bytes=191,
            return_diagnostics=False,
        )


def test_selection_without_outputs_and_mean_only_do_not_charge_diagnostics() -> None:
    x = np.arange(8, dtype=float).reshape(2, 4) + 1
    _, _, _, model = ms.select_n_components(
        x,
        components=[1],
        config=ms.SelectionConfig(
            metric="rms", compute_explained_variance=False, return_best_model=True
        ),
        max_dense_bytes=1,
        **_options(),
    )
    assert model is not None
    assert model.reconstruction_ is None
    rank, _, _, model = ms.select_n_components(
        x,
        components=[0],
        config=ms.SelectionConfig(metric="rms", return_best_model=True),
        max_dense_bytes=1,
    )
    assert rank == 0
    assert model is None


def test_selected_model_helpers_guard_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = VBPCA(1, return_diagnostics=False, **_options()).fit(
        np.arange(8, dtype=float).reshape(2, 4) + 1
    )
    model.opts["max_dense_bytes"] = 1
    monkeypatch.setattr(
        ms, "_reconstruct_data", lambda *_a: pytest.fail("reconstructed")
    )
    monkeypatch.setattr(ms, "_marginal_variance", lambda *_a: pytest.fail("variance"))
    with pytest.raises(ValueError, match="Selected-model diagnostics"):
        ms._compute_evr_for_best(model)
    with pytest.raises(ValueError, match="Selected-model diagnostics"):
        ms._compute_variance_for_best(model)


def test_rank_zero_packing_checks_combined_outputs_defensively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = pf.FinalState(
        a=np.zeros((2, 0)),
        s=np.zeros((0, 4)),
        mu=np.zeros((2, 1)),
        noise_var=1.0,
        av=[],
        sv=[],
        pattern_index=None,
        muv=np.zeros(2),
        va=np.zeros(0),
        vmu=1.0,
        lc={},
        runtime_report=None,
    )
    monkeypatch.setattr(
        pf, "_reconstruct_data", lambda *_a: pytest.fail("reconstructed")
    )
    with pytest.raises(ValueError, match="Fit diagnostics"):
        pf._pack_result(final, max_dense_bytes=191)
    assert (
        pf._pack_result(final, max_dense_bytes=1, include_diagnostics=False)["Xrec"]
        is None
    )


def test_large_shape_estimates_do_not_wrap_or_allocate() -> None:
    shape = (2**40, 2**35)
    assert _memory.estimate_dense_bytes(shape, np.float64) == 2**78
    with pytest.raises(ValueError, match="max_dense_bytes"):
        _memory.check_dense_output_budget(shape, 2**63, arrays=3)
    _memory.check_dense_output_budget((2, 4), 192, arrays=3)
    _memory.check_dense_output_budget((0, 2**65), 0, arrays=3)


def test_restored_diagnostics_and_suppression_keep_the_same_fitted_state() -> None:
    x = np.full((6, 9), np.nan)
    x[:2, :3] = np.arange(6).reshape(2, 3) + 1
    with_outputs = VBPCA(1, max_dense_bytes=1296, **_options()).fit(x)
    without_outputs = VBPCA(
        1, max_dense_bytes=1, return_diagnostics=False, **_options()
    ).fit(x)
    for output in (
        with_outputs.reconstruction_,
        with_outputs.variance_,
        with_outputs.predictive_variance_,
    ):
        assert output is not None
        assert output.shape == x.shape
    for name in ("components_", "scores_", "mean_", "noise_variance_", "n_iter_"):
        np.testing.assert_array_equal(
            getattr(with_outputs, name), getattr(without_outputs, name)
        )


def test_selected_model_combined_outputs_fit_exact_budget() -> None:
    _, _, _, model = ms.select_n_components(
        np.arange(8, dtype=float).reshape(2, 4) + 1,
        components=[1],
        config=ms.SelectionConfig(metric="rms", return_best_model=True),
        max_dense_bytes=192,
        **_options(),
    )
    assert model is not None
    outputs = (model.reconstruction_, model.variance_, model.predictive_variance_)
    assert all(a is not None for a in outputs)
    assert sum(a.nbytes for a in outputs if a is not None) == 192
