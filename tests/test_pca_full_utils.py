"""Cover helper utilities in _pca_full without running the full algorithm."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import vbpca_py._pca_full as pf


def test_coerce_and_float_helpers() -> None:
    assert pf._coerce_int(None, default=5) == 5
    assert pf._coerce_int("bad", default=7) == 7
    assert pf._int_opt({"not": "int"}, default=9) == 9
    assert pf._float_opt(b"not-a-float", default=1.5) == pytest.approx(1.5)


def test_auto_masked_batch_size_branches() -> None:
    dummy_mask = sp.csr_matrix(np.ones((4, 4)))
    prepared_small = pf.PreparedProblem(
        x_data=dummy_mask,
        x_probe=None,
        mask=dummy_mask,
        mask_probe=None,
        n_obs_row=np.ones(4),
        n_data=10,
        n_probe=0,
        ix_obs=np.array([0]),
        jx_obs=np.array([0]),
        n_features=2,
        n_samples=2,
        n1x=2,
        n2x=2,
        row_idx=None,
        col_idx=None,
        n_patterns=10,
        obs_patterns=[],
        pattern_index=None,
    )
    opts: dict[str, object] = {}
    assert pf._auto_masked_batch_size(prepared_small, opts) == 0

    prepared_large = pf.PreparedProblem(
        **{
            **prepared_small.__dict__,
            "n_patterns": 300,
            "n_data": 250_000,
        }
    )
    auto_batch = pf._auto_masked_batch_size(prepared_large, opts)
    assert auto_batch in {192, 256}


def test_adjust_opts_for_explicit_init() -> None:
    class DummyLc:
        rms = [1.0, 2.0]

    class DummyInit:
        lc = DummyLc()

    opts: dict[str, object] = {
        "maxiters": 2,
        "niter_broadprior": 1,
        "init": DummyInit(),
    }
    pf._adjust_opts_for_explicit_init(opts)
    assert opts["niter_broadprior"] > opts["maxiters"]
    assert opts["maxiters"] == 0


def test_validate_dense_mask_budget_and_preflight() -> None:
    x_sparse = sp.csr_matrix(np.ones((2, 2)))
    mask_dense = np.ones((2, 2))
    opts: dict[str, object] = {"max_dense_bytes": 1}
    with pytest.raises(ValueError):
        pf._validate_dense_mask_budget(x_sparse, mask_dense, opts)


def test_build_options_and_select_algorithm_errors() -> None:
    with pytest.raises(ValueError):
        pf._build_options({"compat_mode": "invalid"})

    with pytest.raises(ValueError):
        pf._build_options({"explained_var_solver": "bad"})

    with pytest.raises(ValueError):
        pf._select_algorithm({"algorithm": "wrong"})


def test_explained_variance_paths() -> None:
    xrec = np.array([[1.0, 2.0], [3.0, 4.0]])
    ev, evr = pf._explained_variance(xrec, 1, solver="unknown")
    assert ev.size == 1
    assert evr.size == 1

    tall = np.arange(12, dtype=float).reshape(3, 4)
    ev_tall = pf._explained_variance_tall(
        x_centered=tall, solver="gram", gram_ratio=1.0
    )
    assert np.all(ev_tall >= -1e-12)


def test_pack_result_includes_and_excludes_diagnostics() -> None:
    final = pf.FinalState(
        a=np.ones((2, 1)),
        s=np.ones((1, 2)),
        mu=np.zeros((2, 1)),
        noise_var=0.1,
        av=[np.eye(1), np.eye(1)],
        sv=[np.eye(1), np.eye(1)],
        pattern_index=None,
        muv=np.zeros((2, 1)),
        va=np.ones(1),
        vmu=0.1,
        lc={"rms": [0.5], "prms": [0.4], "cost": [0.3]},
        runtime_report=None,
    )

    res_diag = pf._pack_result(final, include_diagnostics=True)
    assert res_diag["Xrec"] is not None
    assert np.isfinite(res_diag["RMS"])

    res_no_diag = pf._pack_result(final, include_diagnostics=False)
    assert res_no_diag["Xrec"] is None
