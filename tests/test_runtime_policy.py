"""Tests for runtime policy normalization and thread override resolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from vbpca_py._runtime_policy import (
    DenseMaskedAutotuneInputs,
    RuntimeWorkloadProfile,
    SparseAutotuneInputs,
    _default_profile_path,
    _is_explicit_thread_source,
    _load_runtime_profile_data,
    _match_profile_rule,
    _normalize_profile_option,
    _resolve_profile_thread_overrides,
    _resolve_thread_count,
    _safe_autotune_rms_threads,
    _ThreadResolveRequest,
    apply_runtime_policy_defaults,
    autotune_cov_writeback_mode_dense,
    autotune_cov_writeback_mode_sparse,
    resolve_runtime_thread_config,
    resolve_runtime_thread_config_with_report,
)


def test_apply_runtime_policy_defaults_normalizes_fields() -> None:
    opts = {
        "num_cpu": "4",
        "runtime_tuning": "SAFE",
    }

    out = apply_runtime_policy_defaults(opts)

    assert out["num_cpu"] == 4
    assert out["runtime_tuning"] == "safe"


def test_resolve_runtime_thread_config_preserves_legacy_defaults() -> None:
    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 1,
            "_num_cpu_user_set": False,
            "num_cpu_score_update": None,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": None,
        }
    )

    assert cfg.score_update_sparse == 0
    assert cfg.loadings_update_sparse == 0
    assert cfg.noise_sxv_sum == 0
    assert cfg.rms == 1


def test_resolve_runtime_thread_config_explicit_global_num_cpu_applies_to_all() -> None:
    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 6,
            "_num_cpu_user_set": True,
            "num_cpu_score_update": None,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": None,
        }
    )

    assert cfg.score_update_sparse == 6
    assert cfg.loadings_update_sparse == 6
    assert cfg.noise_sxv_sum == 6
    assert cfg.rms == 6


def test_resolve_runtime_thread_config_kernel_specific_overrides_global() -> None:
    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 8,
            "_num_cpu_user_set": True,
            "num_cpu_score_update": 2,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": "3",
            "num_cpu_rms": None,
        }
    )

    assert cfg.score_update_sparse == 2
    assert cfg.loadings_update_sparse == 8
    assert cfg.noise_sxv_sum == 3
    assert cfg.rms == 8


def test_resolve_runtime_thread_config_uses_env_when_no_global_opt(monkeypatch) -> None:
    monkeypatch.setenv("VBPCA_NUM_THREADS", "5")
    monkeypatch.setenv("VBPCA_SCORE_THREADS", "3")

    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 1,
            "_num_cpu_user_set": False,
            "num_cpu_score_update": None,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": None,
        }
    )

    assert cfg.score_update_sparse == 3
    assert cfg.loadings_update_sparse == 5
    assert cfg.noise_sxv_sum == 5
    assert cfg.rms == 5


def test_resolve_runtime_thread_config_invalid_kernel_value_falls_back() -> None:
    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 7,
            "_num_cpu_user_set": True,
            "num_cpu_score_update": "bad",
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": None,
        }
    )

    assert cfg.score_update_sparse == 7


def test_safe_autotune_sets_rms_for_large_sparse_when_unset(monkeypatch) -> None:
    monkeypatch.setattr("vbpca_py._runtime_policy.os.cpu_count", lambda: 16)
    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 1,
            "runtime_tuning": "safe",
            "_num_cpu_user_set": False,
            "num_cpu_score_update": None,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": None,
        },
        workload=RuntimeWorkloadProfile(
            n_features=2000,
            n_samples=1000,
            n_components=24,
            n_observed=3_000_000,
            is_sparse=True,
        ),
    )

    assert cfg.rms == 8


def test_safe_autotune_does_not_override_explicit_rms() -> None:
    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 1,
            "runtime_tuning": "safe",
            "_num_cpu_user_set": False,
            "num_cpu_score_update": None,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": 3,
        },
        workload=RuntimeWorkloadProfile(
            n_features=2000,
            n_samples=1000,
            n_components=24,
            n_observed=3_000_000,
            is_sparse=True,
        ),
    )

    assert cfg.rms == 3


def test_safe_autotune_keeps_dense_rms_single_thread() -> None:
    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 1,
            "runtime_tuning": "safe",
            "_num_cpu_user_set": False,
            "num_cpu_score_update": None,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": None,
        },
        workload=RuntimeWorkloadProfile(
            n_features=120,
            n_samples=400,
            n_components=12,
            n_observed=48_000,
            is_sparse=False,
        ),
    )

    assert cfg.rms == 1


def test_safe_autotune_keeps_wide_sparse_rms_single_thread() -> None:
    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 1,
            "runtime_tuning": "safe",
            "_num_cpu_user_set": False,
            "num_cpu_score_update": None,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": None,
        },
        workload=RuntimeWorkloadProfile(
            n_features=5000,
            n_samples=1000,
            n_components=32,
            n_observed=3_000_000,
            is_sparse=True,
        ),
    )

    assert cfg.rms == 1


def test_runtime_profile_default_threads_override_env(tmp_path, monkeypatch) -> None:
    profile_path = tmp_path / "runtime_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_threads": {
                    "num_cpu_score_update": 3,
                    "num_cpu_loadings_update": 4,
                    "num_cpu_noise_update": 5,
                    "num_cpu_rms": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("VBPCA_NUM_THREADS", "7")

    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 1,
            "runtime_tuning": "off",
            "runtime_profile": str(profile_path),
            "_num_cpu_user_set": False,
            "num_cpu_score_update": None,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": None,
        },
        workload=RuntimeWorkloadProfile(
            n_features=2000,
            n_samples=1000,
            n_components=24,
            n_observed=3_000_000,
            is_sparse=True,
        ),
    )

    assert cfg.score_update_sparse == 3
    assert cfg.loadings_update_sparse == 4
    assert cfg.noise_sxv_sum == 5
    assert cfg.rms == 2


def test_runtime_profile_rule_overrides_default_for_matching_workload(tmp_path) -> None:
    profile_path = tmp_path / "runtime_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_threads": {
                    "num_cpu_rms": 2,
                },
                "workload_rules": [
                    {
                        "match": {
                            "is_sparse": True,
                            "min_features": 4000,
                        },
                        "threads": {
                            "num_cpu_rms": 1,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 1,
            "runtime_tuning": "off",
            "runtime_profile": str(profile_path),
            "_num_cpu_user_set": False,
            "num_cpu_score_update": None,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": None,
        },
        workload=RuntimeWorkloadProfile(
            n_features=5000,
            n_samples=1000,
            n_components=32,
            n_observed=3_000_000,
            is_sparse=True,
        ),
    )

    assert cfg.rms == 1


def test_runtime_profile_respects_explicit_global_num_cpu(tmp_path) -> None:
    profile_path = tmp_path / "runtime_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_threads": {
                    "num_cpu_score_update": 2,
                    "num_cpu_loadings_update": 2,
                    "num_cpu_noise_update": 2,
                    "num_cpu_rms": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 6,
            "runtime_tuning": "off",
            "runtime_profile": str(profile_path),
            "_num_cpu_user_set": True,
            "num_cpu_score_update": None,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": None,
        },
        workload=RuntimeWorkloadProfile(
            n_features=5000,
            n_samples=1000,
            n_components=32,
            n_observed=3_000_000,
            is_sparse=True,
        ),
    )

    assert cfg.score_update_sparse == 6
    assert cfg.loadings_update_sparse == 6
    assert cfg.noise_sxv_sum == 6
    assert cfg.rms == 6


def test_runtime_profile_bad_json_falls_back_to_env(monkeypatch, tmp_path) -> None:
    profile_path = tmp_path / "runtime_profile.json"
    profile_path.write_text("{not json", encoding="utf-8")

    monkeypatch.setenv("VBPCA_NUM_THREADS", "5")

    cfg = resolve_runtime_thread_config(
        {
            "num_cpu": 1,
            "runtime_tuning": "off",
            "runtime_profile": str(profile_path),
            "_num_cpu_user_set": False,
            "num_cpu_score_update": None,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": None,
        },
    )

    assert cfg.score_update_sparse == 5
    assert cfg.loadings_update_sparse == 5
    assert cfg.noise_sxv_sum == 5
    assert cfg.rms == 5


def test_resolve_runtime_thread_config_with_report_includes_sources() -> None:
    cfg, report = resolve_runtime_thread_config_with_report(
        {
            "num_cpu": 4,
            "runtime_tuning": "off",
            "_num_cpu_user_set": True,
            "num_cpu_score_update": 2,
            "num_cpu_loadings_update": None,
            "num_cpu_noise_update": None,
            "num_cpu_rms": None,
        }
    )

    assert cfg.score_update_sparse == 2
    assert cfg.loadings_update_sparse == 4
    assert cfg.noise_sxv_sum == 4
    assert cfg.rms == 4

    kernel_values = report.get("kernel_values", {})
    kernel_sources = report.get("kernel_sources", {})
    assert kernel_values.get("score_update_sparse") == 2
    assert kernel_values.get("loadings_update_sparse") == 4
    assert kernel_values.get("noise_sxv_sum") == 4
    assert kernel_values.get("rms") == 4
    assert kernel_sources.get("score_update_sparse") == "option"
    assert kernel_sources.get("loadings_update_sparse") == "global_num_cpu"
    assert kernel_sources.get("noise_sxv_sum") == "global_num_cpu"
    assert kernel_sources.get("rms") == "global_num_cpu"


def test_normalize_profile_option_auto_blank_and_expanduser(
    monkeypatch, tmp_path
) -> None:
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr("vbpca_py._runtime_policy.Path.home", lambda: fake_home)
    monkeypatch.setenv("HOME", str(fake_home))

    assert _normalize_profile_option(None) is None
    assert _normalize_profile_option("   ") is None
    assert _normalize_profile_option("auto") == _default_profile_path()

    expanded = _normalize_profile_option("~/profile.json")
    assert expanded == Path(str(fake_home / "profile.json"))


def test_load_runtime_profile_data_returns_none_for_non_dict_json(tmp_path) -> None:
    profile_path = tmp_path / "runtime_profile.json"
    profile_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    assert _load_runtime_profile_data(str(profile_path)) is None


def test_match_profile_rule_rejects_out_of_bounds() -> None:
    workload = RuntimeWorkloadProfile(
        n_features=100,
        n_samples=200,
        n_components=10,
        n_observed=10_000,
        is_sparse=True,
    )

    assert not _match_profile_rule({"min_features": 101}, workload)
    assert not _match_profile_rule({"max_features": 99}, workload)
    assert not _match_profile_rule({"min_observed": 20_000}, workload)
    assert not _match_profile_rule({"max_observed": 9_000}, workload)


def test_resolve_profile_thread_overrides_rejects_invalid_schema() -> None:
    out = _resolve_profile_thread_overrides(
        profile_data={
            "schema_version": 2,
            "default_threads": {"num_cpu_rms": 4},
        },
        workload=None,
    )
    assert out == {}


def test_resolve_profile_thread_overrides_handles_malformed_rules_and_negatives() -> (
    None
):
    profile_data = {
        "schema_version": 1,
        "default_threads": {
            "num_cpu_rms": -1,
            "num_cpu_score_update": "3",
        },
        "workload_rules": [
            "not-a-dict",
            {"match": "not-a-dict", "threads": {"num_cpu_rms": 8}},
            {
                "match": {"is_sparse": True, "min_features": 4000},
                "threads": {"num_cpu_rms": 2},
            },
        ],
    }

    workload = RuntimeWorkloadProfile(
        n_features=2000,
        n_samples=500,
        n_components=20,
        n_observed=200_000,
        is_sparse=True,
    )
    out = _resolve_profile_thread_overrides(
        profile_data=profile_data, workload=workload
    )

    assert out == {"num_cpu_score_update": 3}


def test_resolve_thread_count_uses_env_global_then_default() -> None:
    env_global = _resolve_thread_count(
        _ThreadResolveRequest(
            option_value=None,
            use_global_opt=False,
            global_opt_value=9,
            profile_value=None,
            env_specific_value=None,
            env_global_value=7,
            default_value=5,
        )
    )
    assert env_global == 7

    default_only = _resolve_thread_count(
        _ThreadResolveRequest(
            option_value=None,
            use_global_opt=False,
            global_opt_value=9,
            profile_value=None,
            env_specific_value=None,
            env_global_value=None,
            default_value=5,
        )
    )
    assert default_only == 5


def test_safe_autotune_rms_threads_thresholds_and_caps(monkeypatch) -> None:
    monkeypatch.setattr("vbpca_py._runtime_policy.os.cpu_count", lambda: 4)

    sparse_60k = RuntimeWorkloadProfile(
        n_features=200,
        n_samples=100,
        n_components=8,
        n_observed=60_000,
        is_sparse=True,
    )
    assert _safe_autotune_rms_threads(sparse_60k) == 2

    sparse_600k = RuntimeWorkloadProfile(
        n_features=200,
        n_samples=100,
        n_components=8,
        n_observed=600_000,
        is_sparse=True,
    )
    assert _safe_autotune_rms_threads(sparse_600k) == 4

    sparse_3m = RuntimeWorkloadProfile(
        n_features=200,
        n_samples=100,
        n_components=8,
        n_observed=3_000_000,
        is_sparse=True,
    )
    assert _safe_autotune_rms_threads(sparse_3m) == 4


def test_is_explicit_thread_source_detects_env_global_only() -> None:
    assert _is_explicit_thread_source(
        option_value=None,
        global_opt_set=False,
        profile_value=None,
        env_specific=None,
        env_global=3,
    )


def test_autotune_cov_writeback_mode_dense_smoke() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3))
    mask = np.ones_like(x)
    loadings = rng.standard_normal((4, 2))
    scores = rng.standard_normal((2, 3))
    prior_prec = np.eye(2)

    inputs = DenseMaskedAutotuneInputs(
        x_data=x,
        mask=mask,
        loadings=loadings,
        scores=scores,
        noise_var=0.2,
        prior_prec=prior_prec,
    )

    mode, timings, elapsed = autotune_cov_writeback_mode_dense(
        inputs,
        modes=("python", "bulk"),
        reps=1,
        max_total_time=0.2,
        num_cpu_score=1,
        num_cpu_load=1,
    )

    assert mode in {"python", "bulk"}
    assert set(timings) == {"python", "bulk"}
    assert elapsed >= 0.0


def test_autotune_cov_writeback_mode_sparse_smoke() -> None:
    rng = np.random.default_rng(1)
    x_dense = rng.standard_normal((4, 3))
    x_csc = sp.csc_matrix(x_dense)
    x_csr = sp.csr_matrix(x_dense)
    loadings = rng.standard_normal((4, 2))
    scores = rng.standard_normal((2, 3))
    prior_prec = np.eye(2)

    inputs = SparseAutotuneInputs(
        n_features=4,
        n_samples=3,
        x_csc_data=x_csc.data,
        x_csc_indices=x_csc.indices,
        x_csc_indptr=x_csc.indptr,
        x_csr_data=x_csr.data,
        x_csr_indices=x_csr.indices,
        x_csr_indptr=x_csr.indptr,
        loadings=loadings,
        scores=scores,
        noise_var=0.2,
        prior_prec=prior_prec,
    )

    mode, timings, elapsed = autotune_cov_writeback_mode_sparse(
        inputs,
        modes=("python", "bulk"),
        reps=1,
        max_total_time=0.2,
        num_cpu_score=1,
        num_cpu_load=1,
    )

    assert mode in {"python", "bulk"}
    assert set(timings) == {"python", "bulk"}
    assert elapsed >= 0.0
