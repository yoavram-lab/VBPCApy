"""Tests for runtime policy normalization and thread override resolution."""

from __future__ import annotations

import json

from vbpca_py._runtime_policy import (
    RuntimeWorkloadProfile,
    apply_runtime_policy_defaults,
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
