"""Independent execution-path, allocation and persistent-profile contracts."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vbpca_py import _runtime_profiles as profiles
from vbpca_py import dense_update_kernels as duk
from vbpca_py._cpu_limits import available_cpu_count, cgroup_cpu_limit
from vbpca_py._runtime_fingerprint import runtime_environment
from vbpca_py._runtime_policy import (
    DenseMaskedAutotuneInputs,
    RuntimeWorkloadProfile,
    SparseAutotuneInputs,
    _benchmark_pair_candidates,
    _resolve_profile_thread_overrides,
    autotune_cov_writeback_mode_dense,
    autotune_dense_masked_threads,
    autotune_sparse_nopattern_threads,
)


@pytest.mark.parametrize("covariances", [False, True])
@pytest.mark.parametrize("complement", [False, True])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_dense_probe_measures_covariance_layout_and_statistics(
    monkeypatch, covariances, complement, layout
):
    x = np.array(np.arange(24).reshape(6, 4), dtype=float, order=layout)
    a = np.array(np.ones((6, 2)), order=layout)
    s = np.array(np.ones((2, 4)), order=layout)
    av = np.repeat(np.eye(2)[None], 6, axis=0) if covariances else None
    sv = np.repeat(np.eye(2)[None], 4, axis=0) if covariances else None
    calls = []

    def score(**kwargs):
        calls.append("score")
        assert kwargs["loading_covariances"] is av
        assert kwargs["return_covariances"] is covariances
        assert kwargs["use_complement"] is complement
        assert kwargs["x_data"].strides == x.strides
        assert kwargs["loadings"].strides == a.strides

    def loading(**kwargs):
        calls.append("loadings")
        assert kwargs["score_covariances"] is sv
        assert kwargs["return_covariances"] is covariances
        assert kwargs["use_complement"] is complement
        assert kwargs["scores"].strides == s.strides

    monkeypatch.setattr(
        "vbpca_py._runtime_policy.duk.score_update_dense_masked_nopattern", score
    )
    monkeypatch.setattr(
        "vbpca_py._runtime_policy.duk.loadings_update_dense_masked_nopattern", loading
    )
    inputs = DenseMaskedAutotuneInputs(
        x,
        np.ones(x.shape, dtype=bool),
        a,
        s,
        1.0,
        np.eye(2),
        av,
        sv,
        covariances,
        covariances,
        complement,
        complement,
    )
    _, _, report = autotune_dense_masked_threads(
        inputs, candidates=[1, 1], max_total_time=1.0
    )
    assert calls == ["score", "score", "loadings", "loadings"]
    for phase in ["score", "loadings"]:
        evidence = report["cold_warm_samples"][phase]["1"]
        assert evidence["cold_seconds"] >= 0
        assert len(evidence["warm_seconds"]) == 1


def test_sparse_probe_preserves_full_covariances(monkeypatch):
    av = np.repeat(np.eye(2)[None], 3, axis=0)
    sv = np.repeat(np.eye(2)[None], 4, axis=0)

    def score(**kwargs):
        assert kwargs["loading_covariances"] is av
        assert kwargs["return_covariances"]

    def load(**kwargs):
        assert kwargs["score_covariances"] is sv
        assert kwargs["return_covariances"]

    monkeypatch.setattr(
        "vbpca_py._runtime_policy.suk.score_update_sparse_nopattern", score
    )
    monkeypatch.setattr(
        "vbpca_py._runtime_policy.suk.loadings_update_sparse_nopattern", load
    )
    inputs = SparseAutotuneInputs(
        3,
        4,
        np.ones(1),
        np.zeros(1),
        np.zeros(5),
        np.ones(1),
        np.zeros(1),
        np.zeros(4),
        np.ones((3, 2)),
        np.ones((2, 4)),
        1.0,
        np.eye(2),
        av,
        sv,
    )
    _, _, report = autotune_sparse_nopattern_threads(
        inputs, candidates=[1], max_total_time=1.0
    )
    assert report["workload"]["loading_covariances"]
    assert report["workload"]["score_covariances"]


def test_clamped_and_writeback_aliases_do_not_duplicate_probes(monkeypatch):
    inputs = DenseMaskedAutotuneInputs(
        np.ones((3, 4)),
        np.ones((3, 4), dtype=bool),
        np.ones((3, 2)),
        np.ones((2, 4)),
        1.0,
        np.eye(2),
    )
    _, timings, _ = autotune_cov_writeback_mode_dense(
        inputs, modes=["bulk", "kernel", "python"], max_total_time=2.0
    )
    assert set(timings) == {"bulk", "python"}
    calls = []
    _benchmark_pair_candidates(
        [1],
        [1, 2],
        lambda n: calls.append(("score", n)) or 1.0,
        lambda n: calls.append(("load", n)) or 1.0,
        1.0,
    )
    assert calls == [("score", 1), ("load", 1), ("load", 2)]


def test_quota_includes_nested_ancestors_and_fractional_limits(tmp_path):
    group = tmp_path / "group" / "job"
    group.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("0::/group/job\n")
    (tmp_path / "cpu.max").write_text("max 100000")
    (group / "cpu.max").write_text("350000 100000")
    (group.parent / "cpu.max").write_text("180000 100000")
    assert cgroup_cpu_limit(root=tmp_path, membership=membership) == 1
    (group.parent / "cpu.max").write_text("malformed")
    assert cgroup_cpu_limit(root=tmp_path, membership=membership) == 3
    membership.write_text("2:cpu,cpuacct:/job\n")
    v1 = tmp_path / "cpu" / "job"
    v1.mkdir(parents=True)
    (v1 / "cpu.cfs_quota_us").write_text("25000")
    (v1 / "cpu.cfs_period_us").write_text("100000")
    assert cgroup_cpu_limit(root=tmp_path, membership=membership) == 1


def test_native_effective_counts_use_actual_allocation(monkeypatch):
    assert duk.effective_thread_count(100000, 1000000) == available_cpu_count()
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert duk.effective_thread_count(8, 100) == 1
    assert available_cpu_count() == 1


@pytest.mark.parametrize(
    "key",
    [
        "affinity",
        "cpu_model",
        "versions",
        "native_sha256",
        "blas_pools",
        "thread_environment",
        "cgroup_cpu_limit",
    ],
)
def test_profiles_reject_changed_environment_and_keep_old_file(
    tmp_path, monkeypatch, key
):
    environment = runtime_environment()
    monkeypatch.setattr(
        profiles, "runtime_environment", lambda: copy.deepcopy(environment)
    )
    path = tmp_path / "profile.json"
    assert profiles.save_profile_rule(
        path, {"match": {"min_features": 1}, "threads": {}}
    )
    original = path.read_bytes()
    assert profiles.load_profile(path)
    environment[key] = "changed"
    assert profiles.load_profile(path) is None
    assert not profiles.save_profile_rule(path, {"match": {}, "threads": {}})
    assert path.read_bytes() == original


def test_profiles_deduplicate_and_atomic_failure_preserves_previous(
    tmp_path, monkeypatch
):
    path = tmp_path / "profile.json"
    rule = {"match": {"min_features": 2}, "threads": {"num_cpu_rms": 1}}
    assert profiles.save_profile_rule(path, rule)
    rule["threads"] = {"num_cpu_rms": 2}
    assert profiles.save_profile_rule(path, rule)
    data = profiles.load_profile(path)
    assert len(data["workload_rules"]) == 1
    assert data["workload_rules"][0]["threads"]["num_cpu_rms"] == 2
    original = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated interrupted replacement")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        profiles.atomic_write_profile(path, profiles.new_profile())
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_profile_writers_preserve_each_distinct_rule(tmp_path):
    path = tmp_path / "profile.json"
    code = """
import sys
from pathlib import Path
from vbpca_py import _runtime_profiles as p
p.runtime_environment = lambda: {"fixed_test_context": True}
rule = {"match":{"min_features":int(sys.argv[2])},"threads":{}}
assert p.save_profile_rule(Path(sys.argv[1]), rule)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(path), str(i)], env=os.environ.copy()
        )
        for i in range(4)
    ]
    assert [process.wait(timeout=30) for process in processes] == [0] * 4
    data = json.loads(path.read_text())
    assert {row["match"]["min_features"] for row in data["workload_rules"]} == set(
        range(4)
    )


def test_profiles_match_execution_semantics():
    workload = RuntimeWorkloadProfile(
        30, 40, 8, 1100, is_sparse=False, execution=(("score_complement", "True"),)
    )
    data = {
        "schema_version": 2,
        "workload_rules": [
            {
                "match": {},
                "execution": {"score_complement": "False"},
                "threads": {"num_cpu_rms": 3},
            }
        ],
    }
    assert _resolve_profile_thread_overrides(profile_data=data, workload=workload) == {}
    workload = replace(workload, execution=(("score_complement", "False"),))
    assert _resolve_profile_thread_overrides(profile_data=data, workload=workload) == {
        "num_cpu_rms": 3
    }


@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
def test_orchestration_resolves_statistics_and_retains_measured_threads(
    monkeypatch, algorithm
):
    import vbpca_py._pca_full as fitter

    rng = np.random.default_rng(293)
    x = rng.normal(size=(80, 96))
    x[rng.random(x.shape) < 0.01] = np.nan
    opts = fitter._build_options({
        "algorithm": algorithm,
        "maxiters": 0,
        "runtime_tuning": "safe",
        "dense_sufficient_statistics": "complement",
        "uniquesv": False,
        "auto_pattern_masked": False,
        "bias": False,
    })
    prior, postvar = fitter._select_algorithm(opts)
    prepared = fitter._prepare_problem(x, opts)
    training = fitter._initialize_model(
        prepared=prepared,
        n_components=8,
        use_prior=prior,
        use_postvar=postvar,
        opts=opts,
    )
    calls = []

    def measure(inputs, **kwargs):
        calls.append(inputs)
        assert inputs.score_use_complement
        assert inputs.loading_use_complement
        assert (inputs.loading_covariances is not None) == (algorithm == "vb")
        assert (inputs.score_covariances is not None) == bool(len(training.model.sv))
        return 2, 1, {"score": {"2": 0.01}, "loadings": {"1": 0.01}}

    monkeypatch.setattr(fitter, "_dense_autotune_candidates", lambda *args: [1, 2])
    monkeypatch.setattr(fitter, "autotune_dense_masked_threads", measure)
    monkeypatch.setattr(
        fitter,
        "autotune_cov_writeback_mode_dense",
        lambda *args, **kwargs: ("bulk", {"bulk": 0.01}, 0.01),
    )
    threads, report = fitter._resolve_runtime_threads_for_training(
        prepared=prepared, training=training, opts=opts
    )
    assert len(calls) == 1
    assert threads.score_update_dense == 2
    assert threads.loadings_update_dense == 1
    assert report["kernel_values"]["score_update_dense"] == 2
    assert report["execution_pools"]["requested"]["score_update_dense"] == 2


def test_shared_patterns_skip_the_unrepresentative_no_pattern_probe(monkeypatch):
    from test_runtime_policy import _build_autotune_ctx_for_masked

    import vbpca_py._pca_full as fitter

    ctx = _build_autotune_ctx_for_masked()

    def fail(*args, **kwargs):
        pytest.fail("shared patterns must not benchmark the no-pattern kernel")

    monkeypatch.setattr(fitter, "autotune_dense_masked_threads", fail)
    assert fitter._autotune_dense_masked_runtime(ctx)[0] is ctx.runtime_threads
    ctx.prepared.pattern_index = None
    ctx.workload = replace(ctx.workload, n_components=129)
    assert fitter._autotune_dense_masked_runtime(ctx)[0] is ctx.runtime_threads
    assert (
        ctx.runtime_report["autotune_probe"]["reason"]
        == "covariance_probe_resource_limit"
    )
