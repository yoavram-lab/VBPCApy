import pytest

from vbpca_py._runtime_policy import resolve_runtime_thread_config


def _base_opts(num_cpu: int | None = None, user_set: bool = False) -> dict[str, object]:
    opts: dict[str, object] = {}
    if num_cpu is not None:
        opts["num_cpu"] = num_cpu
    opts["_num_cpu_user_set"] = user_set
    return opts


def test_dense_threads_follow_global_num_cpu() -> None:
    cfg = resolve_runtime_thread_config(
        _base_opts(num_cpu=2, user_set=True),
        workload=None,
    )
    assert cfg.score_update_dense == 2
    assert cfg.loadings_update_dense == 2


def test_score_env_override_applies_to_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBPCA_SCORE_THREADS", "3")
    cfg = resolve_runtime_thread_config(_base_opts(), workload=None)
    assert cfg.score_update_dense == 3
    assert cfg.score_update_sparse == 3
    monkeypatch.delenv("VBPCA_SCORE_THREADS", raising=False)
