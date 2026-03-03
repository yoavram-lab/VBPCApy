"""Runtime policy helpers for execution-path and threading controls.

Phase 1 scope is intentionally minimal and behavior-preserving:
- centralize parsing/normalization of runtime-related options,
- define common environment override keys,
- prepare structured hooks for future hardware/data-aware autotuning.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import dense_update_kernels as duk
from . import sparse_update_kernels as suk

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class RuntimeEnvKeys:
    """Environment variable keys used by runtime policy."""

    global_threads: str = "VBPCA_NUM_THREADS"
    subtract_threads: str = "VBPCA_SUBTRACT_THREADS"
    score_threads: str = "VBPCA_SCORE_THREADS"
    loadings_threads: str = "VBPCA_LOADINGS_THREADS"
    noise_threads: str = "VBPCA_NOISE_THREADS"
    rms_threads: str = "VBPCA_RMS_THREADS"


@dataclass(frozen=True)
class RuntimePolicyConfig:
    """Normalized runtime policy configuration."""

    num_cpu: int
    runtime_tuning: str


@dataclass(frozen=True)
class RuntimeThreadConfig:
    """Resolved per-kernel thread counts used during iteration updates."""

    score_update_sparse: int
    loadings_update_sparse: int
    score_update_dense: int
    loadings_update_dense: int
    noise_sxv_sum: int
    rms: int


@dataclass(frozen=True)
class RuntimeWorkloadProfile:
    """Workload metadata used for conservative runtime autotuning."""

    n_features: int
    n_samples: int
    n_components: int
    n_observed: int
    is_sparse: bool


@dataclass(frozen=True)
class DenseMaskedAutotuneInputs:
    """Inputs required to benchmark dense masked kernels."""

    x_data: np.ndarray
    mask: np.ndarray
    loadings: np.ndarray
    scores: np.ndarray
    noise_var: float
    prior_prec: np.ndarray


@dataclass(frozen=True)
class SparseAutotuneInputs:
    """Inputs required to benchmark sparse no-pattern kernels."""

    n_features: int
    n_samples: int
    x_csc_data: np.ndarray
    x_csc_indices: np.ndarray
    x_csc_indptr: np.ndarray
    x_csr_data: np.ndarray
    x_csr_indices: np.ndarray
    x_csr_indptr: np.ndarray
    loadings: np.ndarray
    scores: np.ndarray
    noise_var: float
    prior_prec: np.ndarray


@dataclass(frozen=True)
class _DenseBenchmarkArrays:
    x: np.ndarray
    mask: np.ndarray
    loadings: np.ndarray
    scores: np.ndarray
    prior_prec: np.ndarray


@dataclass(frozen=True)
class _SparseBenchmarkArrays:
    x_csc_data: np.ndarray
    x_csc_indices: np.ndarray
    x_csc_indptr: np.ndarray
    x_csr_data: np.ndarray
    x_csr_indices: np.ndarray
    x_csr_indptr: np.ndarray
    loadings: np.ndarray
    scores: np.ndarray
    prior_prec: np.ndarray


@dataclass(frozen=True)
class _ThreadResolutionInputs:
    opts: dict[str, object]
    env_keys: RuntimeEnvKeys
    env_overrides: dict[str, int]
    profile_overrides: dict[str, int]
    global_env: int | None
    global_num_cpu: int
    num_cpu_user_set: bool


_PROFILE_THREAD_KEYS = (
    "num_cpu_score_update",
    "num_cpu_loadings_update",
    "num_cpu_noise_update",
    "num_cpu_rms",
)


def _default_profile_path() -> Path:
    return Path.home() / ".cache" / "vbpca_py" / "runtime_profile.json"


def _normalize_profile_option(value: object | None) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.lower() == "auto":
        return _default_profile_path()
    return Path(raw).expanduser()


def resolve_profile_path(value: object | None) -> Path | None:
    """Public wrapper to resolve a runtime profile path (supports "auto").

    Returns:
        Expanded path or ``None`` if unset/empty.
    """
    return _normalize_profile_option(value)


def _load_runtime_profile_data(
    profile_option: object | None,
) -> dict[str, object] | None:
    profile_path = _normalize_profile_option(profile_option)
    if profile_path is None or not profile_path.exists():
        return None
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _persist_runtime_profile(
    profile_path: Path,
    data: dict[str, object],
) -> None:
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_autotune_profile_rule(
    *,
    profile_path: Path | None,
    workload: RuntimeWorkloadProfile,
    score_threads: int,
    loadings_threads: int,
    source: str = "autotune_dense_masked",
) -> None:
    """Persist a workload-specific thread rule into the runtime profile.

    The rule matches the exact observed workload bounds; callers should pass
    an appropriate profile path (use ``resolve_profile_path`` to honor "auto").
    """
    if profile_path is None:
        return

    data = _load_runtime_profile_data(profile_path) or {
        "schema_version": 1,
        "default_threads": {},
        "workload_rules": [],
    }

    rules = data.get("workload_rules")
    if not isinstance(rules, list):
        rules = []

    rule = {
        "match": {
            "is_sparse": bool(workload.is_sparse),
            "min_features": int(workload.n_features),
            "max_features": int(workload.n_features),
            "min_samples": int(workload.n_samples),
            "max_samples": int(workload.n_samples),
            "min_components": int(workload.n_components),
            "max_components": int(workload.n_components),
            "min_observed": int(workload.n_observed),
            "max_observed": int(workload.n_observed),
        },
        "threads": {
            "num_cpu_score_update": int(score_threads),
            "num_cpu_loadings_update": int(loadings_threads),
        },
        "source": source,
        "updated_at": time.time(),
    }

    rules.append(rule)
    data["workload_rules"] = rules
    _persist_runtime_profile(profile_path, data)


def _coerce_profile_thread_map(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key in _PROFILE_THREAD_KEYS:
        value = raw.get(key)
        parsed = _parse_int_or_none(value)
        if parsed is None:
            continue
        if parsed < 0:
            continue
        out[key] = int(parsed)
    return out


def _match_profile_rule(
    match: dict[str, object],
    workload: RuntimeWorkloadProfile,
) -> bool:
    is_sparse = match.get("is_sparse")
    if isinstance(is_sparse, bool) and bool(workload.is_sparse) != is_sparse:
        return False

    bounds = (
        ("min_features", workload.n_features, True),
        ("max_features", workload.n_features, False),
        ("min_samples", workload.n_samples, True),
        ("max_samples", workload.n_samples, False),
        ("min_components", workload.n_components, True),
        ("max_components", workload.n_components, False),
        ("min_observed", workload.n_observed, True),
        ("max_observed", workload.n_observed, False),
    )
    for key, observed, is_lower in bounds:
        val = _parse_int_or_none(match.get(key))
        if val is None:
            continue
        if is_lower and observed < val:
            return False
        if (not is_lower) and observed > val:
            return False
    return True


def _resolve_profile_thread_overrides(
    *,
    profile_data: dict[str, object] | None,
    workload: RuntimeWorkloadProfile | None,
) -> dict[str, int]:
    if profile_data is None:
        return {}

    schema_version = _parse_int_or_none(profile_data.get("schema_version"))
    if schema_version is not None and schema_version != 1:
        return {}

    resolved = _coerce_profile_thread_map(profile_data.get("default_threads", {}))

    rules_obj = profile_data.get("workload_rules", [])
    if not isinstance(rules_obj, list) or workload is None:
        return resolved

    for rule_obj in rules_obj:
        if not isinstance(rule_obj, dict):
            continue
        match_obj = rule_obj.get("match", {})
        if not isinstance(match_obj, dict):
            continue
        if not _match_profile_rule(match_obj, workload):
            continue
        resolved.update(_coerce_profile_thread_map(rule_obj.get("threads", {})))

    return resolved


def _coerce_int(value: object, default: int) -> int:
    candidate: Any = value
    try:
        return int(candidate)
    except (TypeError, ValueError):
        return default


def normalize_runtime_tuning_mode(mode: object | None) -> str:
    """Normalize runtime tuning mode.

    Allowed values are ``{"off", "safe", "aggressive"}``.
    Any invalid value falls back to ``"off"`` (behavior-preserving).

    Returns:
        Normalized runtime tuning mode string.
    """
    if mode is None:
        return "off"
    normalized = str(mode).strip().lower()
    if normalized in {"off", "safe", "aggressive"}:
        return normalized
    return "off"


def normalize_cov_writeback_mode(mode: object | None) -> str:
    """Normalize covariance writeback mode.

    Allowed values: ``{"python", "bulk", "auto"}``; invalid inputs fall back
    to ``"auto"`` so that callers can apply context-aware defaults.

    Returns:
        Normalized writeback mode string.
    """
    if mode is None:
        return "auto"
    normalized = str(mode).strip().lower()
    if normalized in {"python", "bulk", "auto"}:
        return normalized
    return "auto"


def normalize_log_progress_stride(stride: object | None, *, default: int = 1) -> int:
    """Normalize stride for progress logging loops.

    Negative values are clamped to ``0`` (disabled). ``None`` falls back to
    ``default``.

    Returns:
        Non-negative stride value.
    """
    parsed = _parse_int_or_none(stride)
    if parsed is None:
        return max(0, int(default))
    return max(0, int(parsed))


def normalize_accessor_mode(mode: object | None) -> str:
    """Normalize accessor mode for score/loadings sparse handling.

    Allowed values: ``{"legacy", "buffered", "auto"}``; invalid inputs fall back
    to ``"auto"`` so callers can apply context-aware defaults.

    Returns:
        Normalized accessor mode string.
    """
    if mode is None:
        return "auto"
    normalized = str(mode).strip().lower()
    if normalized in {"legacy", "buffered", "auto"}:
        return normalized
    return "auto"


def _default_num_cpu() -> int:
    """Conservative default thread count when user did not opt in.

    Uses ``os.cpu_count() - 2`` to avoid oversubscribing shared machines while
    ensuring at least one worker.

    Returns:
        Auto-selected worker count respecting a two-core cushion.
    """
    hw = os.cpu_count() or 1
    return max(1, int(hw) - 2)


def normalize_num_cpu(num_cpu_value: object | None, *, default: int = 1) -> int:
    """Normalize ``num_cpu`` while preserving existing semantics.

    - Valid integer-like values are preserved as-is (including 0).
    - Invalid inputs fall back to ``default``.

    Returns:
        Normalized ``num_cpu`` value.
    """
    if num_cpu_value is None:
        return int(default)
    return _coerce_int(num_cpu_value, default=default)


def read_env_thread_overrides(keys: RuntimeEnvKeys | None = None) -> dict[str, int]:
    """Read thread-count overrides from environment variables.

    Returns only positive integer values; invalid entries are ignored.

    Returns:
        Mapping of env var name to parsed positive integer thread count.
    """
    env_keys = keys or RuntimeEnvKeys()
    out: dict[str, int] = {}
    for key in (
        env_keys.global_threads,
        env_keys.subtract_threads,
        env_keys.score_threads,
        env_keys.loadings_threads,
        env_keys.noise_threads,
        env_keys.rms_threads,
    ):
        raw = os.getenv(key)
        if raw is None:
            continue
        parsed = _coerce_int(raw, default=0)
        if parsed > 0:
            out[key] = parsed
    return out


def _parse_int_or_none(value: object | None) -> int | None:
    if value is None:
        return None
    candidate: Any = value
    try:
        return int(candidate)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _ThreadResolveRequest:
    option_value: object | None
    use_global_opt: bool
    global_opt_value: int
    profile_value: int | None
    env_specific_value: int | None
    env_global_value: int | None
    default_value: int


def _resolve_thread_count(request: _ThreadResolveRequest) -> int:
    parsed_option = _parse_int_or_none(request.option_value)
    if parsed_option is not None:
        return parsed_option

    if request.use_global_opt:
        return int(request.global_opt_value)

    if request.profile_value is not None:
        return int(request.profile_value)

    if request.env_specific_value is not None:
        return int(request.env_specific_value)

    if request.env_global_value is not None:
        return int(request.env_global_value)

    return int(request.default_value)


def _resolve_thread_count_with_source(
    request: _ThreadResolveRequest,
) -> tuple[int, str]:
    parsed_option = _parse_int_or_none(request.option_value)
    if parsed_option is not None:
        return int(parsed_option), "option"

    if request.use_global_opt:
        return int(request.global_opt_value), "global_num_cpu"

    if request.profile_value is not None:
        return int(request.profile_value), "profile"

    if request.env_specific_value is not None:
        return int(request.env_specific_value), "env_specific"

    if request.env_global_value is not None:
        return int(request.env_global_value), "env_global"

    return int(request.default_value), "default"


def _resolved_option_value(value: object | None) -> int | None:
    parsed = _parse_int_or_none(value)
    if parsed is None:
        return None
    if parsed <= 0:
        return None
    return int(parsed)


def _build_dense_autotune_candidates(
    *,
    max_threads: int,
    axis_limit: int,
    tuning_mode: str,
) -> list[int]:
    """Conservative candidate set for dense masked threading autotune.

    Returns:
        Sorted list of candidate thread counts.
    """
    hw = os.cpu_count() or 1
    cap = max(1, min(int(max_threads), int(hw), int(axis_limit)))

    seeds = [1, cap]
    seeds.extend([val for val in (2, 4, 8) if val <= cap])
    if cap > 2:
        seeds.append(max(1, cap // 2))

    unique = [max(1, min(cap, int(axis_limit), int(s))) for s in seeds]

    if tuning_mode == "aggressive" and cap > 8:
        unique.extend((min(cap, 12), min(cap, 16)))

    # Deduplicate and keep ascending.
    dedup = []
    for val in sorted(set(unique)):
        if val not in dedup:
            dedup.append(val)
    return dedup


def _normalize_candidate_list(
    candidates: Sequence[int],
    axis_limit: int,
    hw_threads: int,
) -> list[int]:
    """Clamp candidate thread counts to available hardware and axis size.

    Returns:
        Normalized candidate list respecting ``axis_limit`` and hardware bounds.
    """

    def _norm(val: int) -> int:
        if val <= 0:
            return min(hw_threads, axis_limit)
        return max(1, min(int(val), hw_threads, axis_limit))

    return [_norm(c) for c in candidates]


def _benchmark_pair_candidates(
    cand_score: Sequence[int],
    cand_load: Sequence[int],
    bench_score: Callable[[int], float],
    bench_load: Callable[[int], float],
    max_total_time: float,
) -> tuple[int, int, dict[int, float], dict[int, float], float]:
    start = time.perf_counter()
    best_score = cand_score[0]
    best_score_time = float("inf")
    best_load = cand_load[0]
    best_load_time = float("inf")
    score_timings: dict[int, float] = {}
    load_timings: dict[int, float] = {}

    for s_threads, l_threads in zip(cand_score, cand_load, strict=False):
        if time.perf_counter() - start > max_total_time:
            break

        score_t = bench_score(s_threads)
        load_t = bench_load(l_threads)
        score_timings[s_threads] = score_t
        load_timings[l_threads] = load_t

        if score_t < best_score_time:
            best_score_time = score_t
            best_score = s_threads
        if load_t < best_load_time:
            best_load_time = load_t
            best_load = l_threads

    elapsed = time.perf_counter() - start
    return best_score, best_load, score_timings, load_timings, elapsed


def autotune_dense_masked_threads(
    inputs: DenseMaskedAutotuneInputs,
    *,
    candidates: Sequence[int],
    reps: int = 1,
    max_total_time: float = 0.35,
    tuning_mode: str = "safe",
) -> tuple[int, int, dict[str, object]]:
    """Benchmark dense masked kernels to pick best num_cpu for score/loadings.

    Returns:
        Tuple of (score_threads, loadings_threads, benchmark report).
    """
    hw_threads = max(1, int(os.cpu_count() or 1))
    score_axis = int(inputs.x_data.shape[1])
    load_axis = int(inputs.x_data.shape[0])

    cand_score = _normalize_candidate_list(candidates, score_axis, hw_threads)
    cand_load = _normalize_candidate_list(candidates, load_axis, hw_threads)

    arrays = _DenseBenchmarkArrays(
        x=np.asarray(inputs.x_data, dtype=np.float64, order="C"),
        mask=np.asarray(inputs.mask, dtype=np.float64, order="C"),
        loadings=np.asarray(inputs.loadings, dtype=np.float64, order="C"),
        scores=np.asarray(inputs.scores, dtype=np.float64, order="C"),
        prior_prec=np.asarray(inputs.prior_prec, dtype=np.float64, order="C"),
    )

    def _bench_score(num_cpu: int) -> float:
        t0 = time.perf_counter()
        for _ in range(reps):
            duk.score_update_dense_masked_nopattern(
                x_data=arrays.x,
                mask=arrays.mask,
                loadings=arrays.loadings,
                loading_covariances=None,
                noise_var=float(inputs.noise_var),
                return_covariances=False,
                num_cpu=num_cpu,
            )
        return (time.perf_counter() - t0) / float(max(1, reps))

    def _bench_load(num_cpu: int) -> float:
        t0 = time.perf_counter()
        for _ in range(reps):
            duk.loadings_update_dense_masked_nopattern(
                x_data=arrays.x,
                mask=arrays.mask,
                scores=arrays.scores,
                score_covariances=None,
                prior_prec=arrays.prior_prec,
                noise_var=float(inputs.noise_var),
                return_covariances=False,
                num_cpu=num_cpu,
            )
        return (time.perf_counter() - t0) / float(max(1, reps))

    best_score, best_load, score_timings, load_timings, elapsed = (
        _benchmark_pair_candidates(
            cand_score,
            cand_load,
            _bench_score,
            _bench_load,
            max_total_time,
        )
    )

    report: dict[str, object] = {
        "mode": tuning_mode,
        "candidates": sorted(set(candidates)),
        "score": {str(k): v for k, v in score_timings.items()},
        "loadings": {str(k): v for k, v in load_timings.items()},
        "elapsed_sec": float(elapsed),
        "max_total_time_sec": float(max_total_time),
        "reps": int(reps),
    }

    return int(best_score), int(best_load), report


def autotune_sparse_nopattern_threads(
    inputs: SparseAutotuneInputs,
    *,
    candidates: Sequence[int],
    reps: int = 1,
    max_total_time: float = 0.35,
    tuning_mode: str = "safe",
) -> tuple[int, int, dict[str, object]]:
    """Benchmark sparse no-pattern kernels to pick best num_cpu for score/loadings.

    Returns:
        Tuple of (score_threads, loadings_threads, benchmark report).
    """
    hw_threads = max(1, int(os.cpu_count() or 1))

    cand_score = _normalize_candidate_list(candidates, inputs.n_samples, hw_threads)
    cand_load = _normalize_candidate_list(candidates, inputs.n_features, hw_threads)

    arrays = _SparseBenchmarkArrays(
        x_csc_data=np.asarray(inputs.x_csc_data, dtype=np.float64, order="C"),
        x_csc_indices=np.asarray(inputs.x_csc_indices, dtype=np.int32, order="C"),
        x_csc_indptr=np.asarray(inputs.x_csc_indptr, dtype=np.int32, order="C"),
        x_csr_data=np.asarray(inputs.x_csr_data, dtype=np.float64, order="C"),
        x_csr_indices=np.asarray(inputs.x_csr_indices, dtype=np.int32, order="C"),
        x_csr_indptr=np.asarray(inputs.x_csr_indptr, dtype=np.int32, order="C"),
        loadings=np.asarray(inputs.loadings, dtype=np.float64, order="C"),
        scores=np.asarray(inputs.scores, dtype=np.float64, order="C"),
        prior_prec=np.asarray(inputs.prior_prec, dtype=np.float64, order="C"),
    )

    def _bench_score(num_cpu: int) -> float:
        t0 = time.perf_counter()
        for _ in range(reps):
            suk.score_update_sparse_nopattern(
                x_data=arrays.x_csc_data,
                x_indices=arrays.x_csc_indices,
                x_indptr=arrays.x_csc_indptr,
                loadings=arrays.loadings,
                loading_covariances=None,
                noise_var=float(inputs.noise_var),
                return_covariances=False,
                num_cpu=num_cpu,
            )
        return (time.perf_counter() - t0) / float(max(1, reps))

    def _bench_load(num_cpu: int) -> float:
        t0 = time.perf_counter()
        for _ in range(reps):
            suk.loadings_update_sparse_nopattern(
                x_data=arrays.x_csr_data,
                x_indices=arrays.x_csr_indices,
                x_indptr=arrays.x_csr_indptr,
                scores=arrays.scores,
                score_covariances=None,
                prior_prec=arrays.prior_prec,
                noise_var=float(inputs.noise_var),
                return_covariances=False,
                num_cpu=num_cpu,
            )
        return (time.perf_counter() - t0) / float(max(1, reps))

    best_score, best_load, score_timings, load_timings, elapsed = (
        _benchmark_pair_candidates(
            cand_score,
            cand_load,
            _bench_score,
            _bench_load,
            max_total_time,
        )
    )

    report: dict[str, object] = {
        "mode": tuning_mode,
        "candidates": sorted(set(candidates)),
        "score": {str(k): v for k, v in score_timings.items()},
        "loadings": {str(k): v for k, v in load_timings.items()},
        "elapsed_sec": float(elapsed),
        "max_total_time_sec": float(max_total_time),
        "reps": int(reps),
    }

    return int(best_score), int(best_load), report


def _safe_autotune_rms_threads(profile: RuntimeWorkloadProfile) -> int:
    hw_threads = os.cpu_count() or 1
    hw_threads = max(1, int(hw_threads))

    if not profile.is_sparse:
        return 1

    if profile.n_features > 3_000:
        return 1

    if profile.n_observed < 50_000:
        target = 1
    elif profile.n_observed < 500_000:
        target = 2
    elif profile.n_observed < 2_000_000:
        target = 4
    else:
        target = 8

    target = min(target, hw_threads)
    target = min(target, max(1, int(profile.n_features)))
    return max(1, int(target))


def _safe_autotune_kernel_threads(
    profile: RuntimeWorkloadProfile,
    *,
    kind: str,
) -> int:
    hw_threads = os.cpu_count() or 1
    hw_threads = max(1, int(hw_threads))

    if profile.is_sparse:
        n_obs = max(1, int(profile.n_observed))
        if n_obs < 50_000:
            target = 1
        elif n_obs < 500_000:
            target = 2
        elif n_obs < 2_000_000:
            target = 4
        else:
            target = 8
    else:
        total = max(1, profile.n_features * profile.n_samples)
        if total < 200_000:
            target = 1
        elif total < 1_000_000:
            target = 2
        elif total < 4_000_000:
            target = 4
        else:
            target = 8

    # Slightly favor more threads for RMS; others stay conservative.
    if kind == "rms":
        target = min(target + 1, hw_threads)

    return max(1, min(target, hw_threads))


def _is_explicit_thread_source(
    *,
    option_value: object | None,
    global_opt_set: bool,
    profile_value: int | None,
    env_specific: int | None,
    env_global: int | None,
) -> bool:
    if _resolved_option_value(option_value) is not None:
        return True
    if global_opt_set:
        return True
    if profile_value is not None:
        return True
    if env_specific is not None:
        return True
    return env_global is not None


def resolve_runtime_policy(
    *,
    num_cpu_value: object | None,
    runtime_tuning_value: object | None,
    default_num_cpu: int | None = None,
) -> RuntimePolicyConfig:
    """Resolve runtime policy fields from raw option values.

    Returns:
        Resolved runtime policy configuration.
    """
    effective_default = (
        _default_num_cpu() if default_num_cpu is None else default_num_cpu
    )
    return RuntimePolicyConfig(
        num_cpu=normalize_num_cpu(num_cpu_value, default=effective_default),
        runtime_tuning=normalize_runtime_tuning_mode(runtime_tuning_value),
    )


def apply_runtime_policy_defaults(opts: dict[str, Any]) -> dict[str, Any]:
    """Apply normalized runtime-policy fields onto an options mapping.

    Returns:
        Updated options mapping with normalized runtime policy fields.
    """
    resolved = resolve_runtime_policy(
        num_cpu_value=opts.get("num_cpu"),
        runtime_tuning_value=opts.get("runtime_tuning"),
        default_num_cpu=None,
    )
    opts["num_cpu"] = int(resolved.num_cpu)
    opts["runtime_tuning"] = resolved.runtime_tuning
    runtime_profile = opts.get("runtime_profile")
    opts["runtime_profile"] = None if runtime_profile is None else str(runtime_profile)
    if "cov_writeback_mode" in opts:
        opts["cov_writeback_mode"] = normalize_cov_writeback_mode(
            opts.get("cov_writeback_mode")
        )
    if "log_progress_stride" in opts:
        opts["log_progress_stride"] = normalize_log_progress_stride(
            opts.get("log_progress_stride"),
            default=1,
        )
    if "accessor_mode" in opts:
        opts["accessor_mode"] = normalize_accessor_mode(opts.get("accessor_mode"))
    return opts


def resolve_runtime_thread_config(
    opts: dict[str, object],
    *,
    keys: RuntimeEnvKeys | None = None,
    workload: RuntimeWorkloadProfile | None = None,
) -> RuntimeThreadConfig:
    cfg, _ = resolve_runtime_thread_config_with_report(
        opts,
        keys=keys,
        workload=workload,
    )
    return cfg


def _build_workload_report(
    workload: RuntimeWorkloadProfile | None,
) -> dict[str, object] | None:
    if workload is None:
        return None

    total = max(1, workload.n_features * workload.n_samples)
    density = float(workload.n_observed) / float(total)
    return {
        "n_features": int(workload.n_features),
        "n_samples": int(workload.n_samples),
        "n_components": int(workload.n_components),
        "n_observed": int(workload.n_observed),
        "is_sparse": bool(workload.is_sparse),
        "density": float(density),
    }


def _resolve_kernel_threads_with_sources(
    inputs: _ThreadResolutionInputs,
    workload: RuntimeWorkloadProfile | None,
) -> tuple[dict[str, int], dict[str, str], str]:
    opts = inputs.opts
    env_keys = inputs.env_keys
    env_overrides = inputs.env_overrides
    profile_overrides = inputs.profile_overrides
    global_env = inputs.global_env
    global_num_cpu = inputs.global_num_cpu
    num_cpu_user_set = inputs.num_cpu_user_set

    def _resolve_kernel(
        *,
        option_key: str,
        env_key: str,
        profile_key: str,
        default_value: int,
    ) -> tuple[int, str, bool]:
        threads, source = _resolve_thread_count_with_source(
            _ThreadResolveRequest(
                option_value=opts.get(option_key),
                use_global_opt=num_cpu_user_set,
                global_opt_value=global_num_cpu,
                profile_value=profile_overrides.get(profile_key),
                env_specific_value=env_overrides.get(env_key),
                env_global_value=global_env,
                default_value=default_value,
            )
        )
        explicit = _is_explicit_thread_source(
            option_value=opts.get(option_key),
            global_opt_set=num_cpu_user_set,
            profile_value=profile_overrides.get(profile_key),
            env_specific=env_overrides.get(env_key),
            env_global=global_env,
        )
        return int(threads), source, explicit

    kernel_specs = (
        (
            "score_update_sparse",
            "score",
            env_keys.score_threads,
            "num_cpu_score_update",
            0,
        ),
        (
            "loadings_update_sparse",
            "loadings",
            env_keys.loadings_threads,
            "num_cpu_loadings_update",
            0,
        ),
        (
            "score_update_dense",
            "score",
            env_keys.score_threads,
            "num_cpu_score_update",
            0,
        ),
        (
            "loadings_update_dense",
            "loadings",
            env_keys.loadings_threads,
            "num_cpu_loadings_update",
            0,
        ),
        ("noise_sxv_sum", "noise", env_keys.noise_threads, "num_cpu_noise_update", 0),
        (
            "rms",
            "rms",
            env_keys.rms_threads,
            "num_cpu_rms",
            global_num_cpu,
        ),
    )

    kernel_values: dict[str, int] = {}
    kernel_sources: dict[str, str] = {}
    kernel_explicit: dict[str, bool] = {}

    for name, _kind, env_key, option_key, default_value in kernel_specs:
        threads, source, explicit = _resolve_kernel(
            option_key=option_key,
            env_key=env_key,
            profile_key=option_key,
            default_value=default_value,
        )
        kernel_values[name] = threads
        kernel_sources[name] = source
        kernel_explicit[name] = explicit

    tuning_mode = normalize_runtime_tuning_mode(opts.get("runtime_tuning"))
    if tuning_mode == "safe" and workload is not None:
        for name, kind, _, _, _ in kernel_specs:
            if kernel_explicit[name]:
                continue

            if kind == "rms":
                kernel_values[name] = _safe_autotune_rms_threads(workload)
            else:
                kernel_values[name] = _safe_autotune_kernel_threads(workload, kind=kind)
            kernel_sources[name] = "autotune_safe"

    return kernel_values, kernel_sources, tuning_mode


def resolve_runtime_thread_config_with_report(
    opts: dict[str, object],
    *,
    keys: RuntimeEnvKeys | None = None,
    workload: RuntimeWorkloadProfile | None = None,
) -> tuple[RuntimeThreadConfig, dict[str, object]]:
    """Resolve per-kernel thread counts using a unified override hierarchy.

    Hierarchy (highest to lowest):
    1) Kernel-specific option
    2) Global ``num_cpu`` option, when explicitly user-provided
    3) Runtime profile override (default + matching workload rule)
    4) Kernel-specific environment variable
    5) Global thread environment variable
    6) Legacy default for the kernel

    Returns:
        Tuple of resolved thread counts and source-attribution report.
    """
    env_keys = keys or RuntimeEnvKeys()
    env_overrides = read_env_thread_overrides(env_keys)
    profile_path = _normalize_profile_option(opts.get("runtime_profile"))
    profile_data = _load_runtime_profile_data(opts.get("runtime_profile"))
    profile_overrides = _resolve_profile_thread_overrides(
        profile_data=profile_data,
        workload=workload,
    )

    global_num_cpu = normalize_num_cpu(opts.get("num_cpu"), default=_default_num_cpu())
    num_cpu_user_set = bool(opts.get("_num_cpu_user_set"))
    global_env = env_overrides.get(env_keys.global_threads)
    inputs = _ThreadResolutionInputs(
        opts=opts,
        env_keys=env_keys,
        env_overrides=env_overrides,
        profile_overrides=profile_overrides,
        global_env=global_env,
        global_num_cpu=global_num_cpu,
        num_cpu_user_set=num_cpu_user_set,
    )
    kernel_values, kernel_sources, tuning_mode = _resolve_kernel_threads_with_sources(
        inputs,
        workload,
    )

    cfg = RuntimeThreadConfig(
        score_update_sparse=int(kernel_values["score_update_sparse"]),
        loadings_update_sparse=int(kernel_values["loadings_update_sparse"]),
        score_update_dense=int(kernel_values["score_update_dense"]),
        loadings_update_dense=int(kernel_values["loadings_update_dense"]),
        noise_sxv_sum=int(kernel_values["noise_sxv_sum"]),
        rms=max(1, int(kernel_values["rms"])),
    )
    workload_report = _build_workload_report(workload)

    report: dict[str, object] = {
        "runtime_tuning": tuning_mode,
        "runtime_profile_path": str(profile_path) if profile_path is not None else None,
        "runtime_profile_loaded": bool(profile_data is not None),
        "num_cpu_user_set": bool(num_cpu_user_set),
        "kernel_values": {
            "score_update_sparse": int(cfg.score_update_sparse),
            "loadings_update_sparse": int(cfg.loadings_update_sparse),
            "score_update_dense": int(cfg.score_update_dense),
            "loadings_update_dense": int(cfg.loadings_update_dense),
            "noise_sxv_sum": int(cfg.noise_sxv_sum),
            "rms": int(cfg.rms),
        },
        "kernel_sources": {
            "score_update_sparse": kernel_sources["score_update_sparse"],
            "loadings_update_sparse": kernel_sources["loadings_update_sparse"],
            "score_update_dense": kernel_sources["score_update_dense"],
            "loadings_update_dense": kernel_sources["loadings_update_dense"],
            "noise_sxv_sum": kernel_sources["noise_sxv_sum"],
            "rms": kernel_sources["rms"],
        },
    }
    if workload_report is not None:
        report["workload"] = workload_report
    return cfg, report
