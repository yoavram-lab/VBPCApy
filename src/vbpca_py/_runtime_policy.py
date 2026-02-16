"""Runtime policy helpers for execution-path and threading controls.

Phase 1 scope is intentionally minimal and behavior-preserving:
- centralize parsing/normalization of runtime-related options,
- define common environment override keys,
- prepare structured hooks for future hardware/data-aware autotuning.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    default_num_cpu: int = 1,
) -> RuntimePolicyConfig:
    """Resolve runtime policy fields from raw option values.

    Returns:
        Resolved runtime policy configuration.
    """
    return RuntimePolicyConfig(
        num_cpu=normalize_num_cpu(num_cpu_value, default=default_num_cpu),
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
        default_num_cpu=1,
    )
    opts["num_cpu"] = int(resolved.num_cpu)
    opts["runtime_tuning"] = resolved.runtime_tuning
    runtime_profile = opts.get("runtime_profile")
    opts["runtime_profile"] = (
        None if runtime_profile is None else str(runtime_profile)
    )
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


def resolve_runtime_thread_config_with_report(  # noqa: PLR0914
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

    global_num_cpu = normalize_num_cpu(opts.get("num_cpu"), default=1)
    num_cpu_user_set = bool(opts.get("_num_cpu_user_set"))

    global_env = env_overrides.get(env_keys.global_threads)

    score_threads, score_source = _resolve_thread_count_with_source(
        _ThreadResolveRequest(
            option_value=opts.get("num_cpu_score_update"),
            use_global_opt=num_cpu_user_set,
            global_opt_value=global_num_cpu,
            profile_value=profile_overrides.get("num_cpu_score_update"),
            env_specific_value=env_overrides.get(env_keys.score_threads),
            env_global_value=global_env,
            default_value=0,
        )
    )

    loadings_threads, loadings_source = _resolve_thread_count_with_source(
        _ThreadResolveRequest(
            option_value=opts.get("num_cpu_loadings_update"),
            use_global_opt=num_cpu_user_set,
            global_opt_value=global_num_cpu,
            profile_value=profile_overrides.get("num_cpu_loadings_update"),
            env_specific_value=env_overrides.get(env_keys.loadings_threads),
            env_global_value=global_env,
            default_value=0,
        )
    )

    noise_threads, noise_source = _resolve_thread_count_with_source(
        _ThreadResolveRequest(
            option_value=opts.get("num_cpu_noise_update"),
            use_global_opt=num_cpu_user_set,
            global_opt_value=global_num_cpu,
            profile_value=profile_overrides.get("num_cpu_noise_update"),
            env_specific_value=env_overrides.get(env_keys.noise_threads),
            env_global_value=global_env,
            default_value=0,
        )
    )

    rms_threads, rms_source = _resolve_thread_count_with_source(
        _ThreadResolveRequest(
            option_value=opts.get("num_cpu_rms"),
            use_global_opt=num_cpu_user_set,
            global_opt_value=global_num_cpu,
            profile_value=profile_overrides.get("num_cpu_rms"),
            env_specific_value=env_overrides.get(env_keys.rms_threads),
            env_global_value=global_env,
            default_value=global_num_cpu,
        )
    )

    tuning_mode = normalize_runtime_tuning_mode(opts.get("runtime_tuning"))
    if tuning_mode == "safe" and workload is not None:
        rms_explicit = _is_explicit_thread_source(
            option_value=opts.get("num_cpu_rms"),
            global_opt_set=num_cpu_user_set,
            profile_value=profile_overrides.get("num_cpu_rms"),
            env_specific=env_overrides.get(env_keys.rms_threads),
            env_global=global_env,
        )
        if not rms_explicit:
            rms_threads = _safe_autotune_rms_threads(workload)
            rms_source = "autotune_safe"

    cfg = RuntimeThreadConfig(
        score_update_sparse=int(score_threads),
        loadings_update_sparse=int(loadings_threads),
        noise_sxv_sum=int(noise_threads),
        rms=max(1, int(rms_threads)),
    )
    report: dict[str, object] = {
        "runtime_tuning": tuning_mode,
        "runtime_profile_path": str(profile_path) if profile_path is not None else None,
        "runtime_profile_loaded": bool(profile_data is not None),
        "num_cpu_user_set": bool(num_cpu_user_set),
        "kernel_values": {
            "score_update_sparse": int(cfg.score_update_sparse),
            "loadings_update_sparse": int(cfg.loadings_update_sparse),
            "noise_sxv_sum": int(cfg.noise_sxv_sum),
            "rms": int(cfg.rms),
        },
        "kernel_sources": {
            "score_update_sparse": score_source,
            "loadings_update_sparse": loadings_source,
            "noise_sxv_sum": noise_source,
            "rms": rms_source,
        },
    }
    return cfg, report
