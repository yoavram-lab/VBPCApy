#!/usr/bin/env python3
"""Build or extend a VBPCA runtime profile JSON.

This helper creates a schema_version=1 runtime profile with optional:
- default thread overrides, and
- workload-matching rules derived from either explicit thresholds or a
  reference row in a profiler CSV produced by ``scripts/profile_core_vbpca.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

_THREAD_KEYS = (
    "num_cpu_score_update",
    "num_cpu_loadings_update",
    "num_cpu_noise_update",
    "num_cpu_rms",
)


def _parse_nonnegative(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _load_profile(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"schema_version": 1, "default_threads": {}, "workload_rules": []}

    if not path.exists():
        return {"schema_version": 1, "default_threads": {}, "workload_rules": []}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"Profile root must be a JSON object: {path}"
        raise ValueError(msg)

    out = dict(data)
    out.setdefault("schema_version", 1)
    out.setdefault("default_threads", {})
    out.setdefault("workload_rules", [])

    if not isinstance(out["default_threads"], dict):
        out["default_threads"] = {}
    if not isinstance(out["workload_rules"], list):
        out["workload_rules"] = []

    return out


def _thread_map_from_args(args: argparse.Namespace, prefix: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for key in _THREAD_KEYS:
        arg_name = f"{prefix}_{key}" if prefix else key
        value = _parse_nonnegative(getattr(args, arg_name))
        if value is None:
            continue
        mapping[key] = int(value)
    return mapping


def _derive_match_from_csv(args: argparse.Namespace) -> dict[str, object] | None:
    if args.case_csv is None:
        return None

    if args.case_name == "":
        msg = "--case-name is required when --case-csv is provided."
        raise ValueError(msg)

    frame = pd.read_csv(args.case_csv)
    subset = frame.loc[frame["case"] == args.case_name]
    if subset.empty:
        msg = f"No rows found for case={args.case_name!r} in {args.case_csv}."
        raise ValueError(msg)

    row = subset.iloc[0]
    n_features = int(row["n_features"])
    n_samples = int(row["n_samples"])
    missing_rate = float(row["missing_rate"])
    n_observed = int(
        round(float(n_features * n_samples) * max(0.0, 1.0 - missing_rate))
    )
    is_sparse = str(row.get("kind", "")).strip().lower() == "sparse"

    return {
        "is_sparse": is_sparse,
        "min_features": n_features,
        "min_samples": n_samples,
        "min_observed": n_observed,
    }


def _explicit_match_from_args(args: argparse.Namespace) -> dict[str, object] | None:
    match: dict[str, object] = {}

    if args.match_is_sparse is not None:
        match["is_sparse"] = bool(int(args.match_is_sparse))
    if args.match_min_features is not None:
        match["min_features"] = int(args.match_min_features)
    if args.match_min_samples is not None:
        match["min_samples"] = int(args.match_min_samples)
    if args.match_min_components is not None:
        match["min_components"] = int(args.match_min_components)
    if args.match_min_observed is not None:
        match["min_observed"] = int(args.match_min_observed)

    return match or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output profile path.",
    )
    parser.add_argument(
        "--base-profile",
        type=Path,
        default=None,
        help="Optional existing profile to update.",
    )

    for key in _THREAD_KEYS:
        parser.add_argument(
            f"--default-{key.replace('_', '-')}", type=int, default=None
        )
        parser.add_argument(f"--rule-{key.replace('_', '-')}", type=int, default=None)

    parser.add_argument(
        "--case-csv",
        type=Path,
        default=None,
        help="Optional profile_core_vbpca CSV used to seed workload match bounds.",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default="",
        help="Case name in --case-csv used for match threshold seeding.",
    )

    parser.add_argument("--match-is-sparse", choices=("0", "1"), default=None)
    parser.add_argument("--match-min-features", type=int, default=None)
    parser.add_argument("--match-min-samples", type=int, default=None)
    parser.add_argument("--match-min-components", type=int, default=None)
    parser.add_argument("--match-min-observed", type=int, default=None)

    parser.add_argument(
        "--replace-workload-rules",
        action="store_true",
        help="Replace existing workload_rules instead of appending.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSON to stdout without writing output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile = _load_profile(args.base_profile)

    default_threads = _thread_map_from_args(args, prefix="default")
    if default_threads:
        merged_defaults = dict(profile.get("default_threads", {}))
        merged_defaults.update(default_threads)
        profile["default_threads"] = merged_defaults

    csv_match = _derive_match_from_csv(args)
    explicit_match = _explicit_match_from_args(args)
    rule_match = dict(csv_match or {})
    if explicit_match:
        rule_match.update(explicit_match)

    rule_threads = _thread_map_from_args(args, prefix="rule")
    if rule_threads and not rule_match:
        msg = (
            "Rule thread overrides were provided but no rule match criteria were set. "
            "Provide --case-csv/--case-name or explicit --match-* bounds."
        )
        raise ValueError(msg)

    if rule_threads:
        new_rule = {"match": rule_match, "threads": rule_threads}
        rules = (
            []
            if args.replace_workload_rules
            else list(profile.get("workload_rules", []))
        )
        rules.append(new_rule)
        profile["workload_rules"] = rules

    profile["schema_version"] = 1

    serialized = json.dumps(profile, indent=2, sort_keys=True)
    if args.dry_run:
        print(serialized)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(serialized + "\n", encoding="utf-8")
    print(f"Wrote runtime profile: {args.out}")


if __name__ == "__main__":
    main()
