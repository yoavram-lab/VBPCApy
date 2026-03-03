# src/vbpca_py/_options.py
"""
Utilities for merging user options with default options.

This module provides a small helper for combining user-specified keyword
arguments with a dictionary of default options, treating user keys
case-insensitively by lowercasing them.
"""

from collections.abc import Mapping
from typing import Any


def _normalize_keys_to_lower(opts: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``opts`` with all keys lowercased.

    If multiple keys differ only by case, the last one in iteration order wins.
    """
    normalized: dict[str, Any] = {}
    for key, value in opts.items():
        lower_key = key.lower()
        normalized[lower_key] = value
    return normalized


def _options(
    defopts: Mapping[str, Any], **kwargs: object
) -> tuple[dict[str, Any], str]:
    """Merge user-provided options into default options (case-insensitive).

    Keys from ``kwargs`` are normalized to lowercase before merging. Keys
    in ``defopts`` are assumed to already be lowercase; if not, behaviour is
    case-sensitive with respect to those keys.

    Unknown parameters (present in ``kwargs`` but not in ``defopts``) are
    still added to the returned options dictionary, and a warning message
    listing them is returned.

    Args:
        defopts:
            Default options dictionary. Keys are expected to be lowercase.
        **kwargs:
            User-specified options. Keys are treated case-insensitively via
            lowercasing.

    Returns:
        opts:
            A new dictionary containing ``defopts`` updated with user options.
        wrnmsg:
            Empty string if all user keys are known; otherwise a warning message
            of the form ``"Unknown parameter(s): k1, k2, ..."``.
    """
    # Start from defaults (copy so we don't mutate defopts).
    opts: dict[str, Any] = dict(defopts)

    # Normalize user-provided keys to lowercase.
    user_opts = _normalize_keys_to_lower(kwargs)

    unknown_params: list[str] = []
    for key, value in user_opts.items():
        if key not in defopts and not str(key).startswith("_debug_"):
            unknown_params.append(key)
        opts[key] = value

    if unknown_params:
        wrnmsg = f"Unknown parameter(s): {', '.join(unknown_params)}"
    else:
        wrnmsg = ""

    return opts, wrnmsg
