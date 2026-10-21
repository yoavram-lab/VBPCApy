"""Versioned execution profiles with environment isolation and atomic updates."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ._runtime_fingerprint import environment_digest, runtime_environment

if TYPE_CHECKING:
    from collections.abc import Generator

PROFILE_SCHEMA_VERSION = 2


def new_profile() -> dict[str, object]:
    """Create a profile for the current machine, allocation and loaded build.

    Returns:
        An empty versioned profile with complete environment provenance.
    """
    environment = runtime_environment()
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "environment": environment,
        "environment_sha256": environment_digest(environment),
        "default_threads": {},
        "workload_rules": [],
    }


def automatic_profile_path() -> Path:
    """Partition automatic caches by hardware/allocation/build context.

    Returns:
        A home-cache path that does not overwrite another context or v1 file.
    """
    digest = environment_digest(runtime_environment())
    return Path.home() / ".cache" / "vbpca_py" / f"runtime-v2-{digest}.json"


def _compatible(data: object, expected: dict[str, object]) -> bool:
    return (
        isinstance(data, dict)
        and data.get("schema_version") == PROFILE_SCHEMA_VERSION
        and data.get("environment_sha256") == expected["environment_sha256"]
        and data.get("environment") == expected["environment"]
    )


def load_profile(path: Path) -> dict[str, object] | None:
    """Load only a profile matching the current schema and full environment.

    Returns:
        Compatible data, or ``None`` for stale/malformed/unreadable profiles.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = new_profile()
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and _compatible(data, expected) else None


def _try_lock(descriptor: int) -> bool:
    if sys.platform == "win32":
        import msvcrt  # noqa: PLC0415

        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
    else:
        import fcntl  # noqa: PLC0415

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
    return True


@contextlib.contextmanager
def _profile_lock(path: Path) -> Generator[bool, None, None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        deadline = time.monotonic() + 0.5
        locked = _try_lock(stream.fileno())
        while not locked and time.monotonic() < deadline:
            time.sleep(0.01)
            locked = _try_lock(stream.fileno())
        try:
            yield locked
        finally:
            if locked:
                if sys.platform == "win32":
                    import msvcrt  # noqa: PLC0415

                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl  # noqa: PLC0415

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_write_profile(path: Path, data: dict[str, object]) -> None:
    """Atomically replace a complete profile; callers serialize update cycles.

    The temporary file is on the same filesystem. Interrupted writes leave
    either the previous complete JSON or the replacement, never a partial JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_profile_rule(path: Path, rule: dict[str, object]) -> bool:
    """Merge one execution rule under a bounded process lock and atomic replace.

    Identical workload/semantic keys replace their old rule. Other rules are
    retained. Lock contention times out without blocking inference; incompatible
    explicit files are preserved rather than overwritten. Automatic filenames
    already partition different machine/allocation/build contexts.

    Returns:
        Whether a compatible profile was successfully updated.
    """
    expected = new_profile()
    with _profile_lock(path) as locked:
        if not locked:
            return False
        data = load_profile(path) if path.exists() else expected
        if data is None:
            return False
        rules = data.get("workload_rules", [])
        if not isinstance(rules, list):
            rules = []
        retained = [
            old
            for old in rules
            if isinstance(old, dict)
            and (
                old.get("match") != rule.get("match")
                or old.get("execution", {}) != rule.get("execution", {})
            )
        ]
        data["workload_rules"] = [*retained, rule]
        atomic_write_profile(path, data)
    return True
