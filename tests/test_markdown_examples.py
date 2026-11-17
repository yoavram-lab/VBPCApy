"""Execute every Python block on every documentation page.

Blocks on one page share a namespace and run in order, as a reader would run
them. A block that must not run (for example, one that needs data the reader
supplies) is preceded by an HTML comment naming the reason::

    <!-- docs-test: skip: needs the reader's own genotype file -->

Pages that import an optional dependency that is not installed are skipped,
unless ``VBPCA_DOCS_REQUIRE_ALL=1`` is set; the documentation CI job sets it so
that every page runs there.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
# Shared manuscript-writing guidance is not package documentation.
EXCLUDED_PARTS = {"author_style"}
FENCE_OPEN = re.compile(r"^(?P<indent>[ ]*)```python[ ]*$")
SKIP_MARKER = re.compile(r"^[ ]*<!-- docs-test: skip: (?P<reason>\S.*?) -->[ ]*$")
SNIPPET = re.compile(r'^--8<-- "(?P<path>[^"]+)"$')
OPTIONAL_MODULES = ("pandas", "polars", "matplotlib")
# Pages allowed to contain skipped blocks; keep this empty where possible.
SKIP_ALLOWED: frozenset[str] = frozenset()


def _blocks(path: Path) -> list[tuple[int, str, str | None]]:
    """Return (line, source, skip reason) for each Python fence on a page."""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[int, str, str | None]] = []
    previous = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        opened = FENCE_OPEN.match(line)
        if not opened:
            if line.strip():
                previous = line
            index += 1
            continue
        indent = opened.group("indent")
        start = index + 1
        body: list[str] = []
        index += 1
        while index < len(lines) and lines[index].rstrip() != indent + "```":
            body.append(lines[index][len(indent) :])
            index += 1
        if index == len(lines):
            msg = f"{path}:{start}: unterminated Python fence"
            raise AssertionError(msg)
        marker = SKIP_MARKER.match(previous)
        blocks.append((start, "\n".join(_expand(body)), marker and marker["reason"]))
        previous = ""
        index += 1
    return blocks


def _expand(body: list[str]) -> list[str]:
    """Inline pymdownx snippet includes the way the site build does."""
    expanded: list[str] = []
    for line in body:
        snippet = SNIPPET.match(line.strip())
        if snippet:
            expanded.extend(
                (ROOT / snippet["path"]).read_text(encoding="utf-8").splitlines()
            )
        else:
            expanded.append(line)
    return expanded


def _pages() -> list[str]:
    pages = []
    for path in sorted(DOCS.rglob("*.md")):
        relative = path.relative_to(ROOT)
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if _blocks(path):
            pages.append(relative.as_posix())
    return pages


def _missing_optional(source: str) -> list[str]:
    missing = []
    for module in OPTIONAL_MODULES:
        imported = re.search(rf"^\s*(import|from)\s+{module}\b", source, re.MULTILINE)
        if imported and importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def test_documentation_pages_are_collected() -> None:
    """Guard against the collector silently finding nothing."""
    pages = _pages()
    assert "docs/getting-started/quickstart.md" in pages
    assert "docs/tutorials/basic-dense-pca.md" in pages
    assert len(pages) >= 15


@pytest.mark.parametrize("relative", _pages())
def test_documentation_page_executes(
    relative: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run each Python block of a page in a shared namespace."""
    blocks = _blocks(ROOT / relative)
    skipped = [(line, reason) for line, _, reason in blocks if reason]
    if skipped and relative not in SKIP_ALLOWED:
        lines = ", ".join(str(line) for line, _ in skipped)
        pytest.fail(f"{relative} skips blocks at lines {lines} without an allowance")
    missing = _missing_optional("\n".join(source for _, source, _ in blocks))
    if missing:
        if os.environ.get("VBPCA_DOCS_REQUIRE_ALL") == "1":
            pytest.fail(
                f"{relative} needs {', '.join(missing)}, which is not installed"
            )
        pytest.skip(f"optional dependency not installed: {', '.join(missing)}")

    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__markdown_example__"}
    try:
        for line, source, reason in blocks:
            if reason:
                continue
            code = compile(source, f"{relative}:{line}", "exec")
            exec(code, namespace)  # noqa: S102 - executable documentation contract
    finally:
        if importlib.util.find_spec("matplotlib") is not None:
            import matplotlib.pyplot as plt

            plt.close("all")
