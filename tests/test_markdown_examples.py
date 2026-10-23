"""Execute primary Python workflows directly from Markdown sources."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON_FENCE = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


@pytest.mark.parametrize(
    "relative",
    [
        "docs/getting-started/quickstart.md",
        "docs/tutorials/basic-dense-pca.md",
        "docs/tutorials/preprocessing-mixed-data.md",
    ],
)
def test_primary_markdown_workflow_executes(relative: str) -> None:
    """Run each Python fence in a document in its shared tutorial namespace."""
    path = ROOT / relative
    blocks = PYTHON_FENCE.findall(path.read_text(encoding="utf-8"))
    assert blocks, f"no Python fences found in {relative}"
    namespace: dict[str, object] = {"__name__": "__markdown_example__"}
    for index, block in enumerate(blocks, start=1):
        source = textwrap.dedent(block)
        code = compile(source, f"{relative}::python-fence-{index}", "exec")
        exec(code, namespace)  # noqa: S102 - executable documentation contract
