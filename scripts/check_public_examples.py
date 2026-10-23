"""Execute docstrings and the actual quickstart/migration Python blocks."""

import argparse
import doctest
import importlib
import inspect
import re
import sys
import tomllib
from pathlib import Path

import vbpca_py

PUBLIC_OBJECTS = (
    vbpca_py.VBPCA,
    vbpca_py.PCAOptions,
    vbpca_py.ObservationMasker,
    vbpca_py.AutoEncoder,
    vbpca_py.MissingAwareStandardScaler,
    importlib.import_module("vbpca_py.legacy").VBPCA,
)
PAGES = ("docs/getting-started/quickstart.md", "docs/api/migration-05.md")


def check_examples(root, *, require_wheel=False):
    """Collect runnable examples from their real source and fail on any error.

    Returns:
        Source names and executed example counts.

    Raises:
        RuntimeError: An example is missing/fails or a wheel was required.
    """
    if require_wheel and not Path(vbpca_py.__file__).resolve().is_relative_to(
        Path(sys.prefix).resolve()
    ):
        raise RuntimeError(
            "Documentation checks require an installed wheel, not an editable checkout"
        )
    if require_wheel:
        with (root / "pyproject.toml").open("rb") as handle:
            expected_version = tomllib.load(handle)["project"]["version"]
        if vbpca_py.__version__ != expected_version:
            raise RuntimeError(
                f"Installed version {vbpca_py.__version__!r} differs from "
                f"source version {expected_version!r}"
            )
    runner = doctest.DocTestRunner()
    counts = {}
    for obj in PUBLIC_OBJECTS:
        name = f"{obj.__module__}.{obj.__name__}"
        module = inspect.getmodule(obj)
        examples = doctest.DocTestFinder().find(obj, name=name, globs=vars(module))
        count = sum(len(example.examples) for example in examples)
        if not count:
            raise RuntimeError(f"Public API has no executed examples: {name}")
        for example in examples:
            runner.run(example)
        counts[name] = count
    result = runner.summarize()
    if result.failed:
        raise RuntimeError(f"Public API doctests failed: {result.failed}")
    for filename in PAGES:
        source = (root / filename).read_text()
        blocks = re.findall(r"^```python\n(.*?)^```", source, re.MULTILINE | re.DOTALL)
        if not blocks:
            raise RuntimeError(f"Page has no Python examples: {filename}")
        namespace = {"__name__": "__documented_example__"}
        for index, block in enumerate(blocks):
            exec(compile(block, f"{filename}:block-{index + 1}", "exec"), namespace)  # noqa: S102 - execute trusted repository documentation verbatim
        counts[filename] = len(blocks)
    print(
        f"Executed {result.attempted} public API doctests and {sum(counts[p] for p in PAGES)} actual documentation blocks"
    )
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-wheel", action="store_true")
    check_examples(
        Path(__file__).resolve().parents[1],
        require_wheel=parser.parse_args().require_wheel,
    )
