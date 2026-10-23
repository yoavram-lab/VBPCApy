"""Verify rendered/downloaded source, results and figures after a strict build."""

import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from build_gallery import EXAMPLES


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


class _CodeBlocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.blocks = []
        self.current = None

    def handle_starttag(self, tag, attrs):  # noqa: ARG002 - HTMLParser callback signature
        if tag == "code":
            self.current = []

    def handle_endtag(self, tag):
        if tag == "code" and self.current is not None:
            self.blocks.append("".join(self.current).strip())
            self.current = None

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)


def check_site(root, site):
    """Check the output of the actual documentation build, including images.

    Returns:
        Number of executed gallery examples.
    """
    records = json.loads((site / "gallery/manifest.json").read_text())
    _require(
        len(records) == len(EXAMPLES)
        and {r["slug"] for r in records} == {e[0] for e in EXAMPLES},
        "Gallery manifest must include every example exactly once",
    )
    for record in records:
        source = (root / "scripts" / record["script"]).read_text()
        _require(
            hashlib.sha256(source.encode()).hexdigest() == record["source_sha256"],
            "Executed source hash differs",
        )
        directory = site / "gallery" / record["slug"]
        _require(
            (directory / record["script"]).read_text() == source,
            "Downloaded source differs",
        )
        parser = _CodeBlocks()
        parser.feed((directory / "index.html").read_text())
        _require(
            source.strip() in parser.blocks,
            f"Rendered source differs: {record['script']}",
        )
        _require(
            json.loads((directory / "summary.json").read_text()) == record["summary"],
            "Rendered result summary differs",
        )
        for filename in record["figures"]:
            pixels = mpimg.imread(directory / filename)
            _require(
                pixels.shape[0] >= 100 and pixels.shape[1] >= 100,
                "Figure dimensions are too small",
            )
            _require(
                np.isfinite(pixels).all() and np.ptp(pixels) > 0.1,
                "Figure is empty or invalid",
            )
    print(
        f"Verified {len(records)} rendered sources, downloads, result summaries and their nonempty figures"
    )
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    check_site(Path(__file__).resolve().parents[1], parser.parse_args().site_dir)
