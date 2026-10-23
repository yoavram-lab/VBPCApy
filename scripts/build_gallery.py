"""Execute bounded public examples and assemble source-matched gallery pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

EXAMPLES = (
    (
        "dense",
        "Missing dense observations",
        "gallery_dense.py",
        "Training reconstructions and conditional projections of independent new rows use different score definitions.",
    ),
    (
        "sparse",
        "Sparse observations and stored zeros",
        "gallery_sparse.py",
        "Blank cells are missing. Both training and probe coordinates retain measured zeros. The tiny dense reference uses the identical observation pattern.",
    ),
    (
        "pipeline",
        "Mixed-data Pipeline and row validation",
        "example_masked_pipeline.py",
        "The original-variable mask reaches the first Pipeline step. Row CV and GridSearchCV fit preprocessing within training splits. Fresh new rows are decoded to original variables and input units.",
    ),
    (
        "selection",
        "Cost versus probe RMS",
        "gallery_selection.py",
        "Both criteria use the same held-out cells and positive capacities. Rank zero has no variational cost. Cost and probe RMS have different units and are plotted on separate axes; these short sweeps do not qualify a selection policy.",
    ),
    (
        "raw-cv",
        "Original-variable cell validation",
        "gallery_raw_cv.py",
        "Each raw-cell training fold fits a fresh encoder with an externally declared vocabulary. Nominal Brier scores, continuous RMSE in input units and unrounded ordinal level-index RMSE describe different targets. This differs from holding out complete rows.",
    ),
    (
        "plots",
        "Scree, loadings and explained variance",
        "gallery_plots.py",
        "Each plot reads the returned fitted state through public plotting helpers. Explained variance is shown both as a ratio and in absolute units.",
    ),
)


def _record_table(records):
    keys = list(records[0])
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    lines.extend(
        "| " + " | ".join(f"{record[key]:.6g}" for key in keys) + " |"
        for record in records
    )
    return "\n".join(lines)


def build_gallery(root, destination):
    """Execute scripts in child processes; propagate errors and timeouts.

    Returns:
        A relative-path manifest for source and generated artifacts.

    Raises:
        RuntimeError: An example changes during execution or emits no figures.
    """
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for slug, title, filename, description in EXAMPLES:
        source_path = root / "scripts" / filename
        source = source_path.read_text()
        directory = destination / slug
        directory.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        completed = subprocess.run(
            [sys.executable, str(source_path), "--output-dir", str(directory)],
            cwd=directory,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if source_path.read_text() != source:
            raise RuntimeError(f"Example source changed while executing: {filename}")
        (directory / filename).write_text(source)
        summary = json.loads((directory / "summary.json").read_text())
        figures = sorted(path.name for path in directory.glob("*.png"))
        if not figures:
            raise RuntimeError(f"Example produced no figures: {filename}")
        page = [
            f"# {title}",
            description,
            "These synthetic examples run at most eight fit iterations. They demonstrate API behavior; they do not establish posterior convergence, calibration or a recommended capacity.",
            f"This example requires the 0.5 development API with the `plot` extra. From a 0.5 source checkout, install `python -m pip install '.[plot]'`, then run `python scripts/{filename} --output-dir outputs`. The downloaded source can also run directly against that installed development build.",
            f"[Download this standalone source]({filename}).",
            "## Output",
            f"```text\n{completed.stdout.rstrip()}\n```",
        ]
        if summary.get("records"):
            page.extend(["## Candidate records", _record_table(summary["records"])])
        page.extend(
            f"![{title}: {figure.removesuffix('.png')}]({figure})" for figure in figures
        )
        page.extend(["## Executed source", f"```python\n{source.rstrip()}\n```"])
        (directory / "index.md").write_text("\n\n".join(page) + "\n")
        manifest.append({
            "slug": slug,
            "script": filename,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "figures": figures,
            "summary": summary,
        })
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_gallery(Path(__file__).resolve().parents[1], args.output_dir.resolve())
