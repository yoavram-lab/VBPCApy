"""Plot explained variance and loadings from the returned fitted state."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vbpca_py import VBPCA
from vbpca_py.plotting import loadings_barplot, scree_plot, variance_explained_plot


def run_example(output_dir):
    x = np.random.default_rng(25).normal(size=(24, 5))
    model = VBPCA(
        2,
        maxiters=8,
        niter_broadprior=0,
        random_state=25,
        runtime_tuning="off",
        num_cpu=1,
        verbose=0,
    ).fit(x)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in [
        ("scree", scree_plot(model)),
        (
            "loadings",
            loadings_barplot(
                model, feature_names=["north", "south", "east", "west", "centre"]
            ),
        ),
        ("variance", variance_explained_plot(model)),
    ]:
        figure.savefig(output_dir / f"{name}.png", dpi=130)
        plt.close(figure)
    summary = {
        "capacity": model.components_.shape[0],
        "explained_variance": model.explained_variance_.tolist(),
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "component_zero_loadings": model.components_[0].tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run_example(parser.parse_args().output_dir)
