"""Fit missing dense observations and project independent new rows."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vbpca_py import VBPCA


def run_example(output_dir):
    rng = np.random.default_rng(25)
    complete = rng.normal(size=(24, 2)) @ rng.normal(size=(2, 5))
    complete += 0.2 * rng.normal(size=complete.shape)
    observed = rng.random(complete.shape) > 0.15
    observed[:, 0] = True
    x = np.where(observed, complete, np.nan)
    model = VBPCA(
        2,
        maxiters=8,
        niter_broadprior=0,
        random_state=25,
        runtime_tuning="off",
        num_cpu=1,
        verbose=0,
    ).fit(x)
    new_rows = rng.normal(size=(3, 5))
    new_rows[1, 2] = np.nan
    posterior = model.transform_posterior(new_rows)
    reconstruction = model.inverse_transform(posterior.mean)
    np.testing.assert_equal(posterior.mean.shape, (3, 2))
    np.testing.assert_equal(posterior.covariance.shape, (3, 2, 2))
    np.testing.assert_equal(reconstruction.shape, new_rows.shape)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        complete[observed],
        model.reconstruction_[observed],
        label="Training cells",
        alpha=0.6,
    )
    ax.scatter(
        complete[~observed],
        model.reconstruction_[~observed],
        label="Missing cells",
        marker="x",
    )
    limits = [float(complete.min()), float(complete.max())]
    ax.plot(limits, limits, color="grey", linestyle="--", label="Equality")
    ax.set(
        xlabel="Synthetic input value",
        ylabel="Joint training reconstruction",
        title="Eight-iteration dense example",
    )
    ax.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "dense.png", dpi=130)
    plt.close(fig)
    summary = {
        "training_shape": list(x.shape),
        "new_row_shape": list(new_rows.shape),
        "conditional_shape": list(posterior.mean.shape),
        "latent_covariance_shape": list(posterior.covariance.shape),
        "iterations": model.n_iter_,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run_example(parser.parse_args().output_dir)
