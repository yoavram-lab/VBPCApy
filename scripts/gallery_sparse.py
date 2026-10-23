"""Retain observed sparse zeros when splitting probes and fitting native PCA."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

from vbpca_py import VBPCA, make_xprobe_mask


def run_example(output_dir):
    rng = np.random.default_rng(25)
    values = rng.normal(size=(24, 5))
    values[0, 0] = 0.0
    observed = rng.random(values.shape) > 0.2
    observed[:, 0], observed[0, :] = True, True
    rows, cols = np.nonzero(observed)
    sparse = csr_matrix((values[rows, cols], (rows, cols)), shape=values.shape)
    train, probe = make_xprobe_mask(sparse, 0.05, np.random.default_rng(19))
    np.testing.assert_equal(train.nnz + probe.nnz, sparse.nnz)
    np.testing.assert_equal(
        np.count_nonzero(train.data == 0) + np.count_nonzero(probe.data == 0), 1
    )
    options = {
        "maxiters": 8,
        "niter_broadprior": 0,
        "random_state": 25,
        "runtime_tuning": "off",
        "num_cpu": 1,
        "verbose": 0,
    }
    model = VBPCA(2, **options).fit(train, xprobe=probe)
    dense_probe = np.full(values.shape, np.nan)
    coordinates = probe.tocoo()
    dense_probe[coordinates.row, coordinates.col] = coordinates.data
    dense = VBPCA(2, **options).fit(
        np.where(observed, values, np.nan), xprobe=dense_probe
    )
    np.testing.assert_allclose(
        model.components_, dense.components_, rtol=2e-9, atol=2e-11
    )
    status = np.full(values.shape, np.nan)
    status[observed] = 0
    status[coordinates.row, coordinates.col] = 1
    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(
        status.T,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(
        image, ax=ax, ticks=[0, 1], label="0: training; 1: probe; blank: missing"
    )
    ax.set(
        xlabel="Sample",
        ylabel="Feature",
        title="Sparse observation coordinates, including zero",
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "sparse.png", dpi=130)
    plt.close(fig)
    summary = {
        "shape": list(sparse.shape),
        "input_observations": sparse.nnz,
        "training_observations": train.nnz,
        "probe_observations": probe.nnz,
        "stored_observed_zeros": 1,
        "dense_reference_agrees": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run_example(parser.parse_args().output_dir)
