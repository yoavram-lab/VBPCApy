"""Fit preprocessing within raw-variable cell-validation folds."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vbpca_py import AutoEncoder, CVConfig, cross_validate_raw_components


def make_encoder():
    return AutoEncoder(
        column_types=["categorical", "continuous", "ordinal"],
        column_levels=[["west", "east", "north"], None, ["low", "medium", "high"]],
        drop="first",
        mean_center_ohe=True,
        block_weighting="equal_variance",
        handle_unknown="raise",
    )


def run_example(output_dir):
    rng = np.random.default_rng(25)
    x = np.empty((36, 3), dtype=object)
    x[:, 0] = np.tile(["west", "east", "north"], 12)
    x[:, 1] = rng.normal(100, 15, len(x))
    x[:, 2] = np.tile(["low", "medium", "high"], 12)
    x[2, 0], x[5, 1] = np.nan, np.nan
    chosen, records, folds = cross_validate_raw_components(
        x,
        encoder_factory=make_encoder,
        components=[0, 1, 2],
        config=CVConfig(
            n_splits=2,
            seed=25,
            metric="brier",
            selection_rule="first_minimum",
            early_stop=False,
        ),
        maxiters=8,
        niter_broadprior=0,
        random_state=25,
        runtime_tuning="off",
        num_cpu=1,
        verbose=0,
    )
    means = [fold.schema.variables[1].offset[0] for fold in folds]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(means)), means)
    ax.set(
        xlabel="Raw-cell validation fold",
        ylabel="Training-only continuous offset (input units)",
        xticks=range(len(means)),
        title="A fresh encoder is fitted within each fold",
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "raw-cv.png", dpi=130)
    plt.close(fig)
    summary = {
        "selected_capacity": chosen,
        "fold_training_offsets": means,
        "schema_kinds": [variable.kind for variable in folds[0].schema.variables],
        "records": [
            {
                key: record[key]
                for key in [
                    "k",
                    "mean_brier",
                    "mean_rmse_variable_1",
                    "mean_rmse_variable_2",
                ]
            }
            for record in records
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run_example(parser.parse_args().output_dir)
