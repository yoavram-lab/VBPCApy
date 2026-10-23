"""Fit, search, project and reconstruct a missing-aware mixed-data Pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn import config_context
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline

from vbpca_py import VBPCA, AutoEncoder, ObservationMasker


def run_example(output="default"):
    """Run a bounded API example; eight iterations do not establish convergence.

    Returns:
        Input tables, fitted search, fold estimators and reconstructed new rows.

    Raises:
        RuntimeError: Scores or reconstruction violate the demonstrated contract.
    """
    rng = np.random.default_rng(23)
    raw = np.empty((36, 2), dtype=object)
    raw[:, 0] = np.tile(["west", "east", "north"], 12)
    raw[:, 1] = rng.normal(100, 5, len(raw))
    mask = rng.random(raw.shape) > 0.1
    raw[4, 1], raw[5, 0] = np.nan, np.nan
    mask[4, 1], mask[5, 0] = True, True
    data = np.where(mask, raw, "unobserved placeholder")
    new_data = np.array(
        [
            ["west", 90.0],
            ["unobserved placeholder", np.nan],
            ["north", "unobserved placeholder"],
            ["east", 115.0],
        ],
        dtype=object,
    )
    new_mask = np.array([[True, True], [False, True], [True, False], [True, True]])
    if output != "default":
        import pandas as pd

        data = pd.DataFrame(
            data,
            columns=["region", "size"],
            index=[f"train-{i}" for i in range(len(raw))],
        )
        new_data = pd.DataFrame(
            new_data,
            columns=data.columns,
            index=[f"new-{i}" for i in range(len(new_data))],
        )
    pipeline = Pipeline([
        ("observe", ObservationMasker()),
        (
            "encode",
            AutoEncoder(column_types=["categorical", "continuous"], binary="both"),
        ),
        (
            "model",
            VBPCA(
                1,
                maxiters=8,
                niter_broadprior=0,
                verbose=0,
                random_state=23,
                num_cpu=1,
                runtime_tuning="off",
            ),
        ),
    ])
    if output != "default":
        pipeline.set_output(transform=output)
        if output == "polars":
            # Raw object observations stay in pandas; numerical output uses Polars.
            pipeline.named_steps["observe"].set_output(transform="pandas")
    splitter = KFold(n_splits=3, shuffle=True, random_state=23)
    with config_context(enable_metadata_routing=True):
        result = cross_validate(
            pipeline,
            data,
            cv=splitter,
            params={"observation_mask": mask},
            error_score="raise",
            return_estimator=True,
        )
        search = GridSearchCV(
            pipeline, {"model__n_components": [0, 1]}, cv=splitter, error_score="raise"
        ).fit(data, observation_mask=mask)
        fitted = search.best_estimator_
        projected = fitted.transform(new_data, observation_mask=new_mask)
        # The masker has no inverse. Reverse the fitted model and encoder only.
        reconstructed = fitted[1:].inverse_transform(projected)
    if not np.isfinite(result["test_score"]).all():
        raise RuntimeError("The synthetic CV example returned a nonfinite score")
    if projected.shape != (len(new_data), search.best_params_["model__n_components"]):
        raise RuntimeError("The synthetic projection has the wrong shape")
    if reconstructed.shape != new_data.shape:
        raise RuntimeError("Reconstruction must use the original variable layout")
    if not np.isfinite(reconstructed[:, 1].astype(float)).all():
        raise RuntimeError("Reconstructed continuous values must be finite")
    return {
        "data": data,
        "mask": mask,
        "new_data": new_data,
        "new_mask": new_mask,
        "pipeline": pipeline,
        "splitter": splitter,
        "cv": result,
        "search": search,
        "projected": projected,
        "reconstructed": reconstructed,
    }


def export_gallery(result, output_dir):
    """Render learned fold offsets and original-unit reconstruction summaries."""
    import matplotlib.pyplot as plt

    fitted = result["search"].best_estimator_
    encoder = fitted.named_steps["encode"]
    schema = encoder.encoding_schema_
    schema.validate(len(encoder.get_feature_names_out()), strict_numeric=True)
    offsets = [
        fold.named_steps["encode"].encoding_schema_.variables[1].offset[0]
        for fold in result["cv"]["estimator"]
    ]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(offsets)), offsets)
    ax.set(
        xlabel="Row-validation fold",
        ylabel="Training-only size offset (input units)",
        xticks=range(len(offsets)),
        title="Preprocessing is fitted on each training split",
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pipeline.png", dpi=130)
    plt.close(fig)
    summary = {
        "training_shape": list(result["data"].shape),
        "new_rows": len(result["new_data"]),
        "encoded_features": encoder.get_feature_names_out().tolist(),
        "schema_kinds": [v.kind for v in schema.variables],
        "fold_training_offsets": offsets,
        "selected_capacity": result["search"].best_params_["model__n_components"],
        "decoded_size_input_units": result["reconstructed"][:, 1]
        .astype(float)
        .tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", choices=["default", "pandas", "polars"], default="default"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run_example(args.output)
    fitted = result["search"].best_estimator_
    print(
        "Encoded features:",
        fitted.named_steps["encode"].get_feature_names_out().tolist(),
    )
    print("Selected capacity:", result["search"].best_params_["model__n_components"])
    print("Conditional score shape:", result["projected"].shape)
    print("Reconstructed rows (region, size):")
    print(result["reconstructed"])
    print("Masked Pipeline, row CV, GridSearchCV and new-row projection passed")
    if args.output_dir is not None:
        export_gallery(result, args.output_dir)


if __name__ == "__main__":
    main()
