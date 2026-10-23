"""Execute the documented mixed-data workflow and check independent outcomes."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import multivariate_normal
from sklearn import config_context
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

SCRIPT = Path(__file__).parents[1] / "scripts/example_masked_pipeline.py"


def _observed_numeric(data, mask):
    values = np.asarray(data, dtype=object)[:, 1]
    observed = mask[:, 1] & ~pd.isna(values)
    return values[observed].astype(float)


def _row_score_reference(fitted, data, mask):
    observed = fitted.named_steps["observe"].transform(data, mask=mask)
    encoded = np.asarray(fitted.named_steps["encode"].transform(observed), dtype=float)
    model = fitted.named_steps["model"]
    scores = []
    for row in encoded:
        keep = ~np.isnan(row)
        loadings = model.components_[:, keep].T
        covariance = loadings @ loadings.T + model.noise_variance_ * np.eye(keep.sum())
        scores.append(
            multivariate_normal.logpdf(row[keep], model.mean_[keep], covariance)
            if keep.any()
            else 0.0
        )
    return np.mean(scores)


@pytest.mark.parametrize("output", ["default", "pandas", "polars"])
def test_actual_pipeline_example_routes_folds_and_decodes_new_rows(output):
    result = runpy.run_path(str(SCRIPT))["run_example"](output)
    data, mask = result["data"], result["mask"]
    new_data, new_mask = result["new_data"], result["new_mask"]
    assert data.shape == (36, 2)
    assert new_data.shape == (4, 2)
    assert isinstance(
        result["projected"],
        {"default": np.ndarray, "pandas": pd.DataFrame, "polars": pl.DataFrame}[output],
    )
    with pytest.raises(NotFittedError):
        check_is_fitted(result["pipeline"].named_steps["model"])
    copied = clone(result["pipeline"])
    assert copied.get_params()["model__n_components"] == 1
    fitted = result["search"].best_estimator_
    encoder, model = fitted.named_steps["encode"], fitted.named_steps["model"]
    schema = encoder.encoding_schema_
    schema.validate(model.n_features_in_, strict_numeric=True)
    np.testing.assert_allclose(
        schema.variables[1].offset, _observed_numeric(data, mask).mean()
    )
    assert set(schema.variables[0].categories) == {"west", "east", "north"}
    assert "unobserved placeholder" not in schema.variables[0].categories
    if output != "default":
        assert (
            list(result["projected"].columns) == fitted.get_feature_names_out().tolist()
        )
        assert not new_data.index.isin(data.index).any()
        if output == "pandas":
            assert result["projected"].index.equals(new_data.index)
    # CV learns offsets from its training rows; scoring uses its held-out rows.
    with config_context(enable_metadata_routing=True):
        for index, (training, test) in enumerate(result["splitter"].split(data)):
            fold = result["cv"]["estimator"][index]
            variable = fold.named_steps["encode"].encoding_schema_.variables[1]
            np.testing.assert_allclose(
                variable.offset,
                _observed_numeric(np.asarray(data)[training], mask[training]).mean(),
            )
            test_data = data.iloc[test] if output != "default" else data[test]
            reference = _row_score_reference(fold, test_data, mask[test])
            np.testing.assert_allclose(result["cv"]["test_score"][index], reference)
        components, training_scores = model.components_.copy(), model.scores_.copy()
        repeated = fitted.transform(new_data, observation_mask=new_mask)
        np.testing.assert_allclose(
            np.asarray(repeated), np.asarray(result["projected"])
        )
    np.testing.assert_array_equal(model.components_, components)
    np.testing.assert_array_equal(model.scores_, training_scores)
    conditional = np.asarray(result["projected"])
    reconstructed = result["reconstructed"]
    assert reconstructed.shape == new_data.shape
    assert set(reconstructed[:, 0]) <= set(schema.variables[0].categories)
    numeric = schema.variables[1]
    encoded_prediction = conditional @ components + model.mean_
    reference = (
        encoded_prediction[:, numeric.columns[0]] / numeric.weight + numeric.offset[0]
    )
    np.testing.assert_allclose(reconstructed[:, 1].astype(float), reference)
    # The completely missing new row receives the latent prior mean.
    np.testing.assert_array_equal(conditional[1], 0)
