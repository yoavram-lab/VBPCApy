"""Execute maintained narrative notebooks without optional notebook tooling."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELL_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def test_mixed_missing_table_notebook_executes() -> None:
    """Run code cells in order and verify the documented public-API contract."""
    path = ROOT / "examples" / "mixed-missing-table.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__notebook_example__"}

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(
        isinstance(cell_id, str) and CELL_ID.fullmatch(cell_id) for cell_id in cell_ids
    )
    assert len(set(cell_ids)) == len(cell_ids)
    markdown = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    ]
    assert any("entirely synthetic" in source for source in markdown)
    assert any("do not establish" in source for source in markdown)
    assert any("samples on rows" in source for source in markdown)

    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        code = compile(source, f"{path.name}::cell-{index}", "exec")
        exec(code, namespace)  # noqa: S102 - executable documentation contract

    table = np.asarray(namespace["table"])
    observed = np.asarray(namespace["observed"])
    encoded = np.asarray(namespace["encoded"])
    encoder = namespace["encoder"]
    model = namespace["model"]
    joint_scores = np.asarray(namespace["joint_training_scores"])
    scores = np.asarray(namespace["conditional_scores"])
    reconstructed = np.asarray(namespace["reconstructed_table"])
    assert table.shape == (90, 4)
    assert encoded.shape[0] == table.shape[0]
    assert model.n_features_in_ == encoded.shape[1]
    assert joint_scores.shape == (table.shape[0], 2)
    assert scores.shape == (table.shape[0], 2)
    assert reconstructed.shape == table.shape

    encoded_finite = np.isfinite(encoded)
    feature_groups = np.asarray(encoder.feature_groups_)
    for column in range(table.shape[1]):
        group = feature_groups == column
        expected = np.broadcast_to(
            observed[:, [column]], encoded_finite[:, group].shape
        )
        np.testing.assert_array_equal(encoded_finite[:, group], expected)

    missing_continuous = np.asarray(namespace["missing_continuous"])
    missing_habitat = np.asarray(namespace["missing_habitat"])
    assert np.count_nonzero(missing_continuous) > 0
    assert np.count_nonzero(missing_habitat) > 0

    continuous_mae = namespace["continuous_mae"]
    assert set(continuous_mae) == {
        "temperature_c",
        "nitrate_mg_l",
        "biomass_mg_m3",
    }
    assert all(np.isfinite(value) for value in continuous_mae.values())
    assert np.isfinite(namespace["habitat_accuracy"])

    fitted_habitats = set(table[observed[:, 3], 3])
    hidden_predictions = set(
        np.asarray(namespace["habitat_prediction"])[missing_habitat]
    )
    assert hidden_predictions <= fitted_habitats
