"""Run the actual sparse-construction source included in the documentation."""

import runpy
from pathlib import Path

import numpy as np


def test_documented_stored_zero_construction(capsys):
    script = Path(__file__).parents[1] / "scripts/snippets/stored_observations.py"
    namespace = runpy.run_path(str(script))
    matrix = namespace["x_sparse"]
    assert matrix.shape == (2, 2)
    assert matrix.nnz == 4
    assert np.count_nonzero(matrix.data == 0) == 2
    assert capsys.readouterr().out == "4\n"
