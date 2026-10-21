"""Exercise result annotations, examples and actual fit representations."""

from __future__ import annotations

import doctest
import os
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pytest
import scipy.sparse as sp

import vbpca_py
import vbpca_py.results as result_types
from vbpca_py._pca_full import pca_full


def test_public_return_annotation_can_be_resolved() -> None:
    assert get_type_hints(pca_full)["return"] is result_types.PCAResult


@pytest.mark.parametrize(
    "name",
    [
        "PCAResult",
        "LearningCurve",
        "Float64Array",
        "IndexArray",
        "PosteriorCovariances",
        "CovarianceResult",
        "HyperparameterResult",
    ],
)
def test_result_contracts_are_exported_from_the_package_root(name) -> None:
    assert name in vbpca_py.__all__
    assert getattr(vbpca_py, name) is getattr(result_types, name)


def test_result_docstring_examples() -> None:
    result = doctest.testmod(result_types, raise_on_error=True)
    assert result.attempted >= 7
    assert result.failed == 0


def test_pca_full_docstring_examples() -> None:
    runner = doctest.DocTestRunner()
    for example in doctest.DocTestFinder().find(pca_full, globs=pca_full.__globals__):
        runner.run(example)
    result = runner.summarize()
    assert result.attempted == 4
    assert result.failed == 0


@pytest.mark.parametrize("algorithm", ["vb", "ppca", "map"])
@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("patterns", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("writeback", ["python", "bulk"])
def test_fit_result_contract(
    algorithm, diagnostics, patterns, sparse, writeback
) -> None:
    rng = np.random.default_rng(93)
    data = rng.normal(size=(7, 9))
    mask = np.ones_like(data, dtype=bool)
    mask[1, 2:4] = False
    data[0, 0] = 0.0
    rows, columns = np.nonzero(mask)
    sparse_data = sp.csr_matrix(
        (data[rows, columns], (rows, columns)), shape=data.shape
    )
    result = pca_full(
        sparse_data if sparse else data,
        2,
        mask=sp.csr_matrix(mask) if sparse else mask,
        algorithm=algorithm,
        maxiters=3,
        niter_broadprior=0,
        random_state=93,
        uniquesv=patterns,
        return_diagnostics=diagnostics,
        record_prior_trace=True,
        cov_writeback_mode=writeback,
    )
    assert type(result) is dict
    assert result.keys() == result_types.PCAResult.__required_keys__
    for key in ("A", "S", "Mu", "Muv", "Va"):
        assert result[key].dtype == np.float64
    for key in ("Xrec", "Vr", "Vpred", "ExplainedVar", "ExplainedVarRatio"):
        if diagnostics:
            assert result[key].dtype == np.float64
        else:
            assert result[key] is None
    for key in ("Av", "Sv"):
        for block in result[key]:
            assert block.dtype == np.float64
    if result["Isv"] is not None:
        assert result["Isv"].dtype == np.dtype(int)
    assert result["cv"]["A"] is result["Av"]
    assert result["cv"]["S"] is result["Sv"]
    curve = result["lc"]
    assert curve.keys() <= result_types.LearningCurve.__annotations__.keys()
    assert isinstance(curve["returned_iteration"], int)
    assert isinstance(curve["convergence_reason"], str)
    assert all(isinstance(point, list) for point in curve.get("prior_trace", []))


def test_mypy_rejects_invalid_result_keys_and_metadata(tmp_path) -> None:
    source = tmp_path / "invalid_results.py"
    source.write_text(
        "from vbpca_py.results import LearningCurve, PCAResult\n"
        "def invalid(result: PCAResult, curve: LearningCurve) -> None:\n"
        "    result['loadings']\n"
        "    curve['returned_iteration'] = [1.0]\n"
        "    curve['rms'] = 'not a numeric history'\n"
    )
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["MYPYPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--cache-dir",
            str(tmp_path / "cache"),
            str(source),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert completed.stdout.count(" error:") == 3, completed.stdout
    assert 'no key "loadings"' in completed.stdout
    assert completed.stdout.count("[typeddict-item]") == 3, completed.stdout
