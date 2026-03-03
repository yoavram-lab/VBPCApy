"""Coverage-focused tests for lightweight utility modules.

These tests target branches that were previously unexecuted to lift per-file
coverage above 90% without changing runtime behavior.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vbpca_py import _types
from vbpca_py._memory import (
    estimate_dense_bytes,
    exceeds_budget,
    format_bytes,
    resolve_max_dense_bytes,
)
from vbpca_py._rms import (
    RmsConfig,
    _validate_sparse_mask_matches_structure,
    compute_rms,
)
from vbpca_py._sparsity import validate_mask_compatibility


def test_types_aliases_exist() -> None:
    # Simple import/attribute presence check to execute type alias statements.
    assert _types.Array is not None
    assert _types.Matrix is not None


def test_memory_helpers_cover_branches() -> None:
    assert estimate_dense_bytes((2, 3), np.float32) == 24

    # Small value returns immediately; large value walks through units to TB.
    assert format_bytes(512) == "512.0 B"
    big_label = format_bytes(5 * 1024**4)
    assert big_label.endswith("TB")

    over, est = exceeds_budget((10, 10), np.float64, max_bytes=100)
    assert over is True and est > 100

    under, _ = exceeds_budget((1, 1), np.float64, max_bytes=None)
    assert under is False

    assert resolve_max_dense_bytes("128") == 128
    assert resolve_max_dense_bytes(None, default=256) == 256


def test_validate_mask_compatibility_incompat_sparse_and_dense() -> None:
    dense_data = np.ones((2, 2))
    sparse_mask = sp.csr_matrix(np.ones((2, 2)))
    with pytest.raises(ValueError):
        validate_mask_compatibility(dense_data, sparse_mask)


def test_validate_mask_compatibility_incompat_dense_for_sparse() -> None:
    sparse_data = sp.csr_matrix(np.ones((2, 2)))
    dense_mask = np.ones((2, 2))
    with pytest.raises(ValueError):
        validate_mask_compatibility(sparse_data, dense_mask)


def test_validate_mask_compatibility_shape_mismatch_with_preflight() -> None:
    data = np.ones((2, 2))
    mask = np.ones((3, 2))
    preflight: list[dict[str, object]] = []
    with pytest.raises(ValueError):
        validate_mask_compatibility(data, mask, preflight=preflight, context="unit")
    assert preflight and preflight[0]["context"] == "unit"


def test_validate_sparse_mask_matches_structure_dense_and_sparse_masks() -> None:
    data = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])

    # Dense mask: wrong zeros on nonzero entries.
    bad_dense_mask = np.zeros((2, 2))
    with pytest.raises(ValueError):
        _validate_sparse_mask_matches_structure(data, bad_dense_mask)

    # Dense mask: wrong extra nonzeros.
    bad_dense_mask2 = np.ones((2, 2))
    with pytest.raises(ValueError):
        _validate_sparse_mask_matches_structure(data, bad_dense_mask2)

    # Sparse mask: indices differ.
    mask_wrong = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        _validate_sparse_mask_matches_structure(data, mask_wrong)

    # Valid sparse mask matches structure exactly.
    mask_ok = sp.csr_matrix(data)
    _validate_sparse_mask_matches_structure(data, mask_ok)


def test_validate_mask_compatibility_none_and_list_shapes() -> None:
    # Early return when mask is None.
    validate_mask_compatibility(np.ones((1, 1)), None)

    # Lists trigger the fallback shape detection path.
    with pytest.raises(ValueError):
        validate_mask_compatibility([[1, 2], [3, 4]], [[1, 2]])


def test_compute_rms_dense_error_paths() -> None:
    data = np.ones((2, 2))
    loadings = np.ones((2, 1))
    scores = np.ones((1, 2))

    with pytest.raises(ValueError):
        compute_rms(data, loadings, scores, mask=None, config=RmsConfig())

    bad_mask_shape = np.ones((3, 2))
    with pytest.raises(ValueError):
        compute_rms(data, loadings, scores, mask=bad_mask_shape, config=RmsConfig())

    empty_mask = np.zeros_like(data)
    with pytest.raises(ValueError):
        compute_rms(data, loadings, scores, mask=empty_mask, config=RmsConfig())

    with pytest.raises(ValueError):
        compute_rms(
            data,
            loadings,
            scores,
            mask=np.ones_like(data),
            config=RmsConfig(n_observed=1),
        )

    # Shape mismatch between loadings/scores triggers validation branch.
    with pytest.raises(ValueError):
        compute_rms(
            data,
            np.ones((3, 1)),
            np.ones((1, 2)),
            mask=np.ones_like(data),
            config=RmsConfig(),
        )


def test_compute_rms_sparse_mask_validation_and_mismatch() -> None:
    data = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    loadings = np.ones((2, 1))
    scores = np.ones((1, 2))
    cfg = RmsConfig(validate_sparse_mask=True)

    bad_mask_sparse = sp.csr_matrix([[1.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        compute_rms(data, loadings, scores, mask=bad_mask_sparse, config=cfg)

    bad_mask_dense = np.ones((2, 2))
    with pytest.raises(ValueError):
        compute_rms(data, loadings, scores, mask=bad_mask_dense, config=cfg)

    # Valid sparse flow.
    rms, err = compute_rms(data, loadings, scores, mask=None, config=cfg)
    assert np.isfinite(rms)
    assert isinstance(err, sp.csr_matrix)
