"""Regression tests for dense native input views."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from vbpca_py import dense_update_kernels as kernels


def _pointer(array: np.ndarray) -> int:
    return int(array.__array_interface__["data"][0])


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    ("mask_dtype", "dtype_name"), [(bool, "bool"), (np.uint8, "uint8")]
)
def test_native_view_diagnostic_reports_original_buffers(
    order: str,
    mask_dtype: type,
    dtype_name: str,
) -> None:
    x = np.array(np.arange(20, dtype=float).reshape(4, 5), order=order)
    mask = np.array(x > 3, dtype=mask_dtype, order=order)

    info = kernels.inspect_dense_input_views(x_data=x, mask=mask)

    assert info["x_pointer"] == _pointer(x)
    assert info["mask_pointer"] == _pointer(mask)
    assert info["x_strides_elements"] == tuple(
        stride // x.itemsize for stride in x.strides
    )
    assert info["mask_strides_bytes"] == mask.strides
    assert info["mask_dtype"] == dtype_name
    assert info[f"x_{order.lower()}_contiguous"]
    assert info[f"mask_{order.lower()}_contiguous"]


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("mask_dtype", [bool, np.uint8])
def test_dense_updates_accept_compact_masks_without_equation_drift(
    order: str,
    mask_dtype: type,
) -> None:
    rng = np.random.default_rng(208)
    x_base = rng.normal(size=(7, 9))
    mask_base = rng.random(x_base.shape) > 0.25
    loadings_base = rng.normal(size=(7, 3))
    scores_base = rng.normal(size=(3, 9))
    prior_base = np.diag([0.2, 0.3, 0.4])

    reference_score = kernels.score_update_dense_masked_nopattern(
        x_data=np.array(x_base, order="C"),
        mask=np.array(mask_base, dtype=float, order="C"),
        loadings=np.array(loadings_base, order="C"),
        loading_covariances=None,
        noise_var=0.5,
        return_covariances=False,
        num_cpu=1,
    )["scores"]
    reference_loadings = kernels.loadings_update_dense_masked_nopattern(
        x_data=np.array(x_base, order="C"),
        mask=np.array(mask_base, dtype=float, order="C"),
        scores=np.array(scores_base, order="C"),
        score_covariances=None,
        prior_prec=np.array(prior_base, order="C"),
        noise_var=0.5,
        return_covariances=False,
        num_cpu=1,
    )["loadings"]

    score_result = kernels.score_update_dense_masked_nopattern(
        x_data=np.array(x_base, order=order),
        mask=np.array(mask_base, dtype=mask_dtype, order=order),
        loadings=np.array(loadings_base, order=order),
        loading_covariances=None,
        noise_var=0.5,
        return_covariances=False,
        num_cpu=2,
    )
    loading_result = kernels.loadings_update_dense_masked_nopattern(
        x_data=np.array(x_base, order=order),
        mask=np.array(mask_base, dtype=mask_dtype, order=order),
        scores=np.array(scores_base, order=order),
        score_covariances=None,
        prior_prec=np.array(prior_base, order=order),
        noise_var=0.5,
        return_covariances=False,
        num_cpu=2,
    )

    assert_allclose(score_result["scores"], reference_score, rtol=1e-13, atol=1e-14)
    assert_allclose(
        loading_result["loadings"],
        reference_loadings,
        rtol=1e-13,
        atol=1e-14,
    )


def test_native_views_reject_hidden_layout_copy() -> None:
    x = np.arange(48, dtype=float).reshape(6, 8)[:, ::2]
    mask = np.ones(x.shape, dtype=bool)

    with pytest.raises(ValueError, match="C- or Fortran-contiguous"):
        kernels.inspect_dense_input_views(x_data=x, mask=mask)


def test_native_views_reject_ambiguous_mask_dtype() -> None:
    x = np.ones((4, 5), dtype=float)
    mask = np.ones(x.shape, dtype=np.int32)

    with pytest.raises(ValueError, match="mask dtype"):
        kernels.inspect_dense_input_views(x_data=x, mask=mask)
