"""Keep layout migration separate from numerical compatibility policy."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from vbpca_py import VBPCA, PCAOptions
from vbpca_py.legacy import VBPCA as LegacyVBPCA  # noqa: N811


def run_example():
    """Demonstrate modern observations and explicit feature-first migration.

    Returns:
        Models fitted to the same small synthetic observations.
    """
    x = np.random.default_rng(19).normal(size=(8, 3))
    x[:, 0] *= 1e-10
    x[0, 1], x[1, 2] = 0.0, np.nan
    options = {
        "maxiters": 4,
        "niter_broadprior": 0,
        "random_state": 19,
        "num_cpu": 1,
        "runtime_tuning": "off",
        "runtime_report": True,
        "verbose": 0,
    }
    native = VBPCA(1, **options).fit(x)
    modern_legacy = LegacyVBPCA(1, compat_mode="modern", **options).fit(x.T)
    # Explicit legacy masks are authoritative: exclude existing NaNs too.
    mask = ~np.isnan(x.T)
    modern_masked = LegacyVBPCA(1, compat_mode="modern", **options).fit(x.T, mask=mask)
    historical = LegacyVBPCA(1, **options).fit(x.T)
    # Construct coordinates so the observed numeric zero remains stored.
    rows, columns = np.nonzero(~np.isnan(x))
    sparse = csr_matrix((x[rows, columns], (rows, columns)), shape=x.shape)
    modern_sparse = VBPCA(1, **options).fit(sparse)
    np.testing.assert_allclose(native.components_, modern_legacy.components_.T)
    np.testing.assert_allclose(
        modern_masked.mean_, modern_legacy.mean_, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        native.mean_, modern_sparse.mean_, rtol=1e-10, atol=1e-14
    )
    return {
        "native": native,
        "modern_legacy": modern_legacy,
        "modern_masked": modern_masked,
        "historical": historical,
        "modern_sparse": modern_sparse,
    }


def main() -> None:
    models = run_example()
    print("Native default:", models["native"].compat_mode)
    print("Legacy default:", models["historical"].compat_mode)
    print("Low-level configuration default:", PCAOptions().compat_mode.value)
    for name in ["native", "historical"]:
        report = models[name].runtime_report_
        print(name, report["bias_update_order"], report["variance_update_order"])
    print("Modern masked values and explicit native/legacy layouts agree")


if __name__ == "__main__":
    main()
