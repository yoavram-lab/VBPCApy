"""Empty latent algebra must return safely through the native boundary."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


def _run_isolated(code: str, *arguments: str) -> None:
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    completed = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", textwrap.dedent(code), *arguments],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, (
        f"subprocess exit {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_empty_native_eigensystem_and_validation_in_subprocess():
    # A signal regression must not terminate the entire pytest process.
    _run_isolated("""
        import numpy as np
        from vbpca_py.rotate_update_kernels import weighted_cov_eigh_psd

        for count in (0, 1, 3):
            base = np.empty((0, 0))
            covs = np.empty((count, 0, 0))
            weights = np.arange(count, dtype=float)
            for array in (base, covs, weights):
                array.setflags(write=False)
            result = weighted_cov_eigh_psd(base, covs, weights, 3.0)
            for key, shape in (("cov", (0, 0)), ("eigvals", (0,)),
                               ("eigvecs", (0, 0))):
                assert result[key].shape == shape
                assert result[key].dtype == np.float64

        invalid = (
            (np.empty((0, 0)), np.empty((1, 0, 0)), np.ones(1), 0.0),
            (np.empty((0, 0)), np.empty((1, 0, 1)), np.ones(1), 1.0),
            (np.empty((0, 0)), np.empty((1, 0, 0)), np.ones(2), 1.0),
            (np.empty((0, 0)), np.empty((1, 0)), np.ones(1), 1.0),
            (np.empty((0, 0)), np.empty((1, 0, 0)), np.ones((1, 1)), 1.0),
            (np.empty((0, 1)), np.empty((1, 0, 0)), np.ones(1), 1.0),
        )
        for arguments in invalid:
            try:
                weighted_cov_eigh_psd(*arguments)
            except ValueError:
                pass
            else:
                raise AssertionError("rank zero bypassed input validation")
    """)


@pytest.mark.parametrize("algorithm", ["ppca", "map", "vb"])
@pytest.mark.parametrize("compat_mode", ["strict_legacy", "modern"])
def test_rank_zero_solver_rotation_in_subprocess(algorithm, compat_mode):
    _run_isolated(
        """
        import sys
        import numpy as np
        from scipy import sparse
        from vbpca_py._pca_full import pca_full

        algorithm, compat_mode = sys.argv[1:]
        rng = np.random.default_rng(30)
        data = rng.normal(size=(8, 18))
        data[0, 0] = 0.0
        mask = rng.random(data.shape) > 0.25
        mask[0, 0] = True
        rows, cols = np.nonzero(mask)
        stored = sparse.coo_matrix(
            (data[rows, cols], (rows, cols)), shape=data.shape
        ).tocsr()
        masked = np.where(mask, data, np.nan)
        for kind, table in (("dense", data), ("masked", masked), ("sparse", stored)):
            original = table.copy()
            for rotate in (False, True):
                print(kind, algorithm, compat_mode, rotate, flush=True)
                result = pca_full(
                    table, 0, algorithm=algorithm, compat_mode=compat_mode,
                    maxiters=4, niter_broadprior=0, random_state=31,
                    runtime_tuning="off", num_cpu=1, return_diagnostics=False,
                    record_cost=True, rotate2pca=rotate,
                )
                assert result["A"].shape == (8, 0)
                assert result["S"].shape == (0, 18)
                assert np.isfinite(result["Mu"]).all()
                assert np.isfinite(result["V"]) and result["V"] > 0
                assert np.isfinite(result["RMS"])
                assert result["Xrec"] is None
                if sparse.issparse(table):
                    for name in ("data", "indices", "indptr"):
                        np.testing.assert_array_equal(
                            getattr(table, name), getattr(original, name)
                        )
                else:
                    np.testing.assert_array_equal(table, original)
        """,
        algorithm,
        compat_mode,
    )
