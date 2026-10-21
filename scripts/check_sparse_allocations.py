"""Require sparse kernel allocations to stay independent of observation count.

Linux/glibc diagnostic using a temporary malloc/calloc/realloc interposer.
It counts allocations for the entire warmed native call, including Python
argument handling, copies and output packaging. The helper is never installed
or linked into production wheels. Numeric and worker-failure gates are separate.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median


def _worker():
    import numpy as np
    import scipy.sparse as sp

    import vbpca_py.sparse_update_kernels as kernels

    counter = ctypes.CDLL(os.environ["VBPCA_ALLOCATION_COUNTER"])
    counter.vbpca_allocation_start.restype = None
    counter.vbpca_allocation_stop.restype = ctypes.c_ulonglong
    records = []
    for rank in (8, 32):
        allocations = {}
        for density in (0.05, 0.8):
            rng = np.random.default_rng(294)
            features, samples = 193, 257
            x = sp.random(features, samples, density=density, format="csr", rng=rng)
            x.data[0] = 0.0  # Explicit observed zeros still execute the native loop.
            csc = x.tocsc()
            a = rng.normal(size=(features, rank))
            s = rng.normal(size=(rank, samples))
            cov_a = np.broadcast_to(np.eye(rank), (features, rank, rank)).copy()
            cov_s = np.broadcast_to(np.eye(rank), (samples, rank, rank)).copy()
            for phase, function, data, factors in (
                (
                    "score",
                    kernels.score_update_sparse_nopattern,
                    csc,
                    {"loadings": a, "loading_covariances": cov_a},
                ),
                (
                    "loadings",
                    kernels.loadings_update_sparse_nopattern,
                    x,
                    {
                        "scores": s,
                        "score_covariances": cov_s,
                        "prior_prec": np.eye(rank),
                    },
                ),
            ):
                kwargs = dict(
                    x_data=data.data,
                    x_indices=data.indices,
                    x_indptr=data.indptr,
                    noise_var=0.4,
                    return_covariances=True,
                    num_cpu=1,
                    **factors,
                )
                function(**kwargs)
                counts = []
                for _ in range(3):
                    counter.vbpca_allocation_start()
                    output = function(**kwargs)
                    counts.append(int(counter.vbpca_allocation_stop()))
                    del output
                allocations[phase, density] = median(counts)
                records.append({
                    "shape": [features, samples],
                    "rank": rank,
                    "nnz": x.nnz,
                    "density": density,
                    "phase": phase,
                    "allocation_counts": counts,
                    "native_threads": 1,
                })
        for phase in ("score", "loadings"):
            # Fixed-size setup/output allocation is allowed. An observed-cell
            # allocation would increase by tens of thousands in this contrast.
            if allocations[phase, 0.8] > allocations[phase, 0.05] + 8:
                raise RuntimeError(
                    f"{phase}, rank {rank}: allocations grow with nnz: {allocations}"
                )
    print(
        json.dumps({
            "cases": records,
            "native_sha256": hashlib.sha256(
                Path(kernels.__file__).read_bytes()
            ).hexdigest(),
            "scope": __doc__,
        })
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        _worker()
        return
    if sys.platform != "linux" or platform.libc_ver()[0] != "glibc":
        parser.error("this allocation diagnostic requires Linux/glibc")
    source = Path(__file__).resolve().parents[1] / "tools" / "allocation_counter.c"
    with tempfile.TemporaryDirectory(prefix="vbpca-allocations-") as directory:
        library = Path(directory) / "counter.so"
        subprocess.run(
            [
                "cc",
                "-std=c11",
                "-O2",
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(library),
            ],
            check=True,
        )
        env = dict(os.environ)
        env.update(
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            VBPCA_ALLOCATION_COUNTER=str(library),
            LD_PRELOAD=str(library)
            + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""),
        )
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            print(result.stderr, file=sys.stderr, end="")
            raise SystemExit(result.returncode)
    if args.output:
        args.output.write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
