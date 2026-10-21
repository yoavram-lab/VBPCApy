"""Bounded probes for remaining VBPCA backend costs; no runtime code is changed."""

from __future__ import annotations

# ruff: noqa: PLC2701, SLF001
# Private implementation access is intentional in this diagnostic script.
import argparse
import hashlib
import inspect
import io
import json
import os
import platform
import sys
from collections import Counter
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

import vbpca_py
import vbpca_py._converge as conv
import vbpca_py._full_update as fu
import vbpca_py._pca_full as pc
from vbpca_py import VBPCA
from vbpca_py.model_selection import SelectionConfig, select_n_components

OFF = dict.fromkeys(
    ("angle", "earlystop", "rms_plateau", "cost", "composite", "slowing_down"), False
)
OPTIONS = {
    "maxiters": 4,
    "niter_broadprior": 0,
    "convergence_criteria": OFF,
    "runtime_tuning": "off",
    "num_cpu": 1,
    "compat_mode": "modern",
    "variance_update_order": "post_rotation",
    "random_state": 321,
    "verbose": 0,
    "return_diagnostics": False,
}


def paired_timed(reference, candidate, repeats=5):
    values = [reference(), candidate()]
    samples = [[], []]
    calls = [reference, candidate]
    for repeat in range(repeats):
        for index in (0, 1) if repeat % 2 == 0 else (1, 0):
            started = perf_counter()
            values[index] = calls[index]()
            samples[index].append(perf_counter() - started)
    return values[0], values[1], median(samples[0]), median(samples[1])


def contractions(a, s, av, sv, muv):
    p, q = a.shape
    n = s.shape[1]
    aa = np.einsum("ik,il->ikl", a, a).reshape(p, q * q)
    ss = np.einsum("kj,lj->jkl", s, s).reshape(n, q * q)
    result = aa @ sv.reshape(n, q * q).T
    result += av.reshape(p, q * q) @ ss.T
    result += av.reshape(p, q * q) @ sv.reshape(n, q * q).T
    result += muv
    return result


def complete_minus_missing(state, mask):
    a, s = state.loadings, state.scores
    av = np.stack(state.loading_covariances)
    sv = np.stack(state.score_covariances)
    sum_av, sum_sv = av.sum(axis=0), sv.sum(axis=0)
    total = float(
        np.sum((a.T @ a) * sum_sv)
        + np.sum((s @ s.T) * sum_av)
        + np.sum(sum_av * sum_sv)
    )
    ix, jx = np.nonzero(~mask)
    if ix.size:
        total -= float(
            fu._noise_sxv_sum_ext(
                ix=ix.astype(np.int32),
                jx=jx.astype(np.int32),
                loadings=a,
                scores=s,
                sv_by_col=sv,
                loading_covariances=av,
                num_cpu=1,
            )
        )
    if state.mu_variances.size:
        total += float(np.sum(mask.sum(axis=1) * state.mu_variances[:, 0]))
    return total


def numeric_probes(args):
    records = []
    for q in args.components:
        rng = np.random.default_rng(321)
        x = rng.normal(size=(args.features, args.samples))
        mask = rng.random(x.shape) >= 0.01
        mask[0, :] = True
        mask[:, 0] = True
        fitted = pc.pca_full(x, q, mask=mask, **OPTIONS)
        state = fu.NoiseState(
            loadings=fitted["A"],
            scores=fitted["S"],
            loading_covariances=fitted["Av"],
            score_covariances=fitted["Sv"],
            mu_variances=fitted["Muv"],
            pattern_index=fitted["Isv"],
            n_data=float(mask.sum()),
            noise_var=float(fitted["V"]),
            sparse_num_cpu=1,
        )
        ix, jx = np.nonzero(mask)
        reference, candidate, seconds, candidate_seconds = paired_timed(
            lambda state=state, ix=ix, jx=jx: fu._update_noise_variance(
                state, 0.0, ix, jx
            )[1],
            lambda state=state, mask=mask: complete_minus_missing(state, mask),
        )
        np.testing.assert_allclose(candidate, reference, rtol=1e-10, atol=1e-10)
        final = SimpleNamespace(
            a=fitted["A"],
            s=fitted["S"],
            av=fitted["Av"],
            sv=fitted["Sv"],
            pattern_index=fitted["Isv"],
            muv=fitted["Muv"],
        )
        if final.pattern_index is not None:
            raise ValueError("These prototypes require pattern-free VB covariances")
        expected, actual, var_seconds, fast_seconds = paired_timed(
            lambda final=final: pc._marginal_variance(final),
            lambda final=final: contractions(
                final.a, final.s, np.stack(final.av), np.stack(final.sv), final.muv
            ),
            repeats=3,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        record = {
            "q": q,
            "shape": [args.features, args.samples],
            "noise_current_seconds": seconds,
            "noise_prototype_seconds": candidate_seconds,
            "noise_speed_ratio": seconds / candidate_seconds,
            "noise_absolute_difference": abs(candidate - reference),
            "variance_current_seconds": var_seconds,
            "variance_prototype_seconds": fast_seconds,
            "variance_speed_ratio": var_seconds / fast_seconds,
            "variance_max_absolute_difference": float(np.max(abs(actual - expected))),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    return records


def orchestration_probes():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(24, 36))
    x[rng.random(x.shape) < 0.2] = np.nan
    mask_conversions = Counter()
    original = np.asarray

    def traced_asarray(a, *args, **kwargs):
        value = original(a, *args, **kwargs)
        if (
            isinstance(a, np.ndarray)
            and a.dtype == bool
            and a.shape == x.shape
            and value.dtype == np.float64
            and not np.shares_memory(a, value)
        ):
            frame = inspect.currentframe().f_back
            mask_conversions[frame.f_code.co_name] += 1
        return value

    np.asarray = traced_asarray
    try:
        pc.pca_full(x, 4, **OPTIONS)
    finally:
        np.asarray = original
    cost_calls = 0
    original_cost = conv.compute_full_cost

    def traced_cost(*args, **kwargs):
        nonlocal cost_calls
        cost_calls += 1
        return original_cost(*args, **kwargs)

    conv.compute_full_cost = traced_cost
    try:
        select_n_components(
            x,
            components=[2, 4],
            config=SelectionConfig(
                metric="prms", compute_explained_variance=False, return_best_model=False
            ),
            record_cost=False,
            **OPTIONS,
        )
    finally:
        conv.compute_full_cost = original_cost
    sparse = sp.csr_matrix(np.nan_to_num(x))
    model = VBPCA(4, max_dense_bytes=1, **{**OPTIONS, "return_diagnostics": True}).fit(
        sparse
    )
    sparse_views = Counter()
    original_csc = sp.csr_matrix.tocsc

    def traced_csc(self, *args, **kwargs):
        frame = inspect.currentframe().f_back
        sparse_views[frame.f_code.co_name] += 1
        return original_csc(self, *args, **kwargs)

    sp.csr_matrix.tocsc = traced_csc
    try:
        pc.pca_full(sparse, 4, rotate2pca=False, **OPTIONS)
    finally:
        sp.csr_matrix.tocsc = original_csc
    return {
        "mask_float64_conversion_calls": dict(mask_conversions),
        "iterations": OPTIONS["maxiters"],
        "prms_two_candidate_cost_calls_despite_record_cost_false": cost_calls,
        "sparse_diagnostics_budget": {
            "configured_bytes": 1,
            "reconstruction_bytes": model.reconstruction_.nbytes,
            "latent_variance_bytes": model.variance_.nbytes,
            "predictive_variance_bytes": model.predictive_variance_.nbytes,
        },
        "csr_to_csc_calls_rotation_off": dict(sparse_views),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--components", type=int, nargs="+", default=[8, 32])
    args = parser.parse_args()
    if (
        min(args.features, args.samples) < max(args.components)
        or min(args.components) < 1
    ):
        raise ValueError("Components must fit positive feature and sample dimensions")
    package = Path(vbpca_py.__file__).parent
    source_hashes = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in (
            "_pca_full.py",
            "_full_update.py",
            "_runtime_policy.py",
            "_rms.py",
            "estimators.py",
            "model_selection.py",
        )
    }
    numpy_build = io.StringIO()
    with redirect_stdout(numpy_build):
        np.show_config()
    result = {
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "thread_environment": {
                key: value
                for key, value in os.environ.items()
                if key.startswith(("OMP_", "OPENBLAS_", "MKL_", "VBPCA_"))
                or key in {"SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"}
            },
            "numpy_build": numpy_build.getvalue(),
        },
        "versions": {name: version(name) for name in ("vbpca_py", "numpy", "scipy")},
        "source_hashes": source_hashes,
        "native_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for suffix in ("*.so", "*.pyd")
            for path in package.glob(suffix)
        },
        "numeric": numeric_probes(args),
        "orchestration": orchestration_probes(),
        "scope": "Local synthetic review probes and unadopted algebraic prototypes. Not a fitted-model stopping/default change, scientific qualification or a Rockfish speed guarantee.",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["orchestration"], indent=2))


if __name__ == "__main__":
    main()
