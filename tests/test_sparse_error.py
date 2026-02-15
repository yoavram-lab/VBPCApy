"""Unit tests for sparse-error thread-count resolution helpers."""

from __future__ import annotations

import os

from vbpca_py._sparse_error import _resolve_num_cpu


def test_resolve_num_cpu_prefers_positive_argument() -> None:
    os.environ["VBPCA_NUM_THREADS"] = "7"
    try:
        assert _resolve_num_cpu(3) == 3
    finally:
        os.environ.pop("VBPCA_NUM_THREADS", None)


def test_resolve_num_cpu_reads_env_when_non_positive() -> None:
    os.environ["VBPCA_NUM_THREADS"] = "5"
    try:
        assert _resolve_num_cpu(0) == 5
        assert _resolve_num_cpu(-1) == 5
    finally:
        os.environ.pop("VBPCA_NUM_THREADS", None)


def test_resolve_num_cpu_falls_back_to_one() -> None:
    os.environ["VBPCA_NUM_THREADS"] = "not_an_int"
    try:
        assert _resolve_num_cpu(0) == 1
    finally:
        os.environ.pop("VBPCA_NUM_THREADS", None)
