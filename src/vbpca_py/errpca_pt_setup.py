# errpca_pt_setup.py
"""
Build script for the errpca_pt C++ extension.

Usage (from the directory containing this file and errpca_pt.cpp):

    python errpca_pt_setup.py build_ext --inplace

This will produce a Python extension module named `errpca_pt` that can be
imported as:

    from errpca_pt import errpca_pt
"""

from __future__ import annotations

from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Path to Eigen headers (adjust if you move Eigen)
ROOT_DIR = Path(__file__).resolve().parent
EIGEN_DIR = ROOT_DIR / "eigen" / "eigen-3.4.0"

ext_modules = [
    Pybind11Extension(
        "errpca_pt",
        ["errpca_pt.cpp"],
        include_dirs=[str(EIGEN_DIR)],
        cxx_std=14,          # C++14 is sufficient for this code
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="errpca_pt",
    version="0.1.0",
    description="Compute sparse matrix of reconstruction errors (X - A*S) using CSR sparsity.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

