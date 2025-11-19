# errpca_pt_setup.py
"""
Build script for the errpca_pt C++ extension.

Usage (from the directory containing this file and errpca_pt.cpp):

    python errpca_pt_setup.py build_ext --inplace

This will produce a Python extension module named `errpca_pt` that you
can import as:

    from errpca_pt import errpca_pt

Eigen headers are expected to be available on the system include path,
or provided explicitly via the EIGEN_INCLUDE_DIR environment variable.

Common Eigen install locations:

  • Homebrew (Apple Silicon or Intel):
        export EIGEN_INCLUDE_DIR=/opt/homebrew/include/eigen3
        export EIGEN_INCLUDE_DIR=/usr/local/include/eigen3

  • Conda/mamba:
        export EIGEN_INCLUDE_DIR="$CONDA_PREFIX/include/eigen3"

  • pip-installed Eigen:
        pip install eigen
        # Usually installs under site-packages/eigen/include/eigen3
        # The extension will pick it up automatically if on INCLUDE path.

If Eigen is not found, the build will fail with a clear error.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


def find_default_eigen() -> list[str]:
    """Return a list of candidate Eigen include directories (minimal + illustrative)."""
    candidates = []

    # 1. User override
    env_path = os.environ.get("EIGEN_INCLUDE_DIR")
    if env_path:
        candidates.append(env_path)

    # 2. Conda/mamba environments
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(str(Path(conda_prefix) / "include" / "eigen3"))

    # 3. Homebrew defaults
    candidates.extend([
        "/opt/homebrew/include/eigen3",   # Apple Silicon
        "/usr/local/include/eigen3",      # Intel macOS / Linux brew
    ])

    # 4. pip-installed Eigen (lightweight heuristic)
    for p in sys.path:
        base = Path(p)
        c1 = base / "eigen" / "include" / "eigen3"
        if c1.exists():
            candidates.append(str(c1))

    # Filter only existing paths
    return [c for c in candidates if Path(c).exists()]


# Resolve Eigen include dirs
eigen_includes = find_default_eigen()
if not eigen_includes:
    raise RuntimeError(
        "Eigen headers not found. Set EIGEN_INCLUDE_DIR to the directory "
        "containing the 'eigen3' folder (e.g., /opt/homebrew/include/eigen3)."
    )

ext_modules = [
    Extension(
        "errpca_pt",
        ["errpca_pt.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            *eigen_includes,
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]


setup(
    name="errpca_pt",
    version="0.2",
    description="ERRPCA_PT: Sparse reconstruction error computation for PPCA",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=["pybind11", "numpy", "scipy"],
    zip_safe=False,
)


