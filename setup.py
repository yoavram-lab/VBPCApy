"""Build configuration for the vbpca_py package."""

import os
from pathlib import Path

import pybind11
from setuptools import Extension, setup


def get_pybind_include() -> str:
    """Return the pybind11 include directory."""
    return pybind11.get_include()


def get_eigen_include_path() -> str:
    """Return the Eigen include directory.

    Priority order:
    1. EIGEN_INCLUDE_DIR env var
    2. $CONDA_PREFIX/include/eigen3 (Conda/Mamba)
    3. /opt/homebrew/include/eigen3 (Homebrew on Apple Silicon)
    4. /usr/include/eigen3 (Ubuntu/Debian standard location)
    5. /usr/local/include/eigen3 (fallback)
    """
    env_path = os.environ.get("EIGEN_INCLUDE_DIR")
    if env_path and Path(env_path).exists():
        return env_path

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        potential_path = Path(conda_prefix, "include", "eigen3")
        if potential_path.exists():
            return str(potential_path)

    # Homebrew (Apple Silicon)
    hb_path = Path("/opt/homebrew/include/eigen3")
    if hb_path.exists():
        return str(hb_path)

    # Ubuntu/Debian standard location
    debian_path = Path("/usr/include/eigen3")
    if debian_path.exists():
        return str(debian_path)

    # Fallback
    return "/usr/local/include/eigen3"


ext_modules = [
    Extension(
        name="vbpca_py.errpca_pt",
        sources=["src/vbpca_py/errpca_pt.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_eigen_include_path(),
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14"],
    ),
    Extension(
        name="vbpca_py.subtract_mu_from_sparse",
        sources=["src/vbpca_py/subtract_mu_from_sparse.cpp"],
        include_dirs=[
            get_pybind_include(),
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
    ),
    Extension(
        name="vbpca_py.sparse_update_kernels",
        sources=["src/vbpca_py/sparse_update_kernels.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_eigen_include_path(),
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14"],
    ),
    Extension(
        name="vbpca_py.dense_update_kernels",
        sources=["src/vbpca_py/dense_update_kernels.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_eigen_include_path(),
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14"],
    ),
    Extension(
        name="vbpca_py.noise_update_kernels",
        sources=["src/vbpca_py/noise_update_kernels.cpp"],
        include_dirs=[
            get_pybind_include(),
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14"],
    ),
    Extension(
        name="vbpca_py.rotate_update_kernels",
        sources=["src/vbpca_py/rotate_update_kernels.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_eigen_include_path(),
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14"],
    ),
]

setup(
    ext_modules=ext_modules,
)
