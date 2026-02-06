"""Build configuration for the vbpca_py package."""

import os
from pathlib import Path

import pybind11
from setuptools import Extension, find_packages, setup


def get_pybind_include() -> str:
    """Return the pybind11 include directory."""
    return pybind11.get_include()


def get_eigen_include_path() -> str:
    """Return the Eigen include directory.

    Priority order:
    1. EIGEN_INCLUDE_DIR env var
    2. $CONDA_PREFIX/include/eigen3 (Conda/Mamba)
    3. /opt/homebrew/include/eigen3 (Homebrew on Apple Silicon)
    4. /usr/local/include/eigen3 (fallback)
    """
    env_path = os.environ.get("EIGEN_INCLUDE_DIR")
    if env_path and pathlib.Path(env_path).exists():
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
]

setup(
    name="vbpca_py",
    version="0.1.0",
    description=(
        "Variational Bayesian PCA (Illin & Raiko 2010) with support for missing data."
    ),
    author="Shany Naim and Joshua Macdonald",
    author_email="shany215.sn@gmail.com, jmacdo16@jh.edu",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    install_requires=[
        "numpy",
        "scipy",
    ],
    python_requires=">=3.11",
    zip_safe=False,
    include_package_data=True,
)
