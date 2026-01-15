import os
import platform
import subprocess
from typing import List

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
    3. /usr/local/include/eigen3 (fallback)
    """
    # 1. Explicit override
    env_path = os.environ.get("EIGEN_INCLUDE_DIR")
    if env_path and os.path.exists(env_path):
        return env_path

    # 2. Conda/Mamba
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        potential_path = os.path.join(conda_prefix, "include", "eigen3")
        if os.path.exists(potential_path):
            return potential_path

    # 3. Fallback
    return "/usr/local/include/eigen3"


def get_macos_sdk_path() -> str | None:
    """Return macOS SDK path if available."""
    sdkroot = os.environ.get("SDKROOT")
    if sdkroot and os.path.exists(sdkroot):
        return sdkroot
    try:
        output = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return output if output and os.path.exists(output) else None


def get_macos_compile_args() -> List[str]:
    """Return extra compile args to ensure libc++ headers are found on macOS."""
    if platform.system() != "Darwin":
        return []
    sdk_path = get_macos_sdk_path()
    if not sdk_path:
        return []
    libcxx_path = os.path.join(sdk_path, "usr", "include", "c++", "v1")
    args = [f"-isysroot{sdk_path}"]
    if os.path.exists(libcxx_path):
        args.append(f"-isystem{libcxx_path}")
    return args


MACOS_COMPILE_ARGS = get_macos_compile_args()


ext_modules = [
    Extension(
        name="vbpca_py.errpca_pt",
        sources=["src/vbpca_py/errpca_pt.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_eigen_include_path(),
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14", *MACOS_COMPILE_ARGS],
    ),
    Extension(
        name="vbpca_py.subtract_mu_from_sparse",
        sources=["src/vbpca_py/subtract_mu_from_sparse.cpp"],
        include_dirs=[
            get_pybind_include(),
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++11", *MACOS_COMPILE_ARGS],
    ),
]

setup(
    name="vbpca_py",
    version="0.1.0",
    description="Variational Bayesian PCA (Illin & Raiko 2010) with support for missing data.",
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
