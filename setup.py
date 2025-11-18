import os

import pybind11
from setuptools import Extension, find_packages, setup


# Helper function to get the include paths for pybind11
def get_pybind_include():
    return pybind11.get_include()


# Dynamically determine the Eigen include path
def get_eigen_include_path():
    # Check CONDA_PREFIX environment variable set by Conda/Mamba
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        # Assumes Conda installs to <CONDA_PREFIX>/include/eigen3
        potential_path = os.path.join(conda_prefix, "include", "eigen3")
        if os.path.exists(potential_path):
            return potential_path

    # Fallback paths (e.g., standard system paths, Homebrew default on macOS)
    # You can add more fallbacks here as needed for other systems
    return "/usr/local/include/eigen3"


# Define the C++ extensions
ext_modules = [
    Extension(
        "vbpca_py.errpca_pt",  # Module name
        ["src/vbpca_py/errpca_pt.cpp"],  # Source file
        include_dirs=[
            get_pybind_include(),  # pybind11 include path
            get_eigen_include_path(),  # Dynamically found Eigen path
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14"],  # Optimization and C++ standard
    ),
    Extension(
        "vbpca_py.subtract_mu_from_sparse",  # Module name
        ["src/vbpca_py/subtract_mu_from_sparse.cpp"],  # Source file
        include_dirs=[
            get_pybind_include()  # pybind11 include path
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],  # Optimization and C++ standard
    ),
]

# Setup configuration
setup(
    name="vbpca_py",
    version="0.1.0",
    description="A Python package with C++ extensions for PCA",
    author="Shany",
    author_email="shany215.sn@gmail.com",
    packages=find_packages(where="src"),  # Locate Python packages under "src"
    package_dir={"": "src"},  # Set the root directory for the packages
    ext_modules=ext_modules,  # Add the C++ extensions
    install_requires=["pybind11", "numpy", "scipy"],
    zip_safe=False,  # Avoid setuptools packaging the project into .egg files
)
